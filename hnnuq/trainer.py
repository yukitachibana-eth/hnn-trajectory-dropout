"""
trainer.py
Training loop for HNNs with dropout-based uncertainty (all arms of the paper).

Training rolls the learned Hamiltonian out with a (leapfrog) integrator and minimises the
trajectory MSE. Per optimiser step one mask index is drawn; Fixed-mask MC resamples the
bank index at every leapfrog sub-step when ``train_mask_resample == "per_substep"``.
"""

import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from hnnuq.config import ExperimentConfig
from hnnuq.integration import numerically_integrate, integrate_mc_samples
from hnnuq.metrics import UQMetrics, EnergyMetrics
from hnnuq.hnn_model import build_hnn_model
from hnnuq.masked_linear import discard_model_materialized_masks, get_model_active_mask_num
from hnnuq.tfim.control import register_tfim_operator_structure

# Fresh-mask semantics used for in-training evaluation (one weight-space mask per MC group).
_EVAL_FRESH_MASK_SEMANTICS = "weight_group"


def _unpack(data_like: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Split a split bundle (``.trajectories`` / ``.input_z0``) or a bare tensor.

    Returns the trajectory tensor (k, T, n_samples, 2*dim) and, if the dataset stores
    explicit initial-condition inputs, the (k, n_samples, 2*dim) tensor of model inputs.
    """
    return getattr(data_like, "trajectories", data_like), getattr(data_like, "input_z0", None)


def _initial_state(trajectory_t0: torch.Tensor, input_z0: Optional[torch.Tensor]) -> torch.Tensor:
    """Model input state: the dataset's explicit input if present, else the t=0 target."""
    return input_z0 if input_z0 is not None else trajectory_t0


class Trainer:
    """Training manager for HNNs with dropout variants."""

    def __init__(self, config: ExperimentConfig, device: torch.device):
        self.config = config
        self.device = device
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[ReduceLROnPlateau] = None
        self.metrics_history: Dict[str, List] = {
            "epoch": [],
            "train_loss": [],
            "train_rollout_loss": [],
            "mean_grad_norm": [],
            "val_loss": [],
            "val_loss_per_epoch": [],
            "test_mse": [],
            "test_nll": [],
            "test_crps": [],
            "test_coverage_90": [],
            "calibration_error": [],
            "energy_drift_mean": [],
            "energy_drift_final": [],
            "learning_rate": [],
            "learning_rate_per_epoch": [],
        }
        self.best_val_loss = float("inf")
        self.current_epoch = 0
        self._last_train_epoch_stats = {
            "train_rollout_loss": float("nan"),
            "mean_grad_norm": float("nan"),
        }

    # ------------------------------------------------------------------
    # Mask-bank helpers
    # ------------------------------------------------------------------

    def _bank_size(self) -> int:
        assert self.model is not None
        return get_model_active_mask_num(self.model)

    def _resample_mask_per_substep(self) -> bool:
        """True iff the bank index is redrawn at every leapfrog sub-step during training."""
        return self.config.train_mask_resample == "per_substep" and self.config.finite_mask_mode == "both"

    def _fresh_mask_index_fn(self, mask_num: int) -> Callable[[int], int]:
        """``mask_index_fn`` for numerically_integrate: a new random bank index per call.

        Drawn on the CPU (a CUDA randint + .item() would sync at every Hamiltonian evaluation).
        """

        def _draw(_eval_idx: int) -> int:
            return int(torch.randint(mask_num, (1,)).item())

        return _draw

    def _refresh_mask_banks(self) -> None:
        """Discard materialised mask banks; the next forward draws a new bank from the global RNG.

        Called at the start of every epoch and around the final test pass, so a materialised
        Fixed-mask bank is re-drawn once per epoch. Seeded banks are unaffected.
        """
        discard_model_materialized_masks(self.model)

    # ------------------------------------------------------------------
    # Model / optimiser
    # ------------------------------------------------------------------

    def build(self, dim: int):
        """Build model, optimiser and scheduler; ``dim`` = degrees of freedom (half the state)."""
        cfg = self.config
        self.model = build_hnn_model(
            dim=dim,
            n_hidden=cfg.n_hidden,
            drop_prob=cfg.drop_prob,
            mask_num=cfg.mask_num,
            nonlin=cfg.nonlin,
            device=self.device,
            model_type=cfg.model_type,
            dropout_variant=cfg.dropout_variant,
            finite_mask_mode=cfg.finite_mask_mode,
            concrete_temp=cfg.concrete_temp,
            scale_weights=cfg.scale_weights,
            mask_last_layer=cfg.mask_last_layer,
            finite_mask_storage=cfg.finite_mask_storage,
            mask_seed=cfg.seed,  # seeded banks are derived from the run seed
        )
        if cfg.model_type == "TFIMSeparableHNN":
            register_tfim_operator_structure(self.model, state_half_dim=dim)

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            [{"params": params, "lr": cfg.lr, "weight_decay": cfg.weight_decay}]
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=cfg.scheduler_patience, factor=cfg.scheduler_factor
        )
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {cfg.model_type}  parameters: {n_params:,}")
        print(f"Dropout: {cfg.dropout_variant}, p={cfg.drop_prob}, masks={cfg.mask_num}")

    def get_horizon(self, epoch: int) -> int:
        """Training rollout length at ``epoch``.

        Without curriculum: T. With curriculum: the first two thirds of the epochs before the
        final ``curriculum_final_full_epochs`` use T/4, T/2, 3T/4 in three stages (each at
        least T_init_seq), then the full horizon T.
        """
        T = self.config.T
        if not self.config.gradual_horizon_increase:
            return T
        T_init = self.config.T_init_seq
        final_full_epochs = self.config.curriculum_final_full_epochs
        full_horizon_start = max(0, int(self.config.n_epochs) - final_full_epochs)
        if full_horizon_start <= 0 or epoch >= full_horizon_start:
            return T
        stage_1_end = max(1, full_horizon_start // 3)
        stage_2_end = max(stage_1_end + 1, (2 * full_horizon_start) // 3)
        if epoch < stage_1_end:
            return min(T, max(T_init, T // 4))
        if epoch < stage_2_end:
            return min(T, max(T_init, T // 2))
        return min(T, max(T_init, int(T * 0.75)))

    # ------------------------------------------------------------------
    # Batched-over-k helpers
    # ------------------------------------------------------------------

    def _can_batch_over_k(self, k_array: Optional[torch.Tensor], compute_uq: bool = False) -> bool:
        """Whether all control cells can be rolled out in a single batched call."""
        return (
            self.config.batch_k_matrices
            and k_array is not None
            and hasattr(self.model, "batch_hamiltonian")
            and not compute_uq
        )

    @staticmethod
    def _expand_k_per_sample(k_values: torch.Tensor, samples_per_k: int) -> torch.Tensor:
        """Repeat each leading-k control matrix over its per-k sample batch."""
        expand_shape = (k_values.shape[0], samples_per_k, *k_values.shape[1:])
        return (
            k_values.unsqueeze(1)
            .expand(expand_shape)
            .reshape(k_values.shape[0] * samples_per_k, *k_values.shape[1:])
            .contiguous()
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _kl_term(self, epoch: int, n_samples: int) -> torch.Tensor:
        """KL regulariser of Concrete / Variational dropout (kl_weight, or 1/n_samples, times KL)."""
        kl = self.model.kl_divergence()
        if torch.isnan(kl) or torch.isinf(kl):
            print(f"WARNING: KL divergence is {kl.item()} at epoch {epoch}")
            kl = torch.zeros_like(kl)
        kl_w = self.config.kl_weight if self.config.kl_weight > 0 else 1.0 / float(n_samples)
        return kl_w * kl

    def _uses_kl(self) -> bool:
        return self.config.dropout_variant in ("concrete_dropout", "variational_dropout")

    def train_epoch(
        self,
        train_data: torch.Tensor,
        k_train: Optional[torch.Tensor],
        epoch: int,
        train_input_z0: Optional[torch.Tensor] = None,
    ) -> float:
        """One epoch of trajectory-MSE training; returns the mean loss over optimiser steps.

        train_data: (n_k, T_data, n_samples, 2*dim); k_train: control per cell (or None).
        Random draws, in order: per-cell sample permutation(s), then one mask index per batch
        (and one bank index per leapfrog evaluation when resampling per sub-step).
        """
        assert self.model is not None
        self.model.train()
        cfg = self.config

        n_k, T_data, n_samples, total_dim = train_data.shape
        dim = total_dim // 2
        batch_size = cfg.batch_size
        max_t = min(self.get_horizon(epoch), T_data)
        mask_num = self._bank_size()
        mask_index_fn = self._fresh_mask_index_fn(mask_num) if self._resample_mask_per_substep() else None

        losses: List[float] = []
        rollout_losses: List[float] = []
        grad_norms: List[float] = []

        def finish_step(loss_value: float, rollout_value: float) -> bool:
            """Clip gradients and step; returns False if the step was skipped."""
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            grad_norms.append(float(grad_norm.detach().item()))
            if torch.isnan(grad_norm) or grad_norm > 1000:
                print(f"WARNING: Gradient explosion at epoch {epoch}: {grad_norm}")
                self.optimizer.zero_grad()
                return False
            self.optimizer.step()
            losses.append(loss_value)
            rollout_losses.append(rollout_value)
            return True

        if self._can_batch_over_k(k_train):
            # All control cells in one batch: per-cell shuffles, then one rollout per batch position.
            perms = [torch.randperm(n_samples, device=self.device) for _ in range(n_k)]
            shuffled = [train_data[k][:, perms[k], :] for k in range(n_k)]
            shuffled_input = [train_input_z0[k][perms[k], :] for k in range(n_k)]
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                all_tgt = torch.cat([shuffled[k][:max_t, start:end, :] for k in range(n_k)], dim=1)
                all_z0 = torch.cat([shuffled_input[k][start:end, :] for k in range(n_k)], dim=0)
                k_batched = self._expand_k_per_sample(k_train, end - start)  # one control per row
                midx = int(torch.randint(mask_num, (1,), device=self.device).item())

                self.optimizer.zero_grad()
                pred = numerically_integrate(
                    p_0=all_z0[:, :dim],
                    q_0=all_z0[:, dim:],
                    model=self.model,
                    T=max_t,
                    dt=cfg.dt,
                    k_matrix=k_batched,
                    mask_index=midx,
                    use_leapfrog=cfg.use_leapfrog,
                    requires_grad=True,
                    mask_index_fn=mask_index_fn,
                )
                loss = F.mse_loss(pred, all_tgt)
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e4:
                    print(f"WARNING: Loss is NaN/Inf at epoch {epoch}, batch {start} (batched-k)")
                    continue
                loss.backward()
                loss_value = loss.item()
                rollout_value = loss.item()
                if self._uses_kl():
                    # KL regulariser backpropagated once per optimiser step.
                    reg = self._kl_term(epoch, n_samples)
                    if reg.requires_grad:
                        reg.backward()
                    loss_value += float(reg.detach().item())
                finish_step(loss_value, rollout_value)
        else:
            for k_idx in range(n_k):
                k_matrix = k_train[k_idx] if k_train is not None else None
                perm = torch.randperm(n_samples, device=self.device)
                data_k = train_data[k_idx][:, perm, :]
                input_k = train_input_z0[k_idx][perm, :] if train_input_z0 is not None else None

                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    batch = data_k[:max_t, start:end, :]
                    batch_input = input_k[start:end, :] if input_k is not None else None
                    midx = torch.randint(mask_num, (1,), device=self.device).item()

                    z_0 = _initial_state(batch[0], batch_input)
                    self.optimizer.zero_grad()
                    pred = numerically_integrate(
                        p_0=z_0[:, :dim],
                        q_0=z_0[:, dim:],
                        model=self.model,
                        T=max_t,
                        dt=cfg.dt,
                        k_matrix=k_matrix,
                        mask_index=midx,
                        use_leapfrog=cfg.use_leapfrog,
                        requires_grad=True,
                        mask_index_fn=mask_index_fn,
                    )
                    rollout_loss = F.mse_loss(pred, batch)
                    loss = rollout_loss
                    if self._uses_kl():
                        loss = loss + self._kl_term(epoch, n_samples)
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e4:
                        print(f"WARNING: Loss is NaN/Inf at epoch {epoch}, batch {start}")
                        continue
                    loss.backward()
                    finish_step(loss.item(), float(rollout_loss.detach().item()))

        if not losses:
            raise RuntimeError(
                "train_epoch produced zero optimizer steps "
                f"(epoch={epoch}, n_k={n_k}, n_samples={n_samples}, batch_size={batch_size}): "
                "every batch was skipped due to an invalid loss or gradient."
            )
        self._last_train_epoch_stats = {
            "train_rollout_loss": float(np.mean(rollout_losses)),
            "mean_grad_norm": float(np.mean(grad_norms)),
        }
        return float(np.mean(losses))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _sample_eval_mask_subset(self, n_mc: int, device: torch.device) -> torch.Tensor:
        """Distinct bank indices (a random subset), reusable across sample chunks."""
        bank = self._bank_size()
        return torch.randperm(bank, device=device)[: min(int(n_mc), bank)]

    @torch.no_grad()
    def _integrate_mc_chunked(
        self,
        p_0: torch.Tensor,
        q_0: torch.Tensor,
        k_matrix: Optional[torch.Tensor],
        T_eval: int,
        n_mc: int,
        mask_subset: torch.Tensor,
    ) -> torch.Tensor:
        """Mean MC prediction (T, n, 2*dim) over ``mask_subset``, in chunks of ``mc_eval_chunk_size`` masks."""
        chunk_size = self.config.mc_eval_chunk_size
        all_mask_idxs = mask_subset[:n_mc].to(device=p_0.device)
        effective_n = int(all_mask_idxs.shape[0])
        pred_sum = None
        total = 0
        for start in range(0, effective_n, chunk_size):
            chunk_n = min(chunk_size, effective_n - start)
            mc_chunk = integrate_mc_samples(
                p_0=p_0,
                q_0=q_0,
                model=self.model,
                T=T_eval,
                dt=self.config.dt,
                n_mc_samples=chunk_n,
                k_matrix=k_matrix,
                trajectory_consistent=True,
                use_leapfrog=self.config.use_leapfrog,
                mask_subset=all_mask_idxs[start : start + chunk_n],
                fresh_mask_semantics=_EVAL_FRESH_MASK_SEMANTICS,
            )
            chunk_sum = mc_chunk.sum(dim=0)
            pred_sum = chunk_sum if pred_sum is None else pred_sum + chunk_sum
            total += mc_chunk.shape[0]
        return pred_sum / max(total, 1)

    def _integrate_eval_predictions(
        self,
        p_0: torch.Tensor,
        q_0: torch.Tensor,
        k_matrix: Optional[torch.Tensor],
        T_eval: int,
        n_mc: int,
        mask_subset: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """MC prediction: mean over trajectory-consistent mask rollouts (and the samples)."""
        if self.config.mc_eval_chunk_size > 0:
            return self._integrate_mc_chunked(p_0, q_0, k_matrix, T_eval, n_mc, mask_subset), None
        mc_trajs = integrate_mc_samples(
            p_0=p_0,
            q_0=q_0,
            model=self.model,
            T=T_eval,
            dt=self.config.dt,
            n_mc_samples=n_mc,
            k_matrix=k_matrix,
            trajectory_consistent=True,
            use_leapfrog=self.config.use_leapfrog,
            mask_subset=mask_subset,
            fresh_mask_semantics=_EVAL_FRESH_MASK_SEMANTICS,
        )
        return mc_trajs.mean(dim=0), mc_trajs

    def _evaluate_batched_k_mse_only(
        self,
        data_tensor: torch.Tensor,
        input_z0_all: Optional[torch.Tensor],
        k_array: torch.Tensor,
        T_eval: int,
        n_mc: int,
    ) -> Dict[str, float]:
        """Rollout MSE for all control cells at once, sample axis processed in chunks.

        One mask subset is drawn and reused across chunks.
        """
        chunk = self.config.eval_sample_chunk_size
        if chunk <= 0:
            raise ValueError("batch_k_matrices evaluation requires eval_sample_chunk_size > 0")
        n_k, _, n_samples, total_dim = data_tensor.shape
        dim = total_dim // 2
        mask_subset = self._sample_eval_mask_subset(n_mc, data_tensor.device)
        mse_sum, mse_count = 0.0, 0
        for s0 in range(0, n_samples, chunk):
            s1 = min(s0 + chunk, n_samples)
            n_chunk = n_k * (s1 - s0)
            chunk_target = (
                data_tensor[:, :T_eval, s0:s1, :]
                .permute(1, 0, 2, 3)
                .reshape(T_eval, n_chunk, total_dim)
                .contiguous()
            )  # (T, n_k * chunk, 2*dim)
            input_chunk = (
                None if input_z0_all is None else input_z0_all[:, s0:s1, :].reshape(n_chunk, total_dim)
            )
            z_0 = _initial_state(data_tensor[:, 0, s0:s1, :].reshape(n_chunk, total_dim), input_chunk)
            pred, _ = self._integrate_eval_predictions(
                z_0[:, :dim],
                z_0[:, dim:],
                self._expand_k_per_sample(k_array, s1 - s0),
                T_eval,
                n_mc,
                mask_subset=mask_subset,
            )
            diff = pred - chunk_target
            mse_sum += float(diff.square().sum().item())
            mse_count += int(diff.numel())
        return {"mse": mse_sum / max(mse_count, 1)}

    @torch.no_grad()
    def evaluate(
        self,
        data: Any,
        k_array: Optional[torch.Tensor],
        T_eval: int,
        compute_uq: bool = False,
        use_all_masks: bool = False,
        max_samples_per_k: int = 0,
        pred_samples_override: Optional[int] = None,
    ) -> Dict[str, float]:
        """MC evaluation: rollout MSE and (optionally) Gaussian NLL / CRPS / coverage.

        Args:
            data: trajectories (k, T, n_samples, 2*dim) or a split bundle.
            k_array: control per cell or None.
            T_eval: evaluation horizon.
            compute_uq: also compute NLL, CRPS, 90% coverage and calibration error.
            use_all_masks: use every bank mask instead of ``pred_samples`` MC samples.
            max_samples_per_k: if > 0, evaluate an evenly spaced subset of samples per cell.
            pred_samples_override: if > 0, replaces ``pred_samples``.
        """
        assert self.model is not None
        self.model.eval()
        cfg = self.config
        data_tensor, input_z0_all = _unpack(data)

        n_k, T_data, n_samples, total_dim = data_tensor.shape
        max_samples_per_k = int(max_samples_per_k or 0)
        if max_samples_per_k > 0 and n_samples > max_samples_per_k:
            sample_idx = torch.linspace(
                0, n_samples - 1, max_samples_per_k, device=data_tensor.device, dtype=torch.float64
            ).long()
            data_tensor = data_tensor.index_select(2, sample_idx)
            if input_z0_all is not None:
                input_z0_all = input_z0_all.index_select(1, sample_idx)
        dim = total_dim // 2
        T_eval = min(T_eval, T_data)

        pred_samples_eval = (
            int(pred_samples_override)
            if pred_samples_override is not None and int(pred_samples_override) > 0
            else int(cfg.pred_samples)
        )
        n_mc = self._bank_size() if (use_all_masks or cfg.val_use_all_masks) else pred_samples_eval

        if self._can_batch_over_k(k_array, compute_uq):
            return self._evaluate_batched_k_mse_only(data_tensor, input_z0_all, k_array, T_eval, n_mc)

        uq_levels = tuple(float(level) for level in np.linspace(0.1, 0.9, 9).round(6))
        uq_weight_total = 0.0
        uq_nll_sum = 0.0
        uq_crps_sum = 0.0
        uq_coverage_sums = {level: 0.0 for level in uq_levels}
        all_preds: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        for k_idx in range(n_k):
            k_matrix = k_array[k_idx] if k_array is not None else None
            target = data_tensor[k_idx, :T_eval, :, :]
            input_z0_k = input_z0_all[k_idx] if input_z0_all is not None else None
            z_0 = _initial_state(target[0], input_z0_k)
            pred, mc_trajs = self._integrate_eval_predictions(
                z_0[:, :dim], z_0[:, dim:], k_matrix, T_eval, n_mc
            )
            all_preds.append(pred)
            all_targets.append(target)
            if compute_uq and mc_trajs is not None:
                mc_flat = mc_trajs.reshape(mc_trajs.shape[0], -1, total_dim)
                targets_flat = target.reshape(-1, total_dim)
                weight = float(targets_flat.numel())
                uq_nll_sum += UQMetrics.compute_nll_gaussian(mc_flat, targets_flat) * weight
                uq_crps_sum += UQMetrics.compute_crps(mc_flat, targets_flat) * weight
                coverage = UQMetrics.compute_coverage(mc_flat, targets_flat, nominal_levels=uq_levels)
                for level, value in coverage.items():
                    uq_coverage_sums[round(float(level), 6)] += float(value) * weight
                uq_weight_total += weight

        results: Dict[str, float] = {
            "mse": F.mse_loss(torch.stack(all_preds), torch.stack(all_targets)).item()
        }
        if compute_uq and uq_weight_total > 0.0:
            coverage_means = {l: uq_coverage_sums[l] / uq_weight_total for l in uq_levels}
            results["nll"] = uq_nll_sum / uq_weight_total
            results["crps"] = uq_crps_sum / uq_weight_total
            results["coverage_90"] = coverage_means.get(0.9, 0.0)
            results["calibration_error"] = float(
                np.mean([abs(coverage_means[l] - l) for l in uq_levels])
            )
        return results

    @torch.no_grad()
    def compute_energy_metrics(
        self,
        data: Any,
        k_array: Optional[torch.Tensor],
        n_mc: int = 20,
    ) -> Dict[str, float]:
        """Model-Hamiltonian drift of MC rollouts, each scored with the mask that generated it.

        Uses the first (up to) five control cells and ``n_test_samples`` initial conditions.
        """
        assert self.model is not None
        self.model.eval()
        data_tensor, input_z0_all = _unpack(data)

        n_k = min(5, data_tensor.shape[0])
        dim = data_tensor.shape[3] // 2
        n_samples = min(self.config.n_test_samples, data_tensor.shape[2])
        T_eval = min(self.config.T_test, data_tensor.shape[1])

        all_stats = []
        for k_idx in range(n_k):
            k_matrix = k_array[k_idx] if k_array is not None else None
            input_z0_k = input_z0_all[k_idx, :n_samples, :] if input_z0_all is not None else None
            z_0 = _initial_state(data_tensor[k_idx, 0, :n_samples, :], input_z0_k)
            mc_trajs, mask_idxs = integrate_mc_samples(
                p_0=z_0[:, :dim],
                q_0=z_0[:, dim:],
                model=self.model,
                T=T_eval,
                dt=self.config.dt,
                n_mc_samples=n_mc,
                k_matrix=k_matrix,
                trajectory_consistent=True,
                use_leapfrog=self.config.use_leapfrog,
                return_mask_idxs=True,
                fresh_mask_semantics=_EVAL_FRESH_MASK_SEMANTICS,
            )
            mask_idxs_int = [int(m.item()) for m in mask_idxs]
            n_mc_actual, T_traj, n_ics_traj, _ = mc_trajs.shape
            energies = torch.zeros(n_mc_actual, T_traj, n_ics_traj, device=mc_trajs.device)
            for t in range(T_traj):
                p_t = mc_trajs[:, t, :, :dim].reshape(-1, dim)
                q_t = mc_trajs[:, t, :, dim:].reshape(-1, dim)
                energies[:, t, :] = self.model.batch_hamiltonian(
                    p_t, q_t, k_matrix, mask_idxs_int
                ).reshape(n_mc_actual, n_ics_traj)
            all_stats.append(EnergyMetrics.compute_drift_stats(energies))

        return {
            "drift_mean": float(np.mean([s["drift_mean"] for s in all_stats])),
            "drift_final": float(np.mean([s["drift_final_mean"] for s in all_stats])),
            "drift_max": float(np.max([s["drift_max"] for s in all_stats])),
        }

    @torch.no_grad()
    def _diagnostic_rollout(
        self, data: Any, k_array: Optional[torch.Tensor], n_samples_plot: int = 3
    ) -> None:
        """Deterministic-index rollout of the first cell's first samples (result discarded).

        The model is evaluated with ``mask_index=0``; for fresh-mask arms and for an
        uninitialised bank this advances the global RNG. The call is part of the training
        protocol, so it is kept even though the prediction itself is not stored.
        """
        self.model.eval()
        data_tensor, input_z0_all = _unpack(data)
        _, T_data, n_samples, total_dim = data_tensor.shape
        dim = total_dim // 2
        T_plot = min(self.config.T_test, T_data)
        n_plot = min(n_samples_plot, n_samples)
        target = data_tensor[0, :T_plot, :n_plot, :]
        z_0 = _initial_state(
            target[0], input_z0_all[0, :n_plot, :] if input_z0_all is not None else None
        )
        numerically_integrate(
            p_0=z_0[:, :dim],
            q_0=z_0[:, dim:],
            model=self.model,
            T=T_plot,
            dt=self.config.dt,
            k_matrix=k_array[0] if k_array is not None else None,
            mask_index=0,
            use_leapfrog=self.config.use_leapfrog,
            requires_grad=False,
        )

    # ------------------------------------------------------------------
    # Checkpoints and logging
    # ------------------------------------------------------------------

    def _log_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        test_metrics: Optional[Dict] = None,
        energy_metrics: Optional[Dict] = None,
    ):
        """Log one evaluation line to the console."""
        parts = [
            f"Epoch {epoch:4d}",
            f"train={train_loss:.4f}",
            f"val={val_loss:.4f}",
            f"lr={self.optimizer.param_groups[0]['lr']:.2e}",
        ]
        if self.config.gradual_horizon_increase:
            parts.append(f"T={self.get_horizon(epoch)}")
        if test_metrics:
            parts.append(f"test={test_metrics['mse']:.4f}")
            if "nll" in test_metrics:
                parts.append(f"nll={test_metrics['nll']:.3f}")
                parts.append(f"cov90={test_metrics['coverage_90']:.2f}")
        if energy_metrics:
            parts.append(f"Edrift={energy_metrics['drift_mean']:.4f}")
        tqdm.write(" | ".join(parts))

    def _best_checkpoint_horizon_eligible(self, epoch: int) -> bool:
        """Whether this epoch may define ``best_model.pt`` (optionally only at the full horizon)."""
        if not self.config.best_checkpoint_requires_full_horizon:
            return True
        return self.get_horizon(epoch) >= int(self.config.T)

    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save a training checkpoint (and ``best_model.pt`` = model weights if ``is_best``)."""
        path = Path(path)
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "metrics_history": self.metrics_history,
                "best_val_loss": self.best_val_loss,
                "config": self.config.to_dict(),
            },
            path,
        )
        if is_best:
            torch.save(self.model.state_dict(), path.parent / "best_model.pt")

    def _save_metrics(self, output_dir: Path):
        with open(output_dir / "metrics.pkl", "wb") as f:
            pickle.dump(self.metrics_history, f)

    def _save_final_model(self, output_dir: Path):
        torch.save(self.model.state_dict(), output_dir / "final_model.pt")

    def train(
        self,
        train_data: Any,
        val_data: Any,
        test_data: Any,
        k_train: Optional[torch.Tensor],
        k_val: Optional[torch.Tensor] = None,
        k_test: Optional[torch.Tensor] = None,
        output_dir: Path = Path("output"),
    ) -> Tuple[nn.Module, Dict[str, List]]:
        """
        Full training loop.

        Args:
            train_data / val_data / test_data: trajectories (k, T, n_samples, 2*dim) or split
                bundles with ``.trajectories`` and optional ``.input_z0``.
            k_train / k_val / k_test: control per cell; ``k_val`` defaults to ``k_train``.
            output_dir: receives best_model.pt, final_model.pt, checkpoints and metrics.pkl.

        Returns:
            Trained model and metrics history.
        """
        cfg = self.config
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def to_dev(x):
            """Move a tensor to the training device as float32 (None passes through)."""
            return None if x is None else x.to(self.device, dtype=torch.float32)

        train_t, train_z0 = _unpack(train_data)
        val_t, val_z0 = _unpack(val_data)
        test_t, test_z0 = _unpack(test_data)
        self.build(dim=train_t.shape[3] // 2)

        train_t, train_z0, val_t, val_z0, test_t, test_z0 = map(
            to_dev, (train_t, train_z0, val_t, val_z0, test_t, test_z0)
        )
        k_train, k_val, k_test = map(to_dev, (k_train, k_val, k_test))
        effective_k_val = k_val if k_val is not None else k_train

        class _Split:  # bare container so evaluate() sees (trajectories, input_z0)
            def __init__(self, trajectories, input_z0):
                self.trajectories = trajectories
                self.input_z0 = input_z0

        val_split = _Split(val_t, val_z0)
        test_split = _Split(test_t, test_z0)

        T_val = cfg.T_val if cfg.T_val > 0 else cfg.T_test
        if val_t.shape[1] < T_val:
            raise ValueError(f"val_data has T={val_t.shape[1]} steps but T_val={T_val} is required.")

        early_stop_counter = 0
        try:
            for epoch in tqdm(range(cfg.n_epochs), desc="Training"):
                self.current_epoch = epoch
                is_final_epoch = epoch == cfg.n_epochs - 1
                is_best = False
                self._refresh_mask_banks()

                train_loss = self.train_epoch(train_t, k_train, epoch, train_input_z0=train_z0)

                h = self.metrics_history
                h["epoch"].append(epoch)
                h["train_loss"].append(train_loss)
                h["train_rollout_loss"].append(self._last_train_epoch_stats["train_rollout_loss"])
                h["mean_grad_norm"].append(self._last_train_epoch_stats["mean_grad_norm"])
                h["val_loss_per_epoch"].append(float("nan"))
                h["learning_rate_per_epoch"].append(float(self.optimizer.param_groups[0]["lr"]))

                if epoch % cfg.test_freq == 0 or is_final_epoch:
                    # Intermediate validation may use fewer samples / MC draws (final epoch: full).
                    val_metrics = self.evaluate(
                        val_split,
                        effective_k_val,
                        T_eval=T_val,
                        compute_uq=False,
                        max_samples_per_k=0 if is_final_epoch else cfg.fast_eval_max_samples_per_k,
                        pred_samples_override=(
                            None if is_final_epoch else (cfg.fast_eval_pred_samples or None)
                        ),
                    )
                    val_loss = val_metrics["mse"]
                    h["val_loss"].append(val_loss)
                    h["val_loss_per_epoch"][-1] = float(val_loss)
                    lr = self.optimizer.param_groups[0]["lr"]
                    h["learning_rate"].append(lr)
                    h["learning_rate_per_epoch"][-1] = float(lr)
                    self.scheduler.step(val_loss)

                    best_eligible = self._best_checkpoint_horizon_eligible(epoch)
                    improved = val_loss < self.best_val_loss
                    is_best = improved and best_eligible
                    if is_best:
                        self.best_val_loss = val_loss
                        early_stop_counter = 0
                        self._diagnostic_rollout(val_split, effective_k_val)
                    else:
                        if epoch >= cfg.early_stop_min_epochs and best_eligible:
                            early_stop_counter += 1
                        if cfg.early_stop_patience > 0 and early_stop_counter >= cfg.early_stop_patience:
                            print(
                                f"Early stopping at epoch {epoch} "
                                f"(no improvement for {early_stop_counter} evaluations)"
                            )
                            self._diagnostic_rollout(val_split, effective_k_val)
                            self.save_checkpoint(output_dir / f"checkpoint_ep{epoch}.pt", is_best=False)
                            raise StopIteration

                    test_metrics = None
                    energy_metrics = None
                    if cfg.run_test_during_training or is_final_epoch:
                        if is_final_epoch:
                            self._refresh_mask_banks()  # final test uses a fresh bank
                        test_metrics = self.evaluate(
                            test_split,
                            k_test,
                            T_eval=cfg.T_test,
                            compute_uq=cfg.compute_calibration,
                            use_all_masks=cfg.val_use_all_masks,
                        )
                        h["test_mse"].append(test_metrics["mse"])
                        if "nll" in test_metrics:
                            h["test_nll"].append(test_metrics["nll"])
                            h["test_crps"].append(test_metrics["crps"])
                            h["test_coverage_90"].append(test_metrics["coverage_90"])
                            h["calibration_error"].append(test_metrics["calibration_error"])
                        if cfg.compute_energy:
                            energy_metrics = self.compute_energy_metrics(test_split, k_test)
                            h["energy_drift_mean"].append(energy_metrics["drift_mean"])
                            h["energy_drift_final"].append(energy_metrics["drift_final"])
                        if is_final_epoch:
                            self._refresh_mask_banks()

                    self._log_metrics(epoch, train_loss, val_loss, test_metrics, energy_metrics)
                    self._diagnostic_rollout(val_split, effective_k_val)

                if epoch % cfg.save_freq == 0 or is_best:
                    self.save_checkpoint(output_dir / f"checkpoint_ep{epoch}.pt", is_best=is_best)

        except StopIteration:
            pass  # early stop: fall through to the final save
        except Exception as e:
            print(f"Training interrupted by error: {e}")
            for name in ("final_model.pt", "metrics.pkl"):
                (output_dir / name).unlink(missing_ok=True)
            raise

        self._save_final_model(output_dir)
        self._save_metrics(output_dir)
        return self.model, self.metrics_history
