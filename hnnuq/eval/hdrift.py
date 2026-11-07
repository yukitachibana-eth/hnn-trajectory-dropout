"""Drift of the model's own sampled Hamiltonian along MC rollouts (columns ``h_drift`` / ``h_drift_final`` of the ledger)."""

import contextlib
from pathlib import Path
from typing import Dict

import numpy as np
import torch

import hnnuq.trainer as trainer_module
from hnnuq.eval.common import load_arm_model, mask_semantics, seed_everything
from hnnuq.integration import model_uses_no_finite_mask, integrate_mc_samples
from hnnuq.masked_linear import discard_model_materialized_masks
from hnnuq.metrics import EnergyMetrics
from hnnuq.train import load_data
from hnnuq.trainer import Trainer


@contextlib.contextmanager
def _trainer_semantics(value: str):
    """Temporarily set the fresh-mask semantics the trainer uses for its evaluation rollouts."""
    old = trainer_module._EVAL_FRESH_MASK_SEMANTICS
    trainer_module._EVAL_FRESH_MASK_SEMANTICS = value
    try:
        yield
    finally:
        trainer_module._EVAL_FRESH_MASK_SEMANTICS = old


@torch.no_grad()
def model_h_drift(model, config, test_data: torch.Tensor, k_test, semantics: str, n_mc: int = 20) -> Dict[str, float]:
    """Mean and last-step drift |H_m(z_t) - H_m(z_0)| of ``n_mc`` rollouts scored with the Hamiltonian of the mask that made them.

    Uses the first (up to) five control cells and the first ``n_test_samples`` test ICs over ``T_test`` steps; masks are drawn
    by ``integrate_mc_samples`` (random subset of the bank). Fresh-mask models under "activation" semantics are scored one
    rollout at a time with a fresh dropout draw per Hamiltonian call (the same noise model as the rollout); all other models use
    the batched per-mask Hamiltonian. Results are averaged over control cells.
    """
    model.eval()
    n_k = min(5, test_data.shape[0])
    dim = test_data.shape[3] // 2
    n_samples = min(config.n_test_samples, test_data.shape[2])
    T_eval = min(config.T_test, test_data.shape[1])
    sequential = semantics == "activation" and model_uses_no_finite_mask(model)
    stats = []
    for i in range(n_k):
        k_matrix = k_test[i] if k_test is not None else None
        z_0 = test_data[i, 0, :n_samples, :]
        trajs, mask_idxs = integrate_mc_samples(
            p_0=z_0[:, :dim], q_0=z_0[:, dim:], model=model, T=T_eval, dt=config.dt, n_mc_samples=n_mc, k_matrix=k_matrix,
            trajectory_consistent=True, use_leapfrog=config.use_leapfrog, return_mask_idxs=True, fresh_mask_semantics=semantics,
        )
        idxs = [int(m.item()) for m in mask_idxs]
        m_act, T, s, _ = trajs.shape
        if not sequential:
            energies = torch.zeros(m_act, T, s, device=trajs.device)
            for t in range(T):
                p_t, q_t = trajs[:, t, :, :dim].reshape(-1, dim), trajs[:, t, :, dim:].reshape(-1, dim)
                energies[:, t, :] = model.batch_hamiltonian(p_t, q_t, k_matrix, idxs).reshape(m_act, s)
        else:
            energies = torch.zeros(m_act, T, s, device=trajs.device)
            for m, midx in enumerate(idxs):
                for t in range(T):
                    h = model(trajs[m, t, :, :dim], trajs[m, t, :, dim:], k_matrix, index=midx)
                    energies[m, t, :] = h.squeeze(-1) if h.dim() > 1 else h
        stats.append(EnergyMetrics.compute_drift_stats(energies))
    return {"h_drift": float(np.mean([s["drift_mean"] for s in stats])),
            "h_drift_final": float(np.mean([s["drift_final_mean"] for s in stats]))}


def hdrift_stage(system: str, runs_root: Path, arm: str, seed: int, device: torch.device, smoke: bool = False) -> Dict[str, float]:
    """H-drift of one (system, arm, seed) following the evaluation-after-training protocol.

    Seeds the global RNG with the run's training seed, rebuilds the model, runs the test-set evaluation pass of the training
    protocol (``Trainer.evaluate`` with all bank masks and calibration scores; its RNG draws precede the drift's masks, which
    matters for the fresh-mask arms), then computes the drift. Also returns the test MSE of that pass. Like the final test of
    training, the pass starts from a freshly drawn mask bank (the checkpoint's bank is discarded, the new one comes from the
    seeded global RNG at the first forward).
    """
    semantics = mask_semantics(system, "hdrift", arm)
    seed_everything(0 if arm == "deep_ensemble" else seed)  # model construction draws from the global RNG
    model, config = load_arm_model(runs_root, system, arm, seed, device)
    discard_model_materialized_masks(model)
    if smoke:
        config.n_test_samples, config.T_test, config.k_test_samples, config.pred_samples = 4, 8, 2, 3
    _, _, test_data, _, k_test = load_data(config)
    test_data = test_data.to(device, dtype=torch.float32)
    k_test = k_test.to(device, dtype=torch.float32)
    trainer = Trainer(config, device)
    trainer.model = model
    with _trainer_semantics(semantics):
        test = trainer.evaluate(test_data, k_test, T_eval=config.T_test, compute_uq=config.compute_calibration)
        out = model_h_drift(model, config, test_data, k_test, semantics)
    out["test_mse"] = float(test["mse"])
    return out
