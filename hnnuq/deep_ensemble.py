"""
deep_ensemble.py
Wraps M independently trained HNNs as a single model with the interface of a dropout model
(``mask_num`` = M members), so ``integrate_mc_samples`` and the evaluation stages treat the
members as a mask bank.
"""

from pathlib import Path
from typing import List, Optional
import re

import torch
import torch.nn as nn
from torch import Tensor


class DeepEnsembleWrapper(nn.Module):
    """Wraps M independently-trained HNNs as one model with mask_num=M.

    Interface is drop-in compatible with integrate_mc_samples:
      - forward(p, q, k_matrix=None, index=i)  -> dispatches to member (i % M)
      - batch_hamiltonian(p, q, k_matrix, mask_idxs_int)  -> dispatches each
        group [i*n_ics : (i+1)*n_ics] to member mask_idxs_int[i] % M
      - mask_num property returns M

    Each member is deterministic (drop_prob=0.0), so all stochasticity comes
    from ensemble diversity, not within-member dropout noise.
    """

    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(models)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mask_num(self) -> int:
        """Number of ensemble members (the ensemble's "mask bank" size)."""
        return len(self.members)

    def get_active_mask_num(self) -> int:
        """Return ensemble size so get_model_active_mask_num works correctly."""
        return len(self.members)

    # ------------------------------------------------------------------
    # Forward / Hamiltonian interface
    # ------------------------------------------------------------------

    def forward(
        self,
        p: Tensor,
        q: Tensor,
        k_matrix: Optional[Tensor] = None,
        index: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        """Dispatch to one member.

        Args:
            p: Momenta (batch, dim)
            q: Positions (batch, dim)
            k_matrix: Optional connection matrix
            index: Which member to use; None -> member 0.

        Returns:
            Hamiltonian values (batch,)
        """
        idx = 0 if index is None else (int(index) % self.mask_num)
        member = self.members[idx]
        if k_matrix is not None:
            return member(p, q, k_matrix, index=None)
        return member(p, q, index=None)

    def batch_hamiltonian(
        self,
        p: Tensor,
        q: Tensor,
        k_matrix: Optional[Tensor] = None,
        mask_idxs_int: Optional[List[int]] = None,
    ) -> Tensor:
        """Vectorised Hamiltonian for the batched leapfrog integration path.

        The caller (integrate_mc_samples) lays out the batch as:
            p[i*n_ics : (i+1)*n_ics]  <->  mask mask_idxs_int[i]
        We dispatch each group to the corresponding ensemble member.

        Args:
            p: (n_mc * n_ics, dim)
            q: (n_mc * n_ics, dim)
            k_matrix: optional connection matrix or (batch, dim, dim)
            mask_idxs_int: list of M integer mask indices, one per group

        Returns:
            Hamiltonian values (n_mc * n_ics,)
        """
        if mask_idxs_int is None or len(mask_idxs_int) == 1:
            return self.forward(
                p, q, k_matrix, index=(mask_idxs_int[0] if mask_idxs_int else 0)
            )

        n_groups = len(mask_idxs_int)
        n_total = p.shape[0]
        n_ics = n_total // n_groups

        chunks: List[Tensor] = []
        for i, midx in enumerate(mask_idxs_int):
            row_start = i * n_ics
            row_end = (i + 1) * n_ics
            p_i = p[row_start:row_end]
            q_i = q[row_start:row_end]
            member = self.members[int(midx) % self.mask_num]
            if k_matrix is not None:
                if k_matrix.dim() == 3:
                    k_i = k_matrix[row_start:row_end]
                else:
                    k_i = k_matrix
                chunks.append(member(p_i, q_i, k_i, index=None))
            else:
                chunks.append(member(p_i, q_i, index=None))
        return torch.cat(chunks, dim=0)

    # ------------------------------------------------------------------
    # Loading helper
    # ------------------------------------------------------------------

    @classmethod
    def from_dirs(
        cls,
        member_dirs: List[Path],
        device: str = "cpu",
    ) -> "DeepEnsembleWrapper":
        """Load M trained members from run directories (``config.yaml`` + ``best_model.pt``).

        Args:
            member_dirs: run directories, one per ensemble member.
            device: target device string, e.g. ``"cuda:0"`` or ``"cpu"``.
        """
        from hnnuq.hnn_model import build_hnn_model
        from hnnuq.config import ExperimentConfig
        from hnnuq.masked_linear import load_state_dict_with_optional_masks
        from hnnuq.tfim.control import register_tfim_operator_structure

        members: List[nn.Module] = []
        for run_dir in member_dirs:
            run_dir = Path(run_dir)
            config = ExperimentConfig.load(run_dir / "config.yaml")
            model_path = run_dir / "best_model.pt"
            if not model_path.exists():
                model_path = run_dir / "final_model.pt"
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            dim = infer_model_dim(config, state_dict)

            model = build_hnn_model(
                dim=dim,
                n_hidden=config.n_hidden,
                drop_prob=config.drop_prob,
                mask_num=config.mask_num,
                nonlin=config.nonlin,
                device=torch.device(device),
                model_type=config.model_type,
                dropout_variant=config.dropout_variant,
                finite_mask_mode=config.finite_mask_mode,
                concrete_temp=config.concrete_temp,
                scale_weights=config.scale_weights,
                mask_last_layer=config.mask_last_layer,
                finite_mask_storage=config.finite_mask_storage,
                mask_seed=config.seed,
            )
            if config.model_type == "TFIMSeparableHNN":
                register_tfim_operator_structure(model, state_half_dim=dim)
            load_state_dict_with_optional_masks(model, state_dict, strict=False, device=device)
            model.to(device)
            model.eval()
            members.append(model)
        return cls(members)


def infer_model_dim(config, state_dict) -> int:
    """Degrees of freedom of a trained model, from its config / checkpoint.

    TFIM: half the canonical state size; coupled pendulum: P1 input size minus the appended
    log mass ratio; spring chain: N from the dataset name ``chain<N>_...``.
    """
    if config.model_type == "TFIMSeparableHNN":
        return int(config.state_dim) // 2
    if config.model_type == "MassRatioSimpleHNN":
        return int(state_dict["linear_P1.weight"].shape[-1]) - 1
    match = re.search(r"chain(\d+)", config.dataset_index)
    if match is None:
        raise ValueError(f"Cannot infer the chain length from dataset_index {config.dataset_index!r}")
    return int(match.group(1))
