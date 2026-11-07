"""TFIM operator controls: the sparse computational-basis operator A(g) fed to the HNN.

The transverse-field Ising chain (open boundaries, J = 1) is
    A(g) = -J sum_i Z_i Z_{i+1} - g sum_i X_i
in the computational basis (dimension 2**L): a diagonal part plus L single-bit-flip
off-diagonals. The edge structure does not depend on g, so it is registered explicitly on
the model; the model receives A(g) as its control tensor.
"""

import math
from typing import Optional

import torch

TFIM_J = 1.0


def _site_bitmask(L: int, site: int) -> int:
    return 1 << (L - 1 - site)


def _basis_spin(basis_index: int, L: int, site: int) -> int:
    return 1 if (basis_index & _site_bitmask(L, site)) == 0 else -1


def infer_tfim_qubit_count_from_state_half_dim(state_half_dim: int) -> int:
    """Infer L from the computational-basis amplitude count 2**L."""
    dim = int(state_half_dim)
    L = int(round(math.log2(dim))) if dim >= 1 else -1
    if L < 0 or (1 << L) != dim:
        raise ValueError(f"TFIM state half-dimension must be 2**L, got {dim}")
    return L


def build_tfim_operator_structure(
    L: int, device: Optional[torch.device] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed sparse operator structure: ``(edge_index (2, E), edge_type (E,))``.

    ``edge_type`` is 1 for diagonal (self) edges and 0 for the L single-bit-flip edges.
    """
    dim = 1 << L
    rows: list[int] = []
    cols: list[int] = []
    edge_type: list[int] = []
    for basis_index in range(dim):
        rows.append(basis_index)
        cols.append(basis_index)
        edge_type.append(1)
        for site in range(L):
            rows.append(basis_index ^ _site_bitmask(L, site))
            cols.append(basis_index)
            edge_type.append(0)
    return (
        torch.tensor([rows, cols], dtype=torch.long, device=device),
        torch.tensor(edge_type, dtype=torch.long, device=device),
    )


def build_tfim_operator(
    g: torch.Tensor | float,
    L: int,
    J: float = TFIM_J,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dense operator A(g): shape (2**L, 2**L) for scalar g, (*g.shape, 2**L, 2**L) otherwise."""
    dim = 1 << L
    g_tensor = torch.as_tensor(g, device=device, dtype=dtype).detach()
    operator = torch.zeros((*g_tensor.shape, dim, dim), device=g_tensor.device, dtype=g_tensor.dtype)

    diag_values = []
    for basis_index in range(dim):
        diag_energy = 0.0
        for site in range(L - 1):
            diag_energy += (
                -J * _basis_spin(basis_index, L, site) * _basis_spin(basis_index, L, site + 1)
            )
        diag_values.append(diag_energy)
    diag_tensor = torch.tensor(diag_values, device=g_tensor.device, dtype=g_tensor.dtype)
    diag_idx = torch.arange(dim, device=g_tensor.device)
    operator[..., diag_idx, diag_idx] = diag_tensor

    for basis_index in range(dim):
        for site in range(L):
            operator[..., basis_index ^ _site_bitmask(L, site), basis_index] += -g_tensor

    if g_tensor.ndim == 0:
        return operator.reshape(dim, dim)
    return operator


def register_tfim_operator_structure(model: object, state_half_dim: int) -> None:
    """Register the fixed TFIM edge list on an operator-conditioned model."""
    L = infer_tfim_qubit_count_from_state_half_dim(state_half_dim)
    device = next(model.parameters()).device
    edge_index, edge_type = build_tfim_operator_structure(L, device=device)
    model.register_operator_structure(edge_index=edge_index, edge_type=edge_type)


def resolve_tfim_control_tensor(
    g: torch.Tensor | float,
    state_half_dim: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Operator control A(g) for the stored scalar field values ``g``."""
    L = infer_tfim_qubit_count_from_state_half_dim(state_half_dim)
    return build_tfim_operator(g, L=L, device=device, dtype=dtype)
