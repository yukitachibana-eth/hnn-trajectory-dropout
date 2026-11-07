"""hnn_model.py

Hamiltonian networks with dropout-perturbed layers.

- ``MassRatioSimpleHNN``: coupled pendulum, H(p, q, r) = T([p_i, log r]) + V([q, log r]).
- ``SeparableHNN``: spring chain, H = T(p) + sum over edges f(q_i, q_j, k_ij).
- ``OperatorSeparableHNN``: TFIM, H = E_p(p; g) + E_q(q; g) with sparse operator edge energies.

Every model has a two-layer MLP per energy term; exactly one layer is a ``MaskedLinear``
(the first one by default, the last one with ``mask_last_layer=True``). All models expose
``forward(p, q, k, index)`` for one mask index and ``batch_hamiltonian`` which evaluates
many mask groups with a single ``torch.bmm`` per masked layer.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable
from hnnuq.masked_linear import (
    MaskedLinear,
    DropoutVariant,
    FiniteMaskMode,
    configure_masked_linear_seeds,
)


def get_activation_fn(name: str) -> Callable:
    """Activation by name; the paper uses ``intsin2x`` = integral of sin^2(x) + x, i.e.
    0.5 x - 0.25 sin(2x) + 0.5 x^2."""
    if name != "intsin2x":
        raise ValueError(f"Unknown activation: {name}; only 'intsin2x' is supported")

    def int_sin2x(x):
        """Activation 0.5 x - 0.25 sin(2x) + 0.5 x^2."""
        return 0.5 * x - 0.25 * torch.sin(2 * x) + 0.5 * x.pow(2)

    return int_sin2x


def _mlp2(x, first, second, nonlin, mask_last_layer, mask_index=None, mask_idxs_int=None):
    """Two-layer MLP ``second(nonlin(first(x)))`` in which one layer is a MaskedLinear.

    With ``mask_idxs_int`` the masked layer runs grouped (``batch_forward_grouped``, group i
    uses mask ``mask_idxs_int[i]``); otherwise it uses ``mask_index`` (or a fresh draw).
    """
    def run(layer, h):
        """Apply ``layer`` to ``h``, grouped or with the single mask index."""
        if mask_idxs_int is not None:
            return layer.batch_forward_grouped(h, mask_idxs_int)
        return layer(h, mask_index=mask_index)

    if not mask_last_layer:
        h = nonlin(run(first, x))
        return second(h)
    h = nonlin(first(x))
    return run(second, h)


def _build_edge_mlp(input_feature_dim, n_hidden, mask_last_layer, bias, dropout_kwargs):
    """Layers (first, second) of the edge-energy MLP; one of them is masked."""
    if not mask_last_layer:
        return (
            MaskedLinear(input_feature_dim, n_hidden, bias=bias, **dropout_kwargs),
            nn.Linear(n_hidden, 1, bias=bias),
        )
    return (
        nn.Linear(input_feature_dim, n_hidden, bias=bias),
        MaskedLinear(n_hidden, 1, bias=bias, **dropout_kwargs),
    )


class SparseEdgeEnergy:
    """Sparse edge-energy helper: sum over edges (i, j) of an MLP of the edge features.

    The control ``k`` holds the edge weights: (dim, dim) shared by the batch, or
    (batch, dim, dim) per sample. The edge list is either registered explicitly (TFIM operator
    control, where edges must not depend on the value of g) or inferred from the non-zeros of
    a 2-D ``k`` (spring chain).

    Edge features are [x_i, x_j, k_ij] (``wall_edge``: x_i is zeroed on self edges, which model
    the wall) or, with ``use_edge_type_feature``, [x_i, x_j, k_ij, edge_type] (``self_edge``).
    """

    def __init__(
        self,
        *,
        linear_1: nn.Module,
        linear_2: nn.Module,
        nonlin: Callable,
        mask_last_layer: bool,
        edge_feature_mode: str = "wall_edge",
        use_edge_type_feature: bool = False,
        allow_nonzero_fallback: bool = True,
    ) -> None:
        self.linear_1 = linear_1
        self.linear_2 = linear_2
        self.nonlin = nonlin
        self.mask_last_layer = bool(mask_last_layer)
        self.edge_feature_mode = edge_feature_mode  # "wall_edge" | "self_edge"
        self.use_edge_type_feature = bool(use_edge_type_feature)
        self.allow_nonzero_fallback = bool(allow_nonzero_fallback)
        self._edge_index: Optional[torch.Tensor] = None
        self._edge_type: Optional[torch.Tensor] = None

    def register_structure(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> None:
        """Register the explicit edge list: edge_index (2, n_edges), edge_type (n_edges,)."""
        self._edge_index = torch.as_tensor(edge_index, dtype=torch.long).detach().contiguous()
        self._edge_type = torch.as_tensor(edge_type, dtype=torch.long).detach().contiguous()

    def _resolve_structure(
        self, k_mat: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (i_idx, j_idx, edge_type) of the edges."""
        if self._edge_index is not None:
            edge_index = self._edge_index.to(device=device)
            return edge_index[0], edge_index[1], self._edge_type.to(device=device)
        if not self.allow_nonzero_fallback:
            raise ValueError("SparseEdgeEnergy requires an explicit edge structure.")
        i_idx, j_idx = k_mat.nonzero(as_tuple=True)
        return i_idx, j_idx, (i_idx == j_idx).to(dtype=torch.long)

    @staticmethod
    def _expand_grouped_k(k_mat: torch.Tensor, total_batch: int, n_groups: int) -> torch.Tensor:
        """Repeat a per-sample (samples_per_group, d, d) control across the mask groups."""
        if k_mat.ndim != 3 or k_mat.shape[0] == total_batch:
            return k_mat
        return (
            k_mat.unsqueeze(0)
            .expand(n_groups, -1, -1, -1)
            .reshape(total_batch, *k_mat.shape[1:])
            .contiguous()
        )

    def _build_edge_inputs(
        self,
        x: torch.Tensor,
        i_idx: torch.Tensor,
        j_idx: torch.Tensor,
        edge_type: torch.Tensor,
        k_mat: torch.Tensor,
    ) -> torch.Tensor:
        """Edge features (batch, n_edges, 3 or 4) from x (batch, dim)."""
        batch_size = x.shape[0]
        x_i = x[:, i_idx]
        x_j = x[:, j_idx]
        if self.edge_feature_mode == "wall_edge":
            x_i = x_i.masked_fill((i_idx == j_idx).unsqueeze(0), 0.0)
        if k_mat.ndim == 2:
            k_exp = k_mat[i_idx, j_idx].unsqueeze(0).expand(batch_size, -1)
        else:
            k_exp = k_mat[:, i_idx, j_idx]
        features = [x_i, x_j, k_exp]
        if self.use_edge_type_feature:
            edge_type_exp = edge_type.to(device=x.device, dtype=x.dtype).unsqueeze(0)
            features.append(edge_type_exp.expand(batch_size, -1))
        return torch.stack(features, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        k_mat: torch.Tensor,
        *,
        mask_index: Optional[int] = None,
    ) -> torch.Tensor:
        """Edge energy (batch, 1) for one mask index."""
        k_values = torch.as_tensor(k_mat, device=x.device, dtype=x.dtype).detach()
        i_idx, j_idx, edge_type = self._resolve_structure(k_values, x.device)
        edge_inputs = self._build_edge_inputs(x, i_idx, j_idx, edge_type, k_values)
        return _mlp2(
            edge_inputs, self.linear_1, self.linear_2, self.nonlin, self.mask_last_layer,
            mask_index=mask_index,
        ).sum(dim=1)

    def batch_forward_grouped(
        self,
        x_all: torch.Tensor,
        k_mat: torch.Tensor,
        mask_idxs_int: list,
    ) -> torch.Tensor:
        """Edge energy (n_mc*n_ics, 1) with group i evaluated under mask ``mask_idxs_int[i]``."""
        k_values = torch.as_tensor(k_mat, device=x_all.device, dtype=x_all.dtype).detach()
        k_values = self._expand_grouped_k(k_values, x_all.shape[0], len(mask_idxs_int))
        i_idx, j_idx, edge_type = self._resolve_structure(k_values, x_all.device)
        edge_inputs = self._build_edge_inputs(x_all, i_idx, j_idx, edge_type, k_values)
        return _mlp2(
            edge_inputs, self.linear_1, self.linear_2, self.nonlin, self.mask_last_layer,
            mask_idxs_int=mask_idxs_int,
        ).sum(dim=1)


def _sum_kl(model: nn.Module) -> torch.Tensor:
    """Sum of the KL regularisers of all MaskedLinear layers."""
    kl = torch.tensor(0.0, device=next(model.parameters()).device)
    for module in model.modules():
        if isinstance(module, MaskedLinear):
            kl = kl + module.kl_divergence()
    return kl


class SeparableHNN(nn.Module):
    """Spring-chain HNN, H(p, q, K) = T(p) + V(q, K).

    T(p) is a per-particle MLP summed over particles; V(q, K) = sum over the non-zero edges
    (i, j) of K of an MLP f(q_i, q_j, k_ij) (O(nnz) instead of O(N^2)).
    """

    def __init__(
        self,
        n_hidden: int,
        dim: int,
        drop_prob: float = 0.2,
        mask_num: int = 100,
        nonlin: str = "intsin2x",
        bias: bool = False,
        dropout_variant: DropoutVariant = DropoutVariant.MC_DROPOUT,
        finite_mask_mode: FiniteMaskMode = FiniteMaskMode.BOTH,
        concrete_temp: float = 0.1,
        scale_weights: Optional[bool] = True,
        mask_last_layer: bool = False,
        finite_mask_storage: str = "materialized",
        mask_seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.dim = dim
        self.mask_last_layer = mask_last_layer
        dropout_kwargs = dict(
            drop_prob=drop_prob, mask_num=mask_num, dropout_variant=dropout_variant,
            finite_mask_mode=finite_mask_mode, concrete_temp=concrete_temp,
            scale_weights=scale_weights, finite_mask_storage=finite_mask_storage,
            mask_seed=mask_seed,
        )
        if not mask_last_layer:
            self.linear_K1 = MaskedLinear(1, n_hidden, bias=bias, **dropout_kwargs)
            self.linear_K2 = nn.Linear(n_hidden, 1, bias=bias)
        else:
            self.linear_K1 = nn.Linear(1, n_hidden, bias=bias)
            self.linear_K2 = MaskedLinear(n_hidden, 1, bias=bias, **dropout_kwargs)
        self.linear_P1, self.linear_P2 = _build_edge_mlp(
            3, n_hidden, mask_last_layer, bias, dropout_kwargs
        )
        self.nonlin = get_activation_fn(nonlin)
        self._potential_edge_energy = SparseEdgeEnergy(
            linear_1=self.linear_P1,
            linear_2=self.linear_P2,
            nonlin=self.nonlin,
            mask_last_layer=self.mask_last_layer,
            edge_feature_mode="wall_edge",
            use_edge_type_feature=False,
            allow_nonzero_fallback=True,
        )

    def forward(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        index: Optional[int] = None,
    ) -> torch.Tensor:
        """H = T(p) + V(q, K) for one mask index; p, q: (batch, dim), k: (dim, dim) or (batch, dim, dim)."""
        h = _mlp2(p.unsqueeze(-1), self.linear_K1, self.linear_K2, self.nonlin,
                  self.mask_last_layer, mask_index=index)
        T = h.sum(dim=1)  # (batch, 1)
        V = self._potential_edge_energy.forward(q, k, mask_index=index)
        return (T + V).squeeze(-1)

    def kl_divergence(self) -> torch.Tensor:
        """Sum of the layers' KL regularisers (Concrete / Variational dropout)."""
        return _sum_kl(self)

    def batch_hamiltonian(
        self,
        p_all: torch.Tensor,
        q_all: torch.Tensor,
        k: torch.Tensor,
        mask_idxs_int: list,
    ) -> torch.Tensor:
        """H for n_mc mask groups; group i is rows [i*n_ics:(i+1)*n_ics]. Returns (n_mc*n_ics,)."""
        h = _mlp2(p_all.unsqueeze(-1), self.linear_K1, self.linear_K2, self.nonlin,
                  self.mask_last_layer, mask_idxs_int=mask_idxs_int)
        T = h.sum(dim=1)  # (n_mc*n_ics, 1)
        V = self._potential_edge_energy.batch_forward_grouped(q_all, k, mask_idxs_int)
        return (T + V).squeeze(-1)


class MassRatioSimpleHNN(nn.Module):
    """Coupled-pendulum HNN conditioned on the scalar mass ratio r = m2 / m1.

    H(p, q, r) = T_nn([p_i, log r]) + V_nn([q_1..q_n, log r]): log r is appended to every
    per-particle momentum input and to the joint position vector.

    Args:
        dim: number of particles (4 for the coupled pendulum).
    """

    def __init__(
        self,
        dim: int,
        n_hidden: int = 256,
        drop_prob: float = 0.2,
        mask_num: int = 100,
        nonlin: str = "intsin2x",
        dropout_variant: DropoutVariant = DropoutVariant.MC_DROPOUT,
        finite_mask_mode: FiniteMaskMode = FiniteMaskMode.BOTH,
        concrete_temp: float = 0.1,
        scale_weights: Optional[bool] = True,
        mask_last_layer: bool = False,
        finite_mask_storage: str = "materialized",
        mask_seed: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.mask_last_layer = mask_last_layer
        n_p_features = 2  # [p_i, log_r] per particle
        n_q_features = dim + 1  # [q_1..q_dim, log_r]
        dropout_kwargs = dict(
            drop_prob=drop_prob, mask_num=mask_num, dropout_variant=dropout_variant,
            finite_mask_mode=finite_mask_mode, concrete_temp=concrete_temp,
            scale_weights=scale_weights, finite_mask_storage=finite_mask_storage,
            mask_seed=mask_seed,
        )
        if not mask_last_layer:
            self.linear_K1 = MaskedLinear(n_p_features, n_hidden, bias=False, **dropout_kwargs)
            self.linear_K2 = nn.Linear(n_hidden, 1, bias=False)
            self.linear_P1 = MaskedLinear(n_q_features, n_hidden, **dropout_kwargs)
            self.linear_P2 = nn.Linear(n_hidden, 1, bias=False)
        else:
            self.linear_K1 = nn.Linear(n_p_features, n_hidden, bias=False)
            self.linear_K2 = MaskedLinear(n_hidden, 1, bias=False, **dropout_kwargs)
            self.linear_P1 = nn.Linear(n_q_features, n_hidden)
            self.linear_P2 = MaskedLinear(n_hidden, 1, bias=False, **dropout_kwargs)
        self.nonlin = get_activation_fn(nonlin)

    def _log_r(self, k: torch.Tensor, device: torch.device) -> torch.Tensor:
        """log(mass ratio) as a (1,) tensor."""
        return torch.log(k.to(device).float().reshape(-1)[0:1])

    def _energy(self, p, q, k, mask_index=None, mask_idxs_int=None) -> torch.Tensor:
        """T + V for (batch, dim) inputs; one mask index or grouped mask indices. Returns (batch,)."""
        log_r = self._log_r(k, p.device)
        batch, n_particles = p.shape
        log_r_T = log_r.view(1, 1, 1).expand(batch, n_particles, 1)
        h = torch.cat([p.unsqueeze(-1), log_r_T], dim=-1)  # (batch, n_particles, 2)
        h = _mlp2(h, self.linear_K1, self.linear_K2, self.nonlin, self.mask_last_layer,
                  mask_index=mask_index, mask_idxs_int=mask_idxs_int)
        T = h.sum(dim=1).squeeze(-1)  # (batch,)
        q_aug = torch.cat([q, log_r.expand(batch, 1)], dim=-1)  # (batch, dim + 1)
        x = _mlp2(q_aug, self.linear_P1, self.linear_P2, self.nonlin, self.mask_last_layer,
                  mask_index=mask_index, mask_idxs_int=mask_idxs_int)
        return T + x.squeeze(-1)

    def forward(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        index: Optional[int] = None,
    ) -> torch.Tensor:
        """H(p, q, r) for one mask index; k is the scalar mass ratio."""
        return self._energy(p, q, k, mask_index=index)

    def batch_hamiltonian(
        self,
        p_all: torch.Tensor,
        q_all: torch.Tensor,
        k: torch.Tensor,
        mask_idxs_int: list,
    ) -> torch.Tensor:
        """H for n_mc mask groups of (n_mc*n_ics, dim) inputs. Returns (n_mc*n_ics,)."""
        return self._energy(p_all, q_all, k, mask_idxs_int=mask_idxs_int)

    def kl_divergence(self) -> torch.Tensor:
        """Sum of the layers' KL regularisers (Concrete / Variational dropout)."""
        return _sum_kl(self)


class OperatorSeparableHNN(nn.Module):
    """TFIM HNN, H(p, q, g) = E_p(p; g) + E_q(q; g).

    Both terms are sparse operator energies over the operator edge list, evaluated by one
    edge MLP shared between p and q (``self_edge`` features plus an edge-type flag).
    """

    def __init__(
        self,
        dim: int,
        n_hidden: int,
        drop_prob: float = 0.2,
        mask_num: int = 100,
        nonlin: str = "intsin2x",
        bias: bool = False,
        dropout_variant: DropoutVariant = DropoutVariant.MC_DROPOUT,
        finite_mask_mode: FiniteMaskMode = FiniteMaskMode.BOTH,
        concrete_temp: float = 0.1,
        scale_weights: Optional[bool] = True,
        mask_last_layer: bool = False,
        finite_mask_storage: str = "materialized",
        mask_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_hidden = int(n_hidden)
        self.mask_last_layer = bool(mask_last_layer)
        dropout_kwargs = dict(
            drop_prob=drop_prob, mask_num=mask_num, dropout_variant=dropout_variant,
            finite_mask_mode=finite_mask_mode, concrete_temp=concrete_temp,
            scale_weights=scale_weights, finite_mask_storage=finite_mask_storage,
            mask_seed=mask_seed,
        )
        self.nonlin = get_activation_fn(nonlin)
        self.linear_E1, self.linear_E2 = _build_edge_mlp(
            4, n_hidden, mask_last_layer, bias, dropout_kwargs
        )
        shared_energy = SparseEdgeEnergy(
            linear_1=self.linear_E1,
            linear_2=self.linear_E2,
            nonlin=self.nonlin,
            mask_last_layer=self.mask_last_layer,
            edge_feature_mode="self_edge",
            use_edge_type_feature=True,
            allow_nonzero_fallback=False,
        )
        self.edge_energy_p = shared_energy
        self.edge_energy_q = shared_energy

    def register_operator_structure(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> None:
        """Register the sparse operator edge list (shared by the p and q terms)."""
        self.edge_energy_p.register_structure(edge_index, edge_type=edge_type)

    def forward(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        index: Optional[int] = None,
    ) -> torch.Tensor:
        """H(p, q, g) = E_p + E_q for one mask index; k is the operator control tensor."""
        E_p = self.edge_energy_p.forward(p, k, mask_index=index)
        E_q = self.edge_energy_q.forward(q, k, mask_index=index)
        return (E_p + E_q).squeeze(-1)

    def batch_hamiltonian(
        self,
        p_all: torch.Tensor,
        q_all: torch.Tensor,
        k: torch.Tensor,
        mask_idxs_int: list,
    ) -> torch.Tensor:
        """Grouped H for (n_mc*n_ics, dim) inputs. Returns (n_mc*n_ics,)."""
        E_p = self.edge_energy_p.batch_forward_grouped(p_all, k, mask_idxs_int)
        E_q = self.edge_energy_q.batch_forward_grouped(q_all, k, mask_idxs_int)
        return (E_p + E_q).squeeze(-1)

    def kl_divergence(self) -> torch.Tensor:
        """Sum of the layers' KL regularisers (Concrete / Variational dropout)."""
        return _sum_kl(self)


def build_hnn_model(
    dim: int,
    n_hidden: int,
    drop_prob: float,
    mask_num: int,
    nonlin: str,
    device: torch.device,
    model_type: str,
    dropout_variant: str = "mc_dropout",
    finite_mask_mode: str = "both",
    concrete_temp: float = 0.1,
    scale_weights: Optional[bool] = True,
    mask_last_layer: bool = False,
    finite_mask_storage: str = "materialized",
    mask_seed: Optional[int] = None,
) -> nn.Module:
    """Build the HNN of the given ``model_type`` and seed its seeded-mask banks.

    ``dim`` is the number of degrees of freedom (half the canonical state dimension).
    """
    shared_kwargs = dict(
        n_hidden=n_hidden,
        drop_prob=drop_prob,
        mask_num=mask_num,
        nonlin=nonlin,
        dropout_variant=DropoutVariant(dropout_variant),
        finite_mask_mode=FiniteMaskMode(finite_mask_mode),
        concrete_temp=concrete_temp,
        scale_weights=scale_weights,
        mask_last_layer=mask_last_layer,
        finite_mask_storage=finite_mask_storage,
        mask_seed=mask_seed,
    )
    if model_type == "TFIMSeparableHNN":
        model = OperatorSeparableHNN(dim=dim, **shared_kwargs)
    elif model_type == "MassRatioSimpleHNN":
        model = MassRatioSimpleHNN(dim=dim, **shared_kwargs)
    elif model_type == "SeparableHNN":
        model = SeparableHNN(dim=dim, **shared_kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    model = model.to(device)
    configure_masked_linear_seeds(model, mask_seed)
    return model
