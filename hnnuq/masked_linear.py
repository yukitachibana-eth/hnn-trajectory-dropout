"""masked_linear.py

Linear layer with MC / Concrete / Variational dropout and an optional finite mask bank.

Two mask semantics are used in the paper:

- Fresh-mask MC (``finite_mask_mode == NONE``, MC dropout): every call draws a new
  mask. ``forward`` applies ``F.dropout`` to the layer's output activations with
  ``training=True`` (active in eval mode too); ``batch_forward_grouped`` draws one
  independent weight-space Bernoulli mask per requested group and runs a single
  ``torch.bmm``. ``mask_index`` is ignored on this path.
- Fixed-mask MC (``finite_mask_mode == BOTH``, MC dropout): a bank of ``mask_num``
  weight-space Bernoulli masks is indexed by ``mask_index``. The same index always
  gives the same effective weight matrix, so a rollout that reuses one index at every
  leapfrog sub-step integrates one fixed (symplectic) Hamiltonian. The bank is either
  materialised (a ``(mask_num, out, in)`` buffer drawn lazily from the global RNG) or
  seeded (each mask regenerated from a stable hash of seed, module path and index).

Concrete and Variational dropout always draw fresh noise (weight-space Gumbel-Softmax
relaxation resp. Gaussian multiplicative noise); their masks are never banked.
"""

import math
import hashlib
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropoutVariant(Enum):
    """Dropout family of a MaskedLinear layer."""
    MC_DROPOUT = "mc_dropout"
    CONCRETE_DROPOUT = "concrete_dropout"
    VARIATIONAL_DROPOUT = "variational_dropout"


class FiniteMaskMode(Enum):
    """Whether a layer draws fresh masks or uses a finite mask bank."""
    NONE = "none"  # fresh masks on every call
    BOTH = "both"  # finite bank in training and at inference


class MaskedLinear(nn.Linear):
    """``nn.Linear`` whose weights are perturbed by the configured dropout variant."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dropout_variant: DropoutVariant = DropoutVariant.MC_DROPOUT,
        drop_prob: float = 0.2,
        mask_num: int = 100,
        finite_mask_mode: FiniteMaskMode = FiniteMaskMode.BOTH,
        concrete_temp: float = 0.1,
        device: Optional[str] = None,
        scale_weights: Optional[bool] = True,
        finite_mask_storage: str = "materialized",
        mask_seed: Optional[int] = None,
        mask_cache_size: int = 64,
    ):
        super().__init__(in_features, out_features, bias=bias)

        self.dropout_variant = dropout_variant
        self.drop_prob = float(drop_prob)
        self.mask_num = mask_num
        self.finite_mask_mode = finite_mask_mode
        self.concrete_temp = concrete_temp
        self.scale_weights = scale_weights
        self.finite_mask_storage = str(finite_mask_storage).lower()
        self.mask_seed = int(mask_seed) if mask_seed is not None else 0
        self.mask_cache_size = int(mask_cache_size)
        self.mask_key = f"MaskedLinear:{in_features}:{out_features}"
        self.drop_logit_min = -4.6  # sigmoid(-4.6) ~ 0.01
        self.drop_logit_max = 1.0
        self.log_alpha_min = -8.0
        self.log_alpha_max = 3.0

        if self.finite_mask_storage not in {"materialized", "seeded"}:
            raise ValueError(f"Unknown finite_mask_storage: {self.finite_mask_storage}")
        if finite_mask_mode != FiniteMaskMode.NONE and dropout_variant != DropoutVariant.MC_DROPOUT:
            raise ValueError("Finite mask banks are supported only for mc_dropout.")

        if dropout_variant == DropoutVariant.CONCRETE_DROPOUT:
            init_logit = torch.log(torch.tensor(drop_prob / (1 - drop_prob + 1e-8)))
            self.drop_logit = nn.Parameter(torch.full((1,), init_logit.item()))
        elif dropout_variant == DropoutVariant.VARIATIONAL_DROPOUT:
            init_log_alpha = torch.log(torch.tensor(drop_prob / (1 - drop_prob + 1e-8)))
            self.log_alpha = nn.Parameter(torch.full(self.weight.shape, init_log_alpha.item()))

        # Materialised bank, drawn lazily; seeded banks are regenerated, never stored.
        self.register_buffer(
            "masks", None, persistent=(self.finite_mask_storage == "materialized")
        )
        self._mask_cache = OrderedDict()
        self._mask_cache_limit = self.mask_cache_size

    def configure_mask_seed(self, mask_seed: Optional[int], mask_key: str):
        """Set the seed / module path that define seeded masks."""
        if mask_seed is not None:
            self.mask_seed = int(mask_seed)
        self.mask_key = str(mask_key)
        self._mask_cache.clear()

    def _uses_seeded_finite_masks(self) -> bool:
        return (
            self.finite_mask_storage == "seeded"
            and self.dropout_variant == DropoutVariant.MC_DROPOUT
            and self.finite_mask_mode != FiniteMaskMode.NONE
        )

    def _validate_mask_index(self, mask_index: int) -> int:
        return int(mask_index) % int(self.mask_num)

    def _validate_mask_indices(self, mask_indices: List[int]) -> List[int]:
        return [self._validate_mask_index(mask_index) for mask_index in mask_indices]

    def _stable_mask_seed(self, mask_index: int) -> int:
        payload = f"{self.mask_seed}:{self.mask_key}:{int(mask_index)}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)

    def _sample_seeded_mc_mask(self, mask_index: int) -> torch.Tensor:
        """One deterministic Bernoulli keep-mask (bool, weight shape) for a bank index."""
        mask_index = self._validate_mask_index(mask_index)
        device = self.weight.device
        keep_prob = 1.0 - self.drop_prob
        cache_key = (mask_index, str(device), round(keep_prob, 12))
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            self._mask_cache.move_to_end(cache_key)
            return cached

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._stable_mask_seed(mask_index))
        mask = (torch.rand(self.weight.shape, generator=generator, device="cpu") < keep_prob).to(
            device=device
        )
        self._mask_cache[cache_key] = mask
        self._prune_mask_cache()
        return mask

    def _prune_mask_cache(self):
        while len(self._mask_cache) > self._mask_cache_limit:
            self._mask_cache.popitem(last=False)

    def _get_finite_mask(self, mask_index: int) -> torch.Tensor:
        mask_index = self._validate_mask_index(mask_index)
        if self._uses_seeded_finite_masks():
            return self._sample_seeded_mc_mask(mask_index)
        self._initialize_masks()
        return self.masks[mask_index]

    def _get_finite_masks(self, mask_idxs_int: List[int]) -> torch.Tensor:
        """Stack the masks of several bank indices: (n_mc, out, in)."""
        mask_idxs_int = self._validate_mask_indices(mask_idxs_int)
        if self._uses_seeded_finite_masks():
            # A grouped rollout reuses the whole group at every Hamiltonian call; keep it cached.
            n_requested = len(set(mask_idxs_int))
            self._mask_cache_limit = max(self.mask_cache_size, n_requested)
            masks = torch.stack([self._sample_seeded_mc_mask(int(i)) for i in mask_idxs_int])
            self._prune_mask_cache()
            return masks
        self._initialize_masks()
        idx_t = torch.tensor(mask_idxs_int, dtype=torch.long, device=self.weight.device)
        return self.masks[idx_t]

    def _initialize_masks(self):
        """Lazily draw the materialised Bernoulli keep-bank (mask_num, out, in) from the global RNG."""
        if self.masks is None:
            keep_prob = 1.0 - self.drop_prob
            mask_shape = (self.mask_num, *self.weight.shape)
            self.masks = torch.bernoulli(torch.full(mask_shape, keep_prob, device=self.weight.device))

    def discard_materialized_masks(self):
        """Drop the materialised bank so the next forward draws a new one (RNG order matters).

        Seeded banks are deterministic and need no reset.
        """
        self._mask_cache.clear()
        if (
            self.finite_mask_storage == "materialized"
            and self.dropout_variant == DropoutVariant.MC_DROPOUT
            and self.finite_mask_mode != FiniteMaskMode.NONE
        ):
            self.masks = None

    def _use_finite_masks(self) -> bool:
        return self.finite_mask_mode != FiniteMaskMode.NONE

    @property
    def effective_drop_prob(self) -> torch.Tensor:
        """Learned drop probability of a Concrete layer (clamped sigmoid of the logit)."""
        clamped = self.drop_logit.clamp(self.drop_logit_min, self.drop_logit_max)
        return torch.sigmoid(clamped)

    def _sample_concrete_mask(self, shape: tuple) -> torch.Tensor:
        """Concrete (Gumbel-Softmax) relaxed Bernoulli keep-mask of the given shape."""
        p = self.effective_drop_prob
        u = torch.rand(shape, device=self.weight.device).clamp(1e-8, 1 - 1e-8)
        s = (torch.log(u) - torch.log(1 - u) + torch.log(1 - p) - torch.log(p)) / self.concrete_temp
        return torch.sigmoid(s)

    def _mc_dropout_forward(self, input: torch.Tensor, mask_index: Optional[int]) -> torch.Tensor:
        if self._use_finite_masks():
            if mask_index is None:
                raise ValueError("Fixed-mask MC needs a mask_index")
            mask = self._get_finite_mask(mask_index).to(dtype=self.weight.dtype)
            scale = 1.0 / (1.0 - self.drop_prob + 1e-8) if self.scale_weights else 1.0
            return F.linear(input, mask * self.weight * scale, self.bias)
        return F.dropout(
            F.linear(input, self.weight, self.bias), p=self.drop_prob, training=True
        )

    def _concrete_dropout_forward(self, input: torch.Tensor, mask_index: Optional[int]) -> torch.Tensor:
        # Fresh relaxed mask on every call so that drop_logit receives gradients.
        p = self.effective_drop_prob.item()
        scale = 1.0 / (1.0 - p + 1e-8)
        mask = self._sample_concrete_mask(self.weight.shape)
        return F.linear(input, mask * self.weight * scale, self.bias)

    def _variational_dropout_forward(self, input: torch.Tensor, mask_index: Optional[int]) -> torch.Tensor:
        log_alpha = self.log_alpha.clamp(self.log_alpha_min, self.log_alpha_max)
        std = torch.sqrt(torch.exp(log_alpha) + 1e-8)
        epsilon = torch.randn_like(self.weight)  # fresh Gaussian noise, also at inference
        return F.linear(input, self.weight * (1.0 + std * epsilon), self.bias)

    def forward(self, input: torch.Tensor, mask_index: Optional[int] = None) -> torch.Tensor:
        """Forward pass; ``mask_index`` selects the bank mask (Fixed-mask MC only)."""
        if self.dropout_variant == DropoutVariant.MC_DROPOUT:
            return self._mc_dropout_forward(input, mask_index)
        if self.dropout_variant == DropoutVariant.CONCRETE_DROPOUT:
            return self._concrete_dropout_forward(input, mask_index)
        return self._variational_dropout_forward(input, mask_index)

    def kl_divergence(self) -> torch.Tensor:
        """KL regulariser: Kingma-style approximation (Variational) or Bernoulli KL (Concrete)."""
        if self.dropout_variant == DropoutVariant.VARIATIONAL_DROPOUT:
            log_alpha = self.log_alpha.clamp(-10, 10)
            k1, k2, k3 = 0.63576, 1.87320, 1.48695
            return -torch.sum(
                k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * F.softplus(-log_alpha) - k1
            )
        if self.dropout_variant == DropoutVariant.CONCRETE_DROPOUT:
            p = self.effective_drop_prob
            p0 = torch.tensor(self.drop_prob, device=p.device)
            kl = p * torch.log((p + 1e-8) / (p0 + 1e-8)) + (1 - p) * torch.log(
                (1 - p + 1e-8) / (1 - p0 + 1e-8)
            )
            return kl.sum()
        raise ValueError("kl_divergence is defined for Concrete and Variational dropout only")

    def batch_forward_grouped(self, input: torch.Tensor, mask_idxs_int: list) -> torch.Tensor:
        """Grouped forward: group ``i`` of the input uses the mask / noise of ``mask_idxs_int[i]``.

        One ``torch.bmm`` replaces ``n_mc`` sequential forward calls.

        Args:
            input: (n_mc * n_ics, *inner_dims, in_features); group i is input[i*n_ics:(i+1)*n_ics].
            mask_idxs_int: n_mc bank indices (ignored by fresh-noise variants).

        Returns:
            (n_mc * n_ics, *inner_dims, out_features)
        """
        n_mc = len(mask_idxs_int)
        total = input.shape[0]
        inner_dims = input.shape[1:-1]
        in_f = input.shape[-1]
        device = self.weight.device

        use_finite = self._use_finite_masks()
        mask_idxs_int = self._validate_mask_indices(mask_idxs_int)

        # Effective weights for all groups: (n_mc, out_f, in_f)
        if self.dropout_variant == DropoutVariant.MC_DROPOUT:
            scale = 1.0 / (1.0 - self.drop_prob + 1e-8) if self.scale_weights else 1.0
            if use_finite:
                masks = self._get_finite_masks(mask_idxs_int).to(dtype=self.weight.dtype)
                W_eff = masks * self.weight.unsqueeze(0) * scale
            else:
                keep_prob = 1.0 - self.drop_prob
                fresh = torch.bernoulli(
                    torch.full((n_mc, *self.weight.shape), keep_prob, device=device)
                )
                W_eff = fresh * self.weight.unsqueeze(0) * scale
        elif self.dropout_variant == DropoutVariant.CONCRETE_DROPOUT:
            p = self.effective_drop_prob.item()
            scale = 1.0 / (1.0 - p + 1e-8)
            fresh = self._sample_concrete_mask((n_mc, *self.weight.shape))
            W_eff = fresh * self.weight.unsqueeze(0) * scale
        else:  # VARIATIONAL_DROPOUT
            log_alpha = self.log_alpha.clamp(self.log_alpha_min, self.log_alpha_max)
            std = torch.sqrt(torch.exp(log_alpha) + 1e-8)
            noise = torch.randn(n_mc, *self.weight.shape, device=device)
            W_eff = self.weight.unsqueeze(0) * (1.0 + std * noise)

        # (n_mc*n_ics, *inner, in_f) -> (n_mc, n_ics*prod(inner), in_f) -> bmm
        n_ics = total // n_mc
        inner_prod = math.prod(inner_dims) if inner_dims else 1
        x_g = input.reshape(n_mc, n_ics * inner_prod, in_f)
        out_g = torch.bmm(x_g, W_eff.transpose(1, 2))
        if self.bias is not None:
            out_g = out_g + self.bias
        return out_g.reshape(total, *inner_dims, self.out_features)


def configure_masked_linear_seeds(model: nn.Module, mask_seed: Optional[int]) -> None:
    """Give every MaskedLinear a seed and a stable module-path key for seeded masks."""
    for module_name, module in model.named_modules():
        if isinstance(module, MaskedLinear):
            module.configure_mask_seed(mask_seed, module_name)


def discard_model_materialized_masks(model: nn.Module) -> None:
    """Reset the materialised mask bank of every MaskedLinear (see ``discard_materialized_masks``)."""
    for module in model.modules():
        if isinstance(module, MaskedLinear):
            module.discard_materialized_masks()


def get_model_active_mask_num(model: nn.Module) -> int:
    """Size of the mask bank (a model-level override, e.g. an ensemble wrapper, takes precedence)."""
    if not isinstance(model, MaskedLinear) and hasattr(model, "get_active_mask_num"):
        return int(model.get_active_mask_num())
    for module in model.modules():
        if isinstance(module, MaskedLinear):
            return int(module.mask_num)
    raise ValueError("Model does not contain any MaskedLinear modules")


def load_state_dict_with_optional_masks(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = False,
    device: Optional[torch.device] = None,
):
    """Load weights, then attach the saved materialised mask bank (``*.masks``) of each MaskedLinear.

    The bank is a lazily created buffer (None until first use), so it is set after ``load_state_dict``
    rather than loaded with it. Seeded banks are not stored in checkpoints.
    """
    state_dict_copy = OrderedDict(state_dict)
    if hasattr(state_dict, "_metadata"):
        state_dict_copy._metadata = state_dict._metadata  # type: ignore[attr-defined]
    mask_tensors = {k: state_dict_copy.pop(k) for k in list(state_dict_copy) if ".masks" in k}
    result = model.load_state_dict(state_dict_copy, strict=strict)

    for mask_key, mask_tensor in mask_tensors.items():
        module = model.get_submodule(mask_key.rsplit(".masks", 1)[0])
        module.masks = mask_tensor.to(device if device is not None else module.weight.device)
    return result
