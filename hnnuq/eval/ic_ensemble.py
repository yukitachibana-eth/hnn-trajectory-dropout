"""IC-noise x mask ensembles: rollout, noise-scale fitting, per-case scores.

An ensemble perturbs the initial state of each test case with N iid Gaussian draws (scale sigma) and rolls every perturbed IC out
under the same M masks. Epistemic variance is the within-IC spread across masks, aleatoric the between-IC spread of the mask means.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar

from hnnuq.eval.common import central_coverage, gaussian_nll, variance_decomposition
from hnnuq.integration import integrate_mc_samples

# offsets of the two CPU generators that draw the measurement noise added to predictive samples (M-only and N x M)
NOISE_SEED_OFFSET_MASKS_ONLY = 606060
NOISE_SEED_OFFSET_ENSEMBLE = 707070


def mask_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """CPU generator for mask-index selection (None -> global RNG)."""
    if seed is None:
        return None
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return gen


def select_masks(n_masks: int, bank_size: int, generator: Optional[torch.Generator], device: torch.device) -> torch.Tensor:
    """First ``n_masks`` of a random permutation of the bank (drawn on the generator's device, moved to ``device``)."""
    gen_device = generator.device if generator is not None else device
    perm = torch.randperm(bank_size, device=gen_device, generator=generator)[: min(n_masks, bank_size)]
    return perm.to(device)


@torch.no_grad()
def ic_noise_ensemble(z_0: torch.Tensor, model: nn.Module, noise_std: float, n_ic: int, n_mc: int, T: int, dt: float,
                      bank_size: int, k_matrix: Optional[torch.Tensor], use_leapfrog: bool, seed_offset: int,
                      mask_subset: Optional[torch.Tensor] = None, fresh_mask_semantics: str = "weight_group",
                      generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """N x M ensemble (N, M, T, s, 2d) for test ICs ``z_0`` (s, 2d), all N perturbed ICs rolled out in one batch.

    IC i is perturbed with noise drawn from a generator seeded ``seed_offset + i`` on ``z_0``'s device: p-noise then q-noise,
    each ``noise_std * randn(s, d)``. The M masks are shared by all ICs: ``mask_subset`` if given, else the first M of a random
    permutation of the ``bank_size`` masks drawn from ``generator``.
    """
    dim = z_0.shape[-1] // 2
    s = z_0.shape[0]
    noise_p = torch.empty(n_ic, s, dim, device=z_0.device)
    noise_q = torch.empty(n_ic, s, dim, device=z_0.device)
    for i in range(n_ic):
        gen = torch.Generator(device=z_0.device)
        gen.manual_seed(seed_offset + i)
        noise_p[i] = noise_std * torch.randn(s, dim, device=z_0.device, generator=gen)
        noise_q[i] = noise_std * torch.randn(s, dim, device=z_0.device, generator=gen)
    all_p = z_0[:, :dim].unsqueeze(0) + noise_p  # (N, s, d)
    all_q = z_0[:, dim:].unsqueeze(0) + noise_q
    masks = mask_subset if mask_subset is not None else select_masks(n_mc, bank_size, generator, z_0.device)
    traj = integrate_mc_samples(
        p_0=all_p.reshape(n_ic * s, dim), q_0=all_q.reshape(n_ic * s, dim), model=model, T=T, dt=dt, n_mc_samples=n_mc,
        k_matrix=k_matrix, trajectory_consistent=True, use_leapfrog=use_leapfrog, mask_subset=masks,
        fresh_mask_semantics=fresh_mask_semantics, generator=generator,
    )  # (M, T, N*s, 2d)
    m = traj.shape[0]
    return traj.reshape(m, T, n_ic, s, 2 * dim).permute(2, 0, 1, 3, 4).contiguous()


@torch.no_grad()
def fit_noise_scale(model: nn.Module, val_data: torch.Tensor, k_val: Optional[torch.Tensor], dt: float, n_ic: int, n_mc: int,
                    bank_size: int, use_leapfrog: bool, seed: int, mask_seed: Optional[int],
                    bounds: Tuple[float, float] = (0.001, 0.5)) -> Tuple[float, float]:
    """IC-noise scale minimising the ensemble NLL on validation data (bounded scalar search, xatol 1e-3, <= 50 evaluations).

    ``val_data`` (n_k, T, n_val, 2d) holds one validation set per control cell (``k_val[i]``); the objective is the cell-mean NLL
    of the N x M ensemble. Every candidate scale reuses the same masks (a fresh mask generator per cell) and the same IC noise
    seeds (``seed * 1000 + 100 * cell``). Returns (sigma_hat, NLL at sigma_hat).
    """
    n_k, T = val_data.shape[0], val_data.shape[1]

    def objective(sigma: float) -> float:
        """Cell-mean validation NLL of the ensemble at IC-noise scale ``sigma``."""
        total = 0.0
        for i in range(n_k):
            target = val_data[i]
            ens = ic_noise_ensemble(
                target[0], model, sigma, n_ic, n_mc, T, dt, bank_size, k_val[i] if k_val is not None else None,
                use_leapfrog, seed_offset=seed * 1000 + i * 100, generator=mask_generator(mask_seed),
            )
            total += gaussian_nll(ens.cpu().flatten(0, 1), target.cpu())
        return total / n_k

    result = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-3, "maxiter": 50})
    return float(result.x), float(result.fun)


def add_measurement_noise(masks_only: torch.Tensor, ensemble: torch.Tensor, noise_std: float, seed: int):
    """Copies of the M-only samples and the N x M ensemble with iid N(0, noise_std^2) added (two dedicated CPU generators)."""
    gen_a = torch.Generator(device="cpu")
    gen_a.manual_seed(int(seed) + NOISE_SEED_OFFSET_MASKS_ONLY)
    a = masks_only.clone()
    a.add_(torch.empty_like(a).normal_(mean=0.0, std=noise_std, generator=gen_a))
    gen_b = torch.Generator(device="cpu")
    gen_b.manual_seed(int(seed) + NOISE_SEED_OFFSET_ENSEMBLE)
    b = ensemble.clone()
    b.add_(torch.empty_like(b).normal_(mean=0.0, std=noise_std, generator=gen_b))
    return a, b


def ic_case_scores(masks_only: torch.Tensor, ensemble: torch.Tensor, targets: torch.Tensor, skip_first_step: bool,
                   measurement_noise_std: float = 0.0, noise_seed: int = 0, level: float = 0.9) -> Dict[str, float]:
    """Scores of one evaluation case (CPU tensors).

    masks_only (M, T, s, d) and ensemble (N, M, T, s, d) are scored against targets (T, s, d): Gaussian NLL, empirical coverage
    at ``level`` and its absolute error, for the masks-only samples and for the N x M ensemble; with ``measurement_noise_std > 0``
    iid noise is added to both sample sets first (variances are computed from the noise-free ensemble).
    """
    var_epi, var_alea = variance_decomposition(ensemble, skip_first_step)
    a, b = masks_only, ensemble
    if measurement_noise_std > 0.0:
        a, b = add_measurement_noise(masks_only, ensemble, measurement_noise_std, noise_seed)
    flat = b.flatten(0, 1)  # (N*M, T, s, d)
    cov_masks, cov_ens = central_coverage(a, targets, level), central_coverage(flat, targets, level)
    return {
        "nll_masks_only": gaussian_nll(a, targets), "nll_ensemble": gaussian_nll(flat, targets),
        "cov90_masks_only": cov_masks, "cov90_ensemble": cov_ens,
        "cov90_abs_error_ensemble": abs(cov_ens - level), "var_epi": var_epi, "var_alea": var_alea,
    }
