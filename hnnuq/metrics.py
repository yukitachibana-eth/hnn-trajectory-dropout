"""Uncertainty and energy-drift metrics."""

import torch
import numpy as np
from typing import Dict, Tuple


class UQMetrics:
    """Uncertainty quantification metrics for MC predictions of shape (n_mc, ...)."""

    @staticmethod
    @torch.no_grad()
    def compute_nll_gaussian(
        predictions: torch.Tensor,  # (n_mc, ...)
        targets: torch.Tensor,      # (...)
        eps: float = 1e-6,
    ) -> float:
        """Negative log-likelihood under a Gaussian fitted to the MC samples (variance floor eps)."""
        mean = predictions.mean(dim=0)
        var = predictions.var(dim=0).clamp(min=eps)
        nll = 0.5 * (torch.log(2 * np.pi * var) + (targets - mean) ** 2 / var)
        return nll.mean().item()

    @staticmethod
    @torch.no_grad()
    def compute_crps(
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Continuous Ranked Probability Score."""
        n_mc = predictions.shape[0]
        if n_mc < 2:
            return 0.0
        
        sorted_preds, _ = predictions.sort(dim=0)
        targets_expanded = targets.unsqueeze(0)
        
        term1 = (sorted_preds - targets_expanded).abs().mean(dim=0)
        
        indices = torch.arange(1, n_mc + 1, device=predictions.device, dtype=torch.float32)
        weights = (2 * indices - n_mc - 1) / (n_mc * (n_mc - 1))
        term2 = (sorted_preds * weights.view(-1, *([1] * (predictions.dim() - 1)))).sum(dim=0).abs()
        
        crps = term1 - term2
        return crps.mean().item()
    
    @staticmethod
    @torch.no_grad()
    def compute_coverage(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        nominal_levels: Tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
    ) -> Dict[float, float]:
        """Empirical coverage at different confidence levels."""
        n_mc = predictions.shape[0]
        sorted_preds, _ = predictions.sort(dim=0)
        coverage = {}
        
        for level in nominal_levels:
            alpha = 1 - level
            lower_idx = max(0, int(np.floor(alpha / 2 * n_mc)))
            upper_idx = min(n_mc - 1, int(np.ceil((1 - alpha / 2) * n_mc)) - 1)
            
            lower = sorted_preds[lower_idx]
            upper = sorted_preds[upper_idx]
            
            in_interval = (targets >= lower) & (targets <= upper)
            coverage[level] = in_interval.float().mean().item()
        
        return coverage


class EnergyMetrics:
    """Energy conservation metrics for Hamiltonian systems."""

    @staticmethod
    def compute_drift_stats(energies: torch.Tensor) -> Dict[str, float]:
        """Mean, last-step mean and maximum of the energy drift |H(t) - H(0)| for energies (n_mc, T, n_ics)."""
        drift = (energies - energies[:, 0:1, :]).abs()
        return {
            "drift_mean": drift.mean().item(),
            "drift_final_mean": drift[:, -1, :].mean().item(),
            "drift_max": drift.max().item(),
        }
