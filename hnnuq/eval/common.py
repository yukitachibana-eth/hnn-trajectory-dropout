"""Shared evaluation helpers: model/ensemble loading, run-tree layout, mask-semantics choice, predictive scores.

Run tree (one directory per system / arm / seed; the evaluation stages write into ``<run>/eval/``):

    runs/<system>/<arm>/seed<N>/{config.yaml, best_model.pt, ...}      arm in ARM_DIRS
    runs/<system>/deep_ensemble/member<i>/{config.yaml, best_model.pt}  deterministic members, i = 0..4
"""

import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from hnnuq.config import ExperimentConfig
from hnnuq.deep_ensemble import DeepEnsembleWrapper, infer_model_dim
from hnnuq.hnn_model import build_hnn_model
from hnnuq.masked_linear import load_state_dict_with_optional_masks

# arm directory name -> paper arm label (ledger `arm` column)
ARM_LABELS: Dict[str, str] = {
    "deterministic": "Deterministic",
    "fresh_mask": "Fresh-mask MC",
    "concrete": "Concrete",
    "variational": "Variational",
    "fixed_mask": "Fixed-mask MC",
    "deep_ensemble": "Deep ensemble",
}
ARM_DIRS = list(ARM_LABELS)
SYSTEM_DIRS = {"CP": "cp", "SC": "sc"}


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU and all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_run_dir(path: Path) -> Path:
    """``path`` itself if it holds a checkpoint, else its single child directory that does (training output nesting)."""
    path = Path(path)
    if (path / "best_model.pt").exists() or (path / "final_model.pt").exists():
        return path
    children = [c for c in sorted(path.glob("*")) if c.is_dir() and (c / "best_model.pt").exists()]
    if len(children) != 1:
        raise FileNotFoundError(f"no unique run directory with best_model.pt under {path}")
    return children[0]


def run_base(runs_root: Path, system: str, arm: str, seed: int) -> Path:
    """Directory of one run in the run tree (``system`` in {cp, sc})."""
    return Path(runs_root) / system / arm / f"seed{seed}"


def member_dirs(runs_root: Path, system: str, n_members: int = 5) -> List[Path]:
    """Run directories of the deep-ensemble members."""
    return [resolve_run_dir(Path(runs_root) / system / "deep_ensemble" / f"member{i}") for i in range(n_members)]


def load_model(run_dir: Path, device: torch.device) -> Tuple[nn.Module, ExperimentConfig]:
    """Rebuild the trained HNN from ``config.yaml`` and load ``best_model.pt`` (including a saved mask bank), in eval mode."""
    run_dir = resolve_run_dir(run_dir)
    config = ExperimentConfig.load(run_dir / "config.yaml")
    ckpt = run_dir / "best_model.pt"
    state_dict = torch.load(ckpt if ckpt.exists() else run_dir / "final_model.pt", map_location=device, weights_only=False)
    model = build_hnn_model(
        dim=infer_model_dim(config, state_dict), n_hidden=config.n_hidden, drop_prob=config.drop_prob,
        mask_num=config.mask_num, nonlin=config.nonlin, device=device, model_type=config.model_type,
        dropout_variant=config.dropout_variant, finite_mask_mode=config.finite_mask_mode,
        concrete_temp=config.concrete_temp, scale_weights=config.scale_weights, mask_last_layer=config.mask_last_layer,
        finite_mask_storage=config.finite_mask_storage, mask_seed=config.seed,
    )
    load_state_dict_with_optional_masks(model, state_dict, strict=False, device=device)
    model.eval()
    return model, config


def load_deep_ensemble(dirs: Sequence[Path], device: torch.device) -> Tuple[nn.Module, ExperimentConfig]:
    """Deep ensemble as one model whose "mask bank" is its members (mask_num = number of members)."""
    wrapper = DeepEnsembleWrapper.from_dirs([resolve_run_dir(d) for d in dirs], device=str(device))
    wrapper.eval()
    config = ExperimentConfig.load(resolve_run_dir(dirs[0]) / "config.yaml")
    config.mask_num = wrapper.mask_num
    config.pred_samples = wrapper.mask_num
    config.drop_prob = 0.0
    config.finite_mask_mode = "none"
    return wrapper, config


def load_arm_model(runs_root: Path, system: str, arm: str, seed: int, device: torch.device):
    """Model and config of (system, arm, seed); the deep ensemble ignores ``seed``."""
    if arm == "deep_ensemble":
        return load_deep_ensemble(member_dirs(runs_root, system), device)
    return load_model(run_base(runs_root, system, arm, seed), device)


def mask_semantics(system: str, stage: str, arm: str) -> str:
    """Fresh-mask semantics ("weight_group" | "activation") used for (system, stage in {ic, ood, hdrift}, arm).

    "activation" rolls mask groups out sequentially with per-element dropout draws; "weight_group" draws one weight-space mask
    per group. The choice only changes results for arms that draw fresh masks (Fresh-mask MC, Concrete, Variational); for the
    others it selects the batched or the sequential code path, chosen as in the paper. The coupled-pendulum IC ensemble scores
    Concrete, Variational, Fixed-mask MC and the deep ensemble with weight-space groups; the other CP/SC stages use activation
    semantics for the fresh-mask arms.
    """
    if stage == "ic":
        if system == "sc":
            return "activation"
        return "activation" if arm in ("fresh_mask", "deterministic") else "weight_group"
    if stage == "ood":
        return "weight_group" if (system == "sc" and arm == "deterministic") else "activation"
    return "activation" if arm in ("fresh_mask", "concrete", "variational") else "weight_group"


# ----------------------------------------------------------------------------------------------------------------------
# Predictive scores. ensemble: (N, M, T, s, d) flattened over N*M samples; targets: (T, s, d); all on CPU.
# ----------------------------------------------------------------------------------------------------------------------

def gaussian_nll(samples: torch.Tensor, targets: torch.Tensor) -> float:
    """Gaussian NLL per element under the mean / std (ddof=1, floor 1e-6) of ``samples`` (S, T, s, d), averaged over T, s, d."""
    if samples.shape[0] < 2:
        return float("nan")
    mean = samples.mean(dim=0)
    std = samples.std(dim=0).clamp(min=1e-6)
    nll = 0.5 * (((targets - mean) / std) ** 2 + 2 * std.log() + np.log(2 * np.pi))
    return nll.mean().item()


def central_coverage(samples: torch.Tensor, targets: torch.Tensor, level: float = 0.9) -> float:
    """Fraction of target elements inside the central ``level`` empirical interval of ``samples`` (S, T, s, d)."""
    lo = (1.0 - level) / 2
    lower = torch.quantile(samples, lo, dim=0)
    upper = torch.quantile(samples, 1.0 - lo, dim=0)
    return ((targets >= lower) & (targets <= upper)).float().mean().item()


def variance_decomposition(ensemble: torch.Tensor, skip_first_step: bool) -> Tuple[float, float]:
    """Epistemic and aleatoric variance of an IC-noise x mask ensemble (N, M, T, s, d), averaged over steps, cases and dims.

    Epistemic = mean over ICs of the variance across masks; aleatoric = variance over ICs of the mask-mean; both ddof=1.
    With ``skip_first_step`` the first frame (identical for all masks of an IC) is excluded from the averages (t >= 1).
    alpha follows as epi / (epi + alea) from these two averages (and from their means over cases of a split).
    """
    if ensemble.shape[1] < 2:  # a single mask has no epistemic spread
        return float("nan"), ensemble.mean(dim=1).var(dim=0, unbiased=True)[1 if skip_first_step else 0:].mean().item()
    epistemic = ensemble.var(dim=1, unbiased=True).mean(dim=0)  # (T, s, d)
    aleatoric = ensemble.mean(dim=1).var(dim=0, unbiased=True)  # (T, s, d)
    if skip_first_step:
        epistemic, aleatoric = epistemic[1:], aleatoric[1:]
    return epistemic.mean().item(), aleatoric.mean().item()
