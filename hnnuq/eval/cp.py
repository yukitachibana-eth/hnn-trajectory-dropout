"""Coupled-pendulum evaluation stages (mass-ratio conditioned HNN).

Stages (``python -m hnnuq.eval.cp --runs runs --arm fixed_mask --seed 42 [--device cuda:0] [--smoke]``), written to ``<run>/eval/``:
    ic_fitted.csv  IC-noise x mask ensemble per mass ratio, IC-noise scale fitted on validation data
    ic_fixed.csv   same with the true scale 0.1 and measurement noise 0.1 added to the predictive samples
    ood.csv        mask-index sweep (first 10 masks) per mass ratio: MSE, mask variance, physical energy drift
    hdrift.json    drift of the model's own sampled Hamiltonian (see hnnuq.eval.hdrift)
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from hnnuq.config import ExperimentConfig
from hnnuq.data.coupled_pendulum import CoupledPendulumMRSystem, KAPPA
from hnnuq.eval.common import ARM_DIRS, load_arm_model, mask_semantics, resolve_run_dir, run_base, seed_everything
from hnnuq.eval.hdrift import hdrift_stage
from hnnuq.eval.ic_ensemble import fit_noise_scale, ic_case_scores, ic_noise_ensemble, mask_generator
from hnnuq.integration import integrate_mc_samples, numerically_integrate

SYSTEM = "cp"
SYSTEM_TYPE = "coupled_pendulum_mr"
SIGMA_FIXED = 0.1  # true measurement-noise scale of the dataset


def load_data(config: ExperimentConfig, max_ics: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Train / test / OOD trajectories (n_ratios, T, n_ics, 4) and their mass ratios (n_ratios,) from ``config.data_dir``."""
    root = Path(config.data_dir) / SYSTEM_TYPE
    out = {}
    for name in ("train_data", "train_mass_ratios", "test_data", "test_mass_ratios", "ood_test_data", "ood_mass_ratios"):
        out[name] = torch.from_numpy(np.load(root / f"{name}_{SYSTEM_TYPE}_{config.dataset_index}.npy")).float()
    if max_ics is not None:
        for name in ("train_data", "test_data", "ood_test_data"):
            out[name] = out[name][:, :, :max_ics, :]
    return out


def log_distance_from_train_range(r: float, r_min: float, r_max: float) -> float:
    """log-ratio distance of a mass ratio outside the training interval (0 inside)."""
    if r_min <= r <= r_max:
        return 0.0
    return float(np.log(r_min / r)) if r < r_min else float(np.log(r / r_max))


def dynamics_distance(r: float, r_min: float, r_max: float, train_data: torch.Tensor) -> float:
    """Distance of mass ratio ``r`` to the training interval induced by the change of the Hamiltonian vector field.

    sqrt(C_kin (1/r - 1/r*)^2 + C_force (r - r*)^2) with r* = clip(r, r_min, r_max), C_kin = E[p2^2], C_force = E[sin^2 of the second angle]
    over all training states.
    """
    states = train_data.numpy().astype(np.float64)
    c_kin, c_force = float(np.mean(states[..., 1] ** 2)), float(np.mean(np.sin(states[..., 3]) ** 2))
    r_star = min(max(r, r_min), r_max)
    return float(np.sqrt(max(c_kin * (1.0 / r - 1.0 / r_star) ** 2 + c_force * (r - r_star) ** 2, 0.0)))


def case_list(data: Dict[str, torch.Tensor]):
    """Evaluation cases in dataset order: (split, mass ratio, trajectory (T, s, 4)); in-distribution ratios first, then OOD."""
    cases = []
    for split, traj, ratios in (("in_dist", data["test_data"], data["test_mass_ratios"]),
                                ("ood", data["ood_test_data"], data["ood_mass_ratios"])):
        for i in range(traj.shape[0]):
            cases.append((split, float(ratios[i].item()), traj[i]))
    return cases


def ic_stage(model, config: ExperimentConfig, data, device: torch.device, arm: str, seed: int, fit_sigma: bool,
             n_ic: int = 20, n_mc: int = 50, T_eval: int = 60, max_ics: Optional[int] = None) -> List[dict]:
    """IC-noise x mask ensemble evaluation of every mass ratio; one row per mass ratio.

    RNG order: global seed, scale fit (validation ICs: last ``n_val_samples`` training ICs of all training ratios, first
    ``T_eval`` frames), then per ratio the M-only rollout and the N x M ensemble. Mask indices come from a CPU generator
    re-seeded with ``seed`` for every ratio (M-only masks first, then the ensemble's masks). Per-ratio IC-noise seeds are
    ``1000 * seed + 100 * ratio_index`` (ratio index counted over in-distribution, then OOD ratios). The deterministic arm
    uses a single mask. ``fit_sigma`` False scores with the true scale and adds measurement noise to the predictive samples.
    """
    n_mc = 1 if arm == "deterministic" else n_mc
    semantics = mask_semantics(SYSTEM, "ic", arm)
    seed_everything(seed)
    model.eval()
    train_ratios = data["train_mass_ratios"]
    r_min, r_max = float(train_ratios.min()), float(train_ratios.max())
    t_eval = min(T_eval, data["test_data"].shape[1])
    sigma, val_nll = SIGMA_FIXED, float("nan")
    if fit_sigma:
        n_val = min(config.n_val_samples, data["train_data"].shape[2])
        val = data["train_data"][:, :, -n_val:, :][:, :t_eval].to(device)
        sigma, val_nll = fit_noise_scale(model, val, train_ratios.to(device), config.dt, n_ic, n_mc, config.mask_num,
                                         config.use_leapfrog, seed, mask_seed=seed)
    rows = []
    for idx, (split, ratio, traj) in enumerate(case_list(data)):
        target = traj[:t_eval].to(device)
        z_0, dim = target[0], target.shape[-1] // 2
        ratio_t = torch.tensor(ratio, dtype=torch.float32, device=device)
        gen = mask_generator(seed)
        with torch.no_grad():
            masks_only = integrate_mc_samples(
                p_0=z_0[:, :dim], q_0=z_0[:, dim:], model=model, T=t_eval, dt=config.dt, n_mc_samples=n_mc,
                k_matrix=ratio_t, trajectory_consistent=True, use_leapfrog=config.use_leapfrog, generator=gen,
            )  # (M, T, s, 2d)
            ensemble = ic_noise_ensemble(
                z_0, model, sigma, n_ic, n_mc, t_eval, config.dt, config.mask_num, ratio_t, config.use_leapfrog,
                seed_offset=seed * 1000 + idx * 100, fresh_mask_semantics=semantics, generator=gen,
            )  # (N, M, T, s, 2d)
        scores = ic_case_scores(
            masks_only.cpu(), ensemble.cpu(), target.cpu(), skip_first_step=True,
            measurement_noise_std=0.0 if fit_sigma else SIGMA_FIXED, noise_seed=seed * 1000 + idx * 100,
        )
        rows.append(dict(
            split=split, mass_ratio=ratio, distance=dynamics_distance(ratio, r_min, r_max, data["train_data"]),
            sigma=sigma, sigma_val_nll=val_nll, **scores,
        ))
    return rows


def true_energy(p: np.ndarray, q: np.ndarray, mass_ratio: float) -> np.ndarray:
    """True Hamiltonian (float64) of states p, q (..., 2) at one mass ratio."""
    lead = p.shape[:-1]
    ham = CoupledPendulumMRSystem(kappa=KAPPA, mass_ratio=float(mass_ratio)).double()
    with torch.no_grad():
        h = ham(torch.from_numpy(np.ascontiguousarray(p)).reshape(-1, 2), torch.from_numpy(np.ascontiguousarray(q)).reshape(-1, 2))
    return h.numpy().reshape(lead)


def ood_stage(model, config: ExperimentConfig, data, device: torch.device, seed: int = 42, n_mc: int = 10, T_eval: int = 60) -> List[dict]:
    """Per-mass-ratio accuracy and mask spread from rollouts under mask indices 0..n_mc-1 (one row per ratio).

    Each mask index is rolled out separately from the test ICs of the ratio (global RNG seeded once, so fresh-mask arms draw
    their dropout noise sequentially). A rollout of T frames starts with the IC (frame 0) and is paired with the data frames
    1..T, i.e. every prediction is scored against the data one step ahead. Rows: ``mse`` of the mask-mean prediction over all
    T frames, ``mask_variance`` (population variance across masks averaged over frames t >= 1, ICs, dims; frame 0 is shared by
    all masks), and ``phys_e_drift`` = mean over masks and ICs of |E(z_T) - E(z_0)| under the true Hamiltonian (z_0 = IC frame,
    z_T = last frame), evaluated in float64.
    """
    seed_everything(seed)
    model.eval()
    train_ratios = data["train_mass_ratios"]
    r_min, r_max = float(train_ratios.min()), float(train_ratios.max())
    rows = []
    for split, ratio, traj in case_list(data):
        traj = traj.to(device)
        T, _, dim4 = traj.shape
        t_eval = min(T_eval, T - 1)
        ratio_t = torch.tensor(ratio, dtype=torch.float32, device=device)
        target = traj[1: t_eval + 1]
        with torch.no_grad():
            preds = torch.stack([
                numerically_integrate(p_0=traj[0, :, : dim4 // 2], q_0=traj[0, :, dim4 // 2:], model=model, T=t_eval, dt=config.dt,
                                      k_matrix=ratio_t, mask_index=m, use_leapfrog=True)
                for m in range(n_mc)
            ])  # (M, T_eval, s, 4)
        mean_pred = preds.mean(dim=0)
        pred = preds.cpu().numpy().astype(np.float64)
        e = true_energy(pred[..., :2], pred[..., 2:], round(ratio, 8))  # (M, T_eval, s)
        rows.append(dict(
            split=split, mass_ratio=ratio, distance=dynamics_distance(ratio, r_min, r_max, data["train_data"]),
            log_distance_from_train_range=log_distance_from_train_range(ratio, r_min, r_max),
            mse=((mean_pred - target) ** 2).mean().item(), mask_variance=preds.var(dim=0, unbiased=False)[1:].mean().item(),
            phys_e_drift=float(np.abs(e[:, -1, :] - e[:, 0, :]).mean()),
        ))
    return rows


def write_rows(path: Path, rows: List[dict]) -> None:
    """Write dict rows to CSV (columns of the first row)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


STAGES = ("ic_fitted", "ic_fixed", "ood", "hdrift")


def run(runs_root: Path, arm: str, seed: int, device: torch.device, stages=STAGES, smoke: bool = False) -> Path:
    """Run the requested stages for one (arm, seed) and write the stage files under ``<run>/eval/``.

    Training seed ``seed`` is the evaluation seed of the IC stages; the OOD sweep seeds the global RNG with 42 for every run.
    The deep ensemble is evaluated once (it ignores ``seed``) and writes next to its members.
    """
    model, config = load_arm_model(runs_root, SYSTEM, arm, seed, device)
    out = (Path(runs_root) / SYSTEM / arm / "eval") if arm == "deep_ensemble" else resolve_run_dir(run_base(runs_root, SYSTEM, arm, seed)) / "eval"
    max_ics = 8 if smoke else None
    data = load_data(config, max_ics=max_ics)
    kw = dict(n_ic=2, n_mc=2, T_eval=10) if smoke else {}
    if "ic_fitted" in stages:
        write_rows(out / "ic_fitted.csv", ic_stage(model, config, data, device, arm, seed, True, max_ics=max_ics, **kw))
    if "ic_fixed" in stages:
        write_rows(out / "ic_fixed.csv", ic_stage(model, config, data, device, arm, seed, False, max_ics=max_ics, **kw))
    if "ood" in stages:
        write_rows(out / "ood.csv", ood_stage(model, config, data, device, n_mc=3 if smoke else 10, T_eval=10 if smoke else 60))
    if "hdrift" in stages:
        (out).mkdir(parents=True, exist_ok=True)
        (out / "hdrift.json").write_text(json.dumps(hdrift_stage(SYSTEM, runs_root, arm, seed, device, smoke=smoke)))
    return out


def main(argv=None) -> int:
    """Run the coupled-pendulum stages (ic_fitted, ic_fixed, ood, hdrift) for one arm and seed."""
    ap = argparse.ArgumentParser(description="Coupled-pendulum evaluation stages for one (arm, seed)")
    ap.add_argument("--runs", type=Path, default=Path("runs"), help="run tree root (runs/<system>/<arm>/seed<N>)")
    ap.add_argument("--arm", choices=ARM_DIRS, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stages", default=",".join(STAGES), help="comma-separated subset of " + ",".join(STAGES))
    ap.add_argument("--smoke", action="store_true", help="tiny cases, 2 IC draws, 2 masks, short horizon")
    args = ap.parse_args(argv)
    run(args.runs, args.arm, args.seed, torch.device(args.device), tuple(args.stages.split(",")), args.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
