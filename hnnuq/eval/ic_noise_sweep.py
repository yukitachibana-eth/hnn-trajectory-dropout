"""Coupled-pendulum IC-noise sweep: Fixed-mask MC vs Fresh-mask MC with the initial-state noise scale at 0-4x its fitted value.

    python -m hnnuq.eval.ic_noise_sweep run --runs runs --arm fixed_mask --seed 42 [--device cuda:0] [--smoke]
    python -m hnnuq.eval.ic_noise_sweep collect --runs runs --out results [--smoke]

``run`` writes ``<run>/eval/ic_noise_sweep.csv`` with one row per (noise multiplier, mass ratio): the epistemic variance (mean over
IC draws of the across-mask variance) and the aleatoric variance (variance over IC draws of the mask mean), both ddof=1 and averaged
over steps t >= 1, ICs and dimensions. ``collect`` writes ``<out>/extra/ic_noise_sweep_per_seed.csv`` (method, split,
sigma_multiplier, seed, var_epistemic_mean, var_aleatoric_mean = per-seed means over the mass ratios of a split); alpha follows as
the ratio of the two averages.

Protocol. Evaluation seed 42 for every training seed. The base scale is the validation-NLL-fitted noise scale (the fit of the main
IC-ensemble stage); every multiplier m evaluates 20 IC draws x 50 masks at sigma = m * base, 60 frames, from the stored test /
OOD initial states, activation fresh-mask semantics for both arms. Mask indices come from a CPU generator re-seeded with 42 for every
mass ratio (the M-only rollout draws first, then the ensemble). The fitted scale and the Fresh-mask arm's dropout draws use the
global RNG (seeded once with 42 before the scale fit), so Fresh-mask rows are reproducible only in distribution.
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd
import torch

from hnnuq.eval import cp as cp_eval
from hnnuq.eval.common import ARM_LABELS, load_arm_model, resolve_run_dir, run_base, seed_everything, variance_decomposition
from hnnuq.eval.ic_ensemble import fit_noise_scale, ic_noise_ensemble, mask_generator
from hnnuq.integration import integrate_mc_samples

ARMS = ("fresh_mask", "fixed_mask")
SEEDS = (42, 43, 44)
MULTIPLIERS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
EVAL_SEED, N_IC, N_MC, T_EVAL = 42, 20, 50, 60


def result_path(run_dir: Path) -> Path:
    """Per-run output file of the sweep."""
    return Path(run_dir) / "eval" / "ic_noise_sweep.csv"


@torch.no_grad()
def sweep(model, config, data, device: torch.device, multipliers=MULTIPLIERS, n_ic: int = N_IC, n_mc: int = N_MC,
          T_eval: int = T_EVAL) -> List[dict]:
    """Variance decomposition per (multiplier, mass ratio); the noise scale is fitted once and scaled by each multiplier."""
    seed_everything(EVAL_SEED)
    model.eval()
    train_ratios = data["train_mass_ratios"]
    t_eval = min(T_eval, data["test_data"].shape[1])
    n_val = min(config.n_val_samples, data["train_data"].shape[2])
    val = data["train_data"][:, :, -n_val:, :][:, :t_eval].to(device)
    base, _ = fit_noise_scale(model, val, train_ratios.to(device), config.dt, n_ic, n_mc, config.mask_num, config.use_leapfrog,
                              EVAL_SEED, mask_seed=EVAL_SEED)
    rows = []
    for mult in multipliers:
        for idx, (split, ratio, traj) in enumerate(cp_eval.case_list(data)):
            target = traj[:t_eval].to(device)
            z_0, dim = target[0], target.shape[-1] // 2
            ratio_t = torch.tensor(ratio, dtype=torch.float32, device=device)
            gen = mask_generator(EVAL_SEED)
            # the M-only rollout is not scored here; it advances the mask generator exactly as in the IC-ensemble stage
            integrate_mc_samples(p_0=z_0[:, :dim], q_0=z_0[:, dim:], model=model, T=t_eval, dt=config.dt, n_mc_samples=n_mc,
                                 k_matrix=ratio_t, trajectory_consistent=True, use_leapfrog=config.use_leapfrog, generator=gen)
            ensemble = ic_noise_ensemble(
                z_0, model, mult * base, n_ic, n_mc, t_eval, config.dt, config.mask_num, ratio_t, config.use_leapfrog,
                seed_offset=EVAL_SEED * 1000 + idx * 100, fresh_mask_semantics="activation", generator=gen,
            )  # (N, M, T, s, 2d)
            epi, alea = variance_decomposition(ensemble.cpu(), skip_first_step=True)
            rows.append(dict(split=split, mass_ratio=ratio, sigma_multiplier=mult, base_sigma=base,
                             var_epistemic_mean=epi, var_aleatoric_mean=alea))
    return rows


def run(runs_root: Path, arm: str, seed: int, device: torch.device, smoke: bool = False) -> Path:
    """Run the sweep for one (arm, seed) and write ``eval/ic_noise_sweep.csv``."""
    model, config = load_arm_model(runs_root, cp_eval.SYSTEM, arm, seed, device)
    data = cp_eval.load_data(config, max_ics=8 if smoke else None)
    kw = dict(multipliers=(0.0, 1.0), n_ic=2, n_mc=2, T_eval=10) if smoke else {}
    rows = sweep(model, config, data, device, **kw)
    path = result_path(resolve_run_dir(run_base(runs_root, cp_eval.SYSTEM, arm, seed)))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def collect(runs_root: Path, out: Path) -> None:
    """Per-seed means over mass ratios of each (method, split, multiplier)."""
    parts = []
    for arm in ARMS:
        for seed in SEEDS:
            base = run_base(runs_root, cp_eval.SYSTEM, arm, seed)
            path = result_path(resolve_run_dir(base)) if base.exists() else None
            if path is None or not path.exists():
                continue
            d = pd.read_csv(path)
            g = d.groupby(["split", "sigma_multiplier"])[["var_epistemic_mean", "var_aleatoric_mean"]].mean().reset_index()
            g.insert(0, "method", ARM_LABELS[arm])
            g["seed"] = seed
            parts.append(g)
    cols = ["method", "split", "sigma_multiplier", "seed", "var_epistemic_mean", "var_aleatoric_mean"]
    res = pd.concat(parts, ignore_index=True)[cols].sort_values(["method", "split", "sigma_multiplier", "seed"], kind="stable")
    (Path(out) / "extra").mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(out) / "extra/ic_noise_sweep_per_seed.csv", index=False)


def main(argv=None) -> int:
    """Command line: ``run`` sweeps one arm and seed, ``collect`` writes the per-seed table."""
    ap = argparse.ArgumentParser(description="Coupled-pendulum IC-noise sweep")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="sweep one arm and seed")
    r.add_argument("--runs", type=Path, default=Path("runs"))
    r.add_argument("--arm", choices=ARMS, required=True)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--device", default="cpu")
    r.add_argument("--smoke", action="store_true", help="two multipliers, 2 IC draws, 2 masks, short horizon")
    c = sub.add_parser("collect", help="write the per-seed table")
    c.add_argument("--runs", type=Path, default=Path("runs"))
    c.add_argument("--out", type=Path, default=Path("results"))
    c.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.runs, a.arm, a.seed, torch.device(a.device), a.smoke)
    else:
        collect(a.runs, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
