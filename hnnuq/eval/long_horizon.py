"""Long-horizon physical energy drift (appendix table on drift at t=100 and t=1000, and the per-step spring-chain drift figure).

Arms Deterministic, Fresh-mask MC and Fixed-mask MC are rolled out for 1000 steps on both systems. The drift of a rollout at step t is
|E_t - E_1| / |E_1| under the true Hamiltonian (float64), E_1 being the energy of the rollout's own first recorded frame, averaged over
masks and cases; it needs no ground-truth trajectory. The first recorded frame is the first predicted state for the spring chain and
the initial state for the coupled pendulum, so step t is t steps (spring chain) or t-1 steps (coupled pendulum) after the start.

    python -m hnnuq.eval.long_horizon run --runs runs --arm fixed_mask --seed 42 [--system cp|sc] [--device cuda:0] [--smoke]
    python -m hnnuq.eval.long_horizon collect --runs runs --out results [--smoke]

``run`` writes ``<run>/eval/long_horizon_<system>.csv`` (columns split, t, drift; t = 1..T). ``collect`` writes
``<out>/extra/long_horizon_per_seed.csv`` (system, arm, seed, split, t, value; t in {100, 1000}) and
``<out>/figures/sc_energy_drift_by_step.csv`` (arm, seed, step, drift; in-distribution spring chain).

Protocol. Spring chain: first 3 support topologies x stiffness {0.5, 1.5, 4, 8} (1.5 and 4 in distribution), 200 initial states per case
drawn from the case's generator (the noisy-target initial frame of the OOD stage), masks 0..9, activation fresh-mask semantics, global
RNG re-seeded per case; the split curve is the mean over its cases. Coupled pendulum: the 200 stored initial states of every mass ratio
(7 in distribution, 6 OOD), masks 0..9 rolled out one after the other from a global RNG seeded with 42 before the model is built; the
split curve is the mean over its ratios. The deterministic arm uses one rollout (its masks are identical).

Devices. The coupled-pendulum rollouts always run on CPU: over 1000 chaotic steps float32 rounding differs between CPU and GPU
(relative differences up to ~1e-4 at t=1000), and the paper values were computed on CPU. The spring-chain cases draw their initial
states from the evaluation device's generator, so the paper values are reproduced with ``--device cuda``.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from hnnuq.data.coupled_pendulum import CoupledPendulumMRSystem, KAPPA
from hnnuq.eval import cp as cp_eval, sc as sc_eval
from hnnuq.eval.common import ARM_LABELS, load_arm_model, resolve_run_dir, run_base, seed_everything
from hnnuq.integration import integrate_mc_samples, numerically_integrate

ARMS = ("deterministic", "fresh_mask", "fixed_mask")
SYSTEMS = ("sc", "cp")
SEEDS = list(range(42, 52))
HORIZON, N_MC = 1000, 10
REPORT_STEPS = (100, 1000)
SMOKE_HORIZON, SMOKE_STEPS = 10, (5, 10)
SC_K_VALUES = (0.5, 1.5, 4.0, 8.0)
SC_TOPOLOGIES, SC_N_ICS = 3, 200
CP_SEED = 42


def curve_path(run_dir: Path, system: str) -> Path:
    """Per-run output file of the drift curves of ``system``."""
    return Path(run_dir) / "eval" / f"long_horizon_{system}.csv"


def relative_drift(energy: np.ndarray) -> np.ndarray:
    """|E_t - E_1| / |E_1| averaged over masks and cases, for energies (M, T, s); returns (T,)."""
    e1 = energy[:, :1, :]
    return (np.abs(energy - e1) / np.abs(e1)).mean(axis=(0, 2))


@torch.no_grad()
def sc_curves(model, config, data, device: torch.device, arm: str, horizon: int, k_values: Sequence[float], n_topologies: int,
              n_ics: int, n_mc: int) -> Dict[str, np.ndarray]:
    """Per-split drift curves (horizon,) of the spring chain: mean over the (topology, stiffness) cases of the split."""
    n_mc = 1 if arm == "deterministic" else n_mc
    model.eval()
    per_split: Dict[str, List[np.ndarray]] = {"in_dist": [], "ood": []}
    for source_index, support in sc_eval.topologies(data["k_test"])[:n_topologies]:
        for k_value in k_values:
            # Only the (noisy) initial frame is needed: the generator draws the initial states before integrating, and the
            # measurement noise is a CPU draw whose leading block is the first frame.
            traj, k_mat = sc_eval.target_trajectory(support, k_value, source_index, n_ics, 0, config.dt, device)
            z_0, dim = traj[0], traj.shape[-1] // 2
            seed = sc_eval.eval_seed(source_index, k_value)
            torch.manual_seed(seed)
            np.random.seed(seed % (2 ** 32 - 1))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            masks = torch.arange(min(n_mc, config.mask_num), dtype=torch.long)
            preds = integrate_mc_samples(
                p_0=z_0[:, :dim], q_0=z_0[:, dim:], model=model, T=horizon + 1, dt=config.dt, n_mc_samples=len(masks),
                k_matrix=k_mat, trajectory_consistent=True, use_leapfrog=config.use_leapfrog, mask_subset=masks,
                fresh_mask_semantics="activation",
            )[:, 1: horizon + 1]  # (M, T, s, 2N)
            energy = sc_eval.true_energy(preds.cpu().numpy().astype(np.float64), support.numpy(), k_value)  # (M, T, s)
            per_split[sc_eval.split_of(k_value)].append(relative_drift(energy))
    return {sp: np.mean(v, axis=0) for sp, v in per_split.items()}


@torch.no_grad()
def cp_curves(model, config, data, device: torch.device, arm: str, horizon: int, n_mc: int) -> Dict[str, np.ndarray]:
    """Per-split drift curves (horizon,) of the coupled pendulum: mean over the mass ratios of the split.

    Each mask index is rolled out separately from the 200 stored initial states of the ratio; the first frame of a rollout is
    the initial state.
    """
    n_mc = 1 if arm == "deterministic" else n_mc
    model.eval()
    per_split: Dict[str, List[np.ndarray]] = {"in_dist": [], "ood": []}
    for split, ratio, traj in cp_eval.case_list(data):
        traj = traj.to(device)
        dim4 = traj.shape[-1]
        ratio_t = torch.tensor(ratio, dtype=torch.float32, device=device)
        rollouts = torch.stack([
            numerically_integrate(p_0=traj[0, :, : dim4 // 2], q_0=traj[0, :, dim4 // 2:], model=model, T=horizon, dt=config.dt,
                                  k_matrix=ratio_t, mask_index=m, use_leapfrog=True)
            for m in range(n_mc)
        ]).double()  # (M, T, s, 4)
        ham = CoupledPendulumMRSystem(kappa=KAPPA, mass_ratio=ratio).double().to(device)
        p, q = rollouts[..., :2], rollouts[..., 2:]
        energy = ham(p.reshape(-1, 2), q.reshape(-1, 2)).reshape(p.shape[:-1])  # (M, T, s)
        per_split[split].append(relative_drift(energy.cpu().numpy()))
    return {sp: np.mean(v, axis=0) for sp, v in per_split.items()}


def run(runs_root: Path, system: str, arm: str, seed: int, device: torch.device, smoke: bool = False) -> Path:
    """Roll out one (system, arm, seed) and write its per-split drift curves; returns the file path."""
    horizon = SMOKE_HORIZON if smoke else HORIZON
    if system == "cp":
        seed_everything(CP_SEED)  # the global RNG state at the first rollout includes model construction
    model, config = load_arm_model(runs_root, system, arm, seed, device)
    if system == "cp":
        data = cp_eval.load_data(config, max_ics=8 if smoke else None)
        curves = cp_curves(model, config, data, device, arm, horizon, 2 if smoke else N_MC)
    else:
        data = sc_eval.load_data(config)
        curves = sc_curves(model, config, data, device, arm, horizon, sc_eval.SMOKE_K_VALUES if smoke else SC_K_VALUES,
                           1 if smoke else SC_TOPOLOGIES, 6 if smoke else SC_N_ICS, 2 if smoke else N_MC)
    path = curve_path(resolve_run_dir(run_base(runs_root, system, arm, seed)), system)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(split=sp, t=t + 1, drift=float(v)) for sp, c in curves.items() for t, v in enumerate(c)]).to_csv(path, index=False)
    return path


def collect(runs_root: Path, out: Path, smoke: bool = False) -> None:
    """Gather the per-run curves into the appendix table rows and the per-step figure data."""
    steps = SMOKE_STEPS if smoke else REPORT_STEPS
    table, fig = [], []
    for system in SYSTEMS:
        for arm in ARMS:
            for seed in SEEDS:
                base = run_base(runs_root, system, arm, seed)
                if not base.exists():
                    continue
                path = curve_path(resolve_run_dir(base), system)
                if not path.exists():
                    continue
                c = pd.read_csv(path)
                label = ARM_LABELS[arm]
                for _, r in c[c.t.isin(steps)].iterrows():
                    table.append(dict(system=system.upper(), arm=label, seed=seed, split=r.split, t=int(r.t), value=r.drift))
                if system == "sc":
                    for _, r in c[c.split == "in_dist"].iterrows():
                        fig.append(dict(arm=label, seed=seed, step=int(r.t), drift=r.drift))
    out = Path(out)
    (out / "extra").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    cols = ["system", "arm", "seed", "split", "t", "value"]
    pd.DataFrame(table, columns=cols).sort_values(["system", "arm", "seed", "split", "t"], kind="stable").to_csv(
        out / "extra/long_horizon_per_seed.csv", index=False)
    pd.DataFrame(fig, columns=["arm", "seed", "step", "drift"]).sort_values(["arm", "seed", "step"], kind="stable").to_csv(
        out / "figures/sc_energy_drift_by_step.csv", index=False)


def main(argv=None) -> int:
    """Command line: ``run`` rolls out one arm and seed, ``collect`` writes the table and figure CSVs."""
    ap = argparse.ArgumentParser(description="Long-horizon physical energy drift")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="roll out one arm and seed")
    r.add_argument("--runs", type=Path, default=Path("runs"))
    r.add_argument("--arm", choices=ARMS, required=True)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--system", choices=("cp", "sc", "both"), default="both")
    r.add_argument("--device", default="cpu")
    r.add_argument("--smoke", action="store_true", help="short horizon, few cases, 2 masks")
    c = sub.add_parser("collect", help="write the table and figure CSVs")
    c.add_argument("--runs", type=Path, default=Path("runs"))
    c.add_argument("--out", type=Path, default=Path("results"))
    c.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        for system in (SYSTEMS if a.system == "both" else (a.system,)):
            device = torch.device("cpu") if system == "cp" else torch.device(a.device)
            run(a.runs, system, a.arm, a.seed, device, a.smoke)
    else:
        collect(a.runs, a.out, a.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
