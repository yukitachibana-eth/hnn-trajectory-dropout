"""Hidden-parameter detection on the coupled pendulum (Table 15): Fixed-mask MC vs Fresh-mask MC.

Question: can a trained model's own predictive spread reveal that the true mass ratio r_true differs from the ratio r0 it is
conditioned on, when only the first k states of the true trajectory are revealed? The model always receives r0 (never r_true).
Cases are scored with the paper's main-run coupled-pendulum checkpoints; no extra training or dataset is needed.

Protocol (one IC pool per scoring family, drawn once with ``_sample_ics`` and shared by every r_true, so at k = 0 the model
input is identical for all r_true and a score that depends on r_true there would indicate leakage):
    A  predictive variance, r0 = 1.0, r_true in {1.0, 0.1, 0.2, 0.3, 3.0, 4.0, 5.0} (r_true == r0 is the negative class),
       k in {0, 1, 2, 5, 10, 20}, T = 60, 50 cases, 50 MC samples. The k = 0 rows are the leakage control.
    B  Hamiltonian drift of the model's own sampled energy H_m along the revealed prefix (k + 1 states), anchors
       r0 in {0.75, 1.0, 1.5}, r_true in the seven training mass ratios, k in {2, 3, 5, 8, 10, 20, 30}, 50 cases, M = 50 draws.
    C  predictive variance at the B anchors and r_true grid, k in {2, 3, 5, 8, 10}, T = 60, 50 MC samples.
Detection quality is the AUROC of the score for "r_true != r0", computed per (arm, seed, k) (and per anchor for B, C).

Scores. A, C: variance across the MC samples of the forecast from state k over the remaining T - k steps, population ddof,
averaged over steps and state dimensions, per case. B: drift_score = mean over draws of std_t(H_m) / (|mean_t(H_m)| + 1e-8)
over the prefix states t = 0..k (population std), per case; Fixed-mask MC draws are 50 bank masks (a seeded random subset of
the bank, one fixed network per draw), Fresh-mask MC draws are 50 independent fresh dropout draws per state.

RNG order (the evaluation is seeded and reproducible but order dependent): A seeds the global RNG with 1000 + seed before
loading the model, then runs a calibration forecast (calibration IC pool, k = 0, r0) and, for every (r_true, k), the main
forecast followed by a forecast conditioned on r_true; the latter two consume random draws of the same stream and are
discarded here. C seeds with 2000 + seed before loading the model. B seeds the Fresh-mask draws with 20260727 and picks the
Fixed-mask subset with a CPU generator seeded 20260726.

    python -m hnnuq.eval.hidden_param run --runs runs --arm fixed_mask --seed 42 [--device cuda:0] [--smoke]
    python -m hnnuq.eval.hidden_param collect --runs runs --out results [--smoke]

``run`` writes ``<run>/eval/hidden_param/{A,B,C}.csv`` (per-case scores); ``collect`` computes the AUROCs over all seeds found
in the run tree and writes ``results/extra/hidden_param_{A,B,C}_auroc.csv`` and ``hidden_param_leakage_k0.csv``.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from hnnuq.data.coupled_pendulum import KAPPA, CoupledPendulumMRSystem, _sample_ics
from hnnuq.eval.common import load_arm_model, resolve_run_dir, run_base
from hnnuq.integration import integrate_ground_truth, integrate_mc_samples
from hnnuq.masked_linear import get_model_active_mask_num

SYSTEM = "cp"
ARMS = {"fixed_mask": "Fixed-mask MC", "fresh_mask": "Fresh-mask MC"}
DT = 0.1
R0 = 1.0
R_TRUE_A = (1.0, 0.1, 0.2, 0.3, 3.0, 4.0, 5.0)
TRAIN_RATIOS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
ANCHORS = (0.75, 1.0, 1.5)
CALIBRATION_SEED, SCORING_SEED = 20260723, 20260724
MASK_SUBSET_SEED, FRESH_DRAW_SEED = 20260726, 20260727
EPS = 1e-8


@dataclass(frozen=True)
class Protocol:
    """Sizes of the protocol; the defaults are the paper's, ``smoke()`` a tiny variant."""

    n_cases: int = 50
    horizon: int = 60  # T of the A and C forecasts
    n_mc: int = 50
    k_a: Tuple[int, ...] = (0, 1, 2, 5, 10, 20)
    k_b: Tuple[int, ...] = (2, 3, 5, 8, 10, 20, 30)
    k_c: Tuple[int, ...] = (2, 3, 5, 8, 10)
    anchors: Tuple[float, ...] = ANCHORS
    m_draws: int = 50  # H_m draws of B

    @staticmethod
    def smoke() -> "Protocol":
        """Tiny protocol for the smoke run."""
        return Protocol(n_cases=6, horizon=14, n_mc=4, k_a=(0, 2, 10), k_b=(2, 10), k_c=(2, 10), anchors=(1.0,), m_draws=4)


def reference_trajectory(p0: torch.Tensor, q0: torch.Tensor, r_true: float, T: int) -> torch.Tensor:
    """Noiseless leapfrog trajectory at mass ratio ``r_true`` from the shared ICs; (T, n_cases, 4), state [p_a, p_b, q_a, q_b]."""
    return integrate_ground_truth(p_0=p0, q_0=q0, hamiltonian=CoupledPendulumMRSystem(kappa=KAPPA, mass_ratio=r_true), T=T, dt=DT)


def forecast_from_prefix(model, traj: torch.Tensor, k: int, r_cond: float, dt: float, n_mc: int, device: torch.device) -> torch.Tensor:
    """MC forecast from the revealed state ``traj[k]`` over the remaining T - k steps, conditioned on ``r_cond``: (n_mc, T - k, n_cases, 4).

    Mask-bank models draw n_mc distinct bank masks (one per rollout); Fresh-mask models draw per-element activation dropout.
    """
    if traj.shape[0] - k < 2:
        raise ValueError(f"k={k} leaves fewer than 2 forecast steps of T={traj.shape[0]}")
    state = traj[k].to(device)
    dim = state.shape[-1] // 2
    return integrate_mc_samples(
        p_0=state[:, :dim], q_0=state[:, dim:], model=model, T=traj.shape[0] - k, dt=dt, n_mc_samples=n_mc,
        k_matrix=torch.tensor(r_cond, dtype=torch.float32, device=device), trajectory_consistent=True, use_leapfrog=True,
        fresh_mask_semantics="activation",
    )


def raw_variance(forecast: torch.Tensor) -> np.ndarray:
    """Population variance across MC samples, averaged over forecast steps and state dims: (n_cases,)."""
    return forecast.var(dim=0, unbiased=False).mean(dim=(0, 2)).detach().cpu().numpy()


def _case_ics(n_cases: int, seed: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """IC pool (p0, q0), each (n_cases, 2), drawn with a CPU generator seeded by ``seed``."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    p0, q0 = _sample_ics(n_cases, gen)
    return p0.to(device), q0.to(device)


def score_a(load, config, device, seed: int, pr: Protocol) -> pd.DataFrame:
    """Part A rows (r_true, is_mismatch, k, case_index, raw_var_r0); follows the RNG order of the module docstring."""
    torch.manual_seed(1000 + seed)
    model, _ = load()  # loaded after seeding: model construction draws from the global RNG
    cal_p0, cal_q0 = _case_ics(pr.n_cases, CALIBRATION_SEED, device)
    forecast_from_prefix(model, reference_trajectory(cal_p0, cal_q0, R0, pr.horizon), 0, R0, config.dt, pr.n_mc, device)
    p0, q0 = _case_ics(pr.n_cases, SCORING_SEED, device)
    rows = []
    for r_true in R_TRUE_A:
        traj = reference_trajectory(p0, q0, r_true, pr.horizon)
        for k in pr.k_a:
            var = raw_variance(forecast_from_prefix(model, traj, k, R0, config.dt, pr.n_mc, device))
            forecast_from_prefix(model, traj, k, r_true, config.dt, pr.n_mc, device)  # advances the RNG stream only
            rows += [dict(r_true=r_true, is_mismatch=int(r_true != R0), k=k, case_index=c, raw_var_r0=float(var[c])) for c in range(pr.n_cases)]
    return pd.DataFrame(rows)


def score_c(load, config, device, seed: int, pr: Protocol) -> pd.DataFrame:
    """Part C rows (r0, r_true, is_mismatch, k, case_index, raw_var_r0): predictive variance at every anchor."""
    torch.manual_seed(2000 + seed)
    model, _ = load()
    p0, q0 = _case_ics(pr.n_cases, SCORING_SEED, device)
    trajs = {r: reference_trajectory(p0, q0, r, pr.horizon) for r in TRAIN_RATIOS}
    rows = []
    for r0 in pr.anchors:
        for r_true in TRAIN_RATIOS:
            for k in pr.k_c:
                var = raw_variance(forecast_from_prefix(model, trajs[r_true], k, r0, config.dt, pr.n_mc, device))
                rows += [dict(r0=r0, r_true=r_true, is_mismatch=int(r_true != r0), k=k, case_index=c, raw_var_r0=float(var[c])) for c in range(pr.n_cases)]
    return pd.DataFrame(rows)


@torch.no_grad()
def sampled_energy(model, arm: str, states: List[Tuple[torch.Tensor, torch.Tensor]], r0: float, n_draws: int, device) -> torch.Tensor:
    """H_m of ``n_draws`` network draws on every state: (n_states, n_draws * n_base), draw-major rows (draw i = block i of n_base).

    Fixed-mask MC: draw i uses bank mask ``subset[i]``, a random subset of the bank from a CPU generator seeded with
    MASK_SUBSET_SEED. Fresh-mask MC: every Hamiltonian call draws fresh dropout masks (global RNG seeded once with
    FRESH_DRAW_SEED, never reseeded between states).
    """
    r_t = torch.tensor(r0, dtype=torch.float32, device=device)
    if arm == "fixed_mask":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(MASK_SUBSET_SEED)
        subset = torch.randperm(get_model_active_mask_num(model), generator=gen)[:n_draws].tolist()
    else:
        torch.manual_seed(FRESH_DRAW_SEED)
    energies = torch.empty(len(states), n_draws * states[0][0].shape[0])
    for t, (p_t, q_t) in enumerate(states):
        p_all, q_all = p_t.repeat(n_draws, 1).to(device), q_t.repeat(n_draws, 1).to(device)
        h = model.batch_hamiltonian(p_all, q_all, r_t, subset) if arm == "fixed_mask" else model(p_all, q_all, k=r_t, index=None)
        energies[t] = h.detach().cpu()
    return energies


def drift_scores(energies: torch.Tensor, n_draws: int, k: int) -> torch.Tensor:
    """drift_score of each row-case from the first k + 1 states: mean over draws of std_t / (|mean_t| + eps); (n_base,)."""
    e = energies.reshape(energies.shape[0], n_draws, -1)[: k + 1]  # (k+1, draws, n_base)
    mean_t, std_t = e.mean(dim=0), e.std(dim=0, unbiased=False)
    return (std_t / (mean_t.abs() + EPS)).mean(dim=0)


def score_b(model, arm: str, device, pr: Protocol) -> pd.DataFrame:
    """Part B rows (r0, r_true, is_mismatch, k, case_index, drift_score) for every anchor."""
    p0, q0 = _case_ics(pr.n_cases, SCORING_SEED, device)
    n_states = max(pr.k_b) + 1
    trajs = {r: reference_trajectory(p0, q0, r, n_states).to(device) for r in TRAIN_RATIOS}
    dim = 2
    # rows are r_true-major, case-minor, identical at every state (fresh dropout draws depend on the row position)
    states = [(torch.cat([trajs[r][t][:, :dim].cpu() for r in TRAIN_RATIOS]), torch.cat([trajs[r][t][:, dim:].cpu() for r in TRAIN_RATIOS]))
              for t in range(n_states)]
    rows = []
    for r0 in pr.anchors:
        energies = sampled_energy(model, arm, states, r0, pr.m_draws, device)
        for k in pr.k_b:
            score = drift_scores(energies, pr.m_draws, k)
            for i, r_true in enumerate(r for r in TRAIN_RATIOS for _ in range(pr.n_cases)):
                rows.append(dict(r0=r0, r_true=r_true, is_mismatch=int(r_true != r0), k=k, case_index=i % pr.n_cases, drift_score=float(score[i])))
    return pd.DataFrame(rows)


def run(runs_root: Path, arm: str, seed: int, device: torch.device, smoke: bool = False) -> Path:
    """Score parts A, B, C for one (arm, seed) and write them to ``<run>/eval/hidden_param/``."""
    pr = Protocol.smoke() if smoke else Protocol()
    load = lambda: load_arm_model(runs_root, SYSTEM, arm, seed, device)
    config = load()[1]
    out = resolve_run_dir(run_base(runs_root, SYSTEM, arm, seed)) / "eval" / "hidden_param"
    out.mkdir(parents=True, exist_ok=True)
    score_a(load, config, device, seed, pr).to_csv(out / "A.csv", index=False)
    score_c(load, config, device, seed, pr).to_csv(out / "C.csv", index=False)
    score_b(load()[0], arm, device, pr).to_csv(out / "B.csv", index=False)
    return out


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve of score ``s`` for labels ``y`` (ties get average ranks); NaN if one class is missing."""
    n1 = int(y.sum())
    if n1 == 0 or n1 == len(y):
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def _gather(runs_root: Path, part: str) -> pd.DataFrame:
    """Per-case scores of ``part`` for every (arm, seed) with a hidden_param run, with ``method`` and ``seed`` columns."""
    frames = []
    for arm, label in ARMS.items():
        for run_dir in sorted((Path(runs_root) / SYSTEM / arm).glob("seed*")):
            f = resolve_run_dir(run_dir) / "eval" / "hidden_param" / f"{part}.csv"
            if f.exists():
                frames.append(pd.read_csv(f).assign(method=label, seed=int(run_dir.name[len("seed"):])))
    if not frames:
        raise FileNotFoundError(f"no hidden_param/{part}.csv under {runs_root}/{SYSTEM}")
    return pd.concat(frames, ignore_index=True)


def _auroc_table(df: pd.DataFrame, keys: List[str], score: str) -> pd.DataFrame:
    rows = [dict(zip(keys, key if isinstance(key, tuple) else (key,)), auroc=auroc(g.is_mismatch.to_numpy(), g[score].to_numpy()))
            for key, g in df.groupby(keys)]
    return pd.DataFrame(rows)


def collect(runs_root: Path, out_root: Path) -> Dict[str, Path]:
    """Write the per-seed AUROC tables (A: method, seed, k; B and C: r0, method, seed, k) and the k = 0 leakage-control rows."""
    out = Path(out_root) / "extra"
    out.mkdir(parents=True, exist_ok=True)
    a = _gather(runs_root, "A")
    paths = {
        "A": out / "hidden_param_A_auroc.csv", "B": out / "hidden_param_B_auroc.csv",
        "C": out / "hidden_param_C_auroc.csv", "leakage": out / "hidden_param_leakage_k0.csv",
    }
    _auroc_table(a, ["method", "seed", "k"], "raw_var_r0").to_csv(paths["A"], index=False)
    _auroc_table(_gather(runs_root, "B"), ["r0", "method", "seed", "k"], "drift_score").to_csv(paths["B"], index=False)
    _auroc_table(_gather(runs_root, "C"), ["r0", "method", "seed", "k"], "raw_var_r0").to_csv(paths["C"], index=False)
    a[a.k == 0][["method", "seed", "is_mismatch", "k", "raw_var_r0"]].to_csv(paths["leakage"], index=False)
    return paths


def main(argv=None) -> int:
    """Command line: ``run`` scores one (arm, seed), ``collect`` writes the AUROC tables."""
    ap = argparse.ArgumentParser(description="Hidden-parameter detection (Table 15)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="score one (arm, seed) of the main coupled-pendulum runs")
    r.add_argument("--runs", type=Path, default=Path("runs"))
    r.add_argument("--arm", choices=list(ARMS), required=True)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--device", default="cpu")
    r.add_argument("--smoke", action="store_true", help="6 cases, T=14, 4 MC samples, one anchor")
    c = sub.add_parser("collect", help="AUROC tables from all scored seeds")
    c.add_argument("--runs", type=Path, default=Path("runs"))
    c.add_argument("--out", type=Path, default=Path("results"))
    c.add_argument("--smoke", action="store_true", help="accepted for symmetry with the other stages (collect is size independent)")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run(args.runs, args.arm, args.seed, torch.device(args.device), args.smoke)
    else:
        collect(args.runs, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
