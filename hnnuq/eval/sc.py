"""Spring-chain evaluation stages (SeparableHNN conditioned on the coupling matrix K).

Stages (``python -m hnnuq.eval.sc --runs runs --arm fixed_mask --seed 42 [--device cuda:0] [--smoke]``), written to ``<run>/eval/``:
    ic_fitted.csv  IC-noise x mask ensemble per (topology, stiffness k), IC-noise scale fitted on validation data
    ic_fixed.csv   same with the true scale 0.1 and measurement noise 0.1 added to the predictive samples
    ood.csv        50-mask rollouts per (topology, k) over 100 steps: MSE, mask variance, physical energy drift
    hdrift.json    drift of the model's own sampled Hamiltonian (see hnnuq.eval.hdrift)

Evaluation cases are the distinct support patterns of the test coupling matrices, each rescaled to a scalar stiffness k in
K_VALUES (k in [1, 5] is in-distribution). Ground-truth rollouts are simulated on the evaluation device with per-case seeds and
carry measurement noise 0.1 on every frame; ICs are drawn from the device's generator, so CUDA and CPU evaluations differ.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from hnnuq.config import ExperimentConfig
from hnnuq.data.spring_chain import SpringChainHamiltonianMatrix, simulate_single_k_matrix
from hnnuq.eval.common import ARM_DIRS, load_arm_model, mask_semantics, resolve_run_dir, run_base, seed_everything
from hnnuq.eval.cp import write_rows
from hnnuq.eval.hdrift import hdrift_stage
from hnnuq.eval.ic_ensemble import fit_noise_scale, ic_case_scores, ic_noise_ensemble
from hnnuq.integration import integrate_mc_samples

SYSTEM = "sc"
SYSTEM_TYPE = "chain"
K_VALUES = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
SMOKE_K_VALUES = [0.5, 1.0, 6.0]
TRAIN_K_MIN, TRAIN_K_MAX = 1.0, 5.0
TARGET_SEED, EVAL_SEED, NOISE_SEED_OFFSET = 1729, 2718, 9999
MEASUREMENT_NOISE = 0.1


def load_data(config: ExperimentConfig) -> Dict[str, np.ndarray]:
    """Training trajectories (n_k, T, n_ics, 2N), training and test coupling matrices (n_k, N, N) from ``config.data_dir``."""
    root = Path(config.data_dir) / SYSTEM_TYPE
    tag = f"{SYSTEM_TYPE}_{config.dataset_index}"
    return {"train_data": np.load(root / f"train_data_{tag}.npy"),
            "k_train": np.load(root / f"train_data_{tag}k_train_mat.npy"),
            "k_test": np.load(root / f"test_data_{tag}k_test_mat.npy")}


def topologies(k_test: np.ndarray) -> List[Tuple[int, torch.Tensor]]:
    """Distinct binary support patterns of the test coupling matrices as (index of first occurrence, support (N, N) bool)."""
    seen, out = set(), []
    for i, k in enumerate(torch.from_numpy(k_test).float()):
        support = k.abs() > 0
        key = tuple(support.reshape(-1).int().tolist())
        if key not in seen:
            seen.add(key)
            out.append((i, support))
    return out


def k_matrix_from_support(support: torch.Tensor, k_value: float) -> torch.Tensor:
    """Scalar-scaled coupling matrix with the given support; interior diagonal entries stay zero."""
    k_mat = support.to(torch.float32) * float(k_value)
    n = support.shape[0]
    if n > 2:
        idx = torch.arange(1, n - 1)
        k_mat[idx, idx] = 0.0
    return k_mat


def split_of(k_value: float) -> str:
    """Evaluation split of a spring constant: ``in_dist`` inside the training range, else ``ood``."""
    return "in_dist" if TRAIN_K_MIN <= k_value <= TRAIN_K_MAX else "ood"


def force_scale(train_data: np.ndarray, k_train: np.ndarray) -> float:
    """sqrt(E[|K_support q|^2]) over the training set: converts a stiffness offset into an expected force change."""
    n = train_data.shape[-1] // 2
    sq = []
    for i in range(k_train.shape[0]):
        support = (k_train[i] > 0).astype(np.float32)
        q = train_data[i].reshape(-1, n)
        sq.append(float(((q @ support.T) ** 2).mean()))
    return float(np.sqrt(np.mean(sq)))


def distance_from_train_range(k_value: float, scale: float) -> float:
    """|k - nearest training stiffness| times the force scale (0 inside [1, 5])."""
    if TRAIN_K_MIN <= k_value <= TRAIN_K_MAX:
        return 0.0
    return abs(k_value - (TRAIN_K_MIN if k_value < TRAIN_K_MIN else TRAIN_K_MAX)) * scale


def target_trajectory(support: torch.Tensor, k_value: float, source_index: int, n_ics: int, n_steps: int, dt: float,
                      device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Noisy ground-truth rollout (n_steps + 1, n_ics, 2N) and coupling matrix for one case.

    ICs come from a generator on ``device`` seeded ``1729 + 1000 * source_index + round(100 k)``; the measurement noise
    (std 0.1, every frame including t=0) from a CPU generator seeded with that plus 9999.
    """
    seed = int(TARGET_SEED + 1000 * source_index + round(100 * k_value))
    k_mat = k_matrix_from_support(support, k_value).to(device)
    traj = simulate_single_k_matrix(k_mat, n_ics, n_steps + 1, dt, device=device, seed=seed)
    gen = torch.Generator()
    gen.manual_seed(seed + NOISE_SEED_OFFSET)
    traj = traj + MEASUREMENT_NOISE * torch.randn(traj.shape, generator=gen).to(device)
    return traj, k_mat


def eval_seed(source_index: int, k_value: float) -> int:
    """Per-case seed of the model-side rollouts (IC-noise draws, global RNG)."""
    return int(EVAL_SEED + 1000 * source_index + round(100 * k_value))


def ic_stage(model, config: ExperimentConfig, data, device: torch.device, arm: str, seed: int, fit_sigma: bool,
             n_ic: int = 20, n_mc: int = 50, T_eval: int = 60, n_ics_eval: int = 50, k_values=None, n_topologies=None) -> List[dict]:
    """IC-noise x mask ensemble evaluation of every (topology, k) case; one row per case.

    RNG order: global seed, scale fit (5 validation ICs of every training matrix, first ``T_eval`` frames, 5 IC draws per
    candidate), then the cases in (topology, k) order. Each case rolls out the first ``min(n_mc, mask_num)`` masks (M-only
    rollout with weight-space fresh-mask semantics, ensemble with the arm's semantics) and scores the frames after t=0, so
    variances are averages over t >= 1 by construction. IC-noise seeds are ``2718 + 1000 * topology + round(100 k)``. The
    deterministic arm uses a single mask. ``fit_sigma`` False uses the true scale and adds measurement noise to the samples.
    """
    n_mc = 1 if arm == "deterministic" else n_mc
    semantics = mask_semantics(SYSTEM, "ic", arm)
    seed_everything(seed)
    model.eval()
    k_values = K_VALUES if k_values is None else k_values
    t_eval = min(T_eval, data["train_data"].shape[1])
    scale = force_scale(data["train_data"], data["k_train"])
    sigma, val_nll = MEASUREMENT_NOISE, float("nan")
    if fit_sigma:
        val = torch.from_numpy(data["train_data"][:, :t_eval, -5:, :].astype(np.float32)).to(device)
        k_val = torch.from_numpy(data["k_train"].astype(np.float32)).to(device)
        sigma, val_nll = fit_noise_scale(model, val, k_val, config.dt, 5, n_mc, config.mask_num, config.use_leapfrog, seed, mask_seed=seed)
    rows = []
    tops = topologies(data["k_test"])
    for topo_id, (source_index, support) in enumerate(tops if n_topologies is None else tops[:n_topologies]):
        for k_value in k_values:
            traj, k_mat = target_trajectory(support, k_value, source_index, n_ics_eval, t_eval - 1, config.dt, device)
            z_0, dim = traj[0], traj.shape[-1] // 2
            offset = eval_seed(source_index, k_value)
            masks = torch.arange(min(n_mc, config.mask_num), dtype=torch.long)
            with torch.no_grad():
                masks_only = integrate_mc_samples(
                    p_0=z_0[:, :dim], q_0=z_0[:, dim:], model=model, T=t_eval, dt=config.dt, n_mc_samples=len(masks),
                    k_matrix=k_mat, trajectory_consistent=True, use_leapfrog=config.use_leapfrog, mask_subset=masks,
                )[:, 1:]  # (M, T-1, s, 2d)
                ensemble = ic_noise_ensemble(
                    z_0, model, sigma, n_ic, len(masks), t_eval, config.dt, config.mask_num, k_mat, config.use_leapfrog,
                    seed_offset=offset, mask_subset=masks, fresh_mask_semantics=semantics,
                )[:, :, 1:]  # (N, M, T-1, s, 2d)
            scores = ic_case_scores(masks_only.cpu(), ensemble.cpu(), traj[1:].cpu(), skip_first_step=False,
                                    measurement_noise_std=0.0 if fit_sigma else MEASUREMENT_NOISE, noise_seed=offset)
            rows.append(dict(topology=f"topo_{topo_id:02d}", k_value=k_value, split=split_of(k_value),
                             distance=distance_from_train_range(k_value, scale), sigma=sigma, sigma_val_nll=val_nll, **scores))
    return rows


def true_energy(pred: np.ndarray, support: np.ndarray, k_value: float) -> np.ndarray:
    """True spring-chain energy (float64) of states (..., 2N) with stiffness ``k_value`` on ``support``."""
    n = pred.shape[-1] // 2
    k = support.astype(np.float64) * float(k_value)
    if n > 2:
        idx = np.arange(1, n - 1)
        k[idx, idx] = 0.0
    ham = SpringChainHamiltonianMatrix(torch.from_numpy(0.5 * (k + k.T)), torch.ones(n, dtype=torch.float64))
    lead = pred.shape[:-1]
    flat = torch.from_numpy(np.ascontiguousarray(pred)).reshape(-1, 2 * n)
    with torch.no_grad():
        return ham(flat[:, :n], flat[:, n:]).numpy().reshape(lead)


def ood_stage(model, config: ExperimentConfig, data, device: torch.device, arm: str, mask_seed: int = 42, n_mc: int = 50,
              T_eval: int = 100, n_ics: int = 200, k_values=None, n_topologies=None) -> List[dict]:
    """Per-(topology, k) accuracy and mask spread of ``n_mc`` rollouts (masks 0..n_mc-1) over ``T_eval`` steps; one row per case.

    The global RNG is re-seeded with the case's evaluation seed before each case. Rows: ``mse`` of the mask-mean prediction,
    ``mask_variance`` (population variance across masks averaged over steps, ICs, dims) and ``phys_e_drift`` = mean over
    masks and ICs of |E(z_T) - E(z_1)| under the true Hamiltonian in float64 (z_1 first predicted frame, z_T last).
    """
    semantics = mask_semantics(SYSTEM, "ood", arm)
    model.eval()
    k_values = K_VALUES if k_values is None else k_values
    scale = force_scale(data["train_data"], data["k_train"])
    rows = []
    tops = topologies(data["k_test"])
    for topo_id, (source_index, support) in enumerate(tops if n_topologies is None else tops[:n_topologies]):
        for k_value in k_values:
            traj, k_mat = target_trajectory(support, k_value, source_index, n_ics, T_eval, config.dt, device)
            dim = traj.shape[-1] // 2
            seed = eval_seed(source_index, k_value)
            torch.manual_seed(seed)
            np.random.seed(seed % (2 ** 32 - 1))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            masks = torch.arange(min(n_mc, config.mask_num), dtype=torch.long)
            with torch.no_grad():
                preds = integrate_mc_samples(
                    p_0=traj[0, :, :dim], q_0=traj[0, :, dim:], model=model, T=T_eval + 1, dt=config.dt, n_mc_samples=len(masks),
                    k_matrix=k_mat, trajectory_consistent=True, use_leapfrog=config.use_leapfrog, mask_subset=masks,
                    fresh_mask_semantics=semantics,
                )[:, 1: T_eval + 1]  # (M, T, s, 2N)
            target = traj[1: T_eval + 1]
            e = true_energy(preds.cpu().numpy().astype(np.float64), support.numpy(), k_value)  # (M, T, s)
            rows.append(dict(
                topology=f"topo_{topo_id:02d}", k_value=k_value, split=split_of(k_value), distance=distance_from_train_range(k_value, scale),
                mse=((preds.mean(dim=0) - target) ** 2).mean().item(), mask_variance=preds.var(dim=0, unbiased=False).mean().item(),
                phys_e_drift=float(np.abs(e[:, -1, :] - e[:, 0, :]).mean()),
            ))
    return rows


STAGES = ("ic_fitted", "ic_fixed", "ood", "hdrift")


def run(runs_root: Path, arm: str, seed: int, device: torch.device, stages=STAGES, smoke: bool = False) -> Path:
    """Run the requested stages for one (arm, seed) and write the stage files under ``<run>/eval/``.

    The IC stages use the training seed as evaluation seed; the OOD sweep uses mask seed 42 for every run. The deep ensemble is
    evaluated once (it ignores ``seed``) and writes next to its members.
    """
    model, config = load_arm_model(runs_root, SYSTEM, arm, seed, device)
    out = (Path(runs_root) / SYSTEM / arm / "eval") if arm == "deep_ensemble" else resolve_run_dir(run_base(runs_root, SYSTEM, arm, seed)) / "eval"
    data = load_data(config)
    small = dict(k_values=SMOKE_K_VALUES, n_topologies=1) if smoke else {}
    if "ic_fitted" in stages:
        kw = dict(n_ic=2, n_mc=2, T_eval=8, n_ics_eval=6) if smoke else {}
        write_rows(out / "ic_fitted.csv", ic_stage(model, config, data, device, arm, seed, True, **kw, **small))
    if "ic_fixed" in stages:
        kw = dict(n_ic=2, n_mc=2, T_eval=8, n_ics_eval=6) if smoke else {}
        write_rows(out / "ic_fixed.csv", ic_stage(model, config, data, device, arm, seed, False, **kw, **small))
    if "ood" in stages:
        kw = dict(n_mc=3, T_eval=10, n_ics=8) if smoke else {}
        write_rows(out / "ood.csv", ood_stage(model, config, data, device, arm, **kw, **small))
    if "hdrift" in stages:
        out.mkdir(parents=True, exist_ok=True)
        (out / "hdrift.json").write_text(json.dumps(hdrift_stage(SYSTEM, runs_root, arm, seed, device, smoke=smoke)))
    return out


def main(argv=None) -> int:
    """Run the spring-chain stages (ic_fitted, ic_fixed, ood, hdrift) for one arm and seed."""
    ap = argparse.ArgumentParser(description="Spring-chain evaluation stages for one (arm, seed)")
    ap.add_argument("--runs", type=Path, default=Path("runs"), help="run tree root (runs/<system>/<arm>/seed<N>)")
    ap.add_argument("--arm", choices=ARM_DIRS, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stages", default=",".join(STAGES), help="comma-separated subset of " + ",".join(STAGES))
    ap.add_argument("--smoke", action="store_true", help="one topology, 3 stiffness values, tiny ensembles, short horizon")
    args = ap.parse_args(argv)
    run(args.runs, args.arm, args.seed, torch.device(args.device), tuple(args.stages.split(",")), args.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
