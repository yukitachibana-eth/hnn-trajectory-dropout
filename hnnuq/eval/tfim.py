"""TFIM evaluation stages (paper Tables 8, 11, 21 and Figs 3, 6): norm/energy drift and the nested
initial-condition x mask ensemble.

Usage (repository root; every stage is deterministic given ``--seed``):
    python -m hnnuq.eval.tfim drift  --run-dir <run> --data data/tfim/<dataset> --out <dir>
    python -m hnnuq.eval.tfim nested --run-dir <run> --data data/tfim/<dataset> --out <dir> --epsilon fixed
    python -m hnnuq.eval.tfim nested --run-dir <run> --data data/tfim/<dataset> --out <dir> --epsilon fitted
``--run-dir`` is a training run directory (``config.yaml`` + ``best_model.pt``, else ``final_model.pt``); give
the five member directories to evaluate a deep ensemble. ``--smoke`` shrinks every stage to seconds on CPU.

Stages
  drift   Rollouts from the latent t=0 states of all 5120 test samples at g = 1.0 (10 mask rollouts per state
          for stochastic models). Writes ``drift.json``: mean over states, rollouts and frames of
          | ||psi_t||^2 - 1 | (``norm_drift``) and | E_t - E_0 | (``energy_drift``), E = <psi|H(g)|psi>.
  nested  For each of the 16 test control values g: 20 isotropic state-preparation perturbations (strength
          epsilon) of the intended base state (test state 0), each rolled out with 50 mask draws (a random
          subset of the bank for Fixed-mask MC, the members for a deep ensemble, fresh masks otherwise). The
          targets are the 20 stored latent rollouts of that intended state, the same for every method and
          every epsilon. epsilon is the true preparation noise 0.10 (``--epsilon fixed``) or is fitted per run
          on the validation split (``--epsilon fitted``): the value of a 13-point grid on [0, 0.6] that
          minimises the Gaussian NLL of the validation latent rollouts under the model's perturbed-state
          ensemble. Writes ``per_g.csv`` and ``summary.json``.

Nested quantities per g (rollout layout (N IC, M masks, T, D)):
  var_epi   mean over t >= 1 and dimensions of the IC-average of the mask variance
  var_alea  mean over t >= 1 and dimensions of the variance over ICs of the mask-mean
  alpha     var_epi / (var_epi + var_alea) (ratio of the two averages)
  nll       Gaussian NLL of the stored latent rollouts under the pooled (IC x mask) ensemble, mean over the
            latent rollouts, frames, dims
  cov90_abs_error  | empirical coverage of the central 90% pooled-ensemble band - 0.9 |, mean over latent rollouts
  state_rmse       RMS error of the pooled ensemble-mean wavefunction against each latent rollout, mean over
                   latent rollouts and frames
Variances use ddof = 0; frame t = 0 is excluded because all mask replicas of an IC start from the same state.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from hnnuq.config import ExperimentConfig
from hnnuq.deep_ensemble import DeepEnsembleWrapper, infer_model_dim
from hnnuq.hnn_model import build_hnn_model
from hnnuq.integration import integrate_mc_samples, numerically_integrate
from hnnuq.masked_linear import get_model_active_mask_num, load_state_dict_with_optional_masks
from hnnuq.tfim import exact
from hnnuq.tfim.control import register_tfim_operator_structure, resolve_tfim_control_tensor
from hnnuq.tfim.loader import load_tfim_training_data

SEED = 7
FRESH_MASK_SEMANTICS = "activation"  # fresh-mask models: per-element activation masks, as in training
TRAIN_BAND = (1.3, 1.9)

DRIFT_G = 1.0
DRIFT_N_MC = 10

NESTED_G_REQUESTED = (0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.25, 2.5)
NESTED_N_IC = 20
NESTED_N_MC = 50
PREP_SEED = 13  # generator seed of the first perturbation; sample j uses PREP_SEED + j
EPS_FIXED = 0.10
EPS_GRID = (0.0, 0.6, 13)  # fitted epsilon: linspace(min, max, num)
EPS_FIT_N_IC = 8  # perturbations per validation intended state in the fit
EPS_FIT_N_MC = 4  # mask rollouts per perturbation in the fit
EPS_FIT_MAX_STATES = 8  # validation intended states used in the fit
BASE_STATE = 0
NLL_VAR_FLOOR = 1e-8
COVERAGE_LEVEL = 0.90
BASE_STATE_STRIDE = 100_003  # seed offset between the base states of the epsilon fit

SMOKE_G_REQUESTED = (0.2, 1.0, 1.6)


# ------------------------------------------------------------------------------------- model bundle


@dataclass
class Bundle:
    """A loaded model and how its rollout ensemble is drawn.

    kind: ``deterministic`` (one rollout), ``fresh`` (fresh masks every Hamiltonian call),
    ``bank`` (Fixed-mask MC: one bank mask per rollout) or ``ensemble`` (one member per rollout).
    """

    model: torch.nn.Module
    config: ExperimentConfig
    kind: str

    @property
    def consistent(self) -> bool:
        """True if each rollout keeps one bank mask / ensemble member throughout."""
        return self.kind in ("bank", "ensemble")

    @property
    def mask_num(self) -> int:
        """Size of the mask bank (number of ensemble members for a deep ensemble)."""
        return get_model_active_mask_num(self.model)


def model_kind(config: ExperimentConfig) -> str:
    """Rollout kind of a trained configuration."""
    if config.drop_prob <= 0.0 or config.mask_num <= 1:
        return "deterministic"
    if config.dropout_variant == "mc_dropout" and config.finite_mask_mode == "both":
        return "bank"
    return "fresh"


def _checkpoint(run_dir: Path) -> Path:
    for name in ("best_model.pt", "final_model.pt"):
        if (run_dir / name).exists():
            return run_dir / name
    raise FileNotFoundError(f"no best_model.pt or final_model.pt in {run_dir}")


def load_bundle(run_dirs: Sequence[Path], device: torch.device) -> Bundle:
    """Load one trained run, or a deep ensemble from several member runs. Seeded mask banks are
    regenerated from the run seed."""
    run_dirs = [Path(d) for d in run_dirs]
    if len(run_dirs) > 1:
        model = DeepEnsembleWrapper.from_dirs(run_dirs, device=str(device))
        model.eval()
        return Bundle(model, ExperimentConfig.load(run_dirs[0] / "config.yaml"), "ensemble")
    run_dir = run_dirs[0]
    config = ExperimentConfig.load(run_dir / "config.yaml")
    dim = infer_model_dim(config, None)
    model = build_hnn_model(
        dim=dim,
        n_hidden=config.n_hidden,
        drop_prob=config.drop_prob,
        mask_num=config.mask_num,
        nonlin=config.nonlin,
        device=device,
        model_type=config.model_type,
        dropout_variant=config.dropout_variant,
        finite_mask_mode=config.finite_mask_mode,
        concrete_temp=config.concrete_temp,
        scale_weights=config.scale_weights,
        mask_last_layer=config.mask_last_layer,
        finite_mask_storage=config.finite_mask_storage,
        mask_seed=config.seed,
    )
    register_tfim_operator_structure(model, state_half_dim=dim)
    state_dict = torch.load(_checkpoint(run_dir), map_location=device, weights_only=True)
    load_state_dict_with_optional_masks(model, state_dict, strict=False, device=device)
    model.eval()
    return Bundle(model, config, model_kind(config))


# ------------------------------------------------------------------------------------- test data


@dataclass
class TestData:
    """Test split of the dataset: z (g, T, sample, 2*dim) float32 canonical [q, p], the control grid,
    the time grid, the model input of the base intended state and the intended-state and latent index of every sample."""

    z: np.ndarray
    g_values: np.ndarray
    time_grid: np.ndarray
    base_z0: np.ndarray  # (2*dim,) canonical, float64: intended state ``BASE_STATE``
    flat_to_intended: np.ndarray  # (S,) intended-state index of every test sample
    flat_to_latent: np.ndarray  # (S,) latent-sample index of every test sample
    L: int
    J: float
    train_band: Tuple[float, float]


def load_test_data(data_dir: Path) -> TestData:
    """Load the test split of a TFIM dataset directory (trajectories, control grid, time grid, intended base state, chain parameters)."""
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(data_dir / "test.npz") as f:
        z, g_values, time_grid = f["z"], np.asarray(f["g_values"], dtype=np.float64), np.asarray(f["time_grid"], dtype=np.float64)
        flat_to_intended = np.asarray(f["flat_to_intended_index"], dtype=np.int64)
        flat_to_latent = np.asarray(f["flat_to_latent_index"], dtype=np.int64)
    with np.load(data_dir / "test_aux.npz") as f:
        base_z0 = np.asarray(f["observed_z0_grouped"][BASE_STATE, 0], dtype=np.float64)
    return TestData(z, g_values, time_grid, base_z0, flat_to_intended, flat_to_latent,
                    int(meta["L"]), float(meta["J"]), tuple(meta["train_band"]))


def stored_latent_targets(test: TestData, gi: int, state: int = BASE_STATE) -> np.ndarray:
    """(n_latent, T, D) stored latent rollouts of intended test state ``state`` at control index ``gi``,
    ordered by latent index, in rollout layout concat(p, q), float64."""
    flat = np.flatnonzero(test.flat_to_intended == state)
    flat = flat[np.argsort(test.flat_to_latent[flat], kind="stable")]
    block = np.asarray(test.z[gi][:, flat, :], dtype=np.float64)  # (T, n_latent, 2*dim) canonical
    return exact.canonical_to_rollout(np.transpose(block, (1, 0, 2)))


def nearest_indices(available: Sequence[float], requested: Sequence[float]) -> List[int]:
    """Index of the available control value closest to each requested one (duplicates dropped, order kept)."""
    out: List[int] = []
    for g in requested:
        idx = min(range(len(available)), key=lambda i: abs(available[i] - g))
        if idx not in out:
            out.append(idx)
    return out


def in_train_band(g: float, band: Sequence[float] = TRAIN_BAND) -> bool:
    """True if ``g`` lies inside the training band of control values."""
    return min(band) <= float(g) <= max(band)


# ------------------------------------------------------------------------------------- rollouts


def _control(g: float, dim: int, device: torch.device) -> torch.Tensor:
    return resolve_tfim_control_tensor(float(g), dim, device=device, dtype=torch.float32)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().double().numpy()


def nested_rollouts(
    bundle: Bundle,
    p0: torch.Tensor,
    q0: torch.Tensor,
    g: float,
    T: int,
    dt: float,
    n_mc: int,
    mask_subset: Optional[torch.Tensor],
) -> np.ndarray:
    """(N, M, T, D) float64 rollouts of N initial states with M mask draws, layout concat(p, q).

    Bank / ensemble models roll out the bank members in ``mask_subset`` (one per rollout); a
    deterministic model is rolled out once and repeated; fresh-mask models are rolled out ``n_mc``
    times with fresh masks (drawn from the global torch RNG)."""
    k = _control(g, p0.shape[-1], p0.device)
    if bundle.consistent:
        trajs, _ = integrate_mc_samples(
            p_0=p0, q_0=q0, model=bundle.model, T=T, dt=dt, n_mc_samples=n_mc, k_matrix=k,
            trajectory_consistent=True, use_leapfrog=True, return_mask_idxs=True,
            mask_subset=mask_subset, fresh_mask_semantics=FRESH_MASK_SEMANTICS,
        )
        return np.transpose(_to_numpy(trajs), (2, 0, 1, 3))  # (M, T, N, D) -> (N, M, T, D)
    n_draws = 1 if bundle.kind == "deterministic" else n_mc
    rolls = [
        _to_numpy(numerically_integrate(p_0=p0, q_0=q0, model=bundle.model, T=T, dt=dt, k_matrix=k,
                                        mask_index=None, use_leapfrog=True, requires_grad=False))
        for _ in range(n_draws)
    ]
    ens = np.transpose(np.stack(rolls, axis=0), (2, 0, 1, 3))  # (M, T, N, D) -> (N, M, T, D)
    return np.repeat(ens, n_mc, axis=1) if n_draws == 1 else ens


def _mask_subset(bundle: Bundle, n_mc: int, seed: int) -> Tuple[Optional[torch.Tensor], int]:
    """(bank indices, rollouts per IC): a seeded random subset of the bank for bank / ensemble models."""
    if not bundle.consistent:
        return None, int(max(1, n_mc))
    count = min(int(n_mc), bundle.mask_num)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return torch.randperm(bundle.mask_num, generator=gen)[:count], count


def _split_qp(z_canonical: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Canonical states concat(q, p) -> (q, p)."""
    half = z_canonical.shape[-1] // 2
    return z_canonical[..., :half], z_canonical[..., half:]


# ------------------------------------------------------------------------------------- drift stage


def drift_stage(bundle: Bundle, test: TestData, device: torch.device, n_mc: int, g_target: float) -> Dict[str, float]:
    """Mean |norm^2 - 1| and |E_t - E_0| of model rollouts from every latent test state at g_target."""
    gi = nearest_indices(test.g_values, [g_target])[0]
    g = float(test.g_values[gi])
    z0 = np.asarray(test.z[gi, 0], dtype=np.float64)  # (S, 2*dim) canonical latent states
    dim = z0.shape[-1] // 2
    q0 = torch.tensor(z0[:, :dim], dtype=torch.float32, device=device)
    p0 = torch.tensor(z0[:, dim:], dtype=torch.float32, device=device)
    k = _control(g, dim, device)
    dt = float(test.time_grid[1] - test.time_grid[0])
    T = int(test.z.shape[1])
    h_matrix = exact.hamiltonian_matrix(test.L, g, test.J)
    if bundle.kind == "deterministic":
        trajs = numerically_integrate(p_0=p0, q_0=q0, model=bundle.model, T=T, dt=dt, k_matrix=k,
                                      mask_index=0, use_leapfrog=True, requires_grad=False)[None]
    else:
        subset = None
        if bundle.consistent:
            subset = torch.randperm(bundle.mask_num, device=device)[: min(n_mc, bundle.mask_num)]
        else:  # the draw is unused by fresh-mask rollouts but advances the RNG stream
            subset = torch.randint(0, bundle.mask_num, (n_mc,), device=device)
        trajs, _ = integrate_mc_samples(
            p_0=p0, q_0=q0, model=bundle.model, T=T, dt=dt, n_mc_samples=n_mc, k_matrix=k,
            trajectory_consistent=bundle.consistent, use_leapfrog=True, return_mask_idxs=True,
            mask_subset=subset, fresh_mask_semantics=FRESH_MASK_SEMANTICS,
        )
    norm_sum = energy_sum = 0.0
    count = 0
    for mc in range(trajs.shape[0]):
        block = _to_numpy(trajs[mc])  # (T, S, 2*dim) rollout layout
        norm = exact.norm_squared(block)  # (T, S)
        energy = exact.energy_expectation(block, h_matrix)
        norm_sum += float(np.abs(norm - 1.0).sum())
        energy_sum += float(np.abs(energy - energy[0]).sum())
        count += norm.size
    return {"g": g, "n_rows": count, "norm_drift": norm_sum / count, "energy_drift": energy_sum / count}


# ------------------------------------------------------------------------------------- nested metrics


def nested_variances(ens: np.ndarray) -> Tuple[float, float]:
    """(var_epi, var_alea) of an (N, M, T, D) ensemble averaged over frames t >= 1 and dimensions."""
    var_epi = ens.var(axis=1, ddof=0).mean(axis=0)  # (T, D): IC-average of the mask variance
    var_alea = ens.mean(axis=1).var(axis=0, ddof=0)  # (T, D): IC variance of the mask mean
    return float(var_epi[1:].mean()), float(var_alea[1:].mean())


def pooled_nll(ens: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(N,) Gaussian NLL of target (N, T, D) under the pooled (N*M) ensemble, mean over frames and dims."""
    flat = ens.reshape(-1, *ens.shape[2:])
    mean = flat.mean(axis=0, keepdims=True)
    var = np.maximum(flat.var(axis=0, ddof=0, keepdims=True), NLL_VAR_FLOOR)
    nll = 0.5 * (np.log(2.0 * np.pi * var) + (target - mean) ** 2 / var)
    return nll.mean(axis=(1, 2))


def pooled_coverage(ens: np.ndarray, target: np.ndarray, level: float = COVERAGE_LEVEL) -> np.ndarray:
    """(N,) fraction of (frame, dim) entries of target inside the central ``level`` band of the pooled ensemble."""
    flat = ens.reshape(-1, *ens.shape[2:])
    lo_q = (1.0 - level) / 2.0
    lower, upper = np.quantile(flat, lo_q, axis=0), np.quantile(flat, 1.0 - lo_q, axis=0)
    covered = (target >= lower[None]) & (target <= upper[None])
    return covered.astype(np.float64).mean(axis=(1, 2))


# ------------------------------------------------------------------------------------- epsilon fit


def fit_epsilon(bundle: Bundle, device: torch.device, T: int, dt: float, seed: int, grid: Tuple[float, float, int],
                n_ic: int, n_mc: int, max_states: int) -> Tuple[float, Dict]:
    """Grid value of the state-preparation noise strength that best explains the validation latent rollouts.

    For each candidate epsilon and each validation control value, ``n_ic`` perturbations of every of the
    first ``max_states`` validation intended states (the model input) are rolled out with ``n_mc`` mask
    draws; the objective is the Gaussian NLL of the 20 latent validation rollouts of that intended state
    under the per-intended-state mean and ddof=0 variance (floor 1e-8) of the perturbation-wise mask-mean
    rollouts, averaged over states, frames, dims and control values. Returns (best epsilon, diagnostics)."""
    val = load_tfim_training_data(bundle.config).val_split
    n_int = val.provenance.n_intended_initial_states
    n_lat = val.provenance.n_latent_samples_per_initial_state
    traj = val.trajectories[:, :T].numpy().astype(np.float64)  # (k, T, S, D) rollout layout
    k_g, _, _, D = traj.shape
    targets = traj.reshape(k_g, T, n_int, n_lat, D).transpose(0, 2, 3, 1, 4)[:, :max_states]  # (k, i, l, T, D)
    inputs = val.input_z0[0].numpy().astype(np.float64).reshape(n_int, n_lat, D)[:max_states]
    base_canonical = exact.canonical_to_rollout(inputs[:, 0, :])  # (i, D) canonical [q, p]
    n_base = base_canonical.shape[0]
    n_ic = max(1, min(n_ic, NESTED_N_IC))
    candidates = np.linspace(grid[0], grid[1], grid[2], dtype=np.float64)
    val_g = val.g_values.numpy().astype(np.float64)
    best_eps, best_score, rows = float(EPS_FIXED), float("inf"), []
    for eps in candidates.tolist():
        states = np.stack(
            [exact.complex_to_canonical(exact.perturbed_ensemble(
                exact.canonical_to_complex(base_canonical[b]), n_ic, eps, PREP_SEED + b * BASE_STATE_STRIDE))
             for b in range(n_base)], axis=0).reshape(n_base * n_ic, -1)
        q0, p0 = _split_qp(states)
        q0, p0 = (torch.tensor(a, dtype=torch.float32, device=device) for a in (q0, p0))
        scores = []
        for local, g in enumerate(val_g.tolist()):
            subset, n_inner = _mask_subset(bundle, n_mc, seed * 1009 + PREP_SEED + 10_000 + local)
            ens = nested_rollouts(bundle, p0, q0, g, T, dt, n_inner, subset)
            ens = ens.reshape(n_base, n_ic, n_inner, T, D)
            pred = ens.mean(axis=2)  # (i, ic, T, D)
            mean = pred.mean(axis=1, keepdims=True)
            var = np.maximum(pred.var(axis=1, ddof=0, keepdims=True), NLL_VAR_FLOOR)
            scores.append(float(0.5 * np.mean(np.log(2.0 * np.pi * var) + (targets[local] - mean) ** 2 / var)))
        score = float(np.mean(scores))
        rows.append({"epsilon": float(eps), "objective_nll": score})
        if score < best_score:
            best_eps, best_score = float(eps), score
    return best_eps, {"best_epsilon": best_eps, "best_objective_nll": best_score, "candidates": rows}


# ------------------------------------------------------------------------------------- nested stage


def nested_stage(bundle: Bundle, test: TestData, device: torch.device, convention: str, seed: int,
                 smoke: bool = False) -> Tuple[List[Dict], Dict]:
    """Per-g nested-ensemble metrics (see module docstring). ``convention``: ``fixed`` or ``fitted`` epsilon.

    The model inputs are 20 perturbations (strength epsilon) of the intended base state; every method is scored
    against the stored latent rollouts of that state, which do not depend on the method or on epsilon."""
    T = int(test.z.shape[1])
    dt = float(test.time_grid[1] - test.time_grid[0])
    requested = SMOKE_G_REQUESTED if smoke else NESTED_G_REQUESTED
    n_ic = 3 if smoke else NESTED_N_IC
    n_mc = 3 if smoke else NESTED_N_MC
    summary: Dict = {"convention": convention, "n_ic": n_ic, "n_mc": n_mc, "kind": bundle.kind}
    if convention == "fitted":
        grid = (EPS_GRID[0], EPS_GRID[1], 3) if smoke else EPS_GRID
        epsilon, fit = fit_epsilon(bundle, device, T, dt, seed, grid, 2 if smoke else EPS_FIT_N_IC,
                                   2 if smoke else EPS_FIT_N_MC, 2 if smoke else EPS_FIT_MAX_STATES)
        summary["epsilon_fit"] = fit
    elif convention == "fixed":
        epsilon = EPS_FIXED
    else:
        raise ValueError(f"unknown epsilon convention {convention!r}")
    summary["epsilon"] = epsilon

    base_psi = exact.canonical_to_complex(test.base_z0)
    psi0 = exact.perturbed_ensemble(base_psi, n_ic, epsilon, PREP_SEED)  # (N, dim)
    q0, p0 = _split_qp(exact.complex_to_canonical(psi0))
    q0, p0 = (torch.tensor(a, dtype=torch.float32, device=device) for a in (q0, p0))
    rows: List[Dict] = []
    for gi in nearest_indices(test.g_values, requested):
        g = float(test.g_values[gi])
        target = stored_latent_targets(test, gi)  # (n_latent, T, D)
        subset, n_inner = _mask_subset(bundle, n_mc, seed * 1009 + PREP_SEED + gi)
        ens = nested_rollouts(bundle, p0, q0, g, T, dt, n_inner, subset)  # (N, M, T, D)
        var_epi, var_alea = nested_variances(ens)
        total = var_epi + var_alea
        rmse = np.sqrt(np.mean((ens.mean(axis=(0, 1))[None] - target) ** 2, axis=-1))  # (n_latent, T)
        rows.append({
            "g_index": gi, "g": g, "in_train_band": in_train_band(g, test.train_band), "epsilon": epsilon,
            "n_ic": n_ic, "n_mc": n_inner,
            "nll": float(pooled_nll(ens, target).mean()),
            "cov90_abs_error": float(np.abs(pooled_coverage(ens, target) - COVERAGE_LEVEL).mean()),
            "state_rmse": float(rmse.mean(axis=1).mean()),
            "var_epi": var_epi, "var_alea": var_alea, "alpha": var_epi / total if total > 0 else 0.0,
        })
    return rows, summary


# ------------------------------------------------------------------------------------- CLI


def _write_csv(path: Path, rows: List[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()})


def prepare(args: argparse.Namespace) -> Tuple[Bundle, TestData, torch.device]:
    """Seed the RNGs, load the test data and the model (in this order: model initialisation draws from the RNG)."""
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test = load_test_data(args.data)
    return load_bundle(args.run_dir, device), test, device


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line of the TFIM stages: ``drift`` and ``nested``."""
    parser = argparse.ArgumentParser(description="TFIM evaluation stages")
    parser.add_argument("stage", choices=["drift", "nested"])
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True, help="run directory (several: deep-ensemble members)")
    parser.add_argument("--data", type=Path, required=True, help="dataset directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epsilon", choices=["fixed", "fitted"], default="fixed", help="nested stage: preparation-noise convention")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke", action="store_true", help="tiny sizes (seconds on CPU)")
    args = parser.parse_args(argv)

    bundle, test, device = prepare(args)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "drift":
        if args.smoke:
            test.z = test.z[:, :, :16]
        result = drift_stage(bundle, test, device, 2 if args.smoke else DRIFT_N_MC, DRIFT_G)
        (args.out / "drift.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(result)
    else:
        rows, summary = nested_stage(bundle, test, device, args.epsilon, args.seed, args.smoke)
        _write_csv(args.out / "per_g.csv", rows)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"epsilon={summary['epsilon']:.4f}  {len(rows)} control values -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
