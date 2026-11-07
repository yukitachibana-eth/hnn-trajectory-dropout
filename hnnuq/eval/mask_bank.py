"""Spring-chain Fixed-mask MC as a function of the mask-bank size M in {1, 5, 20, 50, 100, 200} (appendix table, seeds 42-44).

Training (one run per M and seed; M=200 is the main Fixed-mask run, ``runs/sc/fixed_mask``; ``plan`` prints the others):

    python -m hnnuq.train --config configs/sc/fixed_mask.yaml --seed S --mask_num M --output_dir runs/sc/fixed_mask_m<M>/seed<S>     M=5,20,50,100
    python -m hnnuq.train --config configs/sc/fixed_mask_m1_per_rollout.yaml --seed S --output_dir runs/sc/fixed_mask_m1/seed<S>      M=1

Evaluation (arm ``fixed_mask_m<M>``, or ``fixed_mask`` for M=200):

    python -m hnnuq.eval.mask_bank run --runs runs --arm fixed_mask_m20 --seed 42 [--device cuda:0] [--smoke]
    python -m hnnuq.eval.mask_bank collect --runs runs --out results [--smoke]
    python -m hnnuq.eval.mask_bank plan | arms

``run`` reuses the spring-chain stages of ``hnnuq.eval.sc`` with min(M, 50) masks and evaluation seed 42 for every training seed,
and writes ``<run>/eval/mask_bank/{ic_fitted.csv, ic_fixed.csv, ood.csv, hdrift.json}``. ``collect`` writes
``<out>/extra/mask_bank_per_seed.csv`` (M, seed, quantity, value): per-seed means over the evaluation cases of a split of

    mse_{id,ood}                    OOD stage, MSE of the mask-mean prediction
    sigma_hat                       fitted IC-noise scale
    nll_fit_*, cov_fit_*            IC ensemble with the fitted scale: NLL and |coverage-0.9| of the N x M ensemble
    nll_fix_*, cov_fix_*            same with the true scale 0.1 (measurement noise added to the samples)
    nll_mask_*                      NLL of the M-only samples in the fixed-scale evaluation
    var_epi_ood, var_alea_ood       variances of the fitted-scale ensemble (OOD)
    phys_drift_{id,ood}             OOD stage, drift of the true energy over 100 steps
    h_drift                         drift of the model's own sampled Hamiltonian
Quantities that are undefined for a single mask (M=1) are omitted.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from hnnuq.eval import sc as sc_eval
from hnnuq.eval.common import load_arm_model, resolve_run_dir, run_base
from hnnuq.eval.cp import write_rows
from hnnuq.eval.hdrift import hdrift_stage

SIZES = (1, 5, 20, 50, 100, 200)
SEEDS = (42, 43, 44)
EVAL_SEED = 42
N_MC_CAP = 50
STAGES = ("ic_fitted", "ic_fixed", "ood", "hdrift")

# quantity -> (stage file, split, column)
QUANTITIES = {
    "mse_id": ("ood", "in_dist", "mse"), "mse_ood": ("ood", "ood", "mse"),
    "sigma_hat": ("ic_fitted", "in_dist", "sigma"),
    "nll_fit_id": ("ic_fitted", "in_dist", "nll_ensemble"), "nll_fit_ood": ("ic_fitted", "ood", "nll_ensemble"),
    "nll_fix_id": ("ic_fixed", "in_dist", "nll_ensemble"), "nll_fix_ood": ("ic_fixed", "ood", "nll_ensemble"),
    "nll_mask_id": ("ic_fixed", "in_dist", "nll_masks_only"), "nll_mask_ood": ("ic_fixed", "ood", "nll_masks_only"),
    "cov_fit_id": ("ic_fitted", "in_dist", "cov90_abs_error_ensemble"), "cov_fit_ood": ("ic_fitted", "ood", "cov90_abs_error_ensemble"),
    "cov_fix_id": ("ic_fixed", "in_dist", "cov90_abs_error_ensemble"), "cov_fix_ood": ("ic_fixed", "ood", "cov90_abs_error_ensemble"),
    "var_epi_ood": ("ic_fitted", "ood", "var_epi"), "var_alea_ood": ("ic_fitted", "ood", "var_alea"),
    "phys_drift_id": ("ood", "in_dist", "phys_e_drift"), "phys_drift_ood": ("ood", "ood", "phys_e_drift"),
}


def arm_name(size: int) -> str:
    """Run-tree arm directory of the Fixed-mask run with bank size ``size``."""
    return "fixed_mask" if size == 200 else f"fixed_mask_m{size}"


def size_of(arm: str) -> int:
    """Bank size of an arm directory name."""
    return 200 if arm == "fixed_mask" else int(arm.rsplit("_m", 1)[1])


def stage_dir(run_dir: Path) -> Path:
    """Per-run output directory of the stage."""
    return Path(run_dir) / "eval" / "mask_bank"


def plan() -> List[str]:
    """Training commands of the runs this stage needs besides the main Fixed-mask run (one shell command per line)."""
    cmds = []
    for size in SIZES:
        if size == 200:
            continue
        cfg = "configs/sc/fixed_mask_m1_per_rollout.yaml" if size == 1 else "configs/sc/fixed_mask.yaml"
        extra = "" if size == 1 else f" --mask_num {size}"
        for seed in SEEDS:
            cmds.append(f"python -m hnnuq.train --config {cfg} --seed {seed}{extra} --output_dir runs/sc/{arm_name(size)}/seed{seed}")
    return cmds


def arms() -> List[Tuple[str, int]]:
    """(arm, seed) evaluation jobs."""
    return [(arm_name(s), seed) for s in SIZES for seed in SEEDS]


def run(runs_root: Path, arm: str, seed: int, device: torch.device, smoke: bool = False) -> Path:
    """Run the spring-chain stages of (arm, seed) with evaluation seed 42 and min(M, 50) masks."""
    model, config = load_arm_model(runs_root, sc_eval.SYSTEM, arm, seed, device)
    out = stage_dir(resolve_run_dir(run_base(runs_root, sc_eval.SYSTEM, arm, seed)))
    data = sc_eval.load_data(config)
    n_mc = min(size_of(arm), N_MC_CAP)
    small = dict(k_values=sc_eval.SMOKE_K_VALUES, n_topologies=1) if smoke else {}
    ic_kw = dict(n_ic=2, n_mc=min(n_mc, 2), T_eval=8, n_ics_eval=6) if smoke else dict(n_mc=n_mc)
    # the stage's own arm argument only selects the number of masks (deterministic) and the IC-stage mask semantics
    write_rows(out / "ic_fitted.csv", sc_eval.ic_stage(model, config, data, device, "fixed_mask", EVAL_SEED, True, **ic_kw, **small))
    write_rows(out / "ic_fixed.csv", sc_eval.ic_stage(model, config, data, device, "fixed_mask", EVAL_SEED, False, **ic_kw, **small))
    ood_kw = dict(n_mc=min(n_mc, 3), T_eval=10, n_ics=8) if smoke else dict(n_mc=n_mc)
    write_rows(out / "ood.csv", sc_eval.ood_stage(model, config, data, device, "fixed_mask", **ood_kw, **small))
    (out / "hdrift.json").write_text(json.dumps(hdrift_stage(sc_eval.SYSTEM, runs_root, arm, seed, device, smoke=smoke)))
    return out


def collect(runs_root: Path, out: Path) -> None:
    """Per-seed means of each quantity over the cases of its split (NaN cases ignored; all-NaN quantities omitted)."""
    rows = []
    for size in SIZES:
        for seed in SEEDS:
            base = run_base(runs_root, sc_eval.SYSTEM, arm_name(size), seed)
            if not base.exists():
                continue
            d = stage_dir(resolve_run_dir(base))
            if not d.exists():
                continue
            tables = {s: pd.read_csv(d / f"{s}.csv") for s in ("ic_fitted", "ic_fixed", "ood")}
            for q, (stage, split, col) in QUANTITIES.items():
                v = tables[stage][tables[stage].split == split][col].mean()
                if not np.isnan(v):
                    rows.append(dict(M=size, seed=seed, quantity=q, value=float(v)))
            rows.append(dict(M=size, seed=seed, quantity="h_drift", value=json.loads((d / "hdrift.json").read_text())["h_drift"]))
    res = pd.DataFrame(rows, columns=["M", "seed", "quantity", "value"])
    order = {q: i for i, q in enumerate(list(QUANTITIES) + ["h_drift"])}
    res = res.assign(_o=res.quantity.map(order)).sort_values(["_o", "M", "seed"], kind="stable").drop(columns="_o")
    (Path(out) / "extra").mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(out) / "extra/mask_bank_per_seed.csv", index=False)


def main(argv=None) -> int:
    """Command line: ``run`` / ``collect`` evaluate and tabulate; ``plan`` and ``arms`` print the training commands and evaluation jobs."""
    ap = argparse.ArgumentParser(description="Spring-chain Fixed-mask MC across mask-bank sizes")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="evaluate one bank size and seed")
    r.add_argument("--runs", type=Path, default=Path("runs"))
    r.add_argument("--arm", required=True, help="fixed_mask_m<M> (M in 1, 5, 20, 50, 100) or fixed_mask (M=200)")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--device", default="cpu")
    r.add_argument("--smoke", action="store_true", help="one topology, 3 stiffness values, tiny ensembles, short horizon")
    c = sub.add_parser("collect", help="write the per-seed table")
    c.add_argument("--runs", type=Path, default=Path("runs"))
    c.add_argument("--out", type=Path, default=Path("results"))
    c.add_argument("--smoke", action="store_true")
    sub.add_parser("plan", help="print the training commands")
    sub.add_parser("arms", help="print the (arm, seed) evaluation jobs")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.runs, a.arm, a.seed, torch.device(a.device), a.smoke)
    elif a.cmd == "collect":
        collect(a.runs, a.out)
    elif a.cmd == "plan":
        print("\n".join(plan()))
    else:
        print("\n".join(f"{arm} {seed}" for arm, seed in arms()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
