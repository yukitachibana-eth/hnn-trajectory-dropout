"""Ledger of the coupled-pendulum (CP) and spring-chain (SC) evaluation stages.

Reads the stage files under ``runs/<system>/<arm>/seed<N>/eval/`` (see ``hnnuq.eval.cp`` / ``hnnuq.eval.sc``; the deep ensemble's
files live in ``runs/<system>/deep_ensemble/eval/`` and are stored as seed 42) and writes

    results/ledger/classical_per_seed.csv       system, arm, seed, quantity, split, convention, value
    results/figures/cp_ood_by_mass_ratio.csv    arm, seed, mass_ratio, split, distance, var_epi, var_alea       (Figs 2, 4)
    results/figures/sc_ood_by_k.csv             arm, seed, k, split, distance, var_epi, mse                    (Fig 5)

Per-system values are means over the evaluation cases (mass ratios / topology-stiffness pairs) of a split; alpha is the ratio of
those mean variances. ``python -m hnnuq.ledger.classical --runs runs --out results``.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from hnnuq.eval.common import ARM_DIRS, ARM_LABELS, resolve_run_dir, run_base

SYSTEMS = {"CP": "cp", "SC": "sc"}
SEEDS = list(range(42, 52))
LEDGER_COLUMNS = ["system", "arm", "seed", "quantity", "split", "convention", "value"]
SPLIT = {"in_dist": "ID", "ood": "OOD"}
DEEP_ENSEMBLE_SEED = 42  # one 5-member ensemble, stored as a single row


def eval_dir(runs_root: Path, system: str, arm: str, seed: int) -> Path:
    """Directory holding the stage files of (system, arm, seed)."""
    if arm == "deep_ensemble":
        return Path(runs_root) / system / arm / "eval"
    return resolve_run_dir(run_base(runs_root, system, arm, seed)) / "eval"


def _ratio(num: float, den: float) -> float:
    return float("nan") if den == 0 else num / den


def run_rows(system: str, arm: str, seed: int, ev: Path) -> List[dict]:
    """Ledger rows of one run from its stage files (a stage file that is missing contributes no rows)."""
    label, rows = ARM_LABELS[arm], []

    def add(quantity: str, value: float, split: str = "", convention: str = ""):
        """Append one ledger row unless the value is missing."""
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            rows.append(dict(system=system, arm=label, seed=seed if arm != "deep_ensemble" else DEEP_ENSEMBLE_SEED,
                             quantity=quantity, split=split, convention=convention, value=float(value)))

    stochastic = arm != "deterministic"
    if (ev / "ic_fitted.csv").exists():
        d = pd.read_csv(ev / "ic_fitted.csv")
        add("ic_noise_scale", d.sigma.iloc[0], convention="fitted")
        for sp, tag in SPLIT.items():
            s = d[d.split == sp]
            add("g_nll", s.nll_ensemble.mean(), tag, "fitted")
            add("cov90_err", abs(s.cov90_ensemble.mean() - 0.9), tag, "fitted")
            add("var_alea", s.var_alea.mean(), tag)
            if stochastic:
                ve, va = s.var_epi.mean(), s.var_alea.mean()
                add("var_epi", ve, tag)
                add("alpha", _ratio(ve, ve + va), tag, "fitted")
                if sp == "ood":
                    add("var_ratio", _ratio(ve, va), tag)
        if arm == "fresh_mask" and system == "CP":  # dropout-only ID NLL of the fresh-mask masks (T6)
            add("g_nll", d[d.split == "in_dist"].nll_masks_only.mean(), "ID", "dropout-only")
    if (ev / "ic_fixed.csv").exists():
        d = pd.read_csv(ev / "ic_fixed.csv")
        for sp, tag in SPLIT.items():
            s = d[d.split == sp]
            add("g_nll", s.nll_ensemble.mean(), tag, "fixed")
            add("cov90_err", s.cov90_abs_error_ensemble.mean(), tag, "fixed")
            if stochastic:
                add("nll_masks_only", s.nll_masks_only.mean(), tag)
    if (ev / "ood.csv").exists():
        d = pd.read_csv(ev / "ood.csv")
        for sp, tag in SPLIT.items():
            s = d[d.split == sp]
            add("mse", s.mse.mean(), tag)
            add("phys_e_drift", s.phys_e_drift.mean(), tag)
            if system == "CP":
                add("mask_variance", s.mask_variance.mean(), tag)
        if system == "CP" and stochastic:
            o = d[d.split == "ood"]
            if o.mask_variance.nunique() > 1:
                add("spearman_rho", spearmanr(o.log_distance_from_train_range, o.mask_variance)[0], "OOD")
    if (ev / "hdrift.json").exists():
        h = json.loads((ev / "hdrift.json").read_text())
        add("h_drift", h["h_drift"])
        add("h_drift_final", h["h_drift_final"])
    return rows


def build(runs_root: Path, seeds: Optional[List[int]] = None, arms: Optional[List[str]] = None):
    """(ledger, cp_figure, sc_figure) DataFrames for every (system, arm, seed) that has stage files."""
    seeds = SEEDS if seeds is None else seeds
    ledger, cp_fig, sc_fig = [], [], []
    for system, sdir in SYSTEMS.items():
        for arm in arms or ARM_DIRS:
            for seed in ([DEEP_ENSEMBLE_SEED] if arm == "deep_ensemble" else seeds):
                try:
                    ev = eval_dir(runs_root, sdir, arm, seed)
                except FileNotFoundError:
                    continue
                if not ev.exists():
                    continue
                ledger += run_rows(system, arm, seed, ev)
                label = ARM_LABELS[arm]
                if arm != "deterministic" and system == "CP" and (ev / "ic_fitted.csv").exists():
                    d = pd.read_csv(ev / "ic_fitted.csv")
                    cp_fig.append(pd.DataFrame(dict(arm=label, seed=DEEP_ENSEMBLE_SEED if arm == "deep_ensemble" else seed,
                                                    mass_ratio=d.mass_ratio, split=d.split, distance=d.distance,
                                                    var_epi=d.var_epi, var_alea=d.var_alea)))
                if arm != "deterministic" and system == "SC" and (ev / "ood.csv").exists():
                    d = pd.read_csv(ev / "ood.csv")
                    sc_fig.append(pd.DataFrame(dict(arm=label, seed=DEEP_ENSEMBLE_SEED if arm == "deep_ensemble" else seed,
                                                    k=d.k_value, split=d.split, distance=d.distance, var_epi=d.mask_variance, mse=d.mse)))
    keys = ["system", "arm", "seed", "quantity", "split", "convention"]
    led = pd.DataFrame(ledger, columns=LEDGER_COLUMNS).sort_values(keys, na_position="first").reset_index(drop=True)
    cp = pd.concat(cp_fig).sort_values(["arm", "seed", "mass_ratio"]).reset_index(drop=True) if cp_fig else pd.DataFrame()
    sc = pd.concat(sc_fig).reset_index(drop=True) if sc_fig else pd.DataFrame()
    return led, cp, sc


def main(argv=None) -> int:
    """Write the CP / SC ledger and figure tables from the stage files of a run tree."""
    ap = argparse.ArgumentParser(description="Build the CP/SC ledger and figure tables from the evaluation stage files")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = ap.parse_args(argv)
    led, cp, sc = build(args.runs, [int(s) for s in args.seeds.split(",")])
    (args.out / "ledger").mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    led.to_csv(args.out / "ledger" / "classical_per_seed.csv", index=False)
    cp.to_csv(args.out / "figures" / "cp_ood_by_mass_ratio.csv", index=False)
    sc.to_csv(args.out / "figures" / "sc_ood_by_k.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
