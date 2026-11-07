#!/usr/bin/env python3
"""Paired effect sizes (Fixed-mask MC minus comparator), bootstrap CIs and Holm-adjusted paired t-tests.

Protocol (fixed in advance): two-sided paired t-test on the seed-level values of the ten matched seeds 42-51; Holm correction within each
family of tests PER SYSTEM (families: i accuracy and drift, ii likelihood, iii variance decomposition); 95% percentile bootstrap CI of the
mean paired difference (10,000 resamples, numpy default_rng(0) re-created for every test).  Deterministic is a reference row (effect and CI
only, no test, not part of a Holm family).  Negative difference = Fixed-mask MC lower.

`effect_records(ledger)` is imported by 40_build_tables.py.  Standalone: writes out/tables/effect_sizes.csv.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

SEEDS = list(range(42, 52))
N_BOOT, BOOT_SEED = 10_000, 0
FIXED = "Fixed-mask MC"
COMPARATORS = ["Fresh-mask MC", "Concrete", "Variational"]
REFERENCE = "Deterministic"

# family -> [(column header, (quantity, split, convention))]; quantity "var_total" = var_epi + var_alea (derived per seed)
CLS_FAMILIES = {
    "i": [("ID MSE", ("mse", "ID", "")), ("OOD MSE", ("mse", "OOD", "")), ("H-drift", ("h_drift", "", "")),
          ("Phys. E-drift ID", ("phys_e_drift", "ID", "")), ("Phys. E-drift OOD", ("phys_e_drift", "OOD", ""))],
    "ii": [("G-NLL sigma-hat ID", ("g_nll", "ID", "fitted")), ("G-NLL sigma-hat OOD", ("g_nll", "OOD", "fitted")),
           ("G-NLL sigma=0.1 ID", ("g_nll", "ID", "fixed")), ("G-NLL sigma=0.1 OOD", ("g_nll", "OOD", "fixed")),
           ("Cov90 sigma-hat ID", ("cov90_err", "ID", "fitted")), ("Cov90 sigma-hat OOD", ("cov90_err", "OOD", "fitted")),
           ("Cov90 sigma=0.1 ID", ("cov90_err", "ID", "fixed")), ("Cov90 sigma=0.1 OOD", ("cov90_err", "OOD", "fixed"))],
    "iii": [("Var_epi ID", ("var_epi", "ID", "")), ("Var_epi OOD", ("var_epi", "OOD", "")), ("Var_alea ID", ("var_alea", "ID", "")),
            ("Var_alea OOD", ("var_alea", "OOD", "")), ("Var_total ID", ("var_total", "ID", "")), ("Var_total OOD", ("var_total", "OOD", ""))],
}
TFIM_FAMILIES = {
    "i": [("State RMSE ID", ("state_rmse", "ID", "")), ("State RMSE OOD", ("state_rmse", "OOD", "")), ("Norm drift", ("norm_drift", "", "")),
          ("Energy drift", ("energy_drift", "", ""))],
    "ii": [("G-NLL eps-hat ID", ("g_nll", "ID", "fitted")), ("G-NLL eps-hat OOD", ("g_nll", "OOD", "fitted")),
           ("G-NLL eps=0.10 ID", ("g_nll", "ID", "fixed")), ("G-NLL eps=0.10 OOD", ("g_nll", "OOD", "fixed")),
           ("Cov90 eps-hat ID", ("cov90_err", "ID", "fitted")), ("Cov90 eps-hat OOD", ("cov90_err", "OOD", "fitted")),
           ("Cov90 eps=0.10 ID", ("cov90_err", "ID", "fixed")), ("Cov90 eps=0.10 OOD", ("cov90_err", "OOD", "fixed"))],
    "iii": CLS_FAMILIES["iii"],
}


def holm(p: List[float]) -> List[float]:
    """Holm step-down adjusted p-values (monotone)."""
    m, order, run, adj = len(p), np.argsort(p), 0.0, np.empty(len(p))
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj.tolist()


def boot_ci(d: np.ndarray) -> Tuple[float, float]:
    """95% percentile bootstrap CI of the mean over seeds."""
    rng = np.random.default_rng(BOOT_SEED)
    m = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(fx: Optional[pd.Series], cp: Optional[pd.Series]) -> Optional[Dict]:
    """Paired difference of two per-seed series over seeds 42-51: mean, bootstrap CI and two-sided paired t-test (None if a seed is missing)."""
    if fx is None or cp is None:
        return None
    f, c = fx.reindex(SEEDS), cp.reindex(SEEDS)
    if f.isna().any() or c.isna().any():
        return None
    d = (f - c).to_numpy(float)
    lo, hi = boot_ci(d)
    r = sps.ttest_rel(f.to_numpy(float), c.to_numpy(float))
    return dict(n=len(d), mean_diff=float(d.mean()), ci_lo=lo, ci_hi=hi, p_raw=float(r.pvalue))


def make_getter(led: pd.DataFrame, system: str) -> Callable[[str, tuple], Optional[pd.Series]]:
    """(arm, (quantity, split, convention)) -> per-seed series indexed by seed, from the per-seed ledger."""
    d = led[(led.system == system) & (led.seed >= SEEDS[0])]

    def one(arm: str, q: str, sp: str, cv: str) -> Optional[pd.Series]:
        x = d[(d.arm == arm) & (d.quantity == q) & (d.split == sp) & (d.convention == cv)]
        if x.empty:
            return None
        assert x.seed.is_unique
        return x.set_index("seed").value.astype(float).sort_index()

    def get(arm: str, key: tuple) -> Optional[pd.Series]:
        q, sp, cv = key
        if q == "var_total":
            e, a = one(arm, "var_epi", sp, cv), one(arm, "var_alea", sp, cv)
            return None if e is None or a is None else e.add(a)
        return one(arm, q, sp, cv)
    return get


def run_system(system: str, families: Dict[str, list], get) -> List[Dict]:
    """Paired records of every column of every family against every comparator."""
    recs: List[Dict] = []
    for fam, cols in families.items():
        fam_recs = []
        for hdr, key in cols:
            for comp in COMPARATORS + [REFERENCE]:
                res = paired(get(FIXED, key), get(comp, key))
                rec = dict(system=system, family=fam, column=hdr, key="|".join(key), comparator=comp, reference=comp == REFERENCE, n=0,
                           mean_diff=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_raw=np.nan, p_holm=np.nan)
                if res:
                    rec.update(res)
                    if rec["reference"]:
                        rec["p_raw"] = np.nan
                fam_recs.append(rec)
        tests = [r for r in fam_recs if not r["reference"] and r["n"] == 10 and np.isfinite(r["p_raw"])]
        for r, ph in zip(tests, holm([r["p_raw"] for r in tests])):
            r["p_holm"] = ph
        recs += fam_recs
    return recs


def effect_records(led: pd.DataFrame) -> List[Dict]:
    """All paired tests for CP, SC and TFIM, Holm-adjusted within family per system."""
    recs = run_system("CP", CLS_FAMILIES, make_getter(led, "CP")) + run_system("SC", CLS_FAMILIES, make_getter(led, "SC"))
    return recs + run_system("TFIM", TFIM_FAMILIES, make_getter(led, "TFIM"))


if __name__ == "__main__":
    import argparse

    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Paired effect sizes with Holm-adjusted tests from the per-seed ledgers")
    ap.add_argument("--results", type=Path, default=root / "results", help="directory holding ledger/*.csv")
    ap.add_argument("--out", type=Path, default=root / "out/tables")
    a = ap.parse_args()
    led = pd.concat([pd.read_csv(a.results / "ledger" / f) for f in ("classical_per_seed.csv", "tfim_per_seed.csv")])
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(effect_records(led)).to_csv(a.out / "effect_sizes.csv", index=False)
    print("wrote", a.out / "effect_sizes.csv")
