#!/usr/bin/env python3
"""Build the paper's data tables T1 and T6-T21 from results/ only.

Inputs : results/ledger/{classical,tfim}_per_seed.csv (per-seed values; variances averaged over t>=1; alpha = ratio of averaged variances) and results/extra/*.csv (T12-T15, T17, T19).  Means and sd (ddof=1) over seeds 42-51 are recomputed here.
Outputs: out/tables/T<n>.csv (long format, one row per cell) and out/tables/T<n>.tex (plain booktabs).
Bold rule: among the four single-model stochastic arms the lowest mean of a tested column (accuracy/drift or likelihood family) is bold iff
all of its paired tests (41_effect_sizes.py, Holm-adjusted p < 0.05) are significant; Deterministic / Deep ensemble rows are references.
Usage: python scripts/40_build_tables.py [--results results] [--out out/tables]
"""
from __future__ import annotations

import argparse
import importlib.util
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("effect_sizes", HERE / "41_effect_sizes.py")
ES = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ES)

FIXED, FRESH, CONC, VAR, DET, DE = "Fixed-mask MC", "Fresh-mask MC", "Concrete", "Variational", "Deterministic", "Deep ensemble"
STOCH = [FRESH, CONC, VAR, FIXED]
ARMS6, ARMS5, ARMS3 = STOCH + [DET, DE], STOCH + [DET], [FRESH, FIXED, DET]
LAB = {DE: "Deep ens. x5"}


# ------------------------------------------------------------------ number formatting (mean: 3 sig. figs / 2 dp >= 10; sd: 2 sig. figs)
def _exp(x: float, sig: int) -> int:
    return int(f"{abs(x):.{sig - 1}e}".split("e")[1])


def sci(x: float, sig: int) -> str:
    """Scientific notation with ``sig`` significant figures, e.g. 1.2e-4."""
    m, e = f"{x:.{sig - 1}e}".split("e")
    return f"{m}e{int(e)}"


def fmt_mean(x: float) -> str:
    """Mean to 3 significant figures (2 decimals from 10 up; scientific below 1e-3)."""
    a = abs(x)
    if x == 0:
        return "0"
    if a >= 10:
        return f"{x:.2f}"
    if a >= 1:
        return f"{x:.3f}"
    if a >= 1e-3:
        return f"{x:.{max(0, 2 - _exp(x, 3))}f}"
    return sci(x, 2)


def fmt_sd(x: float) -> str:
    """Standard deviation to 2 significant figures."""
    if x == 0:
        return "0"
    return sci(x, 2) if x < 1e-3 else f"{x:.{max(0, 1 - _exp(x, 2))}f}"


def neg(s: str) -> str:
    """Replace a leading minus by a math-mode minus."""
    return "$-$" + s[1:] if s.startswith("-") else s


# ------------------------------------------------------------------ table model
@dataclass
class Cell:
    nums: List[float] = field(default_factory=list)  # numbers in the order they are printed
    display: str = "n/a"
    bold: bool = False
    mark: str = ""

    def tex(self) -> str:
        """LaTeX of the cell (superscript mark, bold)."""
        s = self.display + (f"$^{{{self.mark}}}$" if self.mark else "")
        return r"\textbf{" + s + "}" if self.bold else s


@dataclass
class Row:
    labels: List[str]
    cells: List[Cell]
    block: str = ""


@dataclass
class Part:
    headers: List[str]  # label headers + data headers
    rows: List[Row]
    n_label: int = 1


@dataclass
class Table:
    tid: str
    parts: List[Part]


NA = Cell()


def stat_cell(v: Optional[pd.Series], mode: str) -> Cell:
    """mean (+ sd, ddof=1) over seeds; a single-value series is the 5-member deep ensemble (n=1)."""
    if v is None or len(v.dropna()) == 0:
        return Cell()
    v = v.dropna().to_numpy(float)
    m = float(v.mean())
    if mode == "mean":
        return Cell([m], neg(fmt_mean(m)))
    if len(v) == 1:
        return Cell([m, 1.0], f"{neg(fmt_mean(m))} (n=1)")
    sd = float(v.std(ddof=1))
    if sd < 1e-9 * max(abs(m), 1.0):
        sd = 0.0
    return Cell([m, sd], f"{neg(fmt_mean(m))} $\\pm$ {fmt_sd(sd)}")


def num_cell(x: float, text: Optional[str] = None) -> Cell:
    """Cell holding a single number."""
    return Cell([float(x)], text if text is not None else neg(fmt_mean(float(x))))


# ------------------------------------------------------------------ ledger access and bold rule
class Ledger:
    """Per-seed ledger (system, arm, seed, quantity, split, convention, value) plus the paired-test records used for bold decisions."""

    def __init__(self, res: Path):
        self.led = pd.concat([pd.read_csv(res / "ledger" / f, keep_default_na=False) for f in ("classical_per_seed.csv", "tfim_per_seed.csv")],
                             ignore_index=True)
        self.recs = ES.effect_records(self.led)
        self.idx = {(r["system"], r["key"], r["comparator"]): r for r in self.recs if not r["reference"]}

    def series(self, system: str, arm: str, key: Key) -> Optional[pd.Series]:
        """Per-seed values (seeds 42-51) of one quantity, or None if the ledger has no such rows (quantity undefined for this arm)."""
        q, sp, cv = key
        d = self.led[(self.led.system == system) & (self.led.arm == arm) & (self.led.quantity == q) & (self.led.split == sp) &
                     (self.led.convention == cv) & (self.led.seed >= 42)]
        if d.empty:
            return None
        assert d.seed.is_unique, (system, arm, key)
        return d.set_index("seed").value.astype(float)

    def bold_arm(self, system: str, key: Key, means: Dict[str, float]) -> Optional[str]:
        """Arm to bold in this column (or None) per the Holm-significance rule."""
        cand = {a: m for a, m in means.items() if a in STOCH}
        if len(cand) < 2:
            return None
        best = min(cand, key=cand.get)
        used = ES.COMPARATORS if best == FIXED else [best]
        rr = [self.idx.get((system, "|".join(key), c)) for c in used]
        if any(r is None or r["n"] != 10 for r in rr):
            return None
        ok = all(r["p_holm"] < 0.05 and (r["mean_diff"] < 0 if best == FIXED else r["mean_diff"] > 0) for r in rr)
        return best if ok else None


ELIG = {s: {k for f in ("i", "ii") for _, k in fam[f]} for s, fam in (("CP", ES.CLS_FAMILIES), ("SC", ES.CLS_FAMILIES), ("TFIM", ES.TFIM_FAMILIES))}
Key = Tuple[str, str, str]  # (quantity, split, convention)


@dataclass
class Col:
    header: str
    key: Key


@dataclass
class Blk:
    name: str
    system: str
    arms: List[str]


def main_part(L: Ledger, cols: List[Col], blocks: List[Blk], mode: str, bold: bool = True, first="Method") -> Part:
    """Table part of ``cols`` x arms per block; the bold rule is applied per column when ``bold``."""
    rows: List[Row] = []
    for b in blocks:
        brow: List[Row] = [Row([LAB.get(a, a)], [], b.name) for a in b.arms]
        for ci, c in enumerate(cols):
            key = c.key
            ser = {a: L.series(b.system, a, key) for a in b.arms}
            best = None
            if bold and key in ELIG[b.system]:
                best = L.bold_arm(b.system, key, {a: float(s.mean()) for a, s in ser.items() if s is not None})
            for r, a in zip(brow, b.arms):
                cell = stat_cell(ser[a], mode)
                cell.bold = a == best
                r.cells.append(cell)
        rows += brow
    return Part([first] + [c.header for c in cols], rows)


# ------------------------------------------------------------------ the main-style tables
def K(q: str, sp: str = "", cv: str = "") -> Key:
    """Ledger key (quantity, split, convention)."""
    return (q, sp, cv)


CPSC = [Blk("Coupled pendulum", "CP", ARMS6), Blk("Spring chain", "SC", ARMS6)]
SIG_HAT = Col("sigma-hat", K("ic_noise_scale", "", "fitted"))
NLL_FIT = [Col("G-NLL sigma-hat ID", K("g_nll", "ID", "fitted")), Col("G-NLL sigma-hat OOD", K("g_nll", "OOD", "fitted"))]
NLL_FIX = [Col("G-NLL sigma=0.1 ID", K("g_nll", "ID", "fixed")), Col("G-NLL sigma=0.1 OOD", K("g_nll", "OOD", "fixed"))]
COV_FIT = [Col("Cov90 sigma-hat ID", K("cov90_err", "ID", "fitted")), Col("Cov90 sigma-hat OOD", K("cov90_err", "OOD", "fitted"))]
COV_FIX = [Col("Cov90 sigma=0.1 ID", K("cov90_err", "ID", "fixed")), Col("Cov90 sigma=0.1 OOD", K("cov90_err", "OOD", "fixed"))]
MASKS_ONLY = [Col("NLL masks-only ID", K("nll_masks_only", "ID")), Col("NLL masks-only OOD", K("nll_masks_only", "OOD"))]
T1_COLS = [Col("ID MSE", K("mse", "ID")), Col("OOD MSE", K("mse", "OOD")), SIG_HAT] + NLL_FIT + NLL_FIX + [
    Col("OOD alpha", K("alpha", "OOD", "fitted")), Col("H-drift", K("h_drift"))]
ALPHA2 = [Col("alpha ID", K("alpha", "ID", "fitted")), Col("alpha OOD", K("alpha", "OOD", "fitted"))]


def build_main_family(L: Ledger) -> List[Table]:
    """T1, T6-T11, T16, T18: mean (+/- sd) over seeds per arm, bold by the significance rule."""
    T = [Table("T1", [main_part(L, T1_COLS, CPSC, "mean")]), Table("T9", [main_part(L, T1_COLS, CPSC, "pm")])]
    T.append(Table("T7", [main_part(L, [SIG_HAT] + NLL_FIT + NLL_FIX, CPSC, "pm"), main_part(L, COV_FIT + COV_FIX + MASKS_ONLY, CPSC, "pm")]))
    T.append(Table("T10", [main_part(L, [Col("Var_epi ID", K("mask_variance", "ID")), Col("Var_epi OOD", K("mask_variance", "OOD")), Col("OOD MSE", K("mse", "OOD")),
                                          Col("OOD Spearman rho", K("spearman_rho", "OOD")), Col("Drift mean", K("h_drift")),
                                          Col("Drift final", K("h_drift_final"))], [Blk("Coupled pendulum", "CP", ARMS6)], "pm")]))
    tf = [Blk("TFIM", "TFIM", ARMS5)]
    T.append(Table("T11", [
        main_part(L, [Col("State RMSE ID", K("state_rmse", "ID")), Col("State RMSE OOD", K("state_rmse", "OOD")),
                      Col("eps-hat", K("ic_noise_scale", "", "fitted")), Col("G-NLL eps-hat ID", K("g_nll", "ID", "fitted")),
                      Col("G-NLL eps-hat OOD", K("g_nll", "OOD", "fitted"))], tf, "pm"),
        main_part(L, [Col("G-NLL eps=0.10 ID", K("g_nll", "ID", "fixed")), Col("G-NLL eps=0.10 OOD", K("g_nll", "OOD", "fixed")),
                      Col("Norm drift", K("norm_drift")), Col("Energy drift", K("energy_drift")), Col("OOD alpha", K("alpha", "OOD", "fitted"))], tf, "pm")]))
    fit = main_part(L, [Col("eps", K("ic_noise_scale", "", "fitted")), Col("G-NLL ID", K("g_nll", "ID", "fitted")), Col("G-NLL OOD", K("g_nll", "OOD", "fitted")),
                        Col("Cov90 ID", K("cov90_err", "ID", "fitted")), Col("Cov90 OOD", K("cov90_err", "OOD", "fitted"))] + ALPHA2,
                    [Blk("Fitted scale", "TFIM", ARMS5)], "pm")
    true = main_part(L, [Col("G-NLL ID", K("g_nll", "ID", "fixed")), Col("G-NLL OOD", K("g_nll", "OOD", "fixed")), Col("Cov90 ID", K("cov90_err", "ID", "fixed")),
                         Col("Cov90 OOD", K("cov90_err", "OOD", "fixed")), Col("alpha ID", K("alpha", "ID", "fixed")),
                         Col("alpha OOD", K("alpha", "OOD", "fixed"))], [Blk("True scale eps=0.10", "TFIM", ARMS5)], "pm")
    for r in true.rows:  # the fixed scale is the data's true 0.10
        r.cells.insert(0, num_cell(0.10, "0.10"))
    T.append(Table("T8", [Part(fit.headers, fit.rows + true.rows)]))
    var_cols = [Col("Var_alea ID", K("var_alea", "ID")), Col("Var_alea OOD", K("var_alea", "OOD")), Col("Var_epi ID", K("var_epi", "ID")),
                Col("Var_epi OOD", K("var_epi", "OOD")), Col("Var_epi/Var_alea OOD", K("var_ratio", "OOD"))] + ALPHA2
    T.append(Table("T16", [main_part(L, var_cols, [Blk("Coupled pendulum", "CP", ARMS6), Blk("Spring chain", "SC", ARMS6), Blk("TFIM", "TFIM", ARMS5)],
                                     "pm", bold=False)]))
    T.append(Table("T18", [main_part(L, [Col("Mean H-drift", K("h_drift")), Col("Final H-drift", K("h_drift_final"))], CPSC, "pm"),
                           main_part(L, [Col("Norm drift", K("norm_drift")), Col("Energy drift", K("energy_drift"))], tf, "pm")]))
    # T6: ID G-NLL of the dropout-only baseline vs the IC ensemble; reduction = per-seed dropout-only NLL minus IC-ensemble NLL
    base = L.series("CP", FRESH, K("g_nll", "ID", "dropout-only"))
    rows = []
    for arm, lab, conv in ((FRESH, "Fresh-mask MC (dropout-only)", "dropout-only"), (FRESH, "Fresh-mask MC (IC ensemble)", "IC ensemble"),
                           (FIXED, "Fixed-mask MC (IC ensemble)", "IC ensemble")):
        if conv == "dropout-only":
            rows.append(Row([lab], [Cell([], conv), stat_cell(base, "pm"), Cell(), Cell()]))
            continue
        nll = L.series("CP", arm, K("g_nll", "ID", "fitted"))
        rows.append(Row([lab], [Cell([], conv), stat_cell(nll, "pm"), stat_cell(base - nll, "pm"), stat_cell(L.series("CP", arm, K("ic_noise_scale", "", "fitted")), "pm")]))
    T.append(Table("T6", [Part(["Method", "NLL convention", "ID G-NLL", "Reduction vs dropout-only", "sigma-hat"], rows)]))
    return T


# ------------------------------------------------------------------ effect-size tables T20 / T21
def effect_cell(r: Dict) -> Cell:
    """Cell with a paired effect and its bootstrap interval, starred if Holm-significant."""
    if r["n"] != 10:
        return Cell()
    mark = "*" if (not r["reference"] and r["p_holm"] < 0.05) else ""
    c = Cell([r["mean_diff"], r["ci_lo"], r["ci_hi"]], f"{neg(fmt_mean(r['mean_diff']))}", mark=mark)
    c.display += f" [{neg(fmt_mean(r['ci_lo']))}, {neg(fmt_mean(r['ci_hi']))}]"
    return c


def effect_table(L: Ledger, tid: str, groups: List[tuple], systems: List[tuple]) -> Table:
    """groups: [(fam, [headers])]; systems: [(record system, block label)]; one part per group, comparator rows per system."""
    parts = []
    for fam, hdrs in groups:
        rows = []
        for sysname, lab in systems:
            for comp in ES.COMPARATORS + [ES.REFERENCE]:
                cells = []
                for h in hdrs:
                    r = next(x for x in L.recs if x["system"] == sysname and x["family"] == fam and x["column"] == h and x["comparator"] == comp)
                    cells.append(effect_cell(r))
                rows.append(Row([comp + (" (ref.)" if comp == DET else "")], cells, f"{lab} family ({fam})"))
        parts.append(Part(["Comparator"] + hdrs, rows))
    return Table(tid, parts)


def build_effect_tables(L: Ledger) -> List[Table]:
    """Tables T20 and T21: effect sizes of Fixed-mask MC against each comparator."""
    cls = [("i", ["ID MSE", "OOD MSE", "H-drift"]), ("i", ["Phys. E-drift ID", "Phys. E-drift OOD"]),
           ("ii", ["G-NLL sigma-hat ID", "G-NLL sigma-hat OOD"]), ("ii", ["G-NLL sigma=0.1 ID", "G-NLL sigma=0.1 OOD"]),
           ("ii", ["Cov90 sigma-hat ID", "Cov90 sigma-hat OOD"]), ("ii", ["Cov90 sigma=0.1 ID", "Cov90 sigma=0.1 OOD"])]
    cs = [("CP", "Coupled pendulum"), ("SC", "Spring chain")]
    t20 = effect_table(L, "T20", cls, cs)
    t20.parts += effect_table(L, "T20", [("iii", ["Var_epi ID", "Var_alea ID", "Var_total ID"])], cs).parts
    t20.parts += effect_table(L, "T20", [("iii", ["Var_epi OOD", "Var_alea OOD", "Var_total OOD"])], cs).parts
    tfg = [("i", ["State RMSE ID", "State RMSE OOD"]), ("i", ["Norm drift", "Energy drift"]), ("ii", ["G-NLL eps-hat ID", "G-NLL eps-hat OOD"]),
           ("ii", ["G-NLL eps=0.10 ID", "G-NLL eps=0.10 OOD"]), ("ii", ["Cov90 eps-hat ID", "Cov90 eps-hat OOD"]),
           ("ii", ["Cov90 eps=0.10 ID", "Cov90 eps=0.10 OOD"]), ("iii", ["Var_epi ID", "Var_alea ID", "Var_total ID"]),
           ("iii", ["Var_epi OOD", "Var_alea OOD", "Var_total OOD"])]
    return [t20, effect_table(L, "T21", tfg, [("TFIM", "TFIM")])]


# ------------------------------------------------------------------ extra tables
def build_mask_bank(ex: Path) -> Table:
    """Table T13: spring-chain results by mask-bank size."""
    d = pd.read_csv(ex / "mask_bank_per_seed.csv")
    w = d.pivot_table(index=["M", "seed"], columns="quantity", values="value").reset_index()
    w["alpha"] = w.var_epi_ood / (w.var_epi_ood + w.var_alea_ood)  # ratio of averaged variances
    w["ratio"] = w.var_epi_ood / w.var_alea_ood
    layout = [[("mse_id", 0), ("mse_ood", 0), ("sigma_hat", 0), ("nll_fit_id", 0), ("nll_fit_ood", 0), ("nll_fix_id", 0), ("nll_fix_ood", 0)],
              [("nll_mask_id", 0), ("nll_mask_ood", 0), ("cov_fit_id", 0), ("cov_fit_ood", 0), ("cov_fix_id", 0), ("cov_fix_ood", 0)],
              [("alpha", 0), ("ratio", 0), ("phys_drift_id", 0), ("phys_drift_ood", 0), ("h_drift", 0)]]
    parts = []
    for cols in layout:
        rows = []
        for M in sorted(w.M.unique()):
            g = w[w.M == M].set_index("seed")
            cells = [stat_cell(g[c], "pm") for c, _ in cols]  # quantities undefined for a single mask (M=1) are NaN -> n/a
            rows.append(Row([f"M={int(M)}"], cells))
        parts.append(Part(["Mask bank"] + [c for c, _ in cols], rows))
    return Table("T13", parts)


def build_efficiency(ex: Path) -> Table:
    """Table T14: efficiency benchmark."""
    e = pd.read_csv(ex / "efficiency_summary.csv", dtype=str)
    want = [(FRESH, 0, FRESH), (FIXED, 100, "Fixed-mask MC (M=100)"), (FIXED, 200, "Fixed-mask MC (M=200)"), (CONC, 0, CONC), (VAR, 0, VAR),
            (DET, 0, "Det. HNN"), (DE, 0, "Deep ens. x5")]
    rows = []
    for m, mn, lab in want:
        r = e[(e.arm == m) & (e.mask_bank.astype(int) == mn)]
        assert len(r) == 1, (m, mn)
        r = r.iloc[0]
        cells = [num_cell(float(r.models), r.models), num_cell(float(r.samples), r.samples)]
        cells += [num_cell(float(r[c]), f"{float(r[c]):.3f}") for c in ("rel_train_time", "rel_infer_time", "rel_storage")]
        cells.append(Cell([], r.decomposition_axes))
        rows.append(Row([lab], cells))
    return Table("T14", [Part(["Method", "Models", "Samples", "Train time", "Infer time", "Storage", "Decomp. axes"], rows)])


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve of score ``s`` for binary labels ``y`` (ties get average ranks)."""
    r = rankdata(s)
    n1 = int(y.sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def build_hidden_param(ex: Path) -> Table:
    """Table T15: hidden-parameter detection."""
    B, A, C = (pd.read_csv(ex / f"hidden_param_{x}_auroc.csv") for x in "BAC")
    leak = pd.read_csv(ex / "hidden_param_leakage_k0.csv")
    KS, ANCH = (2, 3, 5, 8, 10), (0.75, 1.0, 1.5)
    mB = B.groupby(["r0", "method", "k"]).auroc.mean()
    rows = []
    for r0 in ANCH:
        p_raw, diffs = [], []
        for k in KS:
            pv = lambda m: B[(B.r0 == r0) & (B.method == m) & (B.k == k)].sort_values("seed").auroc.to_numpy()
            p_raw.append(float(sps.ttest_rel(pv(FIXED), pv(FRESH)).pvalue))
            diffs.append(float((pv(FIXED) - pv(FRESH)).mean()))
        ph = ES.holm(p_raw)  # Holm over the five k of one anchor
        for i, m in enumerate((FIXED, FRESH)):
            rows.append(Row([str(r0) if i == 0 else "", m], [num_cell(mB[(r0, m, k)], f"{mB[(r0, m, k)]:.3f}") for k in KS]))
        rows.append(Row(["", "Delta (Fixed - Fresh)"], [Cell([d], neg(f"{d:.3f}"), mark="dagger" if p < 0.05 else "") for d, p in zip(diffs, ph)]))
    p1 = Part(["r0", "Method"] + [f"k={k}" for k in KS], rows, n_label=2)
    mA = A[A.k.isin((1, 2, 5, 10))].groupby(["method", "k"]).auroc.mean()
    mC = C.groupby(["r0", "method", "k"]).auroc.mean()
    lk = leak[leak.k == 0]
    pk = {m: (mA.loc[m].max(), int(mA.loc[m].idxmax())) for m in (FIXED, FRESH)}
    rg = {m: (mC.xs(m, level="method").min(), mC.xs(m, level="method").max()) for m in (FIXED, FRESH)}
    leak_auc = auroc(lk.is_mismatch.to_numpy(), lk.raw_var_r0.to_numpy())
    p2 = Part(["Predictive-variance score", FIXED, FRESH], [
        Row(["Leakage control, k=0, r0=1.0 (pooled over both methods)"], [Cell([leak_auc], f"{leak_auc:.3f}")]),
        Row(["Peak over k in {1,2,5,10}, r0=1.0"], [Cell([pk[m][0], pk[m][1]], f"{pk[m][0]:.3f} (k={pk[m][1]})") for m in (FIXED, FRESH)]),
        Row(["Range over r0 and k in {2,3,5,8,10}"], [Cell(list(rg[m]), f"{rg[m][0]:.3f}--{rg[m][1]:.3f}") for m in (FIXED, FRESH)])])
    return Table("T15", [p1, p2])


def build_sweep(ex: Path) -> Table:
    """Table T17: IC-noise sweep."""
    d = pd.read_csv(ex / "ic_noise_sweep_per_seed.csv")
    d["alpha"] = d.var_epistemic_mean / (d.var_epistemic_mean + d.var_aleatoric_mean)  # ratio of averaged variances
    mean = d.groupby(["method", "split", "sigma_multiplier"]).mean(numeric_only=True)
    mults = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

    def f(x: float, q: str) -> str:
        if q == "alpha":
            return f"{x:.3f}"
        if x == 0:
            return "0"
        return f"{x:.3f}" if abs(x) >= 1 else (f"{x:.4f}" if abs(x) >= 1e-3 else sci(x, 2))
    rows = []
    for m in (FIXED, FRESH):
        for sp, spl in (("in_dist", "ID"), ("ood", "OOD")):
            for q, ql in (("var_epistemic_mean", "Var_epi"), ("var_aleatoric_mean", "Var_alea"), ("alpha", "alpha")):
                v = [float(mean.loc[(m, sp, x), q]) for x in mults]
                rows.append(Row([m, spl, ql], [Cell([x], f(x, q)) for x in v]))
    return Table("T17", [Part(["Method", "Split", "Quantity"] + [f"{x:g}x" for x in mults], rows, n_label=3)])


def build_long_horizon(ex: Path) -> Table:
    """Table T19: long-horizon energy drift."""
    d = pd.read_csv(ex / "long_horizon_per_seed.csv")
    rows = []
    for sysn, name in (("SC", "Spring chain"), ("CP", "Coupled pendulum")):
        for arm in ARMS3[:2] + [DET]:
            cells = []
            for t in (100, 1000):
                for sp in ("in_dist", "ood"):
                    cells.append(stat_cell(d[(d.system == sysn) & (d.arm == arm) & (d.t == t) & (d.split == sp)].set_index("seed").value, "pm"))
            rows.append(Row([arm], [cells[0], cells[1], cells[2], cells[3]], name))
    return Table("T19", [Part(["Method", "T=100 ID", "T=100 OOD", "T=1000 ID", "T=1000 OOD"], rows)])


def build_poisson(ex: Path) -> Table:
    """Table T12: Poisson-bracket residual with paired tests."""
    d = pd.read_csv(ex / "poisson_bracket_per_seed.csv")
    arms = [("Deterministic", DET), ("Fresh-mask MC", FRESH), ("Fixed-mask per-sub-step", "Fixed-mask MC, per-sub-step training"),
            ("Fixed-mask per-rollout", "Fixed-mask MC, per-rollout training")]
    lg = {a: d[d.arm == a].sort_values("seed").mean_log_resid.to_numpy() for a, _ in arms}
    rows = []
    for a, lab in arms:
        gm = np.exp(lg[a])
        rows.append(Row([lab], [Cell([float(np.exp(lg[a].mean()))], f"{np.exp(lg[a].mean()):.4f}"),
                                Cell([float(gm.min()), float(gm.max())], f"{gm.min():.4f}--{gm.max():.4f}")]))
    p1 = Part(["Method", "Geometric mean", "Range over seeds"], rows)
    rows = []
    for x, y, lab in (("Fixed-mask per-rollout", "Fixed-mask per-sub-step", "Per-rollout - Per-sub-step"), ("Fixed-mask per-rollout", "Fresh-mask MC", "Per-rollout - Fresh"),
                      ("Fixed-mask per-sub-step", "Fresh-mask MC", "Per-sub-step - Fresh"), ("Fixed-mask per-rollout", "Deterministic", "Per-rollout - Deterministic"),
                      ("Fixed-mask per-sub-step", "Deterministic", "Per-sub-step - Deterministic"), ("Fresh-mask MC", "Deterministic", "Fresh - Deterministic")):
        diff = lg[x] - lg[y]
        n, m, se = len(diff), diff.mean(), diff.std(ddof=1) / math.sqrt(len(diff))
        tcrit = sps.t.ppf(0.975, n - 1)
        p = float(sps.ttest_1samp(diff, 0.0).pvalue)
        ptxt = f"{p:.3f}" if p >= 0.01 else f"{p:.0e}"
        rows.append(Row([lab], [Cell([m], f"{m:+.3f}"), Cell([m - tcrit * se, m + tcrit * se], f"[{m - tcrit * se:+.3f},{m + tcrit * se:+.3f}]"), Cell([p], ptxt)]))
    return Table("T12", [p1, Part(["Comparison", "Mean log difference", "95% CI", "p"], rows)])


# ------------------------------------------------------------------ output
def write(t: Table, out: Path) -> None:
    """Write the table as long-format CSV and booktabs LaTeX."""
    recs, tex = [], [r"\begin{table}[t]", r"\centering", r"\small"]
    for pi, p in enumerate(t.parts):
        nd = len(p.headers) - p.n_label
        tex += [r"\begin{tabular}{" + "l" * p.n_label + "c" * nd + "}", r"\toprule", " & ".join(p.headers) + r" \\", r"\midrule"]
        last = None
        for ri, r in enumerate(p.rows):
            if r.block != last and r.block:
                tex.append(r"\multicolumn{%d}{l}{\emph{%s}} \\" % (len(p.headers), r.block))
                last = r.block
            tex.append(" & ".join(r.labels + [""] * (p.n_label - len(r.labels)) + [c.tex() for c in r.cells]) + r" \\")
            for ci, c in enumerate(r.cells):
                recs.append(dict(table=t.tid, part=pi, block=r.block, row_idx=ri, row=" / ".join(r.labels), col_idx=ci, column=p.headers[p.n_label + ci],
                                 value=c.nums[0] if c.nums else np.nan, nums=";".join(repr(float(x)) for x in c.nums), mark=c.mark,
                                 bold=c.bold, display=c.display))
        tex += [r"\bottomrule", r"\end{tabular}", r"\par\medskip" if pi < len(t.parts) - 1 else ""]
    tex.append(r"\end{table}")
    pd.DataFrame(recs).to_csv(out / f"{t.tid}.csv", index=False)
    (out / f"{t.tid}.tex").write_text("\n".join(tex) + "\n")


def main() -> None:
    """Build the tables selected on the command line."""
    root = HERE.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(root / "results"))
    ap.add_argument("--out", default=str(root / "out/tables"))
    ap.add_argument("--lenient", action="store_true", help="skip (and report) tables whose inputs are incomplete, e.g. for smoke-sized results")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    L = Ledger(res)
    ex = res / "extra"
    builders = [("main and effect-size tables", lambda: build_main_family(L) + build_effect_tables(L)), ("T13", lambda: [build_mask_bank(ex)]),
                ("T14", lambda: [build_efficiency(ex)]), ("T15", lambda: [build_hidden_param(ex)]), ("T17", lambda: [build_sweep(ex)]),
                ("T19", lambda: [build_long_horizon(ex)]), ("T12", lambda: [build_poisson(ex)])]
    tables = []
    for name, build in builders:
        try:
            tables += build()
        except Exception as err:  # incomplete inputs of one table must not hide the others in a lenient run
            if not a.lenient:
                raise
            print(f"skipped {name}: {type(err).__name__}: {err}")
    for t in tables:
        write(t, out)
    pd.DataFrame(L.recs).to_csv(out / "effect_sizes.csv", index=False)
    print("wrote", ", ".join(t.tid for t in tables), "to", out)


if __name__ == "__main__":
    main()
