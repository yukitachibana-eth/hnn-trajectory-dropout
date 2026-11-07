#!/usr/bin/env python3
"""Paper figures 2-6 and 8 from the per-seed tables in results/figures/ (Fig 1 is the schematic in assets/, Fig 7 is not built here).

  Fig 2 / 4  coupled pendulum OOD decomposition: epistemic fraction alpha vs mass ratio, epistemic std vs distance from the training support
  Fig 3 / 6  TFIM nested decomposition: epistemic variance and alpha vs transverse field g
  Fig 5      spring chain: epistemic variance and MSE vs stiffness k, epistemic variance vs physics distance
  Fig 8      spring chain physical energy drift vs rollout step (T = 100 prefix and T = 1000)

Every curve uses the same two-stage aggregation: mean over the rows of one (arm, seed, x), then median and inter-quartile range over seeds
at each x (shaded band only for arms with more than one seed).  alpha is formed per seed and per x from the variance components as the ratio of
averages Var_epi / (Var_epi + Var_alea).  Writes out/figures/fig<N>.{pdf,png} and out/figures/fig<N>_data.csv (the plotted series).
Usage: python scripts/50_build_figures.py [--results results/figures] [--out out/figures] [--only 2,3,4,5,6,8]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# arm -> (colour, marker, linestyle, legend/draw order, primary); Okabe-Ito colours, identical in every figure
ARMS = {
    "Deterministic": ("#222222", "x", ":", 0, False),
    "Deep ensemble": ("#6C757D", "s", "-.", 1, False),
    "Fresh-mask MC": ("#009E73", "o", "-", 2, False),
    "Concrete": ("#0072B2", "^", "-", 3, False),
    "Variational": ("#CC79A7", "v", "-", 4, False),
    "Fixed-mask MC": ("#D55E00", "D", "-", 5, True),
}
LABEL = {"Deep ensemble": "Deep ensemble ×5"}
INK, GRAY, BAND_ALPHA = "#222222", "#9A9A9A", 0.18
TFIM_TRAIN_BAND = (1.3, 1.9)  # training range of the transverse field
CP_TRAIN_BAND = (0.5, 2.0)    # training range of the mass ratio
SC_TRAIN_BAND = (1.0, 5.0)    # training range of the stiffness

PLOTTED: List[dict] = []  # rows of the current figure's data csv


def apply_style() -> None:
    """Set the matplotlib style of all figures."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.8, "lines.linewidth": 1.6, "lines.markersize": 5.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white", "figure.facecolor": "white"})


def style_axis(ax, title=None, xlabel=None, ylabel=None, tag=None) -> None:
    """Grid, spines, titles and the bold panel tag."""
    if title:
        ax.set_title(title, pad=6.0)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color="#D8D8D8", alpha=0.5, linewidth=0.6)
    ax.grid(True, axis="x", color="#EAEAEA", alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if tag:
        ax.text(-0.15, 1.06, tag, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", ha="left", clip_on=False,
                color=INK, zorder=10)


def train_band(ax, lo: float, hi: float) -> None:
    """Shade the training range on the x axis."""
    ax.axvspan(lo, hi, color="#D9D9D9", alpha=0.35, zorder=0)


def aggregate(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Two-stage aggregation of one arm: mean per (seed, x), then median / quartiles over seeds at each x."""
    d = df.assign(xk=df[x].round(9)).groupby(["seed", "xk"]).agg(x=(x, "first"), y=(y, "mean")).reset_index()
    g = d.groupby("xk").y
    out = pd.DataFrame({"x": d.groupby("xk").x.first(), "y": g.median(), "lo": g.quantile(0.25), "hi": g.quantile(0.75), "n": g.size()})
    return out.sort_values("x").reset_index(drop=True)


def draw(ax, panel: int, arm: str, s: pd.DataFrame, clip01=False, markevery=None, in_band=None) -> None:
    """Draw one aggregated curve (band, line, markers) and record the plotted values."""
    color, marker, ls, _, primary = ARMS[arm]
    y, lo, hi = (np.clip(s[c].to_numpy(float), 0, 1) if clip01 else s[c].to_numpy(float) for c in ("y", "lo", "hi"))
    x = s["x"].to_numpy(float)
    banded = bool((s["n"] > 1).any())
    lw = 2.2 if primary else 1.5
    if banded:
        ax.fill_between(x, lo, hi, color=color, alpha=BAND_ALPHA, linewidth=0.0, zorder=1)
    if in_band is None:
        ax.plot(x, y, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=5.5, zorder=3, markevery=markevery)
    else:  # TFIM: filled markers inside the training band, open markers outside
        ax.plot(x, y, color=color, linestyle=ls, linewidth=lw, zorder=3)
        inside = np.array([round(v, 9) in in_band for v in x])
        ax.scatter(x[inside], y[inside], color=color, marker=marker, s=30, linewidths=0.8, zorder=4)
        ax.scatter(x[~inside], y[~inside], facecolors="white", edgecolors=color, marker=marker, s=32, linewidths=1.1, zorder=4)
    for i in range(len(x)):
        PLOTTED.append(dict(panel=panel, arm=arm, x=x[i], y=y[i], lo=lo[i] if banded else np.nan, hi=hi[i] if banded else np.nan))


def legend(fig, arms: Dict[str, int], ncol: int, y: float) -> None:
    """Shared figure legend of the arms with their seed counts."""
    handles = [Line2D([0], [0], color=ARMS[a][0], marker=ARMS[a][1], linestyle=ARMS[a][2], linewidth=2.2 if ARMS[a][4] else 1.5,
                      markersize=5.5, label=f"{LABEL.get(a, a)} (n={n})") for a, n in sorted(arms.items(), key=lambda kv: ARMS[kv[0]][3])]
    fig.legend(handles=handles, loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y), fontsize=7.5)


def n_seeds(df: pd.DataFrame, arms) -> Dict[str, int]:
    """Number of seeds per arm in ``df``."""
    return {a: int(df[df.arm == a].seed.nunique()) for a in arms}


def save(fig, out: Path, name: str) -> None:
    """Write the figure as PNG and PDF and its plotted series as CSV."""
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(PLOTTED).to_csv(out / f"{name}_data.csv", index=False)
    PLOTTED.clear()


def alpha_of(df: pd.DataFrame) -> pd.Series:
    """Per-row epistemic fraction Var_epi / (Var_epi + Var_alea) of the (already averaged) variances; 0 when both vanish."""
    tot = df.var_epi + df.var_alea
    return (df.var_epi / tot.where(tot != 0, np.nan)).fillna(0.0)


def fig_cp(res: Path, out: Path, name: str, arms: List[str]) -> None:
    """Fig 2 (main arms) / Fig 4 (all stochastic arms): coupled-pendulum OOD decomposition."""
    df = pd.read_csv(res / "cp_ood_by_mass_ratio.csv")
    df = df[df.arm.isin(arms)].assign(alpha=lambda d: alpha_of(d), std=lambda d: np.sqrt(d.var_epi.clip(lower=0)))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    train_band(axes[0], *CP_TRAIN_BAND)
    axes[1].axvline(0.0, color=GRAY, linestyle=":", linewidth=1.0, zorder=1)
    for a in arms:
        draw(axes[0], 0, a, aggregate(df[df.arm == a], "mass_ratio", "alpha"), clip01=True)
        draw(axes[1], 1, a, aggregate(df[df.arm == a], "distance", "std"))
    axes[0].set_xscale("log")
    axes[0].xaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1.0, 2.0, 5.0]))
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    style_axis(axes[0], "Epistemic share rises OOD", r"Mass ratio $r = m_2/m_1$", r"Epistemic fraction $\alpha$", "(a)")
    style_axis(axes[1], "Epistemic std. grows\nwith physics distance", "Distance from training support", "Epistemic std.", "(b)")
    two_rows = len(arms) > 4
    legend(fig, n_seeds(df, arms), min(len(arms), 4), 1.00 if two_rows else 0.99)
    fig.subplots_adjust(top=0.66 if two_rows else 0.74, bottom=0.20, left=0.09, right=0.99, wspace=0.32)
    save(fig, out, name)


def fig_tfim(res: Path, out: Path, name: str, arms: List[str]) -> None:
    """Fig 3 (main arms) / Fig 6 (all arms): TFIM epistemic variance and alpha vs transverse field."""
    df = pd.read_csv(res / "tfim_by_g.csv")
    df = df[df.arm.isin(arms)].assign(alpha=lambda d: alpha_of(d), var_pos=lambda d: d.var_epi.clip(lower=0))
    in_band = {round(g, 9) for g in df[df.in_train_band].g}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax in axes:
        train_band(ax, *TFIM_TRAIN_BAND)
    for a in arms:
        draw(axes[0], 0, a, aggregate(df[df.arm == a], "g", "var_pos"), in_band=in_band)
        draw(axes[1], 1, a, aggregate(df[df.arm == a], "g", "alpha"), clip01=True, in_band=in_band)
    style_axis(axes[0], "Epistemic variance\nunder quantum shift", "Transverse field g", "Epistemic variance", "(a)")
    style_axis(axes[1], "Epistemic share is\nmethod-dependent", "Transverse field g", r"Epistemic fraction $\alpha$", "(b)")
    legend(fig, n_seeds(df, arms), min(len(arms), 3), 1.00)
    fig.text(0.5, 0.015, "Filled markers: in training band, open markers: OOD", ha="center", va="bottom", fontsize=6.8, color=INK)
    fig.subplots_adjust(top=0.70, bottom=0.24, left=0.09, right=0.99, wspace=0.32)
    save(fig, out, name)


def fig_sc(res: Path, out: Path, name: str) -> None:
    """Fig 5: spring-chain epistemic variance and MSE vs stiffness, epistemic variance vs distance beyond the training range."""
    df = pd.read_csv(res / "sc_ood_by_k.csv").assign(var_pos=lambda d: d.var_epi.clip(lower=0))
    arms = ["Deep ensemble", "Fixed-mask MC", "Fresh-mask MC", "Concrete", "Variational"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    for ax in axes[:2]:
        train_band(ax, *SC_TRAIN_BAND)
    for a in arms:
        d = df[df.arm == a]
        draw(axes[0], 0, a, aggregate(d, "k", "var_pos"))
        draw(axes[1], 1, a, aggregate(d, "k", "mse"))
        draw(axes[2], 2, a, aggregate(d, "distance", "var_pos"))
    style_axis(axes[0], "Epistemic variance vs\nshared stiffness", "Shared stiffness k", "Epistemic variance", "(a)")
    style_axis(axes[1], "MSE vs\nshared stiffness", "Shared stiffness k", "MSE", "(b)")
    style_axis(axes[2], "Epistemic variance vs\nphysics distance", "Physics distance beyond train range", "Epistemic variance", "(c)")
    legend(fig, n_seeds(df, arms), 5, 1.02)
    fig.suptitle("Spring Chain OOD summary (median with IQR band)", fontsize=9.5, y=1.12)
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.08, right=0.99, wspace=0.42)
    save(fig, out, name)


def fig_energy(res: Path, out: Path, name: str) -> None:
    """Fig 8: spring-chain physical energy drift (relative to the first predicted frame) vs rollout step, log scale."""
    df = pd.read_csv(res / "sc_energy_drift_by_step.csv")
    arms = ["Deterministic", "Fresh-mask MC", "Fixed-mask MC"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for panel, (ax, tmax, title, tag) in enumerate([(axes[0], 100, "T=100 (in-distribution)", "(a)"),
                                                    (axes[1], None, "T=1000 (in-distribution)", "(b)")]):
        for a in arms:
            d = df[(df.arm == a) & ((df.step <= tmax) if tmax else True)]
            s = aggregate(d, "step", "drift")
            draw(ax, panel, a, s, markevery=max(1, len(s) // 10))  # ~10 markers per curve
        ax.set_yscale("log")
        style_axis(ax, title, "Rollout step t", "Physical energy drift\n(self-referenced)", tag)
    legend(fig, n_seeds(df, arms), 3, 0.99)
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.11, right=0.99, wspace=0.36)
    save(fig, out, name)


def main() -> None:
    """Render the selected figures."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "figures")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "figures")
    ap.add_argument("--only", default="2,3,4,5,6,8", help="comma-separated figure numbers")
    a = ap.parse_args()
    apply_style()
    only = {int(v) for v in a.only.split(",")}
    main_cp = ["Fixed-mask MC", "Fresh-mask MC", "Deep ensemble"]
    all_cp = main_cp + ["Concrete", "Variational"]  # drawing order; the legend is ordered separately
    all_tfim = ["Fixed-mask MC", "Fresh-mask MC", "Concrete", "Variational", "Deep ensemble"]
    jobs = {2: lambda: fig_cp(a.results, a.out, "fig2", main_cp), 4: lambda: fig_cp(a.results, a.out, "fig4", all_cp),
            3: lambda: fig_tfim(a.results, a.out, "fig3", main_cp), 6: lambda: fig_tfim(a.results, a.out, "fig6", all_tfim),
            5: lambda: fig_sc(a.results, a.out, "fig5"), 8: lambda: fig_energy(a.results, a.out, "fig8")}
    for k in sorted(only):
        jobs[k]()
        print(f"wrote {a.out / f'fig{k}'}.pdf/.png and fig{k}_data.csv")


if __name__ == "__main__":
    main()
