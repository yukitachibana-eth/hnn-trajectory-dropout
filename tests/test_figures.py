"""Checks for scripts/50_build_figures.py.

(a) the figure script runs on results/figures/ and writes PDF, PNG and a data csv for Figs 2-6 and 8;
(b) the plotted series (arm, x, y, band) equal the series of the reference figure build stored in tests/figure_ref/plotted_series.csv
    (relative tolerance 1e-9);
(c) the alpha values behind Figs 2-6, averaged to the ID / OOD level of the results ledger (mean Var_epi and mean Var_alea over the x values
    of a split, then the ratio), equal the ledger's per-seed alpha.
Run `pytest tests/test_figures.py`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REF = pd.read_csv(Path(__file__).resolve().parent / "figure_ref" / "plotted_series.csv")
RES = ROOT / "results"


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("figures")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "50_build_figures.py"), "--out", str(out)], check=True)
    return out


def test_outputs_written(built: Path) -> None:
    for n in (2, 3, 4, 5, 6, 8):
        for ext in ("pdf", "png"):
            assert (built / f"fig{n}.{ext}").stat().st_size > 10_000
        assert (built / f"fig{n}_data.csv").exists()
    assert (ROOT / "assets" / "fig1_trajectory_consistent_dropout.pdf").exists()


@pytest.mark.parametrize("fig", ["fig2", "fig3", "fig4", "fig5", "fig6", "fig8"])
def test_plotted_series_match_reference(built: Path, fig: str) -> None:
    got = pd.read_csv(built / f"{fig}_data.csv")
    ref = REF[REF.figure == fig].drop(columns="figure")
    key = ["panel", "arm", "x"]
    assert len(got) == len(ref)
    got = got.sort_values(key).reset_index(drop=True)
    ref = ref.sort_values(key).reset_index(drop=True)
    assert (got[["panel", "arm"]].values == ref[["panel", "arm"]].values).all()
    for col in ("x", "y", "lo", "hi"):
        np.testing.assert_allclose(got[col].to_numpy(float), ref[col].to_numpy(float), rtol=1e-9, atol=1e-300, equal_nan=True, err_msg=f"{fig} {col}")


def _ledger_alpha(system: str) -> pd.DataFrame:
    led = pd.read_csv(RES / "ledger" / ("tfim_per_seed.csv" if system == "TFIM" else "classical_per_seed.csv"))
    led = led[(led.system == system) & (led.quantity == "alpha") & (led.convention == "fitted")]
    return led.set_index(["arm", "seed", "split"]).value


@pytest.mark.parametrize("system", ["CP", "TFIM"])
def test_alpha_matches_ledger(system: str) -> None:
    if system == "CP":
        df = pd.read_csv(RES / "figures" / "cp_ood_by_mass_ratio.csv")
        df["split"] = np.where(df.split == "in_dist", "ID", "OOD")
    else:
        df = pd.read_csv(RES / "figures" / "tfim_by_g.csv")
        df["split"] = np.where(df.in_train_band, "ID", "OOD")
    m = df.groupby(["arm", "seed", "split"])[["var_epi", "var_alea"]].mean()
    alpha = m.var_epi / (m.var_epi + m.var_alea)
    led = _ledger_alpha(system)
    common = alpha.index.intersection(led.index)
    assert len(common) >= 40
    # the Deep ensemble is one 5-member ensemble whose ledger value comes from a separate evaluation pass (agrees to ~3e-5)
    tol = np.where(common.get_level_values("arm") == "Deep ensemble", 1e-4, 1e-6)
    np.testing.assert_array_less(np.abs(alpha[common].to_numpy() - led[common].to_numpy()), tol * np.abs(led[common].to_numpy()) + 1e-300)
