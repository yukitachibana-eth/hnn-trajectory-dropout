"""Turns TFIM evaluation outputs into the shipped per-seed tables.

Usage (repository root):
    python -m hnnuq.ledger.tfim --runs runs/tfim --out results

Input tree (written by ``hnnuq.eval.tfim`` / ``hnnuq.eval.tfim_bracket``)::

    <runs>/<arm>/seed<N>/eval/{drift,nested_fixed,nested_fitted,bracket}/
    <runs>/deep_ensemble/eval/nested_fitted/           (one ensemble, no seed: stored as seed 42)

with ``<arm>`` in deterministic, fresh_mask, concrete, variational, fixed_mask, fixed_mask_per_rollout.

Output (schemas of ``results/README.md``):
    ledger/tfim_per_seed.csv       system,arm,seed,quantity,split,convention,value
    figures/tfim_by_g.csv          arm,seed,g,in_train_band,var_epi,var_alea
    extra/poisson_bracket_per_seed.csv   arm,seed,mean_log_resid

Ledger quantities of one run (ID = control values inside the training band, OOD = outside; every band
value is the mean over the band's control values, which all carry the same number of rollouts):
    state_rmse, g_nll, cov90_err, var_epi, var_alea   band means of the nested stage columns
        state_rmse: preparation noise fixed at 0.10; g_nll and cov90_err: both conventions
        (``fixed`` = 0.10, ``fitted`` = epsilon fitted on validation data); var_epi, var_alea: fitted
    alpha        band-mean var_epi / (band-mean var_epi + band-mean var_alea), both conventions
    var_ratio    OOD band-mean var_epi / band-mean var_alea (fitted)
    ic_noise_scale   fitted epsilon
    norm_drift, energy_drift   drift stage
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

LEDGER_ARMS = {
    "deterministic": "Deterministic",
    "fresh_mask": "Fresh-mask MC",
    "concrete": "Concrete",
    "variational": "Variational",
    "fixed_mask": "Fixed-mask MC",
}
FIGURE_ARMS = {  # Fig. 3 / Fig. 6 series (the deterministic model has no epistemic variance)
    "fresh_mask": "Fresh-mask MC",
    "concrete": "Concrete",
    "variational": "Variational",
    "fixed_mask": "Fixed-mask MC",
}
BRACKET_ARMS = {
    "deterministic": "Deterministic",
    "fixed_mask": "Fixed-mask per-sub-step",
    "fresh_mask": "Fresh-mask MC",
    "fixed_mask_per_rollout": "Fixed-mask per-rollout",
}
DEEP_ENSEMBLE = "Deep ensemble"
DEEP_ENSEMBLE_SEED = 42  # the single ensemble is stored under the first seed slot


def _read_csv(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing evaluation output: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _seed_dirs(arm_dir: Path) -> List[Tuple[int, Path]]:
    return sorted((int(p.name[4:]), p) for p in arm_dir.glob("seed*") if p.is_dir() and p.name[4:].isdigit())


def _band_means(rows: List[Dict], columns: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """{band: {column: mean over the band's control values}} for band in ID / OOD."""
    out: Dict[str, Dict[str, float]] = {}
    for band, flag in (("ID", "True"), ("OOD", "False")):
        sel = [r for r in rows if r["in_train_band"] == flag]
        out[band] = {c: sum(float(r[c]) for r in sel) / len(sel) for c in columns}
    return out


def _alpha(var_epi: float, var_alea: float) -> float:
    total = var_epi + var_alea
    return 0.0 if total == 0 else var_epi / total


def run_ledger_rows(eval_dir: Path, arm: str, seed: int) -> List[Tuple]:
    """Ledger rows (system, arm, seed, quantity, split, convention, value) of one run."""
    fixed = _band_means(_read_csv(eval_dir / "nested_fixed" / "per_g.csv"), ["nll", "cov90_abs_error", "state_rmse", "var_epi", "var_alea"])
    fitted_rows = _read_csv(eval_dir / "nested_fitted" / "per_g.csv")
    fitted = _band_means(fitted_rows, ["nll", "cov90_abs_error", "var_epi", "var_alea"])
    drift = json.loads((eval_dir / "drift" / "drift.json").read_text(encoding="utf-8"))
    rows: List[Tuple] = []

    def add(quantity: str, split: str, convention: str, value: float) -> None:
        """Append one ledger row."""
        rows.append(("TFIM", arm, seed, quantity, split, convention, float(value)))

    for band in ("ID", "OOD"):
        add("state_rmse", band, "", fixed[band]["state_rmse"])
        for convention, d in (("fixed", fixed), ("fitted", fitted)):
            add("g_nll", band, convention, d[band]["nll"])
            add("cov90_err", band, convention, d[band]["cov90_abs_error"])
            add("alpha", band, convention, _alpha(d[band]["var_epi"], d[band]["var_alea"]))
        add("var_epi", band, "", fitted[band]["var_epi"])
        add("var_alea", band, "", fitted[band]["var_alea"])
    add("var_ratio", "OOD", "", fitted["OOD"]["var_epi"] / fitted["OOD"]["var_alea"])
    add("ic_noise_scale", "", "fitted", float(fitted_rows[0]["epsilon"]))
    add("norm_drift", "", "", drift["norm_drift"])
    add("energy_drift", "", "", drift["energy_drift"])
    return rows


def figure_rows(per_g_csv: Path, arm: str, seed: int) -> List[Tuple]:
    """(arm, seed, g, in_train_band, var_epi, var_alea) per control value of a fitted-epsilon nested stage."""
    return [(arm, seed, float(r["g"]), r["in_train_band"] == "True", float(r["var_epi"]), float(r["var_alea"]))
            for r in _read_csv(per_g_csv)]


def _fmt(v) -> str:
    return repr(v) if isinstance(v, float) else str(v)


def _write(path: Path, header: Sequence[str], rows: List[Tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([_fmt(v) for v in row])


def build(runs: Path, out: Path, outputs: Sequence[str] = ("ledger", "figures", "bracket")) -> Dict[str, int]:
    """Write the requested tables from the run tree; returns the number of rows written per table."""
    counts: Dict[str, int] = {}
    if "ledger" in outputs:
        rows: List[Tuple] = []
        for arm_dir, arm in LEDGER_ARMS.items():
            for seed, seed_dir in _seed_dirs(runs / arm_dir):
                rows += run_ledger_rows(seed_dir / "eval", arm, seed)
        rows.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4], r[5]))
        _write(out / "ledger" / "tfim_per_seed.csv", ["system", "arm", "seed", "quantity", "split", "convention", "value"], rows)
        counts["ledger"] = len(rows)
    if "figures" in outputs:
        rows = []
        for arm_dir, arm in FIGURE_ARMS.items():
            for seed, seed_dir in _seed_dirs(runs / arm_dir):
                rows += figure_rows(seed_dir / "eval" / "nested_fitted" / "per_g.csv", arm, seed)
        de = runs / "deep_ensemble" / "eval" / "nested_fitted" / "per_g.csv"
        if de.exists():
            rows += figure_rows(de, DEEP_ENSEMBLE, DEEP_ENSEMBLE_SEED)
        rows.sort(key=lambda r: (r[0], r[1], r[2]))
        _write(out / "figures" / "tfim_by_g.csv", ["arm", "seed", "g", "in_train_band", "var_epi", "var_alea"], rows)
        counts["figures"] = len(rows)
    if "bracket" in outputs:
        rows = []
        for arm_dir, arm in BRACKET_ARMS.items():
            for seed, seed_dir in _seed_dirs(runs / arm_dir):
                summary = json.loads((seed_dir / "eval" / "bracket" / "summary.json").read_text(encoding="utf-8"))
                rows.append((arm, seed, float(summary["mean_log_resid"])))
        _write(out / "extra" / "poisson_bracket_per_seed.csv", ["arm", "seed", "mean_log_resid"], rows)
        counts["bracket"] = len(rows)
    return counts


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write the TFIM ledger, figure and bracket tables from the stage files of a run tree."""
    parser = argparse.ArgumentParser(description="TFIM ledger writer")
    parser.add_argument("--runs", type=Path, default=Path("runs/tfim"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--outputs", default="ledger,figures,bracket", help="comma-separated subset of ledger,figures,bracket")
    args = parser.parse_args(argv)
    print(build(args.runs, args.out, tuple(args.outputs.split(","))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
