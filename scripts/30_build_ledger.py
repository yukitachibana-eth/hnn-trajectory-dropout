#!/usr/bin/env python
"""Collect the evaluation outputs of a run tree into the shipped results/ schema (see results/README.md).

    python scripts/30_build_ledger.py [--runs runs] [--out results] [--smoke]

Writes <out>/ledger/{classical,tfim}_per_seed.csv, <out>/figures/*.csv and <out>/extra/*.csv. The default ``--out results`` overwrites
the shipped CSVs; pass another directory (e.g. for a partial or smoke run) to keep them.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from _jobs import ROOT

sys.path.insert(0, str(ROOT))

from hnnuq.ledger import classical, tfim  # noqa: E402

APPENDIX_STAGES = ("long_horizon", "ic_noise_sweep", "mask_bank", "hidden_param", "efficiency")


def main(argv=None) -> int:
    """Write the ledger, figure and appendix tables of a run tree."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--smoke", action="store_true", help="pass --smoke to the appendix collectors")
    args = ap.parse_args(argv)
    runs, out = args.runs.resolve(), args.out.resolve()
    classical.main(["--runs", str(runs), "--out", str(out)])
    tfim.main(["--runs", str(runs / "tfim"), "--out", str(out)])
    status = 0
    for stage in APPENDIX_STAGES:
        cmd = [sys.executable, "-m", f"hnnuq.eval.{stage}", "collect", "--runs", str(runs), "--out", str(out)]
        status |= subprocess.run(cmd + (["--smoke"] if args.smoke else []), cwd=ROOT).returncode != 0
    print(f"results written to {out}")
    return int(status)


if __name__ == "__main__":
    sys.exit(main())
