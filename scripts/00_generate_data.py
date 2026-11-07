#!/usr/bin/env python
"""Generate the coupled-pendulum (cp), spring-chain (sc) and TFIM (tfim) datasets.

    python scripts/00_generate_data.py [--systems cp,sc,tfim] [--out data] [--smoke]

Files go to <out>/coupled_pendulum_mr/, <out>/chain/ and <out>/tfim/<dataset name>/ (the dataset name is the
``dataset_index`` of configs/tfim/*.yaml). ``--smoke`` writes tiny datasets for the end-to-end smoke run.
"""

import argparse
import sys
from pathlib import Path

from _jobs import ROOT, SYSTEMS

sys.path.insert(0, str(ROOT))

from hnnuq.config import ExperimentConfig  # noqa: E402
from hnnuq.data import coupled_pendulum, spring_chain, tfim  # noqa: E402


def tfim_dataset_dir(out: Path) -> Path:
    """Directory of the TFIM dataset under the data root ``out`` (named after the configs' dataset_index)."""
    return out / "tfim" / ExperimentConfig.load(ROOT / "configs" / "tfim" / "fixed_mask.yaml").dataset_index


def main(argv=None) -> int:
    """Generate the datasets of the selected systems."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--systems", default=",".join(SYSTEMS), help="comma-separated subset of cp,sc,tfim")
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    flags = ["--smoke"] if args.smoke else []
    for system in args.systems.split(","):
        if system == "cp":
            coupled_pendulum.main(["--out", str(args.out), *flags])
        elif system == "sc":
            spring_chain.main(["--out", str(args.out), *flags])
        elif system == "tfim":
            tfim.main(["--out", str(tfim_dataset_dir(args.out)), *flags])
        else:
            ap.error(f"unknown system {system!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
