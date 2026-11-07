#!/usr/bin/env python
"""Train every model of the paper: system x arm x seed, the deep-ensemble members and the extra training jobs of the appendix stages.

    python scripts/10_train.py [--systems cp,sc,tfim] [--arms ...] [--seeds 42-51] [--gpus 0,1] [--runs runs] [--dry-run] [--smoke]

Each job is ``python -m hnnuq.train --config configs/<system>/<arm>.yaml --seed N --output_dir runs/<system>/<arm>/seed<N>``;
the deep ensemble is trained as five members (seeds 0-4) under runs/<system>/deep_ensemble/member<i>/. The mask-bank-size stage
(``python -m hnnuq.eval.mask_bank plan``) contributes its extra Fixed-mask runs. Jobs are spread one per GPU over ``--gpus``
(CPU if omitted); a run whose best_model.pt exists is skipped, so an interrupted sweep can be resumed. ``--smoke`` shrinks every
run to a couple of epochs on tiny datasets (one seed, CPU).
"""

import argparse
from importlib import import_module
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

from _jobs import ARMS, DEFAULT_SEEDS, N_MEMBERS, ROOT, SYSTEMS, TFIM_EXTRA_ARMS, Job, log_path, parse_gpus, parse_seeds, run_pool

sys.path.insert(0, str(ROOT))

# Overrides that shrink a run to seconds on the tiny datasets written by `00_generate_data.py --smoke`.
SMOKE_OVERRIDES = {
    "cp": ["--n_epochs", "2", "--n_samples", "16", "--n_val_samples", "4", "--k_samples", "2", "--k_val_samples", "2",
           "--k_test_samples", "2", "--n_test_samples", "8", "--T", "5", "--T_val", "9", "--T_test", "9", "--batch_size", "4",
           "--n_hidden", "16", "--pred_samples", "3", "--mask_num", "6", "--test_freq", "1"],
    "sc": ["--n_epochs", "2", "--n_samples", "150", "--n_val_samples", "50", "--k_samples", "2", "--k_val_samples", "2",
           "--k_test_samples", "1", "--n_test_samples", "20", "--T", "5", "--T_val", "14", "--T_test", "14", "--batch_size", "32",
           "--n_hidden", "16", "--pred_samples", "3", "--mask_num", "6", "--test_freq", "1"],
    "tfim": ["--n_epochs", "2", "--n_samples", "4", "--n_val_samples", "2", "--n_test_samples", "4", "--k_samples", "2",
             "--k_val_samples", "1", "--k_test_samples", "4", "--T", "5", "--T_val", "10", "--T_test", "10", "--batch_size", "8",
             "--n_hidden", "16", "--pred_samples", "3", "--mask_num", "6", "--test_freq", "1",
             "--gradual_horizon_increase", "false", "--best_checkpoint_requires_full_horizon", "false"],
}


def data_dir_for(system: str, data_root: Path) -> Path:
    """Value of ``--data_dir`` for ``system`` under the data root (TFIM: the dataset directory itself)."""
    if system == "tfim":
        gen = import_module("00_generate_data")
        return gen.tfim_dataset_dir(data_root)
    return data_root


def train_job(system: str, config: str, seed: int, out: Path, run_index: str, args) -> Job:
    """Training job of one run: ``out`` is the output directory that receives the run directory."""
    cmd = [sys.executable, "-m", "hnnuq.train", "--config", str(ROOT / "configs" / system / f"{config}.yaml"),
           "--seed", str(seed), "--run_index", run_index, "--output_dir", str(out),
           "--data_dir", str(data_dir_for(system, args.data))]
    if args.smoke:
        cmd += SMOKE_OVERRIDES[system]
    if args.smoke or parse_gpus(args.gpus) == [None]:
        cmd += ["--device", "cpu"]
    name = f"train {system}/{out.relative_to(args.runs / system)}"
    return Job(name, cmd, done=str(out / "*" / "best_model.pt"), log=log_path(args.runs, name))


def plan_jobs(args, seeds: List[int]) -> List[Job]:
    """Extra training jobs of the mask-bank-size stage: one ``hnnuq.train`` command per line of ``hnnuq.eval.mask_bank plan``."""
    proc = subprocess.run([sys.executable, "-m", "hnnuq.eval.mask_bank", "plan"], cwd=ROOT, capture_output=True, text=True, check=True)
    jobs = []
    for line in proc.stdout.splitlines():
        argv = shlex.split(line)
        if "hnnuq.train" not in argv:
            continue
        argv = [sys.executable if a in ("python", "python3") else a for a in argv]
        opts = {a: argv[i + 1] for i, a in enumerate(argv[:-1]) if a.startswith("--")}
        system = Path(opts["--config"]).parent.name
        out = Path(opts.get("--output_dir", f"runs/{system}"))
        out = args.runs / out.relative_to("runs") if out.parts[0] == "runs" else out.resolve()
        opts["--config"] = str(ROOT / opts["--config"])
        opts["--output_dir"] = str(out)
        opts.setdefault("--data_dir", str(data_dir_for(system, args.data)))
        if args.smoke:  # one seed; the job's own options (mask-bank size) win over the smoke sizes
            if int(opts["--seed"]) != seeds[0]:
                continue
            smoke = SMOKE_OVERRIDES[system]
            opts.update({k: v for k, v in zip(smoke[::2], smoke[1::2]) if k not in opts and k != "--mask_num"})
        if args.smoke or parse_gpus(args.gpus) == [None]:
            opts["--device"] = "cpu"
        cmd = [sys.executable, "-m", "hnnuq.train"] + [t for kv in opts.items() for t in kv]
        name = f"train {system}/{out.relative_to(args.runs / system)}"
        jobs.append(Job(name, cmd, done=str(out / "*" / "best_model.pt"), log=log_path(args.runs, name)))
    return jobs


def main(argv=None) -> int:
    """Build the job list from the command line and run it."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--arms", default=",".join(ARMS + ("deep_ensemble",)), help="arms (deep_ensemble = five members)")
    ap.add_argument("--seeds", default=None, help="e.g. 42-51 (default; one seed with --smoke)")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--data", type=Path, default=ROOT / "data", help="data root written by 00_generate_data.py")
    ap.add_argument("--gpus", default=None, help="comma-separated GPU ids, one job per GPU (default: CPU)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes when running on CPU")
    ap.add_argument("--no-appendix", action="store_true", help="skip the extra training jobs of the mask-bank-size stage")
    ap.add_argument("--dry-run", action="store_true", help="print the pending commands only")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    args.runs, args.data = args.runs.resolve(), args.data.resolve()
    seeds = parse_seeds(args.seeds) if args.seeds else ([DEFAULT_SEEDS[0]] if args.smoke else list(DEFAULT_SEEDS))
    jobs: List[Job] = []
    for system in args.systems.split(","):
        for arm in args.arms.split(","):
            if arm == "deep_ensemble":
                jobs += [train_job(system, "deep_ensemble", i, args.runs / system / "deep_ensemble" / f"member{i}",
                                   f"deep_ensemble_seed{i}", args) for i in range(N_MEMBERS)]
                continue
            for seed in seeds:
                jobs.append(train_job(system, arm, seed, args.runs / system / arm / f"seed{seed}", "run", args))
        if system == "tfim" and "fixed_mask" in args.arms.split(","):
            for arm in TFIM_EXTRA_ARMS:
                jobs += [train_job(system, arm, s, args.runs / system / arm / f"seed{s}", "run", args) for s in seeds]
    if not args.no_appendix:
        jobs += plan_jobs(args, seeds)
    failed = run_pool(jobs, parse_gpus(args.gpus), args.jobs, args.dry_run)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
