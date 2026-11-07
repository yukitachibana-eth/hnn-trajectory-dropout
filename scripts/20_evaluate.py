#!/usr/bin/env python
"""Evaluate every trained run and the appendix stages.

    python scripts/20_evaluate.py [--systems cp,sc,tfim] [--seeds 42-51] [--gpus 0,1] [--runs runs] [--data data] [--smoke]

Core stages (one subprocess per run and stage group; outputs next to the run, skipped when they already exist):
    cp / sc    ``python -m hnnuq.eval.cp|sc``: ic_fitted, ic_fixed, ood, hdrift   (deep ensemble: once, from its five members)
    tfim       ``python -m hnnuq.eval.tfim drift|nested`` (fixed and fitted preparation noise) and
               ``python -m hnnuq.eval.tfim_bracket`` (Poisson-bracket arms)
Appendix stages (``python -m hnnuq.eval.<stage> run``) follow the contract documented in each stage module.
"""

import argparse
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import List, Optional

from _jobs import ARMS, DEFAULT_SEEDS, N_MEMBERS, ROOT, SYSTEMS, TFIM_EXTRA_ARMS, Job, glob_first, log_path, parse_gpus, parse_seeds, run_pool

sys.path.insert(0, str(ROOT))

TFIM_BRACKET_ARMS = ("deterministic", "fresh_mask", "fixed_mask", "fixed_mask_per_rollout")
# Appendix stages with one ``run`` job per (arm, seed): stage -> (arms, seeds); the seed ranges follow the paper's tables.
APPENDIX = {
    "long_horizon": (("deterministic", "fresh_mask", "fixed_mask"), "42-51"),
    "ic_noise_sweep": (("fresh_mask", "fixed_mask"), "42-44"),
    "hidden_param": (("fresh_mask", "fixed_mask"), "42-51"),
}


def run_dir(runs: Path, system: str, arm: str, seed: int) -> Optional[Path]:
    """Training run directory of (system, arm, seed), or None if it has not been trained."""
    ckpt = glob_first(str(runs / system / arm / f"seed{seed}" / "*" / "best_model.pt"))
    return ckpt.parent if ckpt else None


def member_dirs(runs: Path, system: str) -> Optional[List[Path]]:
    """The five deep-ensemble member run directories, or None if any is missing."""
    found = [glob_first(str(runs / system / "deep_ensemble" / f"member{i}" / "*" / "best_model.pt")) for i in range(N_MEMBERS)]
    return [c.parent for c in found] if all(found) else None


def device_args(args) -> List[str]:
    """``--device`` option: CPU for smoke runs / no GPUs, else the (single, visible) CUDA device."""
    return ["--device", "cpu" if (args.smoke or parse_gpus(args.gpus) == [None]) else "cuda"]


def py(*parts: str) -> List[str]:
    """argv of ``python -m <parts>`` with the current interpreter."""
    return [sys.executable, "-m", *parts]


def classical_jobs(system: str, args, seeds: List[int]) -> List[Job]:
    """CP / SC evaluation jobs: one per trained (arm, seed) plus one for the deep ensemble."""
    jobs = []
    smoke = ["--smoke"] if args.smoke else []
    for arm in ARMS:
        for seed in seeds:
            rd = run_dir(args.runs, system, arm, seed)
            if rd is None:
                print(f"[skip] {system}/{arm}/seed{seed}: not trained")
                continue
            jobs.append(Job(f"eval {system}/{arm}/seed{seed}",
                            py(f"hnnuq.eval.{system}", "--runs", str(args.runs), "--arm", arm, "--seed", str(seed), *device_args(args), *smoke),
                            done=str(rd / "eval" / "hdrift.json")))
    if member_dirs(args.runs, system):
        jobs.append(Job(f"eval {system}/deep_ensemble",
                        py(f"hnnuq.eval.{system}", "--runs", str(args.runs), "--arm", "deep_ensemble", "--seed", "42", *device_args(args), *smoke),
                        done=str(args.runs / system / "deep_ensemble" / "eval" / "hdrift.json")))
    else:
        print(f"[skip] {system}/deep_ensemble: members not trained")
    return jobs


def tfim_jobs(args, seeds: List[int]) -> List[Job]:
    """TFIM evaluation jobs: drift and both nested-ensemble conventions per arm and seed, bracket jobs, and the deep ensemble."""
    data = import_module("00_generate_data").tfim_dataset_dir(args.data)
    smoke = ["--smoke"] if args.smoke else []
    dev = device_args(args)
    jobs = []
    for arm in ARMS + TFIM_EXTRA_ARMS:
        for seed in seeds:
            rd = run_dir(args.runs, "tfim", arm, seed)
            if rd is None:
                print(f"[skip] tfim/{arm}/seed{seed}: not trained")
                continue
            ev = args.runs / "tfim" / arm / f"seed{seed}" / "eval"
            base = ["--run-dir", str(rd), "--data", str(data), *dev, *smoke]
            if arm in ARMS:
                jobs.append(Job(f"eval tfim/{arm}/seed{seed} drift", py("hnnuq.eval.tfim", "drift", *base, "--out", str(ev / "drift")),
                                done=str(ev / "drift" / "drift.json")))
                for eps in ("fixed", "fitted"):
                    jobs.append(Job(f"eval tfim/{arm}/seed{seed} nested_{eps}",
                                    py("hnnuq.eval.tfim", "nested", *base, "--epsilon", eps, "--out", str(ev / f"nested_{eps}")),
                                    done=str(ev / f"nested_{eps}" / "summary.json")))
            if arm in TFIM_BRACKET_ARMS:
                jobs.append(Job(f"eval tfim/{arm}/seed{seed} bracket", py("hnnuq.eval.tfim_bracket", *base, "--out", str(ev / "bracket")),
                                done=str(ev / "bracket" / "summary.json")))
    members = member_dirs(args.runs, "tfim")
    if members:
        out = args.runs / "tfim" / "deep_ensemble" / "eval" / "nested_fitted"
        jobs.append(Job("eval tfim/deep_ensemble nested_fitted",
                        py("hnnuq.eval.tfim", "nested", "--run-dir", *map(str, members), "--data", str(data), *dev, *smoke,
                           "--epsilon", "fitted", "--out", str(out)),
                        done=str(out / "summary.json")))
    else:
        print("[skip] tfim/deep_ensemble: members not trained")
    return jobs


def appendix_job(stage: str, arm: str, seed: int, args) -> Job:
    """``run`` job of an appendix stage for one (arm, seed)."""
    smoke = ["--smoke"] if args.smoke else []
    return Job(f"eval {stage} {arm}/seed{seed}",
               py(f"hnnuq.eval.{stage}", "run", "--runs", str(args.runs), "--arm", arm, "--seed", str(seed), *device_args(args), *smoke),
               done=str(args.runs / "*" / arm / f"seed{seed}" / "*" / "eval" / f"{stage}*"))


def appendix_jobs(args):
    """(parallel jobs, jobs that must run alone). The efficiency benchmark measures wall-clock time, so nothing else may run next to it."""
    jobs, solo = [], []
    for stage, (arms, seed_spec) in APPENDIX.items():
        seeds = [DEFAULT_SEEDS[0]] if args.smoke else parse_seeds(seed_spec)
        jobs += [appendix_job(stage, arm, seed, args) for arm in arms for seed in seeds]
    # arms and seeds of the mask-bank-size grid come from the stage itself
    out = subprocess.run(py("hnnuq.eval.mask_bank", "arms"), cwd=ROOT, capture_output=True, text=True, check=True).stdout
    for line in out.split("\n"):
        if line.strip():
            arm, seed = line.split()
            if not args.smoke or int(seed) == DEFAULT_SEEDS[0]:
                jobs.append(appendix_job("mask_bank", arm, int(seed), args))
    smoke = ["--smoke"] if args.smoke else []
    solo.append(Job("eval efficiency", py("hnnuq.eval.efficiency", "run", "--runs", str(args.runs), "--data", str(args.data),
                                          *device_args(args), *smoke),
                    done=str(args.runs / "cp" / "efficiency" / "*.csv")))
    return jobs, solo


def main(argv=None) -> int:
    """Build the evaluation job list from the command line and run it."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--seeds", default=None, help="e.g. 42-51 (default; one seed with --smoke)")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--data", type=Path, default=ROOT / "data", help="data root written by 00_generate_data.py")
    ap.add_argument("--gpus", default=None, help="comma-separated GPU ids, one job per GPU (default: CPU)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes when running on CPU")
    ap.add_argument("--no-appendix", action="store_true", help="skip the appendix stages")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    args.runs, args.data = args.runs.resolve(), args.data.resolve()
    seeds = parse_seeds(args.seeds) if args.seeds else ([DEFAULT_SEEDS[0]] if args.smoke else list(DEFAULT_SEEDS))
    jobs: List[Job] = []
    for system in args.systems.split(","):
        jobs += tfim_jobs(args, seeds) if system == "tfim" else classical_jobs(system, args, seeds)
    solo: List[Job] = []
    if not args.no_appendix:
        extra, solo = appendix_jobs(args)
        jobs += extra
    gpus = parse_gpus(args.gpus)
    for job in jobs + solo:
        job.log = log_path(args.runs, job.name)
    failed = run_pool(jobs, gpus, args.jobs, args.dry_run)
    failed += run_pool(solo, gpus[:1], 1, args.dry_run)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
