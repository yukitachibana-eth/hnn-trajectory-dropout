"""Shared helpers of the pipeline drivers: system / arm tables, run-tree paths and a subprocess pool.

Run tree (repository root = working directory of every job):

    runs/<system>/<arm>/seed<N>/<system_type>_<dataset>_<run_index>/{config.yaml, best_model.pt, ...}   training output
    runs/<system>/<arm>/seed<N>/eval/...                                                               TFIM stage outputs
    runs/<system>/deep_ensemble/member<i>/<...>/{config.yaml, best_model.pt}                            members, i = 0..4
"""

import glob
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
SYSTEMS = ("cp", "sc", "tfim")
ARMS = ("deterministic", "fresh_mask", "concrete", "variational", "fixed_mask")
TFIM_EXTRA_ARMS = ("fixed_mask_per_rollout",)  # per-rollout mask bank, used by the Poisson-bracket table
N_MEMBERS = 5
DEFAULT_SEEDS = tuple(range(42, 52))


@dataclass
class Job:
    """One subprocess: argv (run from the repository root), the GPU-id environment, and a done-check."""
    name: str
    cmd: List[str]
    done: Optional[str] = None  # absolute path or glob pattern; a match means the job is skipped
    log: Optional[Path] = None
    extra_env: Dict[str, str] = field(default_factory=dict)

    def is_done(self) -> bool:
        """True if ``done`` (an absolute path or glob pattern) matches an existing file."""
        return self.done is not None and bool(glob.glob(str(self.done)))


def glob_first(pattern: str) -> Optional[Path]:
    """First path matching the absolute glob ``pattern`` (sorted), or None."""
    hits = sorted(glob.glob(pattern))
    return Path(hits[0]) if hits else None


def log_path(runs: Path, name: str) -> Path:
    """Log file of a job under ``<runs>/logs/`` (outside the system / arm / seed tree)."""
    return Path(runs) / "logs" / (re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") + ".log")


def parse_gpus(spec: Optional[str]) -> List[Optional[str]]:
    """``"0,1"`` -> ["0", "1"]; None / empty -> [None] (CPU)."""
    ids = [g.strip() for g in (spec or "").split(",") if g.strip()]
    return ids or [None]


def parse_seeds(spec: str) -> List[int]:
    """``"42-44,50"`` -> [42, 43, 44, 50]."""
    out: List[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += range(int(lo), int(hi) + 1)
        elif part.strip():
            out.append(int(part))
    return out


def shell(cmd: Sequence[str]) -> str:
    """argv as a printable shell line."""
    import shlex
    return " ".join(shlex.quote(c) for c in cmd)


def run_pool(jobs: Sequence[Job], gpus: Sequence[Optional[str]], n_parallel: Optional[int] = None, dry_run: bool = False) -> int:
    """Run ``jobs`` as subprocesses, at most one per GPU (``n_parallel`` processes on CPU), skipping finished ones.

    Jobs are assigned round-robin to free GPU slots; a GPU slot sets ``CUDA_VISIBLE_DEVICES``. Returns the number of failed jobs.
    """
    pending = []
    for job in jobs:
        if job.is_done():
            print(f"[skip] {job.name}")
        else:
            pending.append(job)
    if dry_run:
        for job in pending:
            print(shell(job.cmd))
        print(f"{len(pending)} job(s) pending, {len(jobs) - len(pending)} already done")
        return 0
    slots: List[Optional[str]] = list(gpus) if gpus != [None] else [None] * (n_parallel or 1)
    running: Dict[int, tuple] = {}
    failed: List[str] = []
    queue = list(pending)
    t0 = time.time()
    while queue or running:
        for i in range(len(slots)):
            if i not in running and queue:
                job = queue.pop(0)
                env = dict(os.environ, **job.extra_env)
                if slots[i] is not None:
                    env["CUDA_VISIBLE_DEVICES"] = slots[i]
                if slots[i] is None:
                    env.setdefault("OMP_NUM_THREADS", "1")
                    env.setdefault("MKL_NUM_THREADS", "1")
                logf = None
                if job.log is not None:
                    job.log.parent.mkdir(parents=True, exist_ok=True)
                    logf = open(job.log, "w")
                print(f"[start {time.time() - t0:7.0f}s] {job.name}", flush=True)
                proc = subprocess.Popen(job.cmd, env=env, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT if logf else None)
                running[i] = (job, proc, logf, time.time())
        for i, (job, proc, logf, ts) in list(running.items()):
            ret = proc.poll()
            if ret is not None:
                if logf:
                    logf.close()
                status = "ok" if ret == 0 else f"FAILED ({ret}; see {job.log})"
                print(f"[done  {time.time() - t0:7.0f}s] {job.name}: {status} ({time.time() - ts:.0f}s)", flush=True)
                if ret != 0:
                    failed.append(job.name)
                del running[i]
        time.sleep(0.2)
    return len(failed)
