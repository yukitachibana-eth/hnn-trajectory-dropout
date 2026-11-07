"""Computational-cost benchmark of the coupled-pendulum arms (Table 14): relative train-epoch time, inference time and storage.

Each arm is trained for real with the small configurations ``configs/cp/efficiency/<arm>.yaml`` (2 control cells, 64 training
ICs, batch 32, 500 hidden units; the Fixed-mask MC bank is seeded, so a checkpoint stores the seed instead of the masks) and
timed in this protocol: 1 warm-up epoch, then the mean wall time of 10 further epochs; one warm-up rollout, then the mean wall
time of 10 rollouts of 16 test ICs over 60 steps; storage is the byte size of the saved state dict. The rollouts use one
batched call of ``integrate_mc_samples``: Fresh-mask MC, Concrete and Variational draw 10 samples (weight-space groups), Fixed-mask
MC rolls out all M bank masks (M = 100 and M = 200, trajectory-consistent) and the Deterministic arm one trajectory. The
Deep ensemble row is not measured: it is five times one Deterministic member (5 models, 5 samples, 5x time and storage).
Numbers are relative to the Deterministic arm.

Hardware dependence. Models, samples per arm, relative storage and the decomposition-axes column are exact (they follow from the
configurations); the train and inference time ratios are wall-clock measurements, reproducible only up to hardware, library
version and system load (same order of magnitude and ordering of the arms, not the same digits).

    python -m hnnuq.eval.efficiency run --runs runs [--data data] [--arm fixed_mask] [--device cuda:0] [--smoke]
    python -m hnnuq.eval.efficiency collect --runs runs --out results [--smoke]

``run`` (all arms unless ``--arm``) writes ``<runs>/cp/efficiency/<arm>.csv`` with the raw means; ``collect`` normalises them and
writes ``results/extra/efficiency_summary.csv``. Requires the coupled-pendulum dataset under ``--data``.
"""

import argparse
import csv
import gc
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch

from hnnuq.config import ExperimentConfig
from hnnuq.integration import integrate_mc_samples, numerically_integrate
from hnnuq.train import load_data, set_seed
from hnnuq.trainer import Trainer

SYSTEM = "cp"
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / SYSTEM / "efficiency"
# (arm, bank size M of the Fixed-mask rows or None), in table order
CASES: Tuple[Tuple[str, Optional[int]], ...] = (
    ("deterministic", None), ("fresh_mask", None), ("fixed_mask", 100), ("fixed_mask", 200), ("concrete", None), ("variational", None),
)
ARM_LABELS = {"deterministic": "Deterministic", "fresh_mask": "Fresh-mask MC", "fixed_mask": "Fixed-mask MC", "concrete": "Concrete",
              "variational": "Variational"}
DEEP_ENSEMBLE_MEMBERS = 5
MEASURE_COLUMNS = ["arm", "mask_bank", "models", "samples", "train_seconds", "infer_seconds", "checkpoint_bytes"]
SUMMARY_COLUMNS = ["arm", "mask_bank", "models", "samples", "rel_train_time", "rel_infer_time", "rel_storage", "decomposition_axes"]


@dataclass(frozen=True)
class Protocol:
    """Timing protocol; the defaults are the paper's, ``smoke()`` a tiny variant."""

    warmup_epochs: int = 1
    train_repeats: int = 10
    inference_repeats: int = 10
    inference_ics: int = 16
    horizon: int = 60
    pred_samples: int = 10

    @staticmethod
    def smoke() -> "Protocol":
        """Tiny protocol for the smoke run."""
        return Protocol(train_repeats=1, inference_repeats=1, inference_ics=4, horizon=5, pred_samples=4)


def _timed(device: torch.device, fn: Callable[[], object]) -> float:
    """Wall time of one call in seconds, with the device synchronised before and after."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - t0


def _smoke_config(config: ExperimentConfig) -> ExperimentConfig:
    """Shrink the data and the network so that a smoke epoch takes well under a second."""
    config.n_samples, config.n_val_samples, config.n_test_samples = 8, 4, 8
    config.batch_size, config.n_hidden, config.pred_samples = 4, 16, 4
    config.mask_num = min(config.mask_num, 12)
    return config


def measure(arm: str, bank: Optional[int], device: torch.device, data_dir: Path, pr: Protocol, smoke: bool) -> Dict[str, object]:
    """Train and time one case; returns the raw row (``MEASURE_COLUMNS``)."""
    config = ExperimentConfig.load(CONFIG_DIR / f"{arm}.yaml")
    config.data_dir = str(data_dir)
    if bank is not None:
        config.mask_num = bank
    if smoke:
        _smoke_config(config)
    set_seed(config.seed)
    train_data, _, test_data, k_train, k_test = load_data(config)
    train_data, test_data = (x.to(device=device, dtype=torch.float32) for x in (train_data, test_data))
    k_train, k_test = k_train.to(device, dtype=torch.float32), k_test.to(device, dtype=torch.float32)
    trainer = Trainer(config, device)
    trainer.build(dim=train_data.shape[-1] // 2)

    for epoch in range(pr.warmup_epochs):
        trainer.train_epoch(train_data, k_train, epoch=epoch)
    train_s = [_timed(device, lambda e=pr.warmup_epochs + i: trainer.train_epoch(train_data, k_train, epoch=e)) for i in range(pr.train_repeats)]

    dim = test_data.shape[-1] // 2
    p_0, q_0 = test_data[0, 0, : pr.inference_ics, :dim], test_data[0, 0, : pr.inference_ics, dim:]
    horizon, k_matrix = min(pr.horizon, test_data.shape[1]), k_test[0]
    if arm == "deterministic":
        samples = 1
        rollout = lambda: numerically_integrate(p_0=p_0, q_0=q_0, model=trainer.model, T=horizon, dt=config.dt, k_matrix=k_matrix,
                                                mask_index=0, use_leapfrog=config.use_leapfrog)
    else:
        samples = config.mask_num if arm == "fixed_mask" else config.pred_samples  # Fixed-mask MC: the whole bank
        rollout = lambda: integrate_mc_samples(
            p_0=p_0, q_0=q_0, model=trainer.model, T=horizon, dt=config.dt, n_mc_samples=samples, k_matrix=k_matrix,
            trajectory_consistent=config.finite_mask_mode == "both", use_leapfrog=config.use_leapfrog, fresh_mask_semantics="weight_group",
        )
    rollout()  # warm-up
    infer_s = [_timed(device, rollout) for _ in range(pr.inference_repeats)]

    buf = io.BytesIO()
    torch.save(trainer.model.state_dict(), buf)
    return dict(arm=arm, mask_bank=bank or 0, models=1, samples=samples, train_seconds=sum(train_s) / len(train_s),
                infer_seconds=sum(infer_s) / len(infer_s), checkpoint_bytes=buf.tell())


def run(runs_root: Path, arms: List[str], device: torch.device, data_dir: Path, smoke: bool = False) -> Path:
    """Measure the requested arms and write ``<runs>/cp/efficiency/<arm>.csv`` (one row per case of the arm)."""
    pr = Protocol.smoke() if smoke else Protocol()
    out = Path(runs_root) / SYSTEM / "efficiency"
    out.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        rows = [measure(a, bank, device, data_dir, pr, smoke) for a, bank in CASES if a == arm]
        with open(out / f"{arm}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MEASURE_COLUMNS)
            w.writeheader()
            w.writerows(rows)
    return out


def collect(runs_root: Path, out_root: Path) -> Path:
    """Normalise the raw means by the Deterministic arm, append the five-member Deep ensemble row, write the summary table."""
    d = Path(runs_root) / SYSTEM / "efficiency"
    raw = pd.concat([pd.read_csv(d / f"{arm}.csv") for arm in ARM_LABELS], ignore_index=True)
    order = {c: i for i, c in enumerate(CASES)}
    raw = raw.iloc[sorted(range(len(raw)), key=lambda i: order[(raw.arm[i], raw.mask_bank[i] or None)])]
    base = raw[raw.arm == "deterministic"].iloc[0]
    rel = lambda r, c: round(float(r[c]) / float(base[c]), 6)
    rows = [dict(arm=ARM_LABELS[r.arm], mask_bank=int(r.mask_bank), models=int(r.models), samples=int(r.samples),
                 rel_train_time=rel(r, "train_seconds"), rel_infer_time=rel(r, "infer_seconds"), rel_storage=rel(r, "checkpoint_bytes"),
                 decomposition_axes="yes" if r.arm == "fixed_mask" else "no") for _, r in raw.iterrows()]
    n = DEEP_ENSEMBLE_MEMBERS  # five trained deterministic members: n x time and storage, n predictive samples
    rows.append(dict(arm="Deep ensemble", mask_bank=0, models=n, samples=n, rel_train_time=float(n), rel_infer_time=float(n),
                     rel_storage=float(n), decomposition_axes="yes"))
    path = Path(out_root) / "extra" / "efficiency_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(path, index=False)
    return path


def main(argv=None) -> int:
    """Command line: ``run`` trains and times the arms, ``collect`` writes the summary table."""
    ap = argparse.ArgumentParser(description="Efficiency benchmark (Table 14)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="train and time the arms")
    r.add_argument("--runs", type=Path, default=Path("runs"))
    r.add_argument("--data", type=Path, default=Path("data"), help="data root holding coupled_pendulum_mr/")
    r.add_argument("--arm", choices=list(ARM_LABELS), default=None, help="one arm (default: all)")
    r.add_argument("--device", default="cpu")
    r.add_argument("--smoke", action="store_true", help="tiny data and network, 1 timed repeat, 5-step rollout")
    c = sub.add_parser("collect", help="write results/extra/efficiency_summary.csv")
    c.add_argument("--runs", type=Path, default=Path("runs"))
    c.add_argument("--out", type=Path, default=Path("results"))
    c.add_argument("--smoke", action="store_true", help="accepted for symmetry with the other stages")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run(args.runs, [args.arm] if args.arm else list(ARM_LABELS), torch.device(args.device), args.data, args.smoke)
    else:
        collect(args.runs, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
