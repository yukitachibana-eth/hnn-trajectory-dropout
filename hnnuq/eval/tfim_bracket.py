"""Poisson-bracket residual of the learned TFIM Hamiltonians (paper Table 12).

Usage (repository root):
    python -m hnnuq.eval.tfim_bracket --run-dir <run> --data data/tfim/<dataset> --out <dir>

The exact dynamics conserve every quadratic form of (q, p) that commutes with H; the learned
Hamiltonian H_theta should therefore satisfy {G, H_theta} = 0 for G = sum(q^2 + p^2) (the norm).
At each test state the normalised residual is
    | {G, H_theta} | / ( |grad G| |grad H_theta| + 1e-8 ),
    {G, H} = sum_i (dG/dq_i dH/dp_i - dG/dp_i dH/dq_i),   grad = gradient with respect to (q, p).
States: the latent trajectory states of the first latent sample of the first 10 intended initial
states at frames 0, 10, 20, 30, 40, for every one of the 23 test control values (1150 states).
Draws per state: one for a deterministic model; 10 for stochastic models, namely 10 distinct bank
masks (a seeded subset of the bank) for Fixed-mask MC and 10 fresh masks otherwise.
Writes ``residuals.csv`` (one row per control value, frame, intended state, draw) and
``summary.json`` with ``mean_log_resid``, the mean over all rows of log(residual).
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from hnnuq.eval.tfim import SEED, Bundle, TestData, _control, prepare
from hnnuq.integration import compute_hamiltonian_gradients

TIME_INDICES = (0, 10, 20, 30, 40)
N_IC = 10
N_DRAWS = 10
MASK_SUBSET_SEED = 20260724  # bank-mask subset of the Fixed-mask draws (CPU generator)
EPS = 1e-8


def bracket_residual(gq: torch.Tensor, gp: torch.Tensor, hp: torch.Tensor, hq: torch.Tensor) -> torch.Tensor:
    """Normalised |{G, H}| from the gradients (dG/dq, dG/dp, dH/dp, dH/dq), each (n, dim)."""
    bracket = (gq * hp).sum(-1) - (gp * hq).sum(-1)
    norm_g = torch.sqrt((gq**2).sum(-1) + (gp**2).sum(-1))
    norm_h = torch.sqrt((hq**2).sum(-1) + (hp**2).sum(-1))
    return bracket.abs() / (norm_g * norm_h + EPS)


def draw_slots(bundle: Bundle, n_draws: int) -> List[Optional[int]]:
    """Mask index of each draw: ``None`` = fresh masks, an int = that bank mask."""
    if bundle.kind == "deterministic":
        return [0]
    if bundle.kind == "bank":
        gen = torch.Generator()
        gen.manual_seed(MASK_SUBSET_SEED)
        return torch.randperm(bundle.mask_num, generator=gen)[:n_draws].tolist()
    if bundle.kind == "fresh":
        return [None] * n_draws
    raise ValueError("the bracket diagnostic is defined for single models, not deep ensembles")


def bracket_stage(bundle: Bundle, test: TestData, device: torch.device, smoke: bool = False) -> List[dict]:
    """Residual rows (control value, frame, intended state, draw, residual) for the whole test grid."""
    T, dim = int(test.z.shape[1]), int(test.z.shape[-1]) // 2
    n_lat = 20
    n_ic, n_draws = (2, 2) if smoke else (N_IC, N_DRAWS)
    times = tuple(sorted({0, T - 1})) if smoke else TIME_INDICES
    g_indices = range(0, len(test.g_values), 8) if smoke else range(len(test.g_values))
    slots = draw_slots(bundle, n_draws)
    rows: List[dict] = []
    for gi in g_indices:
        g = float(test.g_values[gi])
        keys = [(t, ic) for t in times for ic in range(n_ic)]
        states = np.stack([np.asarray(test.z[gi, t, ic * n_lat], dtype=np.float64) for t, ic in keys])  # (n, 2*dim) [q, p]
        qb = torch.tensor(states[:, :dim], dtype=torch.float32, device=device)
        pb = torch.tensor(states[:, dim:], dtype=torch.float32, device=device)
        k = _control(g, dim, device)
        for draw, idx in enumerate(slots):
            q = qb.clone().requires_grad_(True)
            p = pb.clone().requires_grad_(True)
            hp, hq = compute_hamiltonian_gradients(p, q, lambda pp, qq: bundle.model(pp, qq, k=k, index=idx))
            resid = bracket_residual(2 * qb, 2 * pb, hp.detach(), hq.detach()).cpu().numpy()
            for (t, ic), r in zip(keys, resid):
                rows.append({"g_index": gi, "g": g, "frame": t, "intended_state": ic, "draw": draw, "residual": float(r)})
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line of the Poisson-bracket residual stage."""
    parser = argparse.ArgumentParser(description="TFIM Poisson-bracket residual")
    parser.add_argument("--run-dir", type=Path, nargs=1, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    bundle, test, device = prepare(args)
    rows = bracket_stage(bundle, test, device, args.smoke)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "residuals.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()})
    resid = np.array([r["residual"] for r in rows])
    summary = {"mean_log_resid": float(np.log(np.clip(resid, 1e-300, None)).mean()), "n_rows": len(rows), "kind": bundle.kind}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
