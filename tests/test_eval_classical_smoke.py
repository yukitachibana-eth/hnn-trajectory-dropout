"""Smoke run of the CP / SC evaluation stages and the classical ledger on tiny data (CPU, seconds)."""

from pathlib import Path

import pandas as pd
import pytest
import torch

from hnnuq import train
from hnnuq.data import coupled_pendulum, spring_chain
from hnnuq.eval import cp, sc
from hnnuq.ledger import classical

CP_TRAIN = ["--n_epochs", "2", "--n_samples", "16", "--n_val_samples", "4", "--k_samples", "2", "--k_val_samples", "2",
            "--k_test_samples", "2", "--n_test_samples", "8", "--T", "5", "--T_val", "9", "--T_test", "9", "--batch_size", "4",
            "--n_hidden", "16", "--pred_samples", "3", "--mask_num", "6", "--test_freq", "1"]
SC_TRAIN = ["--n_epochs", "2", "--n_samples", "150", "--n_val_samples", "50", "--k_samples", "2", "--k_val_samples", "2",
            "--k_test_samples", "1", "--n_test_samples", "20", "--T", "5", "--T_val", "14", "--T_test", "14", "--batch_size", "32",
            "--n_hidden", "16", "--pred_samples", "3", "--mask_num", "6", "--test_freq", "1"]
ROOT = Path(__file__).resolve().parents[1]


def _train(system: str, arm: str, data: Path, runs: Path, extra, seed: int = 42):
    train.main(["--config", str(ROOT / "configs" / system / f"{arm}.yaml"), "--seed", str(seed), "--run_index", "run",
                "--output_dir", str(runs / system / arm / f"seed{seed}"), "--data_dir", str(data), "--device", "cpu", *extra])


@pytest.fixture(scope="module")
def smoke_tree(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("classical_smoke")
    data, runs = tmp / "data", tmp / "runs"
    coupled_pendulum.main(["--out", str(data), "--smoke"])
    spring_chain.main(["--out", str(data), "--smoke"])
    for arm in ("fresh_mask", "fixed_mask", "deterministic"):
        _train("cp", arm, data, runs, CP_TRAIN)
        _train("sc", arm, data, runs, SC_TRAIN)
    return runs


def test_stages_and_ledger(smoke_tree):
    for arm in ("fresh_mask", "fixed_mask", "deterministic"):
        cp.run(smoke_tree, arm, 42, torch.device("cpu"), smoke=True)
        sc.run(smoke_tree, arm, 42, torch.device("cpu"), smoke=True)
    led, cp_fig, sc_fig = classical.build(smoke_tree, [42], ["fresh_mask", "fixed_mask", "deterministic"])
    assert list(led.columns) == classical.LEDGER_COLUMNS
    got = set(zip(led.system, led.arm, led.quantity))
    for system in ("CP", "SC"):
        for q in ("mse", "g_nll", "alpha", "phys_e_drift", "h_drift", "ic_noise_scale"):
            assert (system, "Fixed-mask MC", q) in got
        assert (system, "Deterministic", "alpha") not in got
    assert len(cp_fig) == 2 * 4 and len(sc_fig) == 2 * 3  # smoke: 4 mass ratios x 2 arms, 3 stiffness values x 2 arms
    assert led.value.notna().all()
