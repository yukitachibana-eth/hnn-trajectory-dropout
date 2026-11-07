"""Smoke run of the hidden-parameter detection and efficiency stages on tiny data (CPU)."""

from pathlib import Path

import pandas as pd
import pytest
import torch

from hnnuq import train
from hnnuq.data import coupled_pendulum
from hnnuq.eval import efficiency, hidden_param

ROOT = Path(__file__).resolve().parents[1]
CP_TRAIN = ["--n_epochs", "2", "--n_samples", "16", "--n_val_samples", "4", "--k_samples", "2", "--k_val_samples", "2",
            "--k_test_samples", "2", "--n_test_samples", "8", "--T", "5", "--T_val", "9", "--T_test", "9", "--batch_size", "4",
            "--n_hidden", "16", "--pred_samples", "3", "--mask_num", "6", "--test_freq", "1"]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("hidden_efficiency")
    data, runs = tmp / "data", tmp / "runs"
    coupled_pendulum.main(["--out", str(data), "--smoke"])
    for arm in ("fresh_mask", "fixed_mask"):
        train.main(["--config", str(ROOT / "configs/cp" / f"{arm}.yaml"), "--seed", "42", "--run_index", "run",
                    "--output_dir", str(runs / "cp" / arm / "seed42"), "--data_dir", str(data), "--device", "cpu", *CP_TRAIN])
    return data, runs, tmp


def test_hidden_param(tiny):
    _, runs, tmp = tiny
    for arm in ("fresh_mask", "fixed_mask"):
        hidden_param.run(runs, arm, 42, torch.device("cpu"), smoke=True)
    hidden_param.main(["collect", "--runs", str(runs), "--out", str(tmp / "results"), "--smoke"])
    extra = tmp / "results" / "extra"
    for part, cols in (("A", ["method", "seed", "k", "auroc"]), ("B", ["r0", "method", "seed", "k", "auroc"]), ("C", ["r0", "method", "seed", "k", "auroc"])):
        df = pd.read_csv(extra / f"hidden_param_{part}_auroc.csv")
        assert list(df.columns) == cols and df.auroc.between(0, 1).all()
    leak = pd.read_csv(extra / "hidden_param_leakage_k0.csv")
    assert list(leak.columns) == ["method", "seed", "is_mismatch", "k", "raw_var_r0"] and (leak.k == 0).all()
    assert len(leak) == 2 * 7 * 6  # two arms x seven true ratios x six cases


def test_efficiency(tiny):
    data, runs, tmp = tiny
    efficiency.run(runs, list(efficiency.ARM_LABELS), torch.device("cpu"), data, smoke=True)
    path = efficiency.collect(runs, tmp / "results")
    df = pd.read_csv(path)
    assert list(df.columns) == efficiency.SUMMARY_COLUMNS
    assert list(zip(df.arm, df.mask_bank)) == [("Deterministic", 0), ("Fresh-mask MC", 0), ("Fixed-mask MC", 100), ("Fixed-mask MC", 200),
                                                ("Concrete", 0), ("Variational", 0), ("Deep ensemble", 0)]
    assert df.models.tolist() == [1, 1, 1, 1, 1, 1, 5] and df.decomposition_axes.tolist() == ["no", "no", "yes", "yes", "no", "no", "yes"]
    assert df.rel_train_time.iloc[0] == 1.0 and (df.rel_storage > 0).all()
