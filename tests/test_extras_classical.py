"""Smoke run of the long-horizon, IC-noise-sweep and mask-bank stages on tiny data (CPU)."""

from pathlib import Path

import pandas as pd
import pytest
import torch

from hnnuq import train
from hnnuq.data import coupled_pendulum, spring_chain
from hnnuq.eval import ic_noise_sweep, long_horizon, mask_bank
from test_eval_classical_smoke import CP_TRAIN, ROOT, SC_TRAIN

CPU = torch.device("cpu")


def _train(system: str, config: str, arm_dir: str, data: Path, runs: Path, extra, seed: int = 42):
    train.main(["--config", str(ROOT / "configs" / system / f"{config}.yaml"), "--seed", str(seed), "--run_index", "run",
                "--output_dir", str(runs / system / arm_dir / f"seed{seed}"), "--data_dir", str(data), "--device", "cpu", *extra])


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("extras_smoke")
    data, runs = tmp / "data", tmp / "runs"
    coupled_pendulum.main(["--out", str(data), "--smoke"])
    spring_chain.main(["--out", str(data), "--smoke"])
    for arm in ("deterministic", "fresh_mask", "fixed_mask"):
        _train("cp", arm, arm, data, runs, CP_TRAIN)
        _train("sc", arm, arm, data, runs, SC_TRAIN)
    # mask-bank sizes: the two smallest banks stand in for the grid
    _train("sc", "fixed_mask", "fixed_mask_m5", data, runs, SC_TRAIN[:-4] + ["--mask_num", "5", "--test_freq", "1"])
    _train("sc", "fixed_mask_m1_per_rollout", "fixed_mask_m1", data, runs, SC_TRAIN[:-4] + ["--test_freq", "1"])
    return runs


def test_long_horizon(tree, tmp_path):
    for arm in long_horizon.ARMS:
        for system in long_horizon.SYSTEMS:
            long_horizon.run(tree, system, arm, 42, CPU, smoke=True)
    long_horizon.collect(tree, tmp_path, smoke=True)
    table = pd.read_csv(tmp_path / "extra/long_horizon_per_seed.csv")
    assert list(table.columns) == ["system", "arm", "seed", "split", "t", "value"]
    assert len(table) == 2 * 3 * 2 * 2 and table.value.notna().all()  # systems x arms x splits x steps
    fig = pd.read_csv(tmp_path / "figures/sc_energy_drift_by_step.csv")
    assert list(fig.columns) == ["arm", "seed", "step", "drift"] and len(fig) == 3 * long_horizon.SMOKE_HORIZON
    assert (fig[fig.step == 1].drift == 0).all()  # a rollout's drift is measured from its own first frame


def test_ic_noise_sweep(tree, tmp_path):
    for arm in ic_noise_sweep.ARMS:
        ic_noise_sweep.run(tree, arm, 42, CPU, smoke=True)
    ic_noise_sweep.collect(tree, tmp_path)
    d = pd.read_csv(tmp_path / "extra/ic_noise_sweep_per_seed.csv")
    assert list(d.columns) == ["method", "split", "sigma_multiplier", "seed", "var_epistemic_mean", "var_aleatoric_mean"]
    assert len(d) == 2 * 2 * 2 and d.notna().all().all()
    fixed = d[d.method == "Fixed-mask MC"]
    assert (fixed[fixed.sigma_multiplier == 0].var_aleatoric_mean == 0).all()  # same masks and no IC noise: no between-IC spread
    assert (d[d.sigma_multiplier == 1].var_aleatoric_mean > 0).all()


def test_mask_bank(tree, tmp_path, capsys):
    for arm in ("fixed_mask_m1", "fixed_mask_m5"):
        mask_bank.run(tree, arm, 42, CPU, smoke=True)
    mask_bank.collect(tree, tmp_path)
    d = pd.read_csv(tmp_path / "extra/mask_bank_per_seed.csv")
    assert list(d.columns) == ["M", "seed", "quantity", "value"] and set(d.M) == {1, 5}
    assert {"mse_id", "sigma_hat", "nll_fit_ood", "phys_drift_id", "h_drift"} <= set(d[d.M == 5].quantity)
    assert "nll_mask_id" not in set(d[d.M == 1].quantity)  # undefined for a single mask
    assert d.value.notna().all()
    capsys.readouterr()  # drop the output of the runs above
    mask_bank.main(["plan"])
    plan = capsys.readouterr().out.strip().splitlines()
    assert len(plan) == 5 * 3 and all(line.startswith("python -m hnnuq.train") for line in plan)
