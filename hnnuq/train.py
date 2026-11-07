#!/usr/bin/env python
"""
train.py
Single-run training entry point (one system x arm x seed).

Usage (from the repository root):
    python -m hnnuq.train --config configs/cp/fixed_mask.yaml --seed 42 --output_dir runs/cp
    python -m hnnuq.train --config configs/sc/fixed_mask.yaml --seed 42 --mask_num 50   # any config key
"""

import argparse
import sys
import random
from dataclasses import fields
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from hnnuq.config import ExperimentConfig
from hnnuq.tfim.control import resolve_tfim_control_tensor
from hnnuq.tfim.loader import load_tfim_training_data
from hnnuq.trainer import Trainer


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def parse_args(argv: Optional[List[str]] = None):
    """Parse CLI arguments: --config plus one optional override flag per config key."""
    parser = argparse.ArgumentParser(description="HNN-UQ single training run")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, cpu)")
    for f in fields(ExperimentConfig):
        kind = {int: int, float: float, bool: _parse_bool}.get(f.type, str)
        parser.add_argument(f"--{f.name}", type=kind, default=None)
    return parser.parse_args(argv)


def build_config(args) -> ExperimentConfig:
    """Load the YAML config, then apply any CLI overrides."""
    config = ExperimentConfig.load(args.config)
    for f in fields(ExperimentConfig):
        value = getattr(args, f.name, None)
        if value is not None:
            setattr(config, f.name, value)
    config.__post_init__()
    return config


def get_device(device_arg: Optional[str] = None) -> torch.device:
    """Get torch device."""
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data(config: ExperimentConfig):
    """
    Load train/val/test data and per-cell control parameters (CP / SC).

    Expected files under {data_dir}/{system_type}/:
        train_data_{system_type}_{dataset_index}.npy, test_data_*.npy
        coupled_pendulum_mr: train_mass_ratios_*.npy, test_mass_ratios_*.npy
        chain:               train_data_*k_train_mat.npy, test_data_*k_test_mat.npy

    Data shape: (k_samples, T, n_samples, dim*2) with [p, q] concatenated.
    """
    data_dir = Path(config.data_dir) / config.system_type
    tag = f"{config.system_type}_{config.dataset_index}"
    train_file = data_dir / f"train_data_{tag}.npy"
    test_file = data_dir / f"test_data_{tag}.npy"
    for path in (train_file, test_file):
        if not path.exists():
            raise FileNotFoundError(f"Data not found: {path}")

    train_data = torch.from_numpy(np.load(train_file)).float()
    test_data = torch.from_numpy(np.load(test_file)).float()

    n_train_total = config.n_samples + config.n_val_samples
    train_data = train_data[: config.k_samples, :, :n_train_total, :]
    test_data = test_data[: config.k_test_samples, :, : config.n_test_samples, :]

    val_data = train_data[:, :, -config.n_val_samples :, :]
    train_data = train_data[:, :, : -config.n_val_samples, :]

    if config.system_type == "coupled_pendulum_mr":
        k_train_file = data_dir / f"train_mass_ratios_{tag}.npy"
        k_test_file = data_dir / f"test_mass_ratios_{tag}.npy"
    else:  # chain: spring-constant connection matrices
        k_train_file = data_dir / f"train_data_{tag}k_train_mat.npy"
        k_test_file = data_dir / f"test_data_{tag}k_test_mat.npy"
    for path in (k_train_file, k_test_file):
        if not path.exists():
            raise FileNotFoundError(f"Control parameters not found: {path}")
    k_train = torch.from_numpy(np.load(k_train_file)).float()[: config.k_samples]
    k_test = torch.from_numpy(np.load(k_test_file)).float()[: config.k_test_samples]
    print(f"Train: {train_data.shape}  Val: {val_data.shape}  Test: {test_data.shape}")
    return train_data, val_data, test_data, k_train, k_test


def main(argv: Optional[List[str]] = None) -> int:
    """Train one run (system x arm x seed) from the command line; returns the exit code."""
    args = parse_args(argv)
    config = build_config(args)

    output_dir = config.get_run_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")
    print(config)

    set_seed(config.seed)
    device = get_device(args.device)

    if config.system_type == "tfim":
        tfim_data = load_tfim_training_data(config)
        train_data, val_data, test_data = tfim_data.train_data, tfim_data.val_data, tfim_data.test_data
        train_payload, val_payload, test_payload = (
            tfim_data.train_split, tfim_data.val_split, tfim_data.test_split,
        )
        state_half_dim = int(train_data.shape[-1] // 2)
        k_train, k_val, k_test = (
            resolve_tfim_control_tensor(g, state_half_dim, device=device, dtype=torch.float32)
            for g in (tfim_data.g_train, tfim_data.g_val, tfim_data.g_test)
        )
    else:
        train_data, val_data, test_data, k_train, k_test = load_data(config)
        k_val = None
        train_payload, val_payload, test_payload = train_data, val_data, test_data

    trainer = Trainer(config, device)
    model, metrics = trainer.train(
        train_data=train_payload,
        val_data=val_payload,
        test_data=test_payload,
        k_train=k_train,
        k_val=k_val,
        k_test=k_test,
        output_dir=output_dir,
    )
    if metrics["test_mse"]:
        print(f"Final test MSE: {metrics['test_mse'][-1]:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
