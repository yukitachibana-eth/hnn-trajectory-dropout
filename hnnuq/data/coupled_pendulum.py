"""Coupled-pendulum dataset with mass-ratio conditioning.

H(p, q, r) = p_a^2/2 + p_b^2/(2r) + (1 - cos q_a) + r (1 - cos q_b) + kappa/2 (q_a - q_b)^2,  r = m2/m1.

Files written to ``<out>/coupled_pendulum_mr/`` (``<tag>`` = ``coupled_pendulum_mr_<dataset_index>``):
    train_data_<tag>.npy        (n_ratios, T_full, n_train, 4)   canonical state [p_a, p_b, q_a, q_b], measurement noise included
    test_data_<tag>.npy         (n_ratios, T_test, n_test, 4)
    ood_test_data_<tag>.npy     (n_ood, T_test, n_test, 4)
    train_mass_ratios_<tag>.npy, test_mass_ratios_<tag>.npy, ood_mass_ratios_<tag>.npy   float32 (n,)

    python -m hnnuq.data.coupled_pendulum --out data [--smoke]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from hnnuq.integration import integrate_ground_truth

DATASET_INDEX = "cpend_mr01_clean"
SYSTEM_TYPE = "coupled_pendulum_mr"
MASS_RATIOS_TRAIN = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
MASS_RATIOS_OOD = [0.1, 0.2, 0.3, 3.0, 4.0, 5.0]
KAPPA = 1.0
NOISE_STD = 0.1
NOISE_SEED_OFFSET = 9999


class CoupledPendulumMRSystem(nn.Module):
    """True Hamiltonian of the mass-ratio coupled pendulum (m1 = 1)."""

    def __init__(self, kappa: float = KAPPA, mass_ratio: float = 1.0):
        super().__init__()
        self.kappa = kappa
        self.mass_ratio = mass_ratio

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """p, q: (batch, 2) -> H: (batch,)."""
        r = self.mass_ratio
        p_a, p_b, q_a, q_b = p[:, 0], p[:, 1], q[:, 0], q[:, 1]
        kinetic = 0.5 * p_a ** 2 + 0.5 * p_b ** 2 / r
        potential = (1.0 - torch.cos(q_a)) + r * (1.0 - torch.cos(q_b)) + 0.5 * self.kappa * (q_a - q_b) ** 2
        return kinetic + potential


def _sample_ics(n: int, gen: torch.Generator):
    """Sub-separatrix initial conditions: q0 in (-pi/2, pi/2), p0 in (-0.8, 0.8) (q drawn first)."""
    q0 = (torch.rand(n, 2, generator=gen) - 0.5) * np.pi
    p0 = (torch.rand(n, 2, generator=gen) - 0.5) * 1.6
    return p0, q0


def generate_trajectories(n_samples: int, mass_ratio: float, T: int, dt: float, seed: int,
                          kappa: float = KAPPA, noise_std: float = NOISE_STD) -> torch.Tensor:
    """Leapfrog ground truth for one mass ratio plus iid Gaussian measurement noise; returns (T, n_samples, 4)."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    p0, q0 = _sample_ics(n_samples, gen)
    traj = integrate_ground_truth(p_0=p0, q_0=q0, hamiltonian=CoupledPendulumMRSystem(kappa, mass_ratio), T=T, dt=dt)
    if noise_std > 0.0:
        noise_gen = torch.Generator()
        noise_gen.manual_seed(seed + NOISE_SEED_OFFSET)
        traj = traj + noise_std * torch.randn(traj.shape, generator=noise_gen)
    return traj


def generate_dataset(out_dir: Path, n_train: int = 800, n_test: int = 200, T_train: int = 30, T_test: int = 200,
                     dt: float = 0.1, seed: int = 42, train_ratios: List[float] = None,
                     ood_ratios: List[float] = None) -> Dict[str, np.ndarray]:
    """Generate the dataset and write the six ``.npy`` files; seeds are ``seed + {0,1,2} + 1000 * ratio_index`` (train, test, OOD)."""
    train_ratios = MASS_RATIOS_TRAIN if train_ratios is None else train_ratios
    ood_ratios = MASS_RATIOS_OOD if ood_ratios is None else ood_ratios
    T_full = max(T_train, T_test)
    train = torch.stack([generate_trajectories(n_train, r, T_full, dt, seed + i * 1000) for i, r in enumerate(train_ratios)])
    test = torch.stack([generate_trajectories(n_test, r, T_test, dt, seed + 1 + i * 1000) for i, r in enumerate(train_ratios)])
    ood = torch.stack([generate_trajectories(n_test, r, T_test, dt, seed + 2 + i * 1000) for i, r in enumerate(ood_ratios)])
    arrays = {
        "train_data": train.numpy(), "test_data": test.numpy(), "ood_test_data": ood.numpy(),
        "train_mass_ratios": torch.tensor(train_ratios, dtype=torch.float32).numpy(),
        "test_mass_ratios": torch.tensor(train_ratios, dtype=torch.float32).numpy(),
        "ood_mass_ratios": torch.tensor(ood_ratios, dtype=torch.float32).numpy(),
    }
    save_dir = Path(out_dir) / SYSTEM_TYPE
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(save_dir / f"{name}_{SYSTEM_TYPE}_{DATASET_INDEX}.npy", arr)
    return arrays


def main(argv=None) -> int:
    """Command line: write the coupled-pendulum dataset (``--smoke``: tiny)."""
    ap = argparse.ArgumentParser(description="Generate the coupled-pendulum mass-ratio dataset")
    ap.add_argument("--out", default="data", help="data root; files go to <out>/coupled_pendulum_mr/")
    ap.add_argument("--smoke", action="store_true", help="tiny dataset (2 train ratios, 20 ICs, short horizon)")
    args = ap.parse_args(argv)
    if args.smoke:
        generate_dataset(Path(args.out), n_train=20, n_test=10, T_train=5, T_test=10,
                         train_ratios=[0.5, 2.0], ood_ratios=[0.1, 5.0])
    else:
        generate_dataset(Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
