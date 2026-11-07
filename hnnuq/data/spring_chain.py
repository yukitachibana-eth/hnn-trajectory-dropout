"""Four-particle spring-chain dataset with random coupling matrices.

H(p, q; K) = sum_i p_i^2 / (2 m) + sum_{i<j} K_ij (q_i - q_j)^2 / 4 + wall springs on both ends;  fixed endpoints, m = 1.

Files written to ``<out>/chain/`` (``<tag>`` = ``chain_<dataset_index>``):
    train_data_<tag>.npy                 (k_train, T_full, n_ics_train, 2N)   [p, q] with measurement noise
    test_data_<tag>.npy                  (k_test, T_test, n_ics_test, 2N)
    train_data_<tag>k_train_mat.npy      (k_train, N, N)   coupling matrices
    test_data_<tag>k_test_mat.npy        (k_test, N, N)

    python -m hnnuq.data.spring_chain --out data [--smoke]
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from hnnuq.integration import integrate_ground_truth

DATASET_INDEX = "chain4_k5"
SYSTEM_TYPE = "chain"
CHAIN_LENGTH = 4
K_RANGE = (1.0, 5.0)
NOISE_STD = 0.1
DT = 0.1


class SpringChainHamiltonianMatrix(nn.Module):
    """True Hamiltonian of a spring chain with symmetric coupling matrix ``k_mat`` and masses ``m``."""

    def __init__(self, k_mat: torch.Tensor, m, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cpu")
        k_mat = 0.5 * (k_mat + k_mat.T)
        self.register_buffer("k_mat", k_mat.to(self.device))
        self.register_buffer("m", torch.as_tensor(m).to(self.device))

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """p, q: (batch, N) -> H: (batch,)."""
        p, q = p.to(self.device), q.to(self.device)
        kinetic = 0.5 * (p.pow(2) / self.m).sum(dim=1)
        q_diff_sq = (q.unsqueeze(2) - q.unsqueeze(1)).pow(2)  # (batch, N, N)
        potential_off_diag = 0.5 * (self.k_mat.unsqueeze(0) * q_diff_sq).sum(dim=(1, 2)) / 2
        potential_walls = 0.5 * self.k_mat[0, 0] * q[:, 0].pow(2) + 0.5 * self.k_mat[-1, -1] * q[:, -1].pow(2)
        return kinetic + potential_off_diag + potential_walls


def generate_k_matrices_batch(n_matrices: int, chain_length: int, k_range: Tuple[float, float] = (5.0, 10.0),
                              connectivity_prob: float = 0.7, seed: Optional[int] = None) -> torch.Tensor:
    """Random chain couplings: one scalar stiffness per matrix times a random symmetric adjacency (both wall springs always on).

    Seeds the global torch RNG when ``seed`` is given; draw order: stiffness, adjacency.
    """
    if seed is not None:
        torch.manual_seed(seed)
    k_values = torch.empty(n_matrices).uniform_(*k_range)
    prob = (torch.ones(n_matrices, chain_length, chain_length) * connectivity_prob).triu(diagonal=1)
    adj = torch.bernoulli(prob)
    adj = adj + adj.transpose(-1, -2)
    adj[:, 0, 0] = 1.0
    adj[:, -1, -1] = 1.0
    return adj * k_values.view(-1, 1, 1)


def simulate_single_k_matrix(k_mat: torch.Tensor, n_samples: int, T: int, dt: float, init_scale: float = 2.5,
                             fixed_endpoints: bool = True, device: Optional[torch.device] = None,
                             seed: Optional[int] = None) -> torch.Tensor:
    """Leapfrog trajectories (T, n_samples, 2N) of one chain from uniform ICs in (-init_scale, init_scale).

    ``seed`` seeds a generator on ``device`` (so CUDA and CPU draw different ICs); ``None`` uses the global RNG.
    Draw order: p0, then q0.
    """
    device = device or torch.device("cpu")
    n = k_mat.shape[0]
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    m = torch.full((n,), 1.0, device=device)
    hamiltonian = SpringChainHamiltonianMatrix(k_mat, m, device=device)
    p_0 = (torch.rand(n_samples, n, device=device, generator=generator) - 0.5) * 2 * init_scale
    q_0 = (torch.rand(n_samples, n, device=device, generator=generator) - 0.5) * 2 * init_scale
    if fixed_endpoints:
        p_0[:, 0] = 0.0
        p_0[:, -1] = 0.0
        q_0[:, 0] = 0.0
        q_0[:, -1] = 0.0
    return integrate_ground_truth(p_0=p_0, q_0=q_0, hamiltonian=hamiltonian, T=T, dt=dt, use_leapfrog=True)


def simulate_chain_dynamics_batch(k_mats: torch.Tensor, n_samples: int, T: int, dt: float) -> torch.Tensor:
    """Trajectories (n_k, T, n_samples, 2N) for each coupling matrix; ICs come from the global RNG in matrix order."""
    n_k, n = k_mats.shape[0], k_mats.shape[1]
    out = torch.zeros(n_k, T, n_samples, 2 * n)
    for i in range(n_k):
        out[i] = simulate_single_k_matrix(k_mats[i], n_samples, T, dt)
    return out


def generate_dataset(out_dir: Path, n_ics_train: int = 3000, n_ics_test: int = 1000, k_train: int = 20, k_test: int = 10,
                     T_train: int = 10, T_test: int = 100) -> None:
    """Generate train (seed 42) and test (seed 1000) chains and write the four ``.npy`` files.

    Noise is drawn after each split's simulation with the global RNG re-seeded to ``seed + 1``.
    """
    T_full = max(T_train, T_test)
    k_train_mats = generate_k_matrices_batch(k_train, CHAIN_LENGTH, k_range=K_RANGE, seed=42)
    train = simulate_chain_dynamics_batch(k_train_mats, n_ics_train, T_full, DT)
    torch.manual_seed(42 + 1)
    train = train + NOISE_STD * torch.randn_like(train)
    k_test_mats = generate_k_matrices_batch(k_test, CHAIN_LENGTH, k_range=K_RANGE, seed=1000)
    test = simulate_chain_dynamics_batch(k_test_mats, n_ics_test, T_test, DT)
    torch.manual_seed(1000 + 1)
    test = test + NOISE_STD * torch.randn_like(test)
    save_dir = Path(out_dir) / SYSTEM_TYPE
    save_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{SYSTEM_TYPE}_{DATASET_INDEX}"
    np.save(save_dir / f"train_data_{tag}.npy", train.numpy())
    np.save(save_dir / f"test_data_{tag}.npy", test.numpy())
    np.save(save_dir / f"train_data_{tag}k_train_mat.npy", k_train_mats.numpy())
    np.save(save_dir / f"test_data_{tag}k_test_mat.npy", k_test_mats.numpy())


def main(argv=None) -> int:
    """Command line: write the spring-chain dataset (``--smoke``: tiny)."""
    ap = argparse.ArgumentParser(description="Generate the spring-chain dataset")
    ap.add_argument("--out", default="data", help="data root; files go to <out>/chain/")
    ap.add_argument("--smoke", action="store_true", help="tiny dataset (2 train matrices, 200 ICs, short horizon)")
    args = ap.parse_args(argv)
    if args.smoke:
        generate_dataset(Path(args.out), n_ics_train=200, n_ics_test=50, k_train=2, k_test=1, T_train=15, T_test=20)
    else:
        generate_dataset(Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
