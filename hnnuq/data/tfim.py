"""Generates the exact TFIM dataset of the paper (L=4 open chain, hidden initial-state preparation noise).

Usage (repository root):
    python -m hnnuq.data.tfim --out data/tfim/L4_gscan
    python -m hnnuq.data.tfim --out /tmp/tfim_smoke --smoke

Dataset. 256 intended product states per split; each is perturbed by 20 independent isotropic
state-preparation noises of strength 0.10 (fidelity 1 / 1.01 with the intended state) into 20 latent
initial states (the realised preparations). Every latent state is evolved exactly under H(g) for each
control value g (13 training values in the band [1.3, 1.9], 23 test values), 41 frames on [0, 2]. The
model sees only the intended state at t=0 (hidden initial condition); the targets follow the latent state.
The training intended states are the 8 structured product states plus 248 random product states
(``product_state_repertoire``, seed 42); the test intended states are 256 random product states from a
separate seed (4242), so no test state is a training state.

Files in the output directory (the only ones training and evaluation read):
    metadata.json                  chain, control grids, train band, time grid
    {train,test}.npz               z (g, T, sample, 2*dim) float32 canonical [q, p] trajectories of the latent
                                   states, g_values, time_grid, n_intended_initial_states,
                                   n_latent_samples_per_initial_state, flat_to_intended_index,
                                   flat_to_latent_index (sample = intended * n_latent + latent)
    {train,test}_aux.npz           observed_z0_grouped, latent_z0_grouped: (intended, latent, 2*dim) float32
                                   model-input (intended) and latent t=0 states
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from hnnuq.tfim.exact import (
    TFIM_J,
    complex_to_canonical,
    evolve_exact,
    hamiltonian_matrix,
    perturbed_ensemble,
    product_state_repertoire,
)

L = 4
SEED = 42  # training product-state repertoire seed and base of the noise seeds
TEST_STATE_SEED = 4242  # seed of the test intended states
NOISE_LEVEL = 0.10
N_INTENDED = 256
N_LATENT = 20
T_FINAL = 2.0
N_TIME = 41
TRAIN_BAND = (1.3, 1.9)
G_TRAIN = [1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9]
G_TEST = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 1.15, 1.2,
          1.4, 1.6, 1.8, 2.0, 2.2, 2.5]
SPLIT_NOISE_SEED_OFFSET = {"train": 0, "test": 1_000_003}


def latent_initial_states(intended: np.ndarray, split: str) -> np.ndarray:
    """(n_intended, n_latent, 2**L) complex latent states: intended state i gets 20 noise draws with
    generator seeds ``SEED + offset(split) + 10000 * i + j`` (j = 0..19)."""
    seed = SEED + SPLIT_NOISE_SEED_OFFSET[split]
    out = np.empty((intended.shape[0], N_LATENT, intended.shape[1]), dtype=np.complex128)
    for i, psi in enumerate(intended):
        out[i] = perturbed_ensemble(psi, N_LATENT, NOISE_LEVEL, seed + i * 10_000)
    return out


def generate_split(intended: np.ndarray, g_values: Sequence[float], split: str, time_grid: np.ndarray) -> dict:
    """Arrays of one split. z has shape (g, T, sample, 2*dim) float32 in canonical [q, p] order."""
    n_int, dim = intended.shape
    latent = latent_initial_states(intended, split)  # (i, l, dim)
    flat = latent.reshape(n_int * N_LATENT, dim)
    z = np.empty((len(g_values), n_int * N_LATENT, len(time_grid), 2 * dim), dtype=np.float32)
    for gi, g in enumerate(g_values):
        psi = evolve_exact(hamiltonian_matrix(L, float(g), TFIM_J), flat, time_grid)  # (sample, T, dim)
        z[gi] = complex_to_canonical(psi).astype(np.float32)
    observed = np.repeat(intended[:, None, :], N_LATENT, axis=1)  # the model sees the intended state
    return {
        "z": np.transpose(z, (0, 2, 1, 3)),
        "observed_z0_grouped": complex_to_canonical(observed).astype(np.float32),
        "latent_z0_grouped": complex_to_canonical(latent).astype(np.float32),
    }


def write_dataset(out: Path, n_intended: int = N_INTENDED, g_train: Sequence[float] = G_TRAIN,
                  g_test: Sequence[float] = G_TEST, n_time: int = N_TIME) -> None:
    """Generate and write the dataset described in the module docstring."""
    out.mkdir(parents=True, exist_ok=True)
    time_grid = np.linspace(0.0, T_FINAL, n_time, dtype=np.float64)
    intended_by_split = {
        "train": product_state_repertoire(L, n_intended, SEED),
        "test": product_state_repertoire(L, n_intended, TEST_STATE_SEED, named_states=False),
    }
    flat_to_intended = np.repeat(np.arange(n_intended, dtype=np.int64), N_LATENT)
    flat_to_latent = np.tile(np.arange(N_LATENT, dtype=np.int64), n_intended)
    for split, g_values in (("train", g_train), ("test", g_test)):
        arrays = generate_split(intended_by_split[split], g_values, split, time_grid)
        np.savez_compressed(
            out / f"{split}.npz",
            g_values=np.asarray(g_values, dtype=np.float64),
            time_grid=time_grid,
            n_intended_initial_states=np.asarray(n_intended, dtype=np.int64),
            n_latent_samples_per_initial_state=np.asarray(N_LATENT, dtype=np.int64),
            flat_to_intended_index=flat_to_intended,
            flat_to_latent_index=flat_to_latent,
            z=arrays["z"],
        )
        np.savez_compressed(
            out / f"{split}_aux.npz",
            observed_z0_grouped=arrays["observed_z0_grouped"],
            latent_z0_grouped=arrays["latent_z0_grouped"],
        )
        print(f"{split}: z {arrays['z'].shape} -> {out / (split + '.npz')}")
    metadata = {
        "L": L,
        "J": TFIM_J,
        "boundary_condition": "open",
        "train_band": list(TRAIN_BAND),
        "g_train_values": list(g_train),
        "g_test_values": list(g_test),
        "t_final": T_FINAL,
        "n_time_steps": n_time,
        "time_grid": time_grid.tolist(),
        "n_intended_initial_states": n_intended,
        "n_latent_samples_per_initial_state": N_LATENT,
        "noise_level": NOISE_LEVEL,
        "seed": SEED,
        "test_state_seed": TEST_STATE_SEED,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line: write the TFIM dataset (``--smoke``: tiny)."""
    parser = argparse.ArgumentParser(description="Generate the exact TFIM dataset")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny dataset (8 intended states, 3 training / 4 test g values, 11 frames)")
    args = parser.parse_args(argv)
    if args.smoke:
        write_dataset(args.out, n_intended=8, g_train=G_TRAIN[:3], g_test=[0.2, 1.0, 1.6, 2.5], n_time=11)
    else:
        write_dataset(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
