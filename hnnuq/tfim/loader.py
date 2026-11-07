"""TFIM dataset loader for the HNN trainer.

A dataset directory holds ``metadata.json``, ``train.npz`` / ``test.npz`` and the auxiliary
``train_aux.npz`` / ``test_aux.npz``:

- ``z``: flat trajectories (g, T, sample, 2 * dim) in canonical [q, p] order, where
  ``sample = intended_ic * n_latent_samples_per_initial_state + latent_sample``;
- ``observed_z0_grouped`` / ``latent_z0_grouped``: (intended_ic, latent_sample, 2 * dim) model
  input and latent t=0 states (hidden initial condition: the model sees the intended state,
  the trajectory starts from the realised latent state);
- ``g_values``, ``flat_to_intended_index``, ``flat_to_latent_index``.

The trainer uses (k, T, n_samples, 2 * dim) tensors in [p, q] order. Train and validation
share ``train.npz`` and are split by whole intended initial states, and by disjoint g values.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def resolve_tfim_dataset_root(config: Any) -> Path:
    """Dataset directory: ``config.data_dir`` itself, or ``data_dir/dataset_index``."""
    base = Path(config.data_dir).expanduser()
    for candidate in (base, base / str(config.dataset_index)):
        if all((candidate / name).exists() for name in ("metadata.json", "train.npz", "test.npz")):
            return candidate
    raise FileNotFoundError(
        f"TFIM dataset not found under {base} (expected metadata.json, train.npz, test.npz)."
    )


def _load_keys(path: Path, keys: tuple[str, ...]) -> dict[str, Any]:
    with np.load(path) as archive:
        return {key: archive[key] for key in keys}


def _to_pipeline_order(canonical: np.ndarray) -> torch.Tensor:
    """Swap the [q, p] halves of the last axis to the trainer's [p, q] order."""
    half = canonical.shape[-1] // 2
    swapped = np.concatenate([canonical[..., half:], canonical[..., :half]], axis=-1)
    return torch.from_numpy(np.asarray(swapped, dtype=np.float32).copy())


@dataclass
class TFIMSplitProvenance:
    """Which intended initial states / latent samples make up one split."""

    split_name: str
    n_intended_initial_states: int
    n_latent_samples_per_initial_state: int
    selected_intended_initial_state_indices: list[int]
    flat_to_intended_index: list[int]
    flat_to_latent_index: list[int]


@dataclass
class TFIMSplitBundle:
    """One split: trajectories (g, T, sample, 2*dim), model-input and latent t=0 states
    (g, sample, 2*dim), the g values and the sample provenance."""

    split_name: str
    trajectories: torch.Tensor
    input_z0: torch.Tensor
    latent_z0: torch.Tensor
    g_values: torch.Tensor
    provenance: TFIMSplitProvenance


@dataclass
class TFIMTrainingData:
    """Train / validation / test bundles of a TFIM dataset."""

    dataset_root: Path
    train_split: TFIMSplitBundle
    val_split: TFIMSplitBundle
    test_split: TFIMSplitBundle

    @property
    def train_data(self) -> torch.Tensor:
        """Training trajectories (g, T, sample, 2*dim) in [p, q] order."""
        return self.train_split.trajectories

    @property
    def val_data(self) -> torch.Tensor:
        """Validation trajectories."""
        return self.val_split.trajectories

    @property
    def test_data(self) -> torch.Tensor:
        """Test trajectories."""
        return self.test_split.trajectories

    @property
    def g_train(self) -> torch.Tensor:
        """Control values of the training split."""
        return self.train_split.g_values

    @property
    def g_val(self) -> torch.Tensor:
        """Control values of the validation split."""
        return self.val_split.g_values

    @property
    def g_test(self) -> torch.Tensor:
        """Control values of the test split."""
        return self.test_split.g_values


def _load_split(
    arrays: dict[str, Any],
    aux: dict[str, Any],
    split_name: str,
    g_indices: np.ndarray,
    intended_indices: np.ndarray,
) -> TFIMSplitBundle:
    """Slice one split out of the loaded arrays (whole intended-IC blocks, selected g values)."""
    n_latent = int(arrays["n_latent_samples_per_initial_state"])
    flat_to_intended = np.asarray(arrays["flat_to_intended_index"], dtype=np.int64)
    flat_to_latent = np.asarray(arrays["flat_to_latent_index"], dtype=np.int64)

    sample_mask = np.isin(flat_to_intended, intended_indices)
    z = np.asarray(arrays["z"])[g_indices][:, :, sample_mask, :]  # (g, T, sample, 2*dim)
    n_g = z.shape[0]

    def t0_states(key: str) -> torch.Tensor:
        """Grouped t=0 states of the selected intended ICs as (g, sample, 2*dim) in [p, q] order, repeated over g."""
        grouped = np.asarray(aux[key])[intended_indices]  # (intended, latent, 2*dim)
        flat = grouped.reshape(intended_indices.size * n_latent, -1)
        return _to_pipeline_order(flat).unsqueeze(0).expand(n_g, -1, -1).clone()

    g_values = np.asarray(arrays["g_values"], dtype=np.float32)[g_indices]
    return TFIMSplitBundle(
        split_name=split_name,
        trajectories=_to_pipeline_order(z),
        input_z0=t0_states("observed_z0_grouped"),
        latent_z0=t0_states("latent_z0_grouped"),
        g_values=torch.from_numpy(g_values.copy()),
        provenance=TFIMSplitProvenance(
            split_name=split_name,
            n_intended_initial_states=int(intended_indices.size),
            n_latent_samples_per_initial_state=n_latent,
            selected_intended_initial_state_indices=intended_indices.tolist(),
            flat_to_intended_index=flat_to_intended[sample_mask].tolist(),
            flat_to_latent_index=flat_to_latent[sample_mask].tolist(),
        ),
    )


def load_tfim_training_data(config: Any) -> TFIMTrainingData:
    """Load the TFIM train / validation / test splits selected by the config.

    ``k_samples`` / ``k_val_samples`` take consecutive g values of ``train.npz`` (train first,
    then validation), ``k_test_samples`` the first g values of ``test.npz``. ``n_samples`` /
    ``n_val_samples`` / ``n_test_samples`` count intended initial states (all of their latent
    samples are kept); validation uses the intended states following the training ones.
    """
    root = resolve_tfim_dataset_root(config)
    train_keys = (
        "z", "g_values", "n_intended_initial_states",
        "n_latent_samples_per_initial_state", "flat_to_intended_index", "flat_to_latent_index",
    )
    aux_keys = ("observed_z0_grouped", "latent_z0_grouped")
    train_arrays = _load_keys(root / "train.npz", train_keys)
    test_arrays = _load_keys(root / "test.npz", train_keys)
    train_aux = _load_keys(root / "train_aux.npz", aux_keys)
    test_aux = _load_keys(root / "test_aux.npz", aux_keys)

    def count(requested: int, available: int) -> int:
        """Requested count clipped to what is available."""
        return max(0, min(int(requested), int(available)))

    n_g_train_avail = len(train_arrays["g_values"])
    n_g_test_avail = len(test_arrays["g_values"])
    n_g_train = count(config.k_samples, n_g_train_avail)
    n_g_val = count(config.k_val_samples, n_g_train_avail - n_g_train)
    n_g_test = count(config.k_test_samples, n_g_test_avail)
    g_train = np.arange(0, n_g_train, dtype=np.int64)
    g_val = np.arange(n_g_train, n_g_train + n_g_val, dtype=np.int64)
    g_test = np.arange(0, n_g_test, dtype=np.int64)

    n_ic_train_avail = int(train_arrays["n_intended_initial_states"])
    n_ic_test_avail = int(test_arrays["n_intended_initial_states"])
    n_ic_train = count(config.n_samples, n_ic_train_avail)
    n_ic_val = count(config.n_val_samples, n_ic_train_avail - n_ic_train)
    n_ic_test = count(config.n_test_samples, n_ic_test_avail)
    ic_train = np.arange(n_ic_train, dtype=np.int64)
    ic_val = np.arange(n_ic_train, n_ic_train + n_ic_val, dtype=np.int64)
    ic_test = np.arange(n_ic_test, dtype=np.int64)

    return TFIMTrainingData(
        dataset_root=root,
        train_split=_load_split(train_arrays, train_aux, "train", g_train, ic_train),
        val_split=_load_split(train_arrays, train_aux, "val", g_val, ic_val),
        test_split=_load_split(test_arrays, test_aux, "test", g_test, ic_test),
    )
