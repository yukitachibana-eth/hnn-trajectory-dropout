"""Exact dense solver of the open-boundary transverse-field Ising chain.

H(g) = -J sum_i Z_i Z_{i+1} - g sum_i X_i on L qubits, computational basis in big-endian bit
order (site 0 is the most significant bit), dimension 2**L. A wavefunction psi = q + i p is stored
as the canonical real vector z = concat(q, p); the model's rollout layout is concat(p, q).

This module provides what the data generator and the evaluation stages need: the Hamiltonian
matrix, the product-state repertoire, state-preparation perturbations (isotropic noise
in the tangent space of the state), and exact time evolution by diagonalisation.
"""

from typing import List, Sequence

import numpy as np

TFIM_J = 1.0


def _site_bitmask(L: int, site: int) -> int:
    return 1 << (L - 1 - site)


def _basis_spin(basis_index: int, L: int, site: int) -> int:
    return 1 if (basis_index & _site_bitmask(L, site)) == 0 else -1


def hamiltonian_matrix(L: int, g: float, J: float = TFIM_J) -> np.ndarray:
    """Dense complex128 Hamiltonian (2**L, 2**L): diagonal -J sum Z Z plus -g on every single-bit flip."""
    dim = 1 << L
    h = np.zeros((dim, dim), dtype=np.complex128)
    for basis_index in range(dim):
        diag_energy = 0.0
        for site in range(L - 1):
            diag_energy += -J * _basis_spin(basis_index, L, site) * _basis_spin(basis_index, L, site + 1)
        h[basis_index, basis_index] = diag_energy
        for site in range(L):
            h[basis_index ^ _site_bitmask(L, site), basis_index] += -g
    return h


# ---------------------------------------------------------------------------------- state conversions


def complex_to_canonical(psi: np.ndarray) -> np.ndarray:
    """psi = q + i p  ->  z = concat(q, p) along the last axis (float64)."""
    psi = np.asarray(psi, dtype=np.complex128)
    return np.concatenate([psi.real, psi.imag], axis=-1)


def canonical_to_complex(z: np.ndarray) -> np.ndarray:
    """z = concat(q, p)  ->  psi = q + i p (complex128)."""
    z = np.asarray(z, dtype=np.float64)
    half = z.shape[-1] // 2
    return z[..., :half] + 1j * z[..., half:]


def canonical_to_rollout(z: np.ndarray) -> np.ndarray:
    """Swap the halves of the last axis: canonical concat(q, p) <-> rollout layout concat(p, q)."""
    half = z.shape[-1] // 2
    return np.concatenate([z[..., half:], z[..., :half]], axis=-1)


def rollout_to_complex(z_pq: np.ndarray) -> np.ndarray:
    """Rollout layout concat(p, q) -> complex wavefunction."""
    return canonical_to_complex(canonical_to_rollout(np.asarray(z_pq, dtype=np.float64)))


def normalize_state(psi: np.ndarray) -> np.ndarray:
    """psi / ||psi|| for one state vector."""
    psi = np.asarray(psi, dtype=np.complex128)
    return psi / np.linalg.norm(psi)


def norm_squared(z_pq: np.ndarray) -> np.ndarray:
    """||psi||^2 of rollout-layout states, shape (..., 2*dim) -> (...)."""
    return np.sum(np.abs(rollout_to_complex(z_pq)) ** 2, axis=-1)


def energy_expectation(z_pq: np.ndarray, h_matrix: np.ndarray) -> np.ndarray:
    """<psi|H|psi> of rollout-layout states, shape (..., 2*dim) -> (...)."""
    psi = rollout_to_complex(z_pq)
    return np.einsum("...i,ij,...j->...", psi.conj(), np.asarray(h_matrix, dtype=np.complex128), psi).real


# ---------------------------------------------------------------------------------- initial states


def _local_state(kind: str) -> np.ndarray:
    sqrt2 = float(np.sqrt(2.0))
    table = {
        "up": np.array([1.0, 0.0], dtype=np.complex128),
        "down": np.array([0.0, 1.0], dtype=np.complex128),
        "plus": np.array([1.0, 1.0], dtype=np.complex128) / sqrt2,
        "minus": np.array([1.0, -1.0], dtype=np.complex128) / sqrt2,
    }
    return table[kind]


def _kron_product(vectors: Sequence[np.ndarray]) -> np.ndarray:
    state = np.array([1.0], dtype=np.complex128)
    for vector in vectors:
        state = np.kron(state, np.asarray(vector, dtype=np.complex128))
    return state


def _product_state(kinds: Sequence[str]) -> np.ndarray:
    return normalize_state(_kron_product([_local_state(k) for k in kinds]))


def product_state_repertoire(L: int, n_states: int, seed: int, named_states: bool = True) -> np.ndarray:
    """(n_states, 2**L) normalised product states: the 8 structured states (all up, all down, two
    Neel states, two domain walls, |+>^L, |->^L; omitted if ``named_states`` is False), then Haar-random
    single-qubit product states drawn from ``default_rng(seed)`` (two complex normals per qubit, normalised)."""
    half = L // 2
    structured = [
        ["up"] * L,
        ["down"] * L,
        ["up" if s % 2 == 0 else "down" for s in range(L)],
        ["down" if s % 2 == 0 else "up" for s in range(L)],
        ["up"] * half + ["down"] * (L - half),
        ["down"] * half + ["up"] * (L - half),
        ["plus"] * L,
        ["minus"] * L,
    ]
    states: List[np.ndarray] = []
    for kinds in structured if named_states else []:
        if kinds == ["up"] * L:
            basis = np.zeros(1 << L, dtype=np.complex128)
            basis[0] = 1.0
            states.append(basis)  # |0...0> exactly
        elif kinds == ["down"] * L:
            basis = np.zeros(1 << L, dtype=np.complex128)
            basis[-1] = 1.0
            states.append(basis)
        else:
            states.append(_product_state(kinds))
    rng = np.random.default_rng(seed)
    while len(states) < n_states:
        locals_ = []
        for _ in range(L):
            sampled = rng.normal(size=2) + 1j * rng.normal(size=2)
            locals_.append(sampled / np.linalg.norm(sampled))
        states.append(normalize_state(_kron_product(locals_)))
    return np.stack(states[:n_states], axis=0)


# ---------------------------------------------------------------------------------- perturbations


def perturb_state(psi: np.ndarray, epsilon: float, seed: int) -> np.ndarray:
    """Isotropic state-preparation noise: (psi + epsilon t) / sqrt(1 + epsilon^2), with t the unit vector
    along the component of a complex standard-normal draw (``default_rng(seed)``: real parts, then
    imaginary parts) orthogonal to psi. The fidelity with psi is exactly 1 / (1 + epsilon^2)."""
    psi = normalize_state(psi)
    if float(epsilon) == 0.0:
        return psi
    rng = np.random.default_rng(int(seed))
    eta = rng.normal(size=psi.shape[-1]) + 1j * rng.normal(size=psi.shape[-1])
    eta_perp = eta - np.vdot(psi, eta) * psi
    norm = float(np.linalg.norm(eta_perp))
    if norm <= 0.0:
        raise ValueError("noise draw has no component orthogonal to the state")
    tangent = eta_perp / norm
    return normalize_state((psi + float(epsilon) * tangent) / np.sqrt(1.0 + float(epsilon) ** 2))


def perturbed_ensemble(psi: np.ndarray, n_samples: int, epsilon: float, seed: int) -> np.ndarray:
    """(n_samples, 2**L) perturbed copies of psi; sample j uses noise seed ``seed + j``."""
    psi = normalize_state(psi)
    return np.stack([perturb_state(psi, epsilon, int(seed) + j) for j in range(int(n_samples))], axis=0)


# ---------------------------------------------------------------------------------- exact evolution


def evolve_exact(h_matrix: np.ndarray, psi0: np.ndarray, time_grid: Sequence[float]) -> np.ndarray:
    """Exact evolution psi(t) = exp(-i H t) psi0 by diagonalisation.

    Args:
        h_matrix: (dim, dim) Hamiltonian.
        psi0: (n, dim) initial states (each renormalised).
        time_grid: (T,) times.
    Returns:
        psi: (n, T, dim) complex128.
    """
    psi0 = np.stack([normalize_state(s) for s in np.asarray(psi0, dtype=np.complex128).reshape(-1, h_matrix.shape[0])])
    evals, evecs = np.linalg.eigh(np.asarray(h_matrix, dtype=np.complex128))
    coefficients = evecs.conj().T @ psi0.T
    psi = np.empty((psi0.shape[0], len(time_grid), h_matrix.shape[0]), dtype=np.complex128)
    for t_index, t in enumerate(time_grid):
        phase = np.exp(-1j * evals * float(t))
        psi[:, t_index, :] = (evecs @ (coefficients * phase[:, None])).T
    return psi


def evolve_exact_canonical(h_matrix: np.ndarray, psi0: np.ndarray, time_grid: Sequence[float]) -> np.ndarray:
    """Like ``evolve_exact`` but returns canonical z = concat(q, p), shape (n, T, 2*dim)."""
    return complex_to_canonical(evolve_exact(h_matrix, psi0, time_grid))
