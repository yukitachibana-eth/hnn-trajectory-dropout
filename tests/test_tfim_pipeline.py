"""TFIM pipeline checks on a tiny dataset and untrained models (CPU, seconds): the exact solver conserves norm
and energy, the data generator is deterministic, and every evaluation stage plus the ledger writer runs end to end
for each rollout kind (deterministic, fresh masks, mask bank, deep ensemble)."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hnnuq.config import ExperimentConfig
from hnnuq.data import tfim as data_tfim
from hnnuq.eval import tfim as ev
from hnnuq.eval import tfim_bracket as br
from hnnuq.hnn_model import build_hnn_model
from hnnuq.ledger import tfim as led
from hnnuq.tfim import exact
from hnnuq.tfim.control import register_tfim_operator_structure

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "tfim"


def test_exact_solver_conserves_norm_and_energy():
    h = exact.hamiltonian_matrix(4, 1.3)
    assert np.allclose(h, h.conj().T)
    psi0 = exact.perturbed_ensemble(exact.product_state_repertoire(4, 3, 42)[2], 4, 0.1, 5)
    z = exact.canonical_to_rollout(exact.evolve_exact_canonical(h, psi0, np.linspace(0, 2, 11)))
    assert np.allclose(exact.norm_squared(z), 1.0, atol=1e-12)
    e = exact.energy_expectation(z, h)
    assert np.allclose(e, e[:, :1], atol=1e-10)


def test_preparation_noise_is_isotropic_with_known_fidelity():
    psi = exact.product_state_repertoire(4, 3, 42)[2]
    eps = 0.10
    batch = exact.perturbed_ensemble(psi, 50, eps, 7)
    assert np.allclose(np.linalg.norm(batch, axis=1), 1.0)
    assert np.allclose(np.abs(batch @ psi.conj()) ** 2, 1.0 / (1.0 + eps**2))
    assert np.array_equal(exact.perturbed_ensemble(psi, 50, 0.0, 7), np.tile(psi, (50, 1)))
    assert not np.allclose(batch[0], batch[1])


def test_test_states_are_not_training_states():
    train = exact.product_state_repertoire(4, 32, 42)
    test = exact.product_state_repertoire(4, 32, 4242, named_states=False)
    assert np.abs(test.conj() @ train.T).max() < 1.0 - 1e-6


@pytest.fixture(scope="module")
def smoke_data(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("tfim_data")
    data_tfim.main(["--out", str(out), "--smoke"])
    return out


def test_data_generation_is_deterministic(smoke_data, tmp_path):
    data_tfim.main(["--out", str(tmp_path), "--smoke"])
    for name in ("train", "test", "train_aux", "test_aux"):
        with np.load(smoke_data / f"{name}.npz") as a, np.load(tmp_path / f"{name}.npz") as b:
            assert all(np.array_equal(a[k], b[k]) for k in a.files)


def test_nested_targets_are_stored_latent_rollouts(smoke_data):
    test = ev.load_test_data(smoke_data)
    gi = 1
    target = ev.stored_latent_targets(test, gi)
    n_latent = int(np.sum(test.flat_to_intended == ev.BASE_STATE))
    assert target.shape == (n_latent, test.z.shape[1], test.z.shape[-1])
    # frame 0 of the targets are the latent initial states of the base intended state
    with np.load(smoke_data / "test_aux.npz") as f:
        latent = f["latent_z0_grouped"][ev.BASE_STATE].astype(np.float64)  # (latent, 2*dim) canonical
    assert np.allclose(target[:, 0], exact.canonical_to_rollout(latent), atol=1e-6)


def _make_run(root: Path, name: str, data: Path, seed: int = 0) -> Path:
    """Untrained model saved as a run directory of the given arm config (smoke-sized data selection)."""
    cfg = ExperimentConfig.load(CONFIGS / f"{name}.yaml")
    cfg.data_dir, cfg.seed, cfg.mask_num = str(data), seed, min(cfg.mask_num, 6)
    cfg.n_samples, cfg.n_val_samples, cfg.n_test_samples = 4, 2, 4
    cfg.k_samples, cfg.k_val_samples, cfg.k_test_samples = 2, 1, 4
    run = root / f"{name}_seed{seed}"
    run.mkdir(parents=True)
    cfg.save(run / "config.yaml")
    torch.manual_seed(seed)
    model = build_hnn_model(dim=16, n_hidden=cfg.n_hidden, drop_prob=cfg.drop_prob, mask_num=cfg.mask_num, nonlin=cfg.nonlin,
                            device=torch.device("cpu"), model_type=cfg.model_type, dropout_variant=cfg.dropout_variant,
                            finite_mask_mode=cfg.finite_mask_mode, concrete_temp=cfg.concrete_temp, scale_weights=cfg.scale_weights,
                            mask_last_layer=cfg.mask_last_layer, finite_mask_storage=cfg.finite_mask_storage, mask_seed=cfg.seed)
    register_tfim_operator_structure(model, 16)
    torch.save(model.state_dict(), run / "best_model.pt")
    return run


@pytest.mark.parametrize("arm", ["deterministic", "fresh_mask", "fixed_mask"])
def test_stages_and_ledger_smoke(arm, smoke_data, tmp_path):
    run = _make_run(tmp_path / "ckpt", arm, smoke_data)
    ev_dir = tmp_path / "runs" / arm / "seed0" / "eval"
    common = ["--run-dir", str(run), "--data", str(smoke_data), "--device", "cpu", "--smoke"]
    ev.main(["drift", *common, "--out", str(ev_dir / "drift")])
    ev.main(["nested", "--epsilon", "fixed", *common, "--out", str(ev_dir / "nested_fixed")])
    ev.main(["nested", "--epsilon", "fitted", *common, "--out", str(ev_dir / "nested_fitted")])
    br.main([*common, "--out", str(ev_dir / "bracket")])
    per_g = list(csv.DictReader(open(ev_dir / "nested_fitted" / "per_g.csv")))
    assert len(per_g) == 3 and {r["in_train_band"] for r in per_g} == {"True", "False"}
    assert all(np.isfinite(float(r["nll"])) and float(r["var_alea"]) > 0 for r in per_g)
    if arm == "deterministic":
        assert all(float(r["var_epi"]) == 0.0 for r in per_g)
    else:
        assert all(float(r["var_epi"]) > 0 for r in per_g)
    assert np.isfinite(json.loads((ev_dir / "bracket" / "summary.json").read_text())["mean_log_resid"])
    counts = led.build(tmp_path / "runs", tmp_path / "results")
    assert counts == {"ledger": 22, "figures": 0 if arm == "deterministic" else 3, "bracket": 1}


def test_deep_ensemble_smoke(smoke_data, tmp_path):
    members = [_make_run(tmp_path / "ckpt", "deep_ensemble", smoke_data, seed=s) for s in range(3)]
    out = tmp_path / "eval"
    common = ["--run-dir", *map(str, members), "--data", str(smoke_data), "--device", "cpu", "--smoke"]
    ev.main(["nested", "--epsilon", "fitted", *common, "--out", str(out)])
    rows = list(csv.DictReader(open(out / "per_g.csv")))
    assert all(int(r["n_mc"]) == 3 and float(r["var_epi"]) > 0 for r in rows)
