"""
config.py
Experiment configuration dataclass: exactly the keys that the paper configs (configs/) set.
"""

from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

_DROPOUT_VARIANTS = {"mc_dropout", "concrete_dropout", "variational_dropout"}
_MODEL_TYPES = {"MassRatioSimpleHNN", "SeparableHNN", "TFIMSeparableHNN"}


@dataclass
class ExperimentConfig:
    """Hyperparameters of one training run (one system x arm x seed)."""

    # Identification
    system_type: str = "chain"  # "coupled_pendulum_mr" | "chain" | "tfim"
    dataset_index: str = "0"
    run_index: str = "0"
    seed: int = 42

    # Data: k_* count control cells (mass ratio / spring matrix / TFIM g value)
    n_samples: int = 900
    n_val_samples: int = 100
    n_test_samples: int = 100
    k_samples: int = 10
    k_val_samples: int = 10
    k_test_samples: int = 100

    # Model
    model_type: str = "SeparableHNN"  # "MassRatioSimpleHNN" | "SeparableHNN" | "TFIMSeparableHNN"
    n_hidden: int = 800
    nonlin: str = "intsin2x"  # activation (only "intsin2x" is implemented)
    state_dim: int = 0  # TFIM: canonical state size 2 * 2**L (used to rebuild the model); 0 = unset

    # Dropout. Arms: Deterministic (drop_prob 0), Fresh-mask MC (finite_mask_mode none),
    # Fixed-mask MC (finite_mask_mode both), Concrete, Variational.
    dropout_variant: str = "mc_dropout"
    drop_prob: float = 0.1
    mask_num: int = 100  # size of the mask bank (Fixed-mask MC) / number of MC samples at test time
    finite_mask_mode: str = "both"  # "none" (fresh masks) | "both" (finite bank)
    train_mask_resample: str = "per_rollout"  # "per_rollout" | "per_substep" (bank index per leapfrog sub-step)
    concrete_temp: float = 0.1
    scale_weights: bool = True
    mask_last_layer: bool = False  # mask the last MLP layer instead of the first
    finite_mask_storage: str = "materialized"  # "materialized" | "seeded" (bank regenerated from seed)
    kl_weight: float = 0.0  # KL weight for Concrete / Variational; 0 -> 1 / n_samples

    # Optimisation
    batch_size: int = 512
    n_epochs: int = 1000
    lr: float = 0.004
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    scheduler_patience: int = 1  # ReduceLROnPlateau on the validation MSE
    scheduler_factor: float = 0.8
    batch_k_matrices: bool = False  # roll out all control cells in one batch (needs batch_hamiltonian)
    gradual_horizon_increase: bool = True  # horizon curriculum, see Trainer.get_horizon
    curriculum_final_full_epochs: int = 0
    early_stop_patience: int = 0
    early_stop_min_epochs: int = 0
    best_checkpoint_requires_full_horizon: bool = False

    # Horizons
    T: int = 30  # training rollout length
    T_val: int = 0  # validation rollout length (0 -> T_test)
    T_test: int = 100
    dt: float = 0.1
    T_init_seq: int = 10  # minimum curriculum horizon
    use_leapfrog: bool = True

    # Evaluation during training
    pred_samples: int = 10  # MC samples per prediction
    val_use_all_masks: bool = False  # use mask_num samples instead of pred_samples
    mc_eval_chunk_size: int = 0  # >0: chunk the MC-sample axis
    eval_sample_chunk_size: int = 0  # >0: chunk the sample axis (MSE-only evaluation)
    fast_eval_max_samples_per_k: int = 0  # >0: cap samples per cell for intermediate validation
    fast_eval_pred_samples: int = 0  # >0: fewer MC samples for intermediate validation
    test_freq: int = 50
    save_freq: int = 100
    compute_energy: bool = True
    compute_calibration: bool = True
    run_test_during_training: bool = True  # False: test metrics only at the last epoch

    data_dir: str = "./data"
    output_dir: str = "./outputs"

    def __post_init__(self):
        if self.model_type not in _MODEL_TYPES:
            raise ValueError(f"Unknown model_type {self.model_type!r}; expected one of {sorted(_MODEL_TYPES)}")
        if self.dropout_variant not in _DROPOUT_VARIANTS:
            raise ValueError(f"Unknown dropout_variant {self.dropout_variant!r}")
        if self.finite_mask_mode not in {"none", "both"}:
            raise ValueError(f"Unknown finite_mask_mode {self.finite_mask_mode!r}")
        if self.train_mask_resample not in {"per_rollout", "per_substep"}:
            raise ValueError(f"Unknown train_mask_resample {self.train_mask_resample!r}")
        if self.finite_mask_storage not in {"materialized", "seeded"}:
            raise ValueError(f"Unknown finite_mask_storage {self.finite_mask_storage!r}")
        if self.finite_mask_mode == "both" and self.dropout_variant != "mc_dropout":
            raise ValueError("finite_mask_mode='both' is only defined for mc_dropout")
        if self.finite_mask_storage == "seeded" and self.finite_mask_mode == "none":
            raise ValueError("finite_mask_storage='seeded' requires finite_mask_mode='both'")
        if self.early_stop_min_epochs < 0 or self.curriculum_final_full_epochs < 0:
            raise ValueError("early_stop_min_epochs and curriculum_final_full_epochs must be non-negative")
        if self.gradual_horizon_increase and self.curriculum_final_full_epochs <= 0:
            raise ValueError("gradual_horizon_increase requires curriculum_final_full_epochs > 0")
        if (self.mc_eval_chunk_size > 0 or self.eval_sample_chunk_size > 0) and not self.batch_k_matrices:
            raise ValueError("evaluation chunking requires batch_k_matrices")
        for name in ("mc_eval_chunk_size", "eval_sample_chunk_size",
                     "fast_eval_max_samples_per_k", "fast_eval_pred_samples"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def get_run_dir(self) -> Path:
        """Output directory of this run: <output_dir>/<system_type>_<dataset_index>_<run_index>."""
        return Path(self.output_dir) / f"{self.system_type}_{self.dataset_index}_{self.run_index}"

    def to_dict(self) -> dict:
        """All fields as a plain dict."""
        return asdict(self)

    def save(self, path: Path):
        """Write the resolved config as YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        """Build a config from a mapping; unknown keys raise ValueError."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown config key(s): {unknown}")
        return cls(**data)

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load a YAML config; unknown keys raise ValueError."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def __str__(self) -> str:
        return "\n".join(["ExperimentConfig:"] + [f"  {k}: {v}" for k, v in self.to_dict().items()])
