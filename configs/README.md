# Paper configs

One YAML per system and arm. Every key maps to a field of `hnnuq.config.ExperimentConfig`
(unknown keys raise `ValueError`). `seed`, `output_dir`, `device` and any other field can be
overridden on the command line, e.g. `python -m hnnuq.train --config configs/cp/fixed_mask.yaml --seed 42 --mask_num 50`.
`data_dir` is relative to the repository root.

| config | arm |
|---|---|
| `cp/deterministic.yaml`, `sc/deterministic.yaml`, `tfim/deterministic.yaml` | Deterministic (`drop_prob 0`) |
| `cp/fresh_mask.yaml`, `sc/fresh_mask.yaml`, `tfim/fresh_mask.yaml` | Fresh-mask MC (`finite_mask_mode none`, M=200 samples at test time) |
| `cp/concrete.yaml`, `sc/concrete.yaml`, `tfim/concrete.yaml` | Concrete dropout |
| `cp/variational.yaml`, `sc/variational.yaml`, `tfim/variational.yaml` | Variational dropout |
| `cp/fixed_mask.yaml`, `sc/fixed_mask.yaml`, `tfim/fixed_mask.yaml` | Fixed-mask MC: bank of M=200 masks on the last layer, bank index resampled at every leapfrog sub-step in training, one mask per rollout at inference |
| `cp/deep_ensemble.yaml`, `sc/deep_ensemble.yaml`, `tfim/deep_ensemble.yaml` | Deep-ensemble member (deterministic; members are seeds 0..4, trained by `scripts/10_train.py` into `runs/<sys>/deep_ensemble/member<i>/`) |
| `tfim/fixed_mask_per_rollout.yaml` | Fixed-mask MC with a per-rollout mask bank (index fixed over the whole training rollout); used for the Poisson-bracket table |
| `sc/fixed_mask_m1_per_rollout.yaml` | Per-rollout mask bank with M=1; the M=1 row of the mask-bank-size sweep (rows M=5..100 use `sc/fixed_mask.yaml --mask_num M`) |

`cp/efficiency/<arm>.yaml` are the tiny-geometry training configs of the efficiency benchmark (Table 14), read by `hnnuq.eval.efficiency`.

Systems: `cp` = coupled pendulum with mass-ratio conditioning, `sc` = 4-particle spring chain
(`SeparableHNN`), `tfim` = L=4 transverse-field Ising chain (`TFIMSeparableHNN`, operator-separable
backbone, seeded finite-mask banks, horizon curriculum).

Smoke run (a few epochs, short horizons) of any config:
`python -m hnnuq.train --config configs/cp/fixed_mask.yaml --seed 0 --n_epochs 2 --n_samples 128 --n_val_samples 16 --T_val 10 --T_test 10 --n_test_samples 8 --output_dir runs/smoke`.
