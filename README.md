# Trajectory-Consistent Dropout for Uncertainty Decomposition in Hamiltonian Neural Networks

Yuki Tachibana and Stephen J. Roberts, Department of Engineering Science, University of Oxford.

Reproduction code for the paper's tables and figures: data generation, training of all models, evaluation, the per-seed result
tables (`results/`), and the scripts that turn those tables into the paper's tables and figures. Models are Hamiltonian neural networks
for a coupled pendulum with mass-ratio conditioning (CP), a four-particle spring chain (SC) and a four-site transverse-field Ising
chain (TFIM), compared under six arms: Deterministic, Fresh-mask MC, Concrete, Variational, Fixed-mask MC and Deep ensemble.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # Python >= 3.10; PyTorch 2.5.1
```

## Quick check (seconds, from the shipped CSVs)

```bash
scripts/reproduce.sh --from-csv        # writes out/tables/T*.{csv,tex} and out/figures/fig*.{pdf,png}
pytest tests -m "not slow"             # paper numbers (T1, T6-T21), plotted figure series, unit and smoke tests
```

`tests/test_paper_numbers.py` compares every cell of the paper's data tables (T1, T6-T21, `tests/paper_tex/`) with the tables built from
`results/`, `tests/test_figures.py` the plotted series of Figs 2-6 and 8. The format of the CSVs in `results/` is described in
[results/README.md](results/README.md).

## Smoke run (a few minutes, CPU)

```bash
scripts/reproduce.sh --smoke [--workdir DIR]   # tiny datasets, 2-epoch trainings, one seed; results go to DIR/results, tables and figures to DIR/out
pytest tests/test_smoke_pipeline.py            # the same run as a test (marked slow)
```

It exercises every stage (data, training of all arms, all core and appendix evaluations, ledger, tables, figures) on tiny settings. The
numbers are meaningless; the shipped `results/` are never touched. Tables whose grid needs the full-size runs (hidden-parameter detection,
IC-noise sweep) are reported as skipped.

## Full run

```bash
scripts/reproduce.sh --gpus 0,1,2,3    # data -> train -> eval -> ledger -> tables -> figures; overwrites results/
scripts/reproduce.sh data train --gpus 0,1                # single stages; also: eval ledger tables figures
python scripts/10_train.py --dry-run --systems sc         # print the pending training commands
```

| script | what it does |
|---|---|
| `scripts/00_generate_data.py` | writes the CP, SC and TFIM datasets to `data/` |
| `scripts/10_train.py` | trains system x arm x seed (seeds 42-51) with `configs/<system>/<arm>.yaml`, the five deep-ensemble members (seeds 0-4) and the extra models of the appendix stages (mask-bank sizes); one job per GPU; finished runs (`best_model.pt`) are skipped |
| `scripts/20_evaluate.py` | core evaluation (CP/SC: IC ensemble with fitted and fixed noise scale, OOD sweep, Hamiltonian drift; TFIM: drift, nested ensemble with fixed and fitted preparation noise, Poisson bracket) and the appendix stages; stage outputs that exist are skipped |
| `scripts/30_build_ledger.py` | collects the stage outputs into `results/` (`--out DIR` writes elsewhere) |
| `scripts/40_build_tables.py`, `41_effect_sizes.py` | per-seed results -> tables (means, sd, alpha, Holm-adjusted paired tests, bold rule) |
| `scripts/50_build_figures.py` | per-seed results -> Figs 2-6 and 8 |

Run directories are `runs/<system>/<arm>/seed<N>/` (deep-ensemble members `runs/<system>/deep_ensemble/member<i>/`), logs in `runs/logs/`.
A single run can be started by hand: `python -m hnnuq.train --config configs/cp/fixed_mask.yaml --seed 42 --output_dir runs/cp/fixed_mask/seed42`
([configs/README.md](configs/README.md)).

### Compute estimate (approximate)

Per-seed training times on shared RTX A6000 GPUs (upper bounds): about 1.5-2.5 h (CP), 5-7 h (SC) and 2.1-2.5 h (TFIM). With 10 seeds and
five single-model arms per system, plus the deep-ensemble members, the TFIM per-rollout bank runs and the spring-chain mask-bank sizes
(M = 1, 5, 20, 50, 100; seeds 42-44), training is roughly 500-650 GPU-hours. Evaluation is about 50 GPU-hours (CP about 5 min per run,
SC 10-60 min, TFIM 10-15 min; the long-horizon, hidden-parameter and IC-noise stages add a few hours each). Four GPUs therefore take about
one week. The tables and figures from `results/` take seconds.

## Paper items

| paper item | stage(s) | `results/` file(s) | output |
|---|---|---|---|
| T1 main results (CP, SC) | `hnnuq.eval.cp/sc` ic_fitted, ic_fixed, ood, hdrift | `ledger/classical_per_seed.csv` | `out/tables/T1` |
| T6 IC-ensemble NLL gain over dropout-only (CP) | `hnnuq.eval.cp` ic_fitted, ic_fixed | `ledger/classical_per_seed.csv` | `T6` |
| T7, T9 noise conventions, full main table (CP, SC) | `hnnuq.eval.cp/sc` ic_fitted, ic_fixed | `ledger/classical_per_seed.csv` | `T7`, `T9` |
| T8, T11 TFIM noise conventions and results | `hnnuq.eval.tfim` drift, nested (fixed, fitted) | `ledger/tfim_per_seed.csv` | `T8`, `T11` |
| T10 coupled-pendulum OOD | `hnnuq.eval.cp` ood, hdrift | `ledger/classical_per_seed.csv` | `T10` |
| T12 Poisson-bracket residual (TFIM) | `hnnuq.eval.tfim_bracket` | `extra/poisson_bracket_per_seed.csv` | `T12` |
| T13 mask-bank size (SC) | `hnnuq.eval.mask_bank` (+ its training jobs) | `extra/mask_bank_per_seed.csv` | `T13` |
| T14 efficiency (CP) | `hnnuq.eval.efficiency` | `extra/efficiency_summary.csv` | `T14` |
| T15 hidden-parameter detection (CP) | `hnnuq.eval.hidden_param` | `extra/hidden_param_{A,B,C}_auroc.csv`, `extra/hidden_param_leakage_k0.csv` | `T15` |
| T16 variance decomposition | all ic / nested stages | `ledger/{classical,tfim}_per_seed.csv` | `T16` |
| T17 IC-noise sweep (CP) | `hnnuq.eval.ic_noise_sweep` | `extra/ic_noise_sweep_per_seed.csv` | `T17` |
| T18 Hamiltonian / norm and energy drift | `hnnuq.eval.cp/sc` hdrift, `hnnuq.eval.tfim` drift | `ledger/{classical,tfim}_per_seed.csv` | `T18` |
| T19 long-horizon energy drift | `hnnuq.eval.long_horizon` | `extra/long_horizon_per_seed.csv` | `T19` |
| T20, T21 effect sizes | `scripts/41_effect_sizes.py` | `ledger/{classical,tfim}_per_seed.csv` | `T20`, `T21`, `effect_sizes.csv` |
| Figs 2, 4 CP OOD decomposition | `hnnuq.eval.cp` ic_fitted | `figures/cp_ood_by_mass_ratio.csv` | `out/figures/fig2`, `fig4` |
| Fig 5 SC OOD | `hnnuq.eval.sc` ood | `figures/sc_ood_by_k.csv` | `fig5` |
| Figs 3, 6 TFIM decomposition | `hnnuq.eval.tfim nested --epsilon fitted` | `figures/tfim_by_g.csv` | `fig3`, `fig6` |
| Fig 8 SC energy drift | `hnnuq.eval.long_horizon` | `figures/sc_energy_drift_by_step.csv` | `fig8` |
| Fig 1 schematic | not generated | `assets/fig1_trajectory_consistent_dropout.pdf` | |
| Fig 7 | not reproduced | | |
| T2-T5 | text and hyperparameters (T4, T5 are the `configs/`) | | |

Tables are written as `out/tables/T<n>.csv` (one row per cell) and `T<n>.tex`; figures as `out/figures/fig<n>.{pdf,png}` with the plotted
series in `fig<n>_data.csv`.

## Definitions and seeds

- Training seeds 42-51 for every system and arm (10 runs per cell); the Deep ensemble is one five-member ensemble (member seeds 0-4) and
  appears with n = 1; the mask-bank-size, IC-noise-sweep seeds are 42-44.
- The epistemic fraction is alpha = mean(Var_epi) / (mean(Var_epi) + mean(Var_alea)), the ratio of the two variances after averaging over
  rollout steps t >= 1, evaluation cases and state dimensions. Var_epi is the mask variance averaged over initial-state draws, Var_alea the
  variance over initial-state draws of the mask mean. Variances use ddof = 1 for CP and SC and ddof = 0 for TFIM.
- Means and standard deviations in the tables (ddof = 1) are taken over seeds.
- TFIM data: each intended product state is prepared 20 times with isotropic state-preparation noise of strength 0.10 (fidelity 1/1.01 with
  the intended state); the model sees only the intended state, the targets follow the prepared (latent) states. The training intended states
  are the eight structured product states plus 248 random ones; the 256 test intended states are random product states from a separate seed,
  so no test state is a training state.
- TFIM nested ensemble: test state 0 is perturbed 20 times (strength 0.10, or the strength fitted on the validation split by minimising the
  Gaussian NLL over a 13-point grid on [0, 0.6]) and each perturbation is rolled out with 50 mask draws (the 5 members for the Deep
  ensemble). Every method is scored against the same 20 stored latent rollouts of that state, whatever the method and the strength.

## Reproducibility notes

- The Fresh-mask arms (Fresh-mask MC, Concrete, Variational) draw dropout masks from the global random number generator at evaluation;
  the Fresh-mask MC Poisson-bracket values (T12) and the Fresh-mask rows of the IC-noise sweep (T17) depend on RNG state that the paper runs
  did not seed, so a rerun reproduces them within the seed-to-seed scatter, not digit by digit. Concrete fresh-mask sampling is equal in
  distribution, not bitwise, to the sampler used for the paper.
- Spring-chain evaluation targets are drawn on the evaluation device; the paper values are reproduced on CUDA.
- Coupled-pendulum long-horizon rollouts (T19) always run on CPU, as in the paper: over 1000 chaotic steps float32 rounding differs
  between CPU and GPU.
- The efficiency timings (T14) are wall-clock measurements and depend on hardware, library versions and system load; the model counts,
  sample counts and storage ratios are exact. The Deep-ensemble row is five times a deterministic member.
- The Deep ensemble is a single 5-member ensemble, so its cells are one evaluation (n=1); its Hamiltonian drift uses all five members
  and involves no random draws.

## Tests and release check

```bash
pytest tests                  # includes the slow smoke run
scripts/release_check.sh      # private-path / name / e-mail scan, size and binary check, forbidden-term test, tests; --smoke adds the smoke run
```

## Citation and licence

Citation metadata is in `CITATION.cff`. The code is released under the MIT licence (`LICENSE`).
