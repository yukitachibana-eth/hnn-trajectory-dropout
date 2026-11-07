# Shipped results (inputs of `scripts/40_build_tables.py`)

All files are per-seed values (seeds 42-51 unless stated) written by the evaluation stages of the pipeline; the table builder recomputes means,
standard deviations (ddof=1), fitted scales, alpha, effect sizes and bold decisions from them. Running `scripts/30_build_ledger.py` after a
from-scratch run writes `ledger/*.csv` with the same schema.

## `ledger/classical_per_seed.csv`, `ledger/tfim_per_seed.csv`
One row per `system` (CP coupled pendulum, SC spring chain, TFIM) x `arm` (Deterministic, Fresh-mask MC, Concrete, Variational, Fixed-mask MC,
Deep ensemble) x `seed` x `quantity` x `split` (ID / OOD / empty) x `convention` (`fitted` = IC-noise scale fitted per run on validation data,
`fixed` = true scale 0.1 with measurement noise for CP/SC and epsilon 0.10 for TFIM, `dropout-only`, or empty). Feeds T1, T6-T11, T16, T18, T20, T21.

| quantity | meaning (unit) |
|---|---|
| `mse`, `state_rmse` | rollout MSE (CP, SC) / state RMSE of the predicted wavefunction (TFIM) |
| `ic_noise_scale` | fitted IC-noise scale sigma-hat / epsilon-hat (state units) |
| `g_nll` | Gaussian NLL per rollout element of the IC x mask ensemble (nats) |
| `cov90_err` | absolute deviation of the empirical 90% coverage from 0.9 |
| `nll_masks_only` | NLL of the mask ensemble alone plus measurement noise (CP, SC) |
| `var_epi`, `var_alea`, `var_ratio` | epistemic (mask) and aleatoric (IC) variance, averaged over rollout steps t>=1, test cases and dimensions; their ratio |
| `alpha` | epistemic fraction = mean var_epi / (mean var_epi + mean var_alea), with the same t>=1 averages |
| `mask_variance` | mean predictive variance across masks at the observed IC (T10 Var_epi column) |
| `h_drift`, `h_drift_final` | drift of the model's own sampled Hamiltonian, averaged over the rollout / at the last frame |
| `phys_e_drift` | absolute drift of the true energy along each sampled rollout, from the first predicted frame to the last step |
| `spearman_rho` | Spearman correlation between predictive variance and log distance of the OOD mass ratio from the training range |
| `norm_drift`, `energy_drift` | TFIM norm and energy drift of the predicted wavefunction |

Notes. TFIM Concrete and Variational rows come from runs trained with their KL term. The Deep ensemble is one 5-member ensemble, stored as a single
row (seed 42), so its cells are printed with n=1; its `h_drift` / `h_drift_final` use all five members.
Deterministic has no mask-ensemble quantities, hence no `var_epi`, `alpha`, `nll_masks_only` rows for CP/SC.

## `extra/`
- `mask_bank_per_seed.csv` (T13): spring chain, mask-bank sizes M in {1,5,20,50,100,200}, seeds 42-44; per-seed means over evaluation cases of the
  quantities named in the `quantity` column (`mse_*`, `nll_fit_*` / `nll_fix_*` = G-NLL with fitted / fixed sigma, `nll_mask_*`, `cov_fit_*`, `cov_fix_*`,
  `var_epi_ood`, `var_alea_ood`, `phys_drift_*` = physical energy drift, `h_drift`; suffix id/ood = split). M=1 has no mask-ensemble rows.
- `efficiency_summary.csv` (T14): measured relative train time, inference time and storage (relative to the deterministic HNN), model and sample counts.
  The deep-ensemble row is extrapolated from a measured deterministic member (five-member parallel assumption).
- `hidden_param_{A,B,C}_auroc.csv`, `hidden_param_leakage_k0.csv` (T15): per-seed AUROC of hidden-mass-ratio shift detection by revealed prefix length
  k; B scores the Hamiltonian drift along the prefix at anchors r0, A and C score predictive variance (held-out and training-grid mass ratios);
  the leakage file holds the k=0 per-sample variance scores and mismatch labels used for the pooled control AUROC.
- `ic_noise_sweep_per_seed.csv` (T17): coupled pendulum, seeds 42-44, per-seed means over mass ratios of Var_epi and Var_alea at 0-4x the fitted IC-noise scale.
- `long_horizon_per_seed.csv` (T19): relative physical energy drift |E(z_t)-E(z_1)|/|E(z_1)| at t=100 and t=1000 of 1000-step rollouts (10 masks), seeds 42-51.
- `poisson_bracket_per_seed.csv` (T12): TFIM per-seed mean log of the normalised Poisson-bracket residual over draws and the 1150 test states, for
  Deterministic, Fresh-mask MC and the two Fixed-mask banks (per-sub-step, per-rollout training).
