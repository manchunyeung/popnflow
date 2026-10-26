# `pipeline/` — code behind the figures in *popGMM* (arXiv version)

Everything needed to regenerate the 13 PDFs in `popGMM_paper_arXiv.zip:Figures/`,
gathered from `production/` on 2026-10-26.

* **`figures/`** — one dedicated script per paper figure, named
  `fig_<paper file name>.py`. Each contains only what that figure needs. It reads
  the stored results (caches, CSVs, dynesty checkpoints) and writes the PDF.
  These are new, extracted from the multi-purpose production scripts; each
  docstring names the code it came from.
* **`data/`, `inference/`, `sbc/`, `plotting/`** — **verbatim copies** of the
  production code that made those stored results. `SOURCES.md5` maps each copy
  to its original, and `check_sources.sh` reports drift. Exception:
  `data/prep_paper_diagnostics_inputs.py` is new; it reconstructs a never-saved
  one-off conversion bit-for-bit (`--check`).
* Provenance is exact. Every paper PDF is byte-identical (md5) to a file still
  on disk, listed below.

## Figure map

| Paper file | Label | Script | Reads | Paper PDF is byte-identical to |
|---|---|---|---|---|
| `corner_sanity.pdf` | fig:proof_of_concept | `fig_corner_sanity.py` | nothing (runs the toy inference) | `results/figures/corner_sanity.pdf` |
| `mdc_neff_alpha_gamma.pdf` | fig:Neff_plot | `fig_mdc_neff_alpha_gamma.py` | `paper_diagnostics/neff_cache_cat000_100k.npz` | `sim_cat6_sharp_w8/paper_diagnostics/mdc_neff_alpha_gamma.pdf` |
| `mdc_xi_scan.pdf` | fig:xi_plot | `fig_mdc_xi_scan.py` | `paper_diagnostics/xi_cache_cat000_100k.npz` | `…/paper_diagnostics/mdc_xi_scan.pdf` |
| `mdc_variance_Nsamp.pdf` | fig:var | `fig_mdc_variance_Nsamp.py` | `variance_cache_3method_cat000_2e4.npz`, `fitvar_cache_cat000.npz` | `…/paper_diagnostics/mdc_variance_budget_left_numeric.pdf` |
| `pp_plot_per_parameter.pdf` | fig:pp_plot | `fig_pp_plot_per_parameter.py` | `sbc_suite/sbc_n10/pit/{met1,met3_cholfix,met4_cholfix}.npz` | `sbc_suite/sbc_n10/figures/coverage_pp_cholfix.pdf` |
| `pull_boxplot.pdf` | fig:pull_box | `fig_pull_boxplot.py` | `sim_cat6_sharp_w8/pit_bias_cat6_met124_nlive1000.csv` | `paper_figs/fig05_pull_boxplot_mdc_nlive1000.pdf` |
| `mdc_corner_pos.pdf` | fig:mdc_pos | `fig_mdc_corner_pos.py` | sim_cat6 cat000 checkpoints (`popfig.RUNS["mdc"]`) | `…/paper_diagnostics_nlive1000/mdc_0_corner_all.pdf` |
| `mdc_Rm1.pdf` | fig:mdc_mass_distributions | `fig_mdc_Rm1.py` | same + endo3 injections | `…/paper_diagnostics_nlive1000/mdc_Rm1_posteriors.pdf` |
| `mdc_Rz.pdf` | fig:mdc_mass_distributions | `fig_mdc_Rz.py` | same + endo3 injections | `…/paper_diagnostics_nlive1000/mdc_Rz_posteriors.pdf` |
| `gwtc3_corner_pos.pdf` | fig:gwtc3_pos | `fig_gwtc3_corner_pos.py` | `results/data/inference/gwtc3_69met{1,3,4}.pkl` | `results/figures/gwtc3_69_all_params/gwtc3_69_corner_all.pdf` |
| `gwtc3_Rm1.pdf` | fig:gwtc3_mass_distributions | `fig_gwtc3_Rm1.py` | same + endo3 injections | `…/gwtc3_69_all_params/gwtc3_Rm1_posteriors.pdf` |
| `gwtc3_Rz.pdf` | fig:gwtc3_mass_distributions | `fig_gwtc3_Rz.py` | same + endo3 injections | `…/gwtc3_69_all_params/gwtc3_Rz_posteriors.pdf` |
| `gmm_fit_m1.pdf` | fig:KS_AD_tests | `fig_gmm_fit_m1.py` | `gmm_ks_ad/ks_ad_stored.csv`, `null_calibration.npz` | `gmm_ks_ad/ks_ad_stored_m1.pdf` |

`figures/popfig.py` is the one shared module. The six posterior figures
(`fig_{mdc,gwtc3}_{corner_pos,Rm1,Rz}.py`) import it for:
* checkpoint loading;
* cosmology;
* the two population models;
* the R₀ normalisation.

All "mdc" figures are on **`sim_cat6_sharp_w8`**, our own GWMockCat catalog,
not Ray's MDC (retired); the name survives from the code.

## Quick start

```bash
cd production           # or anywhere, with POPNFLOW_PROD=<production dir>
bash pipeline/make_paper_figures.sh                     # all 13
bash pipeline/make_paper_figures.sh mdc_Rz pull_boxplot # a subset
bash pipeline/make_paper_figures.sh --compare <unzipped arXiv>/Figures   # pixel diff vs paper
bash pipeline/make_paper_figures.sh --paper-bug         # the published (buggy) variants, see below

# or one figure directly (paths are relative to production/):
python pipeline/figures/fig_mdc_xi_scan.py --out /tmp/mdc_xi_scan.pdf
```

Output goes to `pipeline/output/Figures/` under the paper's filenames
(git-ignored); logs go to `pipeline/output/logs/`. Nothing in the result
directories is written. Use `~/.conda/envs/pop/bin/python` (override with
`PYTHON=`); the cvmfs python has no jax.

Runtimes:
* Cache/CSV figures: seconds.
* Corners: ~10 s.
* sim_cat6 rate figures: ~1.5 min.
* GWTC-3 rate figures: ~6 min (pairing p(m1) is marginalised on a 300×300 grid for each of ~11k draws).
* Toy model: three nested-sampling runs.

## Verification (2026-10-26, against the arXiv PDFs at 100 dpi)

| Figures | Result |
|---|---|
| `mdc_neff_alpha_gamma`, `mdc_xi_scan`, `mdc_variance_Nsamp`, `pp_plot_per_parameter`, `pull_boxplot`, `gmm_fit_m1` | **pixel-identical** (0.000% differ) |
| `mdc_corner_pos`, `mdc_Rm1`, `gwtc3_corner_pos`, `gwtc3_Rm1` | same posteriors, medians and bands; 1.5–16% of pixels differ (histogram binning, axis ranges, faint trace lines) |
| `mdc_Rz`, `gwtc3_Rz` | as above, and the Ensemble band is now correct (see bugs) |
| `corner_sanity` | new toy realisation (the original was unseeded), with the three toy bugs fixed |

The posterior figures can never match pixel-for-pixel. The production
script resampled each dynesty run with `resample_equal()` and **no seed**
(`make_plots_met4.py:553,609,645`), so the paper shows one unrepeatable draw.
The new scripts seed it (`--seed`, default 0); re-running the original
script gives differences of the same size (1.6–17%).

## Bugs found in the code that made the published figures

Each fix is on by default in `figures/`; `--paper-bug` (on the script or the
driver) reproduces the published version.

1. **`mdc_Rz`, `gwtc3_Rz` — Ensemble band upper edge.**
   `make_plots_met4.py:934` draws `fill_between(z, p5_Rz3, p95_Rz4)`: the
   Ensemble band uses **Mixture's** 95th percentile. Medians and the other
   bands are unaffected. On GWTC-3 the published orange band visibly reaches
   Mixture's upper edge.
2. **`corner_sanity` — Ensemble proposal draw.** `sanity_test_methods.py:496`
   draws the GMM proposal as `μ + L⁻¹z` with Prec = LLᵀ. That has covariance
   (LᵀL)⁻¹, not the GMM's, while `log q` evaluates the intended mixture, so
   the importance weights are inconsistent. It is the same bug that biased
   met3/met4 β, which was fixed in `sim_cat_inference.py` on 2026-08-21. This
   toy predates the fix and was never rerun.
3. **`corner_sanity` — POP draws.** `make_eps(u)` ignores `u` for even N, so
   the m₁ and z population draws used the **same** normal quantiles. They were
   perfectly correlated instead of independent.
4. **`corner_sanity` — POP Jacobian.** POP draws θ from the population and
   averages p_e(θ_det), which needs no Jacobian. `sanity_test_methods.py:408`
   multiplies by J = (dd_L/dz)(1+z) anyway. The published run's log shows it:
   POP logZ −16.85 against −113.66 (PE) and −113.67 (Ensemble). The 96.8-nat
   gap is 10 events × log J at z ≈ 1 (≈ 9.7). J varies with z, so this tilts
   POP's redshift posterior rather than only shifting its evidence.

## Caveats a referee or co-author should know

* **GWTC-3 figures are the seed-69 Colab run of 2026-05-05.** It predates
  four fixes:
  - the Cholesky proposal fix;
  - the PE-bounds box;
  - the selection-Jacobian removal;
  - the spin `p_draw` marginalisation.

  `paper_figs/SOURCES.md` measures the last of these as moving γ from 1.54
  to 1.90. The later nlive-2000 reruns in `colab_gwtc3/` carry the fixes; the
  paper does not show them.
* **Rate normalisation (all four R figures).** R₀ = N_obs / (ε V T), with
  ε taken from the **LVK endo3 injections**. That holds for the sim_cat6
  figures too, even though those events were selected by GWMockCat's own
  injection campaign. There is also a constant spin factor, +log 0.25
  (GWTC-3) and −log 0.25 (sim_cat6). The sim_cat6 "True Distribution" goes
  through the same normaliser, so posterior vs truth is like-for-like. The
  absolute sim_cat6 scale (R₀ ≈ 7 Gpc⁻³ yr⁻¹) is not the simulation's own,
  though; the new scripts reproduce it as published.
* **PP plot run counts.** The PP figure uses 160 PE runs, but only 155
  Ensemble and 157 Mixture runs; the rest were unusable. The caption says
  160 catalogs for all methods.
* **Pull box plot.** It hides outliers (`showfliers=False`).
* **Mixture colour.** It is pink in the R(m₁)/R(z) plots and green in the
  corners.

## Layout and the full chain

```
figures/     stage 4  one script per paper figure (+ popfig.py)      <- start here
data/        stage 1  mock catalog + injections + per-event GMMs
inference/   stage 2  hierarchical inference (HTCondor + Colab), original toy script
sbc/         stage 3  simulation-based calibration suite (PP figure)
plotting/    stage 4' production scripts that WROTE the caches/CSVs figures/ reads
make_paper_figures.sh, check_sources.sh, SOURCES.md5
```

Stages 1–3 and `plotting/` are archived **for the record**. They take hours
to days, and their outputs already exist. Their `.sh`/`.sub` files hard-code
`/home/manchun.yeung/.../production` and call the scripts **there**, so run
`check_sources.sh` before relying on them.

### Stage 1 — data (`data/`)

Run from `production/`. Env `pop`, except the SNR grid, which needs `gppop` (lal).
`sim_cat6_sharp_w8` = sharp PL+peak, q-sampled mock PE, width calibrated at SNR 8.

```bash
gppop/bin/python data/sim_cat_01_build_snr_grid.py           # -> sim_cat/snr_grid_O3actual_H1L1_Xeff.h5 (needs mock-PE/sensitivity)
python data/sim_cat_02_validate_snr_grid.py
python data/sim_cat_03_generate_pool.py --outdir sim_cat6_sharp_w8 \
    --mass-model sharp --mass-ratio-sampling q --width-ref-snr 8.0 \
    --n-draw 1000000 --pool-size 276 --n-pe-samples 5000 --pe-oversample 20 --seed 20260805
python data/sim_cat_04_build_catalogs.py --indir sim_cat6_sharp_w8   # 50 x 69, seed 20260803, mdc_membership.npy
python data/sim_cat_05_injections.py --outdir sim_cat                # 75M drawn, rho_obs>9, seed 20260803
cp sim_cat/injections.h5 sim_cat6_sharp_w8/                          # byte-identical across sim_cat builds
python data/sim_cat_06_validate.py --indir sim_cat6_sharp_w8
python data/sim_cat_fit_event_gmms.py --pe-file sim_cat6_sharp_w8/pe_samples.h5 \
    --out-dir sim_cat6_sharp_w8 --n-jobs 32                          # event_gmms.h5, BIC K, used by met3/met4
python pipeline/data/prep_paper_diagnostics_inputs.py                # m1det/m2det/dL.txt + endo3-schema injections
```

These settings are recorded in the outputs' HDF5 attrs:
* seed 20260805, sharp mass model, `q` sampling;
* rho_obs > 9, width_ref_snr 8;
* 1M drawn → 867 detected → 276 in the pool;
* GWMockCat commit `e595d70`.

**`--pe-oversample 20` is inferred, not recorded.** The default of 8 is too
small for `stage3_stats.json`'s median ESS of 53,614 (more than 8 × 5000),
and 20 is the value noted for these builds.

### Stage 2 — inference (`inference/`)

| Result dir (under `sim_cat6_sharp_w8/`) | Method / settings | Submitted with |
|---|---|---|
| `inference/simcat_cat{i}_met1.h5`, `inference/simcatnm_cat{i}_met3.h5` | PE (met1), Ensemble (met3), nlive 1000, PE box 1.5/98.5 | `run_sim_cat.sub` → `run_sim_cat_catalog.sh` → `sim_cat_inference.py` |
| `inference_met4_nlive200/` | Mixture (met4), nlive 200 | `run_sim_cat_met4_nlive200.sub` |
| `inference_boxfix_cholfix_nlive1000/` | met3, nlive 1000, PE box 0.02/99.98, Cholesky fix | `run_sim_cat_boxfix_cholfix_nlive1000.sub` + `boxfix_cholfix_simcat_nlive1000_jobs.txt` |
| `inference_met4_nlive1000_boxfix_cholfix/` | met4, nlive 1000, same fixes | same submit file (met4 rows, outtag `boxfix_cholfix`) |

Which figure uses which:
* N_eff/ξ: met1 + the *original* met3 (`inference/`).
* Variance: also met4 nlive 200.
* Pull box plot and sim_cat6 corner/rates: met1 + the nlive-1000 boxfix+cholfix met3/met4.

The submit files are for HTCondor on the LDG cluster: Claude prepares them,
Simon submits them.

GWTC-3 (`results/data/inference/gwtc3_69met{1,3,4}.pkl`, 2026-05-05) was run
on **Colab GPU** with `inference/gwtc3_mdc_69clean.ipynb`, the pairing model,
seed 69, against `endo3_bbhpop-LIGO-T2100113-v12.hdf5`. The notebook saves
`69method{1,3,4}.pkl` to Drive; the files were renamed on download.

`inference/sanity_test_methods.py` is the original toy script. `fig_corner_sanity.py`
supersedes it for the figure.

### Stage 3 — SBC suite (`sbc/`), for the PP figure

20 prior-drawn Λ blocks × 8 catalogs of 69 from a 276-event pool each:

```bash
python sbc/sbc_00_draw_hyperparams.py --n-blocks 20 --outdir sbc_n10 --master-seed 20260815
python sbc/make_sbc_dag.py ...          # sbc.dag: gen + met1 (submit_sbc_gen.sub / submit_sbc_infer.sub)
condor_submit submit_sbc_boxfix_cholfix.sub   # met3/met4 cholfix runs (boxfix_cholfix_jobs.txt)
python sbc/sbc_02_extract_pit.py ...    # -> sbc_n10/pit/{met1,met3_cholfix,met4_cholfix}.npz
python sbc/sbc_16_coverage_pp.py --suite sbc_n10 \
    --methods met1 met3_cholfix met4_cholfix --out coverage_pp_cholfix   # + bootstrap bands / p-values (CSV)
```

Two notes on this suite:
* `sbc_suite/sbc_n10/` was later extended to 40 blocks. The paper's figure
  uses the **20-block** PIT files (`pit/*_cholfix.npz`, not `*_n40.npz`).
* The 20-block PIT files were extracted before 5 met3 and 3 met4
  straggler runs finished.

### Stage 4′ — how the caches and CSVs that `figures/` reads were made (`plotting/`)

| Input | Made by |
|---|---|
| `paper_diagnostics/neff_cache_cat000_100k.npz`, `xi_cache_cat000_100k.npz` | `make_diagnostic_plots.py --run mdc --catalog 0 --indir sim_cat6_sharp_w8/inference --sampler-pattern "simcat_cat{i:03d}_{method}.h5" --method-patterns "met3=simcatnm_cat{i:03d}_{method}.h5" --pe-m1det/--pe-m2det/--pe-dL sim_cat6_sharp_w8/paper_diagnostics/{m1det,m2det,dL}.txt --injection-file sim_cat6_sharp_w8/paper_diagnostics/injections_endo3schema.h5 --xi-gmm-sizes 100000 300000 --neff-results-cache … --xi-results-cache … --make-neff-scan-plot --make-xi-scan-plot` (log `run_neff_xi_100k.log`) |
| `paper_diagnostics/variance_cache_3method_cat000_2e4.npz` | same base flags + `--indir sim_cat6_sharp_w8/paper_diagnostics_indir3 --n-pool 1400000 --variance-nsamp-max 20000 --make-variance-plot` (≈ 50 min; indir3 = symlinks to met1/met3/met4-nlive200) |
| `paper_diagnostics/fitvar_cache_cat000.npz` | same base flags + `--make-fitvar-plot --pool-cache …/pool_fitvar_cat000.npz --fitvar-results-cache …` at defaults (200 Λ from the met3 posterior, n_boot 50, pool 20000; log `run_fitvar.log`) |
| `pit_bias_cat6_met124_nlive1000.csv` | `pp_bias_diagnostics.py extract --data-dir sim_cat6_sharp_w8 --n-catalogs 50 --methods met1 met3 met4 --method-patterns "met1=inference/simcat_cat{i:03d}_met1.h5" "met3=inference_boxfix_cholfix_nlive1000/simcatnm_cat{i:03d}_met3.h5" "met4=inference_met4_nlive1000_boxfix_cholfix/simcat_cat{i:03d}_met4.h5" --out sim_cat6_sharp_w8/pit_bias_cat6_met124_nlive1000.csv` |
| `gmm_ks_ad/ks_ad_stored.csv`, `null_calibration.npz` | `plotting/rerun_ks_ad.py --mode stored` (imports `test_gmm_fit.py`) and `gmm_ks_ad/calibrate_thresholds.py`; run them from `production/gmm_ks_ad/`, where their `HERE`-relative paths resolve |

`plot_variance_budget_left_numeric.py` and `replot_ks_ad.py` are the earlier
plotting entry points, kept for the record. `paper_figs/SOURCES.md` has the
full history of each figure, including rejected variants.
