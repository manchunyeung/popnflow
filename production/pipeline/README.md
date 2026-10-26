# `pipeline/` — code behind the figures in *popGMM* (arXiv version)

Everything needed to regenerate the 13 PDFs in `popGMM_paper_arXiv.zip:Figures/`,
gathered from `production/` on 2026-10-26.

* Every script here is a **verbatim copy** of the one that produced the paper
  figure. `SOURCES.md5` maps each copy to its original; `check_sources.sh`
  reports any original that has changed since.
* Exceptions:
  - `data/prep_paper_diagnostics_inputs.py` is **new**. It reconstructs a
    one-off conversion that was never saved, and it reproduces the files on
    disk bit-for-bit (`--check`).
  - `make_paper_figures.sh` is **new**: the figure driver.
* Provenance is exact. Every paper PDF is byte-identical (md5) to a file
  that is still on disk; the table below names it.

## Figure map

| Paper file | Paper label | Byte-identical source on disk | Generator (`plotting/` unless noted) |
|---|---|---|---|
| `corner_sanity.pdf` | fig:proof_of_concept | `results/figures/corner_sanity.pdf` | `inference/sanity_test_methods.py --tag sanity` |
| `mdc_neff_alpha_gamma.pdf` | fig:Neff_plot | `sim_cat6_sharp_w8/paper_diagnostics/mdc_neff_alpha_gamma.pdf` | `make_diagnostic_plots.py --make-neff-scan-plot` |
| `mdc_xi_scan.pdf` | fig:xi_plot | `sim_cat6_sharp_w8/paper_diagnostics/mdc_xi_scan.pdf` | `make_diagnostic_plots.py --make-xi-scan-plot` (same call) |
| `mdc_variance_Nsamp.pdf` | fig:var | `sim_cat6_sharp_w8/paper_diagnostics/mdc_variance_budget_left_numeric.pdf` | `plot_variance_budget_left_numeric.py` |
| `pp_plot_per_parameter.pdf` | fig:pp_plot | `sbc_suite/sbc_n10/figures/coverage_pp_cholfix.pdf` | `sbc/sbc_16_coverage_pp.py` |
| `pull_boxplot.pdf` | fig:pull_box | `paper_figs/fig05_pull_boxplot_mdc_nlive1000.pdf` | `pp_bias_diagnostics.py plot` |
| `mdc_corner_pos.pdf` | fig:mdc_pos | `sim_cat6_sharp_w8/paper_diagnostics_nlive1000/mdc_0_corner_all.pdf` | `make_plots_met4.py --run mdc` |
| `mdc_Rm1.pdf`, `mdc_Rz.pdf` | fig:mdc_mass_distributions | `…/paper_diagnostics_nlive1000/mdc_R{m1,z}_posteriors.pdf` | same call |
| `gwtc3_corner_pos.pdf` | fig:gwtc3_pos | `results/figures/gwtc3_69_all_params/gwtc3_69_corner_all.pdf` | `make_plots_met4.py --run gwtc3` |
| `gwtc3_Rm1.pdf`, `gwtc3_Rz.pdf` | fig:gwtc3_mass_distributions | `…/gwtc3_69_all_params/gwtc3_R{m1,z}_posteriors.pdf` | same call |
| `gmm_fit_m1.pdf` | fig:KS_AD_tests | `gmm_ks_ad/ks_ad_stored_m1.pdf` | `replot_ks_ad.py --mode stored` |

All "MDC" figures in the paper are on **`sim_cat6_sharp_w8`**, our own
GWMockCat catalog, not Ray's MDC (it is retired). `make_diagnostic_plots.py`
takes `--run mdc` for it; the flag picks the likelihood branch, not the
dataset, so every path default must be overridden, as the driver does.

## Quick start: regenerate the figures

```bash
cd production           # or anywhere, with POPNFLOW_PROD=<production dir>
bash pipeline/make_paper_figures.sh                  # all but the toy model
bash pipeline/make_paper_figures.sh --sanity         # + toy model (runs dynesty)
bash pipeline/make_paper_figures.sh pp mdc           # a subset
bash pipeline/make_paper_figures.sh --compare <unzipped arXiv>/Figures   # pixel diff
```

Output goes to `pipeline/output/Figures/` under the paper's filenames
(git-ignored); logs go to `pipeline/output/logs/`. The driver only **reads**
the result directories. The two scripts that write next to their inputs
(`sbc_16`, `make_plots_met4.py`) are pointed at staging dirs under `output/`.
Python is `~/.conda/envs/pop/bin/python` (override with `PYTHON=`); the
cvmfs python has no jax.

The data stay where they are. The worktree/branch carries code only; the
driver `cd`s to the data root (the parent of `pipeline/` if it holds
`sim_cat6_sharp_w8/`, else `$POPNFLOW_PROD`).

## Layout and the full chain

```
data/        stage 1  mock catalog + injections + per-event GMMs
inference/   stage 2  hierarchical inference (HTCondor + Colab) and the toy model
sbc/         stage 3  simulation-based calibration suite (PP figure)
plotting/    stage 4  figure scripts
make_paper_figures.sh   stage 4 driver
```

Stages 1–3 are archived **for the record**. They take days of cluster
time, and their outputs already exist. Their `.sh`/`.sub` files hard-code
`/home/manchun.yeung/.../production` and call the scripts **there**, not
these copies, so run `check_sources.sh` before relying on them.

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

Recorded in the outputs' HDF5 attrs: seed 20260805, sharp, `q`, rho_obs > 9,
width_ref_snr 8, 1M drawn → 867 detected → 276 pool, GWMockCat commit
`e595d70`. **`--pe-oversample 20` is inferred, not recorded.** The default of
8 is too small for `stage3_stats.json`'s median ESS of 53,614 (more than
8 × 5000), and 20 is the value noted for these builds.

### Stage 2 — inference (`inference/`)

| Result dir (under `sim_cat6_sharp_w8/`) | Method / settings | Submitted with |
|---|---|---|
| `inference/simcat_cat{i}_met1.h5`, `inference/simcatnm_cat{i}_met3.h5` | PE (met1), Ensemble (met3), nlive 1000, PE box 1.5/98.5 | `run_sim_cat.sub` → `run_sim_cat_catalog.sh` → `sim_cat_inference.py` |
| `inference_met4_nlive200/` | Mixture (met4), nlive 200 | `run_sim_cat_met4_nlive200.sub` |
| `inference_boxfix_cholfix_nlive1000/` | met3, nlive 1000, PE box 0.02/99.98, Cholesky fix | `run_sim_cat_boxfix_cholfix_nlive1000.sub` + `boxfix_cholfix_simcat_nlive1000_jobs.txt` |
| `inference_met4_nlive1000_boxfix_cholfix/` | met4, nlive 1000, same fixes | same submit file (met4 rows, outtag `boxfix_cholfix`) |

Which figure uses which:
* N_eff/xi: met1 + the *original* met3 (`inference/`).
* Variance: also met4 nlive 200.
* Pull box plot and MDC corner/rates: met1 + the nlive-1000 boxfix+cholfix met3/met4.

The submit files are for HTCondor on the LDG cluster: Claude prepares them,
Simon submits them.

GWTC-3 (`results/data/inference/gwtc3_69met{1,3,4}.pkl`, 2026-05-05) was run
on **Colab GPU** with `inference/gwtc3_mdc_69clean.ipynb`, the pairing model,
seed 69, against `endo3_bbhpop-LIGO-T2100113-v12.hdf5`. The notebook saves
`69method{1,3,4}.pkl` to Drive; the files were renamed on download.

The toy model: `sanity_test_methods.py --outdir results/figures --tag sanity`
(as in `run_once.sh`) runs its own PE/POP/Ensemble inference and plots the corner.

### Stage 3 — SBC suite (`sbc/`), for the PP figure

20 prior-drawn Λ blocks × 8 catalogs of 69 from a 276-event pool each:

```bash
python sbc/sbc_00_draw_hyperparams.py --n-blocks 20 --outdir sbc_n10 --master-seed 20260815
python sbc/make_sbc_dag.py ...          # sbc.dag: gen + met1 (submit_sbc_gen.sub / submit_sbc_infer.sub)
condor_submit submit_sbc_boxfix_cholfix.sub   # met3/met4 cholfix runs (boxfix_cholfix_jobs.txt)
python sbc/sbc_02_extract_pit.py ...    # -> sbc_n10/pit/{met1,met3_cholfix,met4_cholfix}.npz
python sbc/sbc_16_coverage_pp.py --suite sbc_n10 \
    --methods met1 met3_cholfix met4_cholfix --out coverage_pp_cholfix
```

Two notes on this suite:
* `sbc_suite/sbc_n10/` was later extended to 40 blocks. The paper's figure
  uses the **20-block** PIT files (`pit/*_cholfix.npz`, not `*_n40.npz`).
* The 20-block PIT files were extracted before 5 met3 and 3 met4
  straggler runs finished.

`sbc_03_bands.py` (PIT PP bands) is included because `sbc_16` shares its
bootstrap machinery and the README there describes both.

### Stage 4 — figure-specific inputs that are caches, and how they were made

| Cache | Made by |
|---|---|
| `paper_diagnostics/neff_cache_cat000_100k.npz`, `xi_cache_cat000_100k.npz` | the `neff_xi` call without the cache files present (≈ minutes) |
| `paper_diagnostics/variance_cache_3method_cat000_2e4.npz` | `make_diagnostic_plots.py … --make-variance-plot --n-pool 1400000 --variance-nsamp-max 20000 --indir sim_cat6_sharp_w8/paper_diagnostics_indir3` (≈ 50 min; indir3 = symlinks to met1/met3/met4-nlive200) |
| `paper_diagnostics/fitvar_cache_cat000.npz` | `make_diagnostic_plots.py … --make-fitvar-plot --pool-cache sim_cat6_sharp_w8/paper_diagnostics/pool_fitvar_cat000.npz --fitvar-results-cache …` at defaults (200 Λ from the met3 posterior, n_boot 50, pool 20000); log `run_fitvar.log` |
| `pit_bias_cat6_met124_nlive1000.csv` | `pp_bias_diagnostics.py extract --data-dir sim_cat6_sharp_w8 --n-catalogs 50 --methods met1 met3 met4 --method-patterns "met1=inference/simcat_cat{i:03d}_met1.h5" "met3=inference_boxfix_cholfix_nlive1000/simcatnm_cat{i:03d}_met3.h5" "met4=inference_met4_nlive1000_boxfix_cholfix/simcat_cat{i:03d}_met4.h5" --out sim_cat6_sharp_w8/pit_bias_cat6_met124_nlive1000.csv` |
| `gmm_ks_ad/ks_ad_stored.csv`, `null_calibration.npz` | `plotting/rerun_ks_ad.py --mode stored` (imports `test_gmm_fit.py`) and `gmm_ks_ad/calibrate_thresholds.py`; run them from `production/gmm_ks_ad/`, where their `HERE`-relative paths resolve |

`paper_figs/SOURCES.md` has the full history of each figure, including
rejected variants and the bugs found along the way.

## Caveats a referee or co-author should know

* **The GWTC-3 figures are the seed-69 run of 2026-05-05.** It predates
  four fixes: the Cholesky proposal fix, the PE-bounds box, the
  selection-Jacobian removal and the spin `p_draw` marginalisation.
  `SOURCES.md` measures the last of these as moving γ from 1.54 to 1.90.
  The later nlive-2000 Colab reruns (`colab_gwtc3/`) carry the fixes; they
  are **not** what the paper shows.
* `make_plots_met4.py` was edited after these figures were rendered (6a/6b
  parameter split, the "Fiducial" → "Mixture" legend label). Re-renders can
  therefore differ cosmetically from the paper; see `--compare`.
* The Mixture colour differs between figures: pink in the R(m1)/R(z) plots,
  green in the corners.
* The pull box plot hides outliers (`showfliers=False`).
