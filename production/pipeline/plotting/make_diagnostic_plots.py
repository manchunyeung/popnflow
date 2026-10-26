"""
Diagnostic plots for the GMM importance-sampling population-inference paper.

Standalone sibling of `make_plots_met4.py`, separated out because it needs to
load functions (population models + likelihoods) and build/load a GMM-based
importance-sampling pool — work that the lightweight plotting script doesn't
need to do.

Outputs (in --outdir):
    {run}_variance_Nsamp.pdf       (--make-variance-plot)         empirical subsampling
    {run}_variance_analytic.pdf    (--make-variance-analytic-plot) analytical 1/N law
    {run}_neff_alpha_gamma.pdf     (--make-neff-scan-plot)
    {run}_xi_scan.pdf              (--make-xi-scan-plot)          xi vs (alpha, gamma)
    {run}_xi_ratio.pdf             (--make-xi-ratio-plot)         log10(xi_inj/xi_GMM)
    {run}_variance_budget.pdf      (--make-fitvar-plot)           term (A)+(B) budget

The plots are gated independently so you can run them separately. Pass
--make-both for the two original plots, or --make-all for everything.

Sources merged from gwtc3_mdc_69clean_2.ipynb:
    Cell 35  ->  per-event variance vs N_samp
    Cell 36  ->  N_eff vs sigma & gamma scatter
plus analytical-variance / xi-scan / xi-ratio diagnostics added on top.
"""
import argparse
from pathlib import Path
import os
import sys

import numpy as np
import h5py
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import jax
jax.config.update("jax_enable_x64", True)
from jax import jit, vmap, random
from jax import numpy as jnp
from jax.scipy.special import logsumexp, erf, erfinv

import astropy.units as u
from astropy.cosmology import Planck15, FlatLambdaCDM
import astropy.constants as constants

from jaxinterp2d import interp2d
import dynesty
from dynesty.utils import resample_equal

from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde, norm as scipy_norm, linregress
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import logit, expit


# ==========================================================
# 1. CLI
# ==========================================================
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Variance-vs-N_samp, N_eff-vs-(alpha,gamma), and xi diagnostics. "
            "Builds its own GMM-based proposal pool (or loads a cached one) "
            "so it can re-evaluate the IS weights at posterior coordinates."
        )
    )
    # --- shared with make_plots_met4.py ---
    p.add_argument("--indir", type=Path, required=True,
                   help="Directory with sampler outputs (same as make_plots_met4.py).")
    p.add_argument("--outdir", type=Path, required=True,
                   help="Directory for diagnostic plots.")
    p.add_argument("--run", type=str, required=True, choices=["gwtc3", "mdc"],
                   help="gwtc3 or mdc.")
    p.add_argument("--seed", type=int, default=777,
                   help="RNG seed (also used in the legacy .pkl sampler filenames).")
    p.add_argument("--catalog", type=int, default=0,
                   help="Catalog realization index for the h5 sampler files "
                        "(default: 0 -> cat000). MUST match the catalog the PE "
                        "sample files below correspond to.")
    p.add_argument("--sampler-pattern", type=str,
                   default="{run}_cat{i:03d}_{method}.h5",
                   help="Filename pattern for the dynesty checkpoints, with {run}, "
                        "{i} and {method} placeholders "
                        "(default: '{run}_cat{i:03d}_{method}.h5').")
    p.add_argument("--method-patterns", nargs="+", default=["met3={run}nm_cat{i:03d}_{method}.h5"],
                   help="Per-method pattern overrides as 'method=pattern'. Default "
                        "handles the met3 prefix: met3={run}nm_cat{i:03d}_{method}.h5. "
                        "Pass '' to disable. Anything not overridden uses "
                        "--sampler-pattern.")
    p.add_argument("--pe-m1det", type=Path, default=Path("input_data/mdc_m1det_69rand.txt"),
                   help="PE samples, detector-frame m1 (Nobs*nsamp rows).")
    p.add_argument("--pe-m2det", type=Path, default=Path("input_data/mdc_m2det_69rand.txt"),
                   help="PE samples, detector-frame m2.")
    p.add_argument("--pe-dL", type=Path, default=Path("input_data/mdc_dL_69rand.txt"),
                   help="PE samples, luminosity distance.")
    p.add_argument("--injection-file", type=Path,
                   default=Path("input_data/endo3_bbhpop-LIGO-T2100113-v12.hdf5"),
                   help="LVK injection HDF5 used for selection effects.")
    # --- specific to this diagnostic script ---
    p.add_argument("--pool-cache", type=Path, default=None,
                   help="Optional NPZ cache of a previously-built IS pool. "
                        "If the file exists, load it; otherwise build a new pool "
                        "and save to this path. If omitted, the pool is rebuilt "
                        "every run and not cached.")
    p.add_argument("--n-pool", type=int, default=20000,
                   help="Size of the proposal IS pool to build (default: 20000). "
                        "Larger = lower MC noise but slower evaluation.")
    # --- plot selection + budgets ---
    p.add_argument("--make-variance-plot", action="store_true",
                   help="Produce the empirical (subsampling) variance vs N_samp plot.")
    p.add_argument("--make-variance-analytic-plot", action="store_true",
                   help="Produce the analytical 1/N variance plot (V_full scaled by ceiling/N).")
    p.add_argument("--make-neff-scan-plot", action="store_true",
                   help="Produce the N_eff vs alpha/gamma plot.")
    p.add_argument("--make-xi-scan-plot", action="store_true",
                   help="Produce the xi vs parameter projection plot (default panels: alpha, gamma).")
    p.add_argument("--make-xi-ratio-plot", action="store_true",
                   help="Produce the log10(xi_inj/xi_GMM) ratio plot.")
    p.add_argument("--make-both", action="store_true",
                   help="Produce the two original plots (variance + N_eff scan).")
    p.add_argument("--make-all", action="store_true",
                   help="Produce every diagnostic plot.")
    p.add_argument("--variance-results-cache", type=Path, default=None,
                   help="NPZ file to save/load variance sweep results. If the file exists "
                        "the 24-minute sweep is skipped and results are loaded from it.")
    p.add_argument("--variance-n-coords", type=int, default=500,
                   help="Coordinates per method for the variance sweep (default: 500).")
    p.add_argument("--variance-nsamp-points", type=int, default=20,
                   help="Number of points in the log-spaced N_samp grid (default: 20).")
    p.add_argument("--variance-nsamp-max", type=int, default=30000,
                   help="Upper end of the N_samp grid (default: 30000). The GMM lines extend "
                        "to here; PE stops at its own sample ceiling. NOTE: to reach this "
                        "empirically the pool must hold >= variance_nsamp_max * Nobs samples, "
                        "so build with --n-pool accordingly; the analytical plot reaches it for free.")
    p.add_argument("--neff-n-prior-samples", type=int, default=3000,
                   help="Prior draws for the N_eff / xi scans (default: 3000).")
    p.add_argument("--xi-params", nargs=2, default=["alpha", "gamma"],
                   help="Two hyperparameters for the xi scan/ratio panels (default: alpha gamma).")
    p.add_argument("--sel-source-frame-mask", nargs=4, type=float, default=None,
                   metavar=("M_LO", "M_HI", "Z_HI", "DL_LO"),
                   help="Mask the selection-GMM fit on SOURCE-frame masses with "
                        "these INJECTION-CAMPAIGN bounds instead of the legacy "
                        "detector-frame [6,70]+dL>=50 cut. For sim_cat6 use "
                        "'2 100 1.9 0' (injections.h5 reference_distribution). "
                        "Omit to reproduce the published Fig 2/3 exactly.")
    p.add_argument("--sel-mass-domain", nargs=2, type=float, default=None,
                   metavar=("LO", "HI"),
                   help="CDF/KDE domain for mass_1 (default 6 70, matching the "
                        "legacy mask). Pair with --sel-source-frame-mask; the "
                        "notebook uses '2 150', wider than the data.")
    p.add_argument("--xi-gmm-sizes", type=int, nargs=2, default=[80000, 300000],
                   help="Two GMM sample sizes for the xi scan/ratio (default: 80000 300000). "
                        "Requires --n-pool >= the larger value; otherwise it is capped (with a warning).")
    p.add_argument("--neff-results-cache", type=Path, default=None,
                   help="NPZ file to save/load the N_eff-vs-alpha/gamma scan results "
                        "(the found-injection reweighting values and the GMM-based "
                        "selection-function fit + resample, which are the slow parts). "
                        "If the file exists, the scan is skipped and results are loaded "
                        "from it, so replotting (legend, figure size, colors, ...) is instant.")
    p.add_argument("--xi-results-cache", type=Path, default=None,
                   help="NPZ file to save/load the xi-scan results (found-injection "
                        "reweighting + GMM selection-function fit/resample). If the file "
                        "exists, the scan is skipped and results are loaded from it.")
    p.add_argument("--xi-ratio-before-cache", type=Path, default=None,
                   help="Optional NPZ xi-scan cache computed with the PRE-FIX "
                        "(spin-uncancelled) selection_xi_method1, for the referee-facing "
                        "before/after comparison in --make-xi-ratio-plot. If given, its "
                        "xi_inj is used as the 'before' curve; --xi-results-cache (recomputed "
                        "with the current, fixed code) is always the 'after' curve.")
    # --- term (B): fit-variance floor ---
    p.add_argument("--make-fitvar-plot", action="store_true",
                   help="Produce the variance-budget plot: term (A) 1/N curves plus "
                        "the bootstrap fit-variance floor, term (B), and the "
                        "B-vs-PE scatter.")
    p.add_argument("--fitvar-nboot", type=int, default=50,
                   help="Bootstrap resamples per event (default: 50).")
    p.add_argument("--fitvar-n-coords", type=int, default=200,
                   help="Hyperparameter draws at which to evaluate B (default: 200).")
    p.add_argument("--fitvar-coord-source", choices=["posterior", "prior"],
                   default="posterior",
                   help="Where the Lambda draws come from. 'posterior' (default) "
                        "uses the met4 hyper-posterior, so B is on the SAME Lambda "
                        "ensemble as the (A) curves and the two are comparable. "
                        "'prior' samples the prior box, which answers the different "
                        "question of whether the likelihood surface is reliable "
                        "everywhere dynesty looks.")
    p.add_argument("--fitvar-pool-size", type=int, default=20000,
                   help="Evaluation-pool size M_max for the fit-variance integrals "
                        "(default: 20000). The 1/M convergence scan (--fitvar-m-scan) "
                        "evaluates at fractions of this M_max.")
    p.add_argument("--fitvar-m-scan", type=float, nargs="+",
                   default=[0.125, 0.25, 0.5, 1.0],
                   help="Pool fractions of --fitvar-pool-size at which to evaluate "
                        "Vhat(M), for the 1/M convergence fit. The intercept of the "
                        "OLS regression Vhat(M) = B + c/M is the fit variance (B); "
                        "the slope (c) is the residual conditional-MC contamination.")
    p.add_argument("--fitvar-min-snr", type=float, default=3.0,
                   help="Require intercept/sigma_intercept above this before quoting "
                        "a floor; otherwise report a 2-sigma upper limit instead "
                        "(default: 3.0).")
    p.add_argument("--fitvar-defensive-frac", type=float, default=0.1,
                   help="Mixing fraction f for the defensive evaluation proposal "
                        "q* = (1-f)*q_E + f*q_broad (default: 0.1). Bounds the "
                        "importance weights so common-random-numbers cancellation "
                        "does not fail near the m_min edge. Diagnostic-only: does "
                        "not affect the (A) curves or any other plot.")
    p.add_argument("--fitvar-broad-inflate", type=float, default=4.0,
                   help="Covariance inflation factor for q_broad, the K=3 "
                        "full-covariance GMM fitted to the pooled PE samples of all "
                        "events that defines the defensive component (default: 4.0).")
    p.add_argument("--fitvar-train-frac", type=float, default=1.0,
                   help="Fraction of each event's PE samples used to TRAIN the "
                        "production GMM. Must match production: _build_pool uses "
                        "all samples, hence 1.0. Set 0.8 if the production fit "
                        "uses an 80/20 split -- B scales as 1/N_train.")
    p.add_argument("--fitvar-vary-seed", action="store_true",
                   help="Vary the EM random_state across bootstraps. Default is a "
                        "fixed seed, isolating data-resampling variance; varying it "
                        "adds EM optimisation stochasticity. Run both (referee M7).")
    p.add_argument("--fitvar-n-jobs", type=int, default=1,
                   help="Parallel jobs for the GMM refits (joblib). -1 = all cores.")
    p.add_argument("--fitvar-coord-chunk", type=int, default=8,
                   help="Hyperparameter coords per vectorised chunk (memory knob).")
    p.add_argument("--fitvar-results-cache", type=Path, default=None,
                   help="NPZ cache for the bootstrap fit variances. If present, "
                        "the refits are skipped.")
    p.add_argument("--fitvar-moment-check", action="store_true",
                   help="Validate the density-evaluation path against the EM moment "
                        "identity (a converged GMM reproduces its training data's "
                        "first two moments exactly). Cheap; run once.")

    args = p.parse_args()

    if args.make_all:
        args.make_variance_plot = True
        args.make_variance_analytic_plot = True
        args.make_neff_scan_plot = True
        args.make_xi_scan_plot = True
        args.make_xi_ratio_plot = True
        args.make_fitvar_plot = True
    if args.make_both:
        args.make_variance_plot = True
        args.make_neff_scan_plot = True

    if not (args.make_variance_plot or args.make_variance_analytic_plot
            or args.make_neff_scan_plot or args.make_xi_scan_plot
            or args.make_xi_ratio_plot or args.make_fitvar_plot
            or args.fitvar_moment_check):
        p.error("Nothing to do: pass --make-variance-plot, --make-variance-analytic-plot, "
                "--make-neff-scan-plot, --make-xi-scan-plot, --make-xi-ratio-plot, "
                "--make-fitvar-plot, --make-both, or --make-all.")
    return args


args = parse_args()
args.outdir.mkdir(parents=True, exist_ok=True)

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Seaborn first (it resets rcParams), THEN our font settings so they survive.
sns.set_context('talk', font_scale=1.3)
sns.set_style('ticks')
sns.set_palette('colorblind')
C = sns.color_palette('colorblind')   # C[0] blue, C[1] orange, C[2] green, C[3] red

USE_TEX = True
if USE_TEX:
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Computer Modern Roman']
else:
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['cmr10', 'Computer Modern Roman', 'DejaVu Serif']
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['axes.formatter.use_mathtext'] = True
matplotlib.rcParams['axes.unicode_minus'] = False
# Shared axis-label / legend sizes across all diagnostic scripts (see run_all_diagnostics.py).
matplotlib.rcParams['axes.labelsize'] = float(os.environ.get('AXIS_LABEL_FONTSIZE', 20))
matplotlib.rcParams['legend.fontsize'] = float(os.environ.get('LEGEND_FONTSIZE', 16))

jax.config.update('jax_default_matmul_precision', 'tensorfloat32')


# ==========================================================
# 2. Cosmology grids
# ==========================================================
H0Planck = Planck15.H0.value
Om0Planck = Planck15.Om0
c_kms = constants.c.to('km/s').value

zMax = 2.0
zgrid = jnp.expm1(np.linspace(np.log(1), np.log(zMax + 1), 10000))
Om0grid = jnp.linspace(0, 1, 1000)

rs = []
for Om0 in tqdm(Om0grid, desc="Building cosmo grid"):
    cosmo = FlatLambdaCDM(H0=H0Planck, Om0=Om0)
    rs.append(cosmo.comoving_distance(zgrid).to(u.Mpc).value)
rs = jnp.asarray(rs).reshape(len(Om0grid), len(zgrid))


@jit
def E(z, Om0=Om0Planck):
    return jnp.sqrt(Om0 * (1 + z) ** 3 + (1.0 - Om0))


@jit
def r_of_z(z, H0, Om0=Om0Planck):
    return interp2d(Om0, z, Om0grid, zgrid, rs) * (H0Planck / H0)


@jit
def dL_of_z(z, H0, Om0=Om0Planck):
    return (1 + z) * r_of_z(z, H0, Om0)


@jit
def z_of_dL(dL, H0, Om0=Om0Planck):
    return jnp.interp(dL, dL_of_z(zgrid, H0, Om0), zgrid)


@jit
def dV_of_z(z, H0, Om0=Om0Planck):
    return c_kms * r_of_z(z, H0, Om0) ** 2 / (H0 * E(z, Om0))


@jit
def dV_of_z_normed(z, Om0, gamma):
    dV = dV_of_z(zgrid, H0Planck, Om0) * (1 + zgrid) ** (gamma - 1)
    prob = dV / jnp.trapezoid(dV, zgrid)
    return jnp.interp(z, zgrid, prob)


@jit
def ddL_of_z(z, dL, H0, Om0=Om0Planck):
    return dL / (1 + z) + c_kms * (1 + z) / (H0 * E(z, Om0))


# ==========================================================
# 3. Selection injections
# ==========================================================
with h5py.File(args.injection_file, 'r') as f:
    Tobs = f.attrs['analysis_time_s'] / (365.25 * 24 * 3600)
    Ndraw = int(f.attrs['total_generated'])

    m1detsels_all = f['injections/mass1'][:]
    m2detsels_all = f['injections/mass2'][:]
    dLsels_all    = f['injections/distance'][:]
    p_draw_all    = f['injections/sampling_pdf'][:]
    # Exact stored source-frame fields -- used for m1sels/m2sels/zsels instead
    # of recomputing via the z_of_dL(dL) round-trip. That round-trip's
    # interpolation error pushes a handful of found injections (16/76202)
    # just outside the analytic marginal's [2,100] Msun / mass-ratio support,
    # which makes log_pdraw_marginal spuriously -inf and poisons the entire
    # xi estimator with +inf. The exact fields are guaranteed in-bounds by
    # construction (they generated the injection), so this class of bug can't
    # occur. See CLAUDE_CODE_TASK_selection.md Task 1/2.
    m1sels_exact_all = f['injections/mass1_source'][:]
    m2sels_exact_all = f['injections/mass2_source'][:]
    zsels_exact_all  = f['injections/redshift'][:]
    ifar_gstlal      = f['injections/ifar_gstlal'][:]
    ifar_pycbc_bbh   = f['injections/ifar_pycbc_bbh'][:]
    ifar_pycbc_broad = f['injections/ifar_pycbc_hyperbank'][:]

sels = (ifar_gstlal > 1) | (ifar_pycbc_bbh > 1) | (ifar_pycbc_broad > 1)
m1detsels = jnp.asarray(m1detsels_all[sels])
m2detsels = jnp.asarray(m2detsels_all[sels])
dLsels    = jnp.asarray(dLsels_all[sels])
p_draw    = jnp.asarray(p_draw_all[sels])

zsels  = jnp.asarray(zsels_exact_all[sels])
m1sels = jnp.asarray(m1sels_exact_all[sels])
m2sels = jnp.asarray(m2sels_exact_all[sels])
print(f"[selection] {int(sels.sum())} found injections out of {Ndraw}")


# ==========================================================
# 4. Per-event PE samples
# ==========================================================
Nobs = 69

def _load_pe_samples():
    dL_raw    = np.loadtxt(args.pe_dL)
    m1det_raw = np.loadtxt(args.pe_m1det)
    m2det_raw = np.loadtxt(args.pe_m2det)

    nsamp = int(dL_raw.shape[0] // Nobs)
    if dL_raw.shape[0] != Nobs * nsamp:
        raise RuntimeError(
            f"PE samples don't divide evenly: got {dL_raw.shape[0]} rows, "
            f"expected Nobs*nsamp with Nobs={Nobs}."
        )

    dL_ev    = jnp.asarray(dL_raw)
    m1det_ev = jnp.asarray(m1det_raw)
    m2det_ev = jnp.asarray(m2det_raw)
    z_ev     = z_of_dL(dL_ev, H0Planck, Om0Planck)
    m1_ev    = m1det_ev / (1 + z_ev)
    m2_ev    = m2det_ev / (1 + z_ev)

    return dict(
        m1_ev=m1_ev, m2_ev=m2_ev, z_ev=z_ev, dL_ev=dL_ev,
        m1det_ev=m1det_ev, m2det_ev=m2det_ev,
        nsamp=nsamp,
    )

pe = _load_pe_samples()
nsamp = pe['nsamp']
print(f"[PE] Loaded {Nobs} events x {nsamp} samples each.")


# ==========================================================
# 5. Mass + redshift population models
# ==========================================================
mass = jnp.linspace(1, 150, 2000)
mass_ratio = jnp.linspace(1e-5, 1, 2000)


def Sfilter_low(m, m_min, dm_min):
    def f(mm, dM):
        return jnp.exp(dM / mm + dM / (mm - dM))
    S = 1. / (f(m - m_min, dm_min) + 1.)
    S = jnp.where(m < m_min + dm_min, S, 1.)
    S = jnp.where(m > m_min, S, 0.)
    return S


def Sfilter_high(m, m_max, dm_max):
    def f(mm, dM):
        return jnp.exp(dM / mm + dM / (mm - dM))
    S = 1. / (f(m - m_max, -dm_max) + 1.)
    S = jnp.where(m > m_max - dm_max, S, 1.)
    S = jnp.where(m < m_max, S, 0.)
    return S


@jit
def logpm1_powerlaw(m1, m_min, m_max, alpha, dm_min, dm_max):
    pm1 = Sfilter_low(mass, m_min, dm_min) * mass ** (-alpha) * Sfilter_high(mass, m_max, dm_max)
    pm1 = pm1 / jnp.trapezoid(pm1, mass)
    return jnp.log(jnp.interp(m1, mass, pm1))


@jit
def logpm1_peak(m1, mu, sigma):
    pm1 = jnp.exp(-(mass - mu) ** 2 / (2 * sigma ** 2))
    pm1 = pm1 / jnp.trapezoid(pm1, mass)
    return jnp.log(jnp.interp(m1, mass, pm1))


@jit
def logpm1_powerlaw_powerlaw(m1, m_min, m_max, alpha, dm_min, dm_max, mu, sigma, f1):
    p1 = jnp.exp(logpm1_powerlaw(m1, m_min, m_max, alpha, dm_min, dm_max))
    p2 = jnp.exp(logpm1_peak(m1, mu, sigma))
    return jnp.log((1 - f1) * p1 + f1 * p2)


@jit
def logfq(m1, m2, beta):
    q = m2 / m1
    pq = mass_ratio ** beta
    pq = pq / jnp.trapezoid(pq, mass_ratio)
    return jnp.log(jnp.interp(q, mass_ratio, pq))


@jit
def log_p_pop_pl_pl(m1, m2, z, gamma, m_min_1, m_max_1, alpha_1, dm_min_1, dm_max_1,
                    beta, mu, sigma, f1):
    """GWTC-3 PL+Peak with smooth turn-on/off (10-param)."""
    log_dNdm1 = logpm1_powerlaw_powerlaw(m1, m_min_1, m_max_1, alpha_1, dm_min_1, dm_max_1, mu, sigma, f1)
    log_dNdm2 = logpm1_powerlaw_powerlaw(m2, m_min_1, m_max_1, alpha_1, dm_min_1, dm_max_1, mu, sigma, f1)
    log_pq   = logfq(m1, m2, beta)
    log_dvdz = jnp.log(dV_of_z_normed(z, Om0Planck, gamma))
    log_p_sz = np.log(0.25)
    log_p = log_p_sz + log_dNdm1 + log_dNdm2 + log_pq + log_dvdz
    return jnp.where(m2 < m1, log_p, -jnp.inf)


@jit
def logpm1m2_plpeak_massratio(m1, m2, m_min_1, m_max_1, alpha_1, beta, mu, sigma, f):
    """MDC analytic PL+Peak with hard edges (8-param)."""
    q = m2 / m1
    alpha_1 = -alpha_1
    norm_pl = (m_max_1 ** (1. + alpha_1) - m_min_1 ** (1. + alpha_1))
    p_m1_pl = (1. + alpha_1) * m1 ** alpha_1 / norm_pl
    p_m1_pl = jnp.where(m1 > m_max_1, 0.0, p_m1_pl)
    p_m1_pl = jnp.where(m1 < m_min_1, 0.0, p_m1_pl)

    p_m1_peak = jnp.exp(-0.5 * (m1 - mu) ** 2 / sigma ** 2) / jnp.sqrt(2. * jnp.pi * sigma ** 2)
    p_m1 = f * p_m1_peak + (1. - f) * p_m1_pl

    q_min = m_min_1 / m1
    denom = 1 - q_min ** (1. + beta)
    p_q = (1. + beta) * q ** beta / denom
    p_q = jnp.where(q * m1 < m_min_1, 0.0, p_q)

    return jnp.log(p_m1) + jnp.log(p_q) - jnp.log(m1)


@jit
def log_p_pop_lvk(m1, m2, z, gamma, m_min_1, m_max_1, alpha_1, beta, mu, sigma, f1):
    """MDC analytic 8-param population."""
    log_joint_mass = logpm1m2_plpeak_massratio(m1, m2, m_min_1, m_max_1, alpha_1, beta, mu, sigma, f1)
    log_pz = jnp.log(dV_of_z_normed(z, Om0Planck, gamma))
    log_p = log_joint_mass + log_pz
    log_p = jnp.where(m2 < m1, log_p, -jnp.inf)
    return log_p - jnp.log(0.25)


if args.run == "mdc":
    def log_p_pop(m1, m2, z, coord):
        gamma, m_min, m_max, alpha, beta, mu, sigma, f1 = coord
        return log_p_pop_lvk(m1, m2, z, gamma, m_min, m_max, alpha, beta, mu, sigma, f1)
else:
    def log_p_pop(m1, m2, z, coord):
        (gamma, m_min, m_max, alpha, dm_min, dm_max,
         beta, mu, sigma, f1) = coord
        return log_p_pop_pl_pl(m1, m2, z, gamma, m_min, m_max, alpha, dm_min, dm_max,
                               beta, mu, sigma, f1)


# ==========================================================
# 6. Prior box 
# ==========================================================
gamma_low, gamma_high     = 0.0, 10.0
m_min_low, m_min_high     = 6.0, 6.0
m_max_low, m_max_high     = 70.0, 70.0
alpha_low, alpha_high     = 0.0, 6.0
beta_low, beta_high       = 0.0, 6.0
mu_low, mu_high           = 20.0, 50.0
sigma_low, sigma_high     = 1.0, 10.0
f1_low, f1_high           = 0.0, 1.0

if args.run == "mdc":
    labels = ['gamma', 'm_min', 'm_max', 'alpha', 'beta', 'mu', 'sigma', 'f1']
    lower_bound = np.array([gamma_low, m_min_low, m_max_low, alpha_low,
                            beta_low, mu_low, sigma_low, f1_low])
    upper_bound = np.array([gamma_high, m_min_high, m_max_high, alpha_high,
                            beta_high, mu_high, sigma_high, f1_high])
else:
    dm_min_low, dm_min_high = 1.0, 100.0
    dm_max_low, dm_max_high = 1.0, 100.0
    labels = ['gamma', 'm_min', 'm_max', 'alpha', 'dm_min', 'dm_max',
              'beta', 'mu', 'sigma', 'f1']
    lower_bound = np.array([gamma_low, m_min_low, m_max_low, alpha_low,
                            dm_min_low, dm_max_low,
                            beta_low, mu_low, sigma_low, f1_low])
    upper_bound = np.array([gamma_high, m_min_high, m_max_high, alpha_high,
                            dm_min_high, dm_max_high,
                            beta_high, mu_high, sigma_high, f1_high])
ndims = len(lower_bound)

for _pname in args.xi_params:
    if _pname not in labels:
        raise SystemExit(f"--xi-params '{_pname}' not in {labels} for run={args.run}")


# ==========================================================
# 7. Build the IS proposal pool (per-event GMM mixture)
# ==========================================================
@jax.jit
def exact_log_gmm(x_ND, w_K, mu_KD, Prec_KDD, logdetS_K):
    D        = mu_KD.shape[-1]
    diffs    = x_ND[:, None, :] - mu_KD[None, :, :]
    quad     = jnp.einsum('nkd,kdj,nkj->nk', diffs, Prec_KDD, diffs)
    log_comp = -0.5 * (D * jnp.log(2 * jnp.pi) + logdetS_K[None, :] + quad)
    logw     = jnp.log(jnp.clip(w_K, 1e-300, 1.0))[None, :]
    return logsumexp(logw + log_comp, axis=-1)

def _build_pool(n_pool, m1det_ev, m2det_ev, dL_ev, seed):
    rng = np.random.default_rng(seed)

    m1d = np.asarray(m1det_ev).reshape(Nobs, nsamp)
    m2d = np.asarray(m2det_ev).reshape(Nobs, nsamp)
    dLd = np.asarray(dL_ev).reshape(Nobs, nsamp)

    K_PER_EVENT = 7
    gmms_events = []

    print("[pool] Fitting per-event GMMs...")
    for e in tqdm(range(Nobs), desc="Fitting per-event GMMs"):
        X_e = np.column_stack([m1d[e], m2d[e], dLd[e]])
        g = GaussianMixture(n_components=K_PER_EVENT, covariance_type='full',
                            reg_covar=1e-6, n_init=5, random_state=seed).fit(X_e)
        gmms_events.append(g)

    wts_ev     = jnp.asarray(np.stack([g.weights_    for g in gmms_events]), dtype=jnp.float64)
    mus_ev     = jnp.asarray(np.stack([g.means_      for g in gmms_events]), dtype=jnp.float64)
    prec_ev    = jnp.asarray(np.stack([g.precisions_ for g in gmms_events]), dtype=jnp.float64)
    logdets_ev = jnp.asarray(
        np.stack([-2.0 * np.sum(np.log(np.diagonal(g.precisions_cholesky_,
                                                   axis1=-2, axis2=-1)), axis=-1)
                  for g in gmms_events]),
        dtype=jnp.float64,
    )

    print(f"[pool] Drawing {n_pool} samples from per-event GMM mixture...")
    per_event = n_pool // Nobs
    leftover  = n_pool - per_event * Nobs
    sizes = np.full(Nobs, per_event)
    sizes[:leftover] += 1

    chunks = []
    for e in range(Nobs):
        if sizes[e] == 0:
            continue
        samps, _ = gmms_events[e].sample(sizes[e])
        chunks.append(samps)
    X = np.vstack(chunks) # (N, 3)

    X_j = jnp.asarray(X)

    print(f"[pool] Evaluating per-event log-densities at pool samples "
          f"(E={Nobs}, N={X.shape[0]})...")
    logp_E_N = np.zeros((Nobs, X.shape[0]), dtype=np.float32)
    for e in tqdm(range(Nobs), desc="per-event log-density"):
        lp_e = exact_log_gmm(X_j, wts_ev[e], mus_ev[e], prec_ev[e], logdets_ev[e])
        logp_E_N[e] = np.asarray(lp_e)
    logp_E_N_j = jnp.asarray(logp_E_N)

    log_q_j = logsumexp(logp_E_N_j, axis=0) - jnp.log(Nobs)

    m1det = X_j[:, 0]; m2det = X_j[:, 1]; dL_p = X_j[:, 2]
    zs_pop  = z_of_dL(dL_p, H0Planck, Om0Planck)
    m1s_pop = m1det / (1 + zs_pop)
    m2s_pop = m2det / (1 + zs_pop)

    logJ = (
        -jnp.log(ddL_of_z(zs_pop, dL_p, H0Planck, Om0Planck))
        - 2.0 * jnp.log1p(zs_pop)
        - 2.0 * jnp.log(dL_p)
    )
    logJ = jnp.where(jnp.isfinite(logJ), logJ, -jnp.inf)

    # Validity: match analysis box
    valid = (
        (m1s_pop >= 6.0) & (m1s_pop <= 70.0) &
        (m2s_pop >= 6.0) & (m2s_pop <= m1s_pop) &
        (zs_pop  >= 0.0) & (zs_pop  <= 1.9)   &
        (dL_p    >= 50.0) &
        jnp.isfinite(logJ)
    )
    valid_np = np.asarray(valid)
    print(f"[pool] valid samples: {valid_np.sum()}/{valid_np.size} "
          f"({100.0 * valid_np.mean():.1f}%)")

    return dict(
        m1s_pop=m1s_pop[valid],
        m2s_pop=m2s_pop[valid],
        zs_pop=zs_pop[valid],
        dLs_pop=dL_p[valid],
        log_q=log_q_j[valid],
        logp_E_N=logp_E_N_j[:, valid],
        logJ=logJ[valid],
        nsamp_pop=int(valid_np.sum()),
    )


def _save_pool_npz(path: Path, pool: dict):
    np.savez_compressed(
        path,
        m1s_pop  = np.asarray(pool['m1s_pop']),
        m2s_pop  = np.asarray(pool['m2s_pop']),
        zs_pop   = np.asarray(pool['zs_pop']),
        dLs_pop  = np.asarray(pool['dLs_pop']),
        log_q    = np.asarray(pool['log_q']),
        logp_E_N = np.asarray(pool['logp_E_N']),
        logJ     = np.asarray(pool['logJ']),
    )
    print(f"[pool] cached -> {path}")


def _load_pool_npz(path: Path):
    d = np.load(path)
    pool = {k: jnp.asarray(d[k]) for k in
            ('m1s_pop', 'm2s_pop', 'zs_pop', 'dLs_pop',
             'log_q', 'logp_E_N', 'logJ')}
    pool['nsamp_pop'] = int(pool['m1s_pop'].shape[0])
    print(f"[pool] loaded {pool['nsamp_pop']} samples from cache: {path}")
    return pool


if args.pool_cache is not None and args.pool_cache.exists():
    pool = _load_pool_npz(args.pool_cache)
else:
    print("[pool] Building per-event GMMs...")
    pool = _build_pool(args.n_pool, pe['m1det_ev'], pe['m2det_ev'], pe['dL_ev'], seed=args.seed)
    if args.pool_cache is not None:
        args.pool_cache.parent.mkdir(parents=True, exist_ok=True)
        _save_pool_npz(args.pool_cache, pool)


# ==========================================================
# 8. N_eff / xi / variance evaluators 
# ==========================================================
@jit
def _logdiffexp(x, y):
    return x + jnp.log1p(-jnp.exp(y - x))

def selection_neff_method1(coord):
    log_w = log_p_pop(m1sels, m2sels, zsels, coord) - log_pdraw_marginal(m1sels, m2sels, zsels, dLsels)
    log_mu = logsumexp(log_w) - jnp.log(Ndraw)
    log_s2 = logsumexp(2 * log_w) - 2.0 * jnp.log(Ndraw)
    log_sigma2 = _logdiffexp(log_s2, 2.0 * log_mu - jnp.log(Ndraw))
    return jnp.exp(2.0 * log_mu - log_sigma2)

def selection_xi_method1(coord):
    log_w = log_p_pop(m1sels, m2sels, zsels, coord) - log_pdraw_marginal(m1sels, m2sels, zsels, dLsels)
    return float(jnp.exp(logsumexp(log_w) - jnp.log(Ndraw)))

def selection_xi_pool(coord, pool_obj):
    log_w = (
        log_p_pop(pool_obj['m1s_pop'], pool_obj['m2s_pop'], pool_obj['zs_pop'], coord)
        + pool_obj['logJ']
        - pool_obj['log_q']
    )
    log_w = jnp.where(jnp.isfinite(log_w), log_w, -jnp.inf)
    return float(jnp.exp(logsumexp(log_w) - jnp.log(pool_obj['nsamp_pop'])))

# ----------------------------------------------------------------------------
# Selection GMM (Talbot & Thrane) for the CORRECT xi integral.
# ----------------------------------------------------------------------------

@jax.jit
def powerlaw_ln_normalization_jax(alpha, minimum, maximum):
    return jnp.log((1.0 + alpha) / (maximum**(1.0 + alpha) - minimum**(1.0 + alpha)))

@jax.jit
def powerlaw_ln_pdf_jax(value, alpha, minimum, maximum):
    log_prob = alpha * jnp.log(value) + powerlaw_ln_normalization_jax(alpha, minimum, maximum)
    return jnp.where((value >= minimum) & (value <= maximum), log_prob, -jnp.inf)

@jax.jit
def log_pdraw_marginal(m1, m2, z, dL_val):
    """log p_draw(m1, m2, z) -- the MARGINAL injection draw density over exactly
    the parameters p_pop models (source-frame m1, m2, z). No spin factor.

    THE canonical denominator for every selection estimator (Method 1 and the
    selection GMM) -- see CLAUDE_CODE_TASK_selection.md. Validated against the
    file's exact stored marginal
    (mass1_source_mass2_source_sampling_pdf * redshift_sampling_pdf): the
    residual log(exact) - log(analytic) has mean -0.310, std 0.00078 over all
    284672 injections -- i.e. a fixed (harmless, cancels in every ratio used
    here) normalisation offset, not a functional-form error.
    """
    log_p_m1 = powerlaw_ln_pdf_jax(m1, alpha=-2.35, minimum=2.0, maximum=100.0)
    log_p_q  = powerlaw_ln_pdf_jax(m2 / m1, alpha=1.0, minimum=2.0 / m1, maximum=1.0)
    log_dvdz = jnp.log(dV_of_z_normed(z, Om0Planck, gamma=1.0))
    return log_p_m1 + log_p_q + log_dvdz - jnp.log(m1)

# Backward-compat alias: same function, old name. The "injection prior" name
# was misleading -- it is the draw-density MARGINAL over (m1,m2,z), not the
# full injection prior (which also has spin factors; see sampling_pdf).
log_p_injection_prior = log_pdraw_marginal

_eta_sel = float(np.asarray(sels).sum()) / Ndraw      # N_found / N_draw

class ReflectingKDE(gaussian_kde):
    def __init__(self, dataset, bw_method=None, weights=None, minimum=-np.inf, maximum=np.inf):
        super().__init__(dataset=dataset, bw_method=bw_method, weights=weights)
        self.minimum = minimum
        self.maximum = maximum
        self.range   = self.maximum - self.minimum

    def __call__(self, points):
        prob     = self.evaluate(points)
        fraction = (points - self.minimum) / self.range
        if self.minimum > -np.inf:
            prob += self.evaluate(self.minimum - fraction * self.range)
            prob *= (points >= self.minimum)
        if self.maximum < np.inf:
            prob += self.evaluate(self.maximum + (1.0 - fraction) * self.range)
            prob *= (points <= self.maximum)
        return prob

def _build_selection_gmm(sizes, seed, source_frame_mask=None, mass_domain=None):
    """Fit a GMM to the found injections using empirical CDF->probit transform.

    source_frame_mask : (m_lo, m_hi, z_hi, dL_lo) or None
        None  -> LEGACY behaviour, kept so the published Fig 2/3 reproduce
                 bit-for-bit: cut m1/m2 at [6, 70] with dL >= 50, applied to
                 DETECTOR-frame masses.
        tuple -> corrected behaviour: cut on SOURCE-frame masses with the
                 INJECTION-CAMPAIGN bounds.

        Why this matters (measured 2026-08-28). `m1_resam` is `m1detsels`,
        i.e. DETECTOR frame, but 6 and 70 are m_min/m_max of the POPULATION,
        which lives in source frame. A detector-frame ceiling at 70 is a
        SLOPED source-frame cut, m1_src <= 70/(1+z): ~58 at z=0.2 but only
        ~36 at z=1. On sim_cat6 the legacy mask keeps 53.1% of the 103,750
        detections; the campaign bounds in source frame keep 100%. The
        selection density then has no support where the population puts mass,
        and xi is underestimated exactly where the estimator is blind.

        That is harmless for Fig 3's own question -- comparing xi estimators
        at and around Lambda_true, inside the mask -- which is why the
        published figure is sound. It breaks when the same density is used as
        a general selection function across the whole prior box.

        The corrected form follows gwtc3_mdc_69clean.ipynb (cells 15-16),
        which masks with campaign bounds and re-cuts the DRAWS in source frame.

    mass_domain : (lo, hi) or None
        CDF/KDE domain for mass_1. Legacy is (6.0, 70.0) -- identical to the
        mask, which pins the reflecting KDE against a boundary where samples
        still live. The notebook uses (2.0, 150.0): wider than the data.
    """
    lq_sels_np = logit(np.asarray(m2detsels) / np.asarray(m1detsels))
    X_raw      = np.column_stack([np.asarray(m1detsels), lq_sels_np, np.asarray(dLsels)])
    
    m1_resam = X_raw[:, 0]
    lq_resam = X_raw[:, 1]
    dL_resam = X_raw[:, 2]

    # Physical validity mask matching targeted hyper-parameters
    q_resam  = expit(lq_resam)
    m2_resam = m1_resam * q_resam
    z_resam  = np.asarray(z_of_dL(jnp.asarray(dL_resam), H0Planck, Om0Planck))
    if source_frame_mask is None:
        phys_mask = (
            (m1_resam >= 6.0) & (m1_resam <= 70.0) &
            (m2_resam >= 6.0) & (m2_resam <= m1_resam) &
            (q_resam  >= 6.0 / m1_resam) &
            (z_resam  >= 0.0) & (z_resam  <= 1.9) &
            (dL_resam >= 50.0)
        )
    else:
        m_lo, m_hi, z_hi, dL_lo = source_frame_mask
        m1_src = m1_resam / (1.0 + z_resam)      # the frame the population lives in
        m2_src = m2_resam / (1.0 + z_resam)
        phys_mask = (
            (m1_src >= m_lo) & (m1_src <= m_hi) &
            (m2_src >= m_lo) & (m2_src <= m1_src) &
            (z_resam >= 0.0) & (z_resam <= z_hi) &
            (dL_resam >= dL_lo)
        )
        print(f"[xi-scan] SOURCE-frame mask [{m_lo}, {m_hi}], z<={z_hi}, "
              f"dL>={dL_lo}: keeping {phys_mask.sum():,}/{len(phys_mask):,} "
              f"({phys_mask.mean()*100:.1f}%) found injections")
    m1_resam = m1_resam[phys_mask]
    lq_resam = lq_resam[phys_mask]
    dL_resam = dL_resam[phys_mask]

    GRID_N   = 10_000
    FIT_KEYS = ["mass_1", "lq", "dL"]

    data_dict = {"mass_1": m1_resam, "lq": lq_resam, "dL": dL_resam}
    ranges    = {
        "mass_1": (6.0, 70.0) if mass_domain is None else tuple(mass_domain),
        "lq":     (float(np.percentile(lq_resam, 0.05)),
                   float(np.percentile(lq_resam, 99.95))),
        "dL":     (0.0, float(np.max(dLsels)) * 1.05),
    }

    cdf_grids = {}
    grid_meta = {}

    for key in FIT_KEYS:
        arr        = data_dict[key]
        min_, max_ = ranges[key]
        domain     = np.linspace(min_, max_, GRID_N)

        if key == "mass_1":
            kde  = ReflectingKDE(arr, minimum=min_, maximum=max_, weights=arr**0.5)
            epdf = kde(domain) * domain**-0.5
            epdf /= trapezoid(epdf, domain)
        else:
            kde  = ReflectingKDE(arr, minimum=min_, maximum=max_)
            epdf = kde(domain)

        epdf = np.clip(epdf, 1e-300, None)
        ecdf = cumulative_trapezoid(epdf, domain, initial=0)
        ecdf = np.clip(ecdf / ecdf[-1], 1e-12, 1.0 - 1e-12)

        cdf_grids[key]   = jnp.asarray(ecdf)
        grid_meta[key]   = {"min": min_, "delta": (max_ - min_) / (GRID_N - 1)}

    def apply_empirical_transform_np(m1_arr, lq_arr, dL_arr):
        out = np.zeros((len(m1_arr), 3))
        for ii, (key, arr) in enumerate(zip(FIT_KEYS, [m1_arr, lq_arr, dL_arr])):
            meta   = grid_meta[key]
            domain = np.linspace(meta["min"], meta["min"] + meta["delta"] * (GRID_N - 1), GRID_N)
            u = np.interp(arr, domain, np.asarray(cdf_grids[key]))
            u = np.clip(u, 1e-12, 1.0 - 1e-12)
            out[:, ii] = scipy_norm.ppf(u)
        return out

    def unscale_from_white(X_white):
        out = np.zeros_like(X_white)
        for ii, key in enumerate(FIT_KEYS):
            meta   = grid_meta[key]
            domain = np.linspace(meta["min"], meta["min"] + meta["delta"] * (GRID_N - 1), GRID_N)
            u = scipy_norm.cdf(X_white[:, ii])
            u = np.clip(u, 1e-12, 1.0 - 1e-12)
            out[:, ii] = np.interp(u, np.asarray(cdf_grids[key]), domain)
        return out

    X_white = apply_empirical_transform_np(m1_resam, lq_resam, dL_resam)
    print(f"[xi-scan] fitting selection GMM (K=10) to {X_white.shape[0]} whitened found injections...")
    gmm_sel = GaussianMixture(n_components=10, covariance_type='full',
                              reg_covar=1e-6, n_init=3, random_state=seed).fit(X_white)

    n_max = int(max(sizes))
    print(f"[xi-scan] drawing {n_max:,} samples from selection GMM...")
    Sw, _ = gmm_sel.sample(n_max)
    S = unscale_from_white(Sw)
    
    m1d, q_all, dLd = S[:, 0], expit(S[:, 1]), S[:, 2]
    m2d = m1d * q_all

    m1d = jnp.asarray(m1d); m2d = jnp.asarray(m2d); dLd = jnp.asarray(dLd)
    zsrc = z_of_dL(dLd, H0Planck, Om0Planck)
    m1src = m1d / (1.0 + zsrc)
    m2src = m2d / (1.0 + zsrc)
    log_pdraw = log_pdraw_marginal(m1src, m2src, zsrc, dLd)

    # Re-cut the DRAWS in source frame (notebook cell 16). The inverse-CDF
    # interpolation extrapolates slightly past the fitted domain -- measured
    # draws at m1 = 4.87 from a nominal floor of 6 -- so the sampled pool needs
    # its own validity mask, not just the fit.
    if source_frame_mask is not None:
        m_lo, m_hi, z_hi, dL_lo = source_frame_mask
        keep = (np.isfinite(np.asarray(log_pdraw))
                & (np.asarray(m1src) >= m_lo) & (np.asarray(m1src) <= m_hi)
                & (np.asarray(m2src) >= m_lo) & (np.asarray(m2src) <= np.asarray(m1src))
                & (np.asarray(zsrc) >= 0.0) & (np.asarray(zsrc) <= z_hi)
                & (np.asarray(dLd) >= dL_lo))
        k = jnp.asarray(keep)
        m1src, m2src, zsrc, dLd = m1src[k], m2src[k], zsrc[k], dLd[k]
        log_pdraw = log_pdraw[k]
        n_max = int(keep.sum())
        print(f"[xi-scan] source-frame draw mask: kept {n_max:,} "
              f"({keep.mean()*100:.1f}%) of the sampled pool")

    return dict(m1=m1src, m2=m2src, z=zsrc, dL=dLd, log_pdraw=log_pdraw, nfull=n_max)


def selection_xi_gmm(coord, sel, n_use):
    log_w = (log_p_pop(sel['m1'][:n_use], sel['m2'][:n_use], sel['z'][:n_use], coord)
             - sel['log_pdraw'][:n_use])
    log_w = jnp.where(jnp.isfinite(log_w), log_w, -jnp.inf)
    return float(_eta_sel * jnp.exp(logsumexp(log_w) - jnp.log(n_use)))


def selection_neff_gmm(coord, sel, n_use):
    log_w = (log_p_pop(sel['m1'][:n_use], sel['m2'][:n_use], sel['z'][:n_use], coord)
             - sel['log_pdraw'][:n_use])
    log_w = jnp.where(jnp.isfinite(log_w), log_w, -jnp.inf)
    log_num = 2.0 * logsumexp(log_w)
    log_den = logsumexp(2.0 * log_w)
    return float(jnp.exp(log_num - log_den))


_SEL_GMM_CACHE = {}


def _get_selection_gmm(sizes, seed):
    """Memoized `_build_selection_gmm` so the neff-scan and xi-scan plots
    (both fitting a GMM to the same found injections) don't redo the fit.

    --sel-source-frame-mask switches the fit to the corrected source-frame
    campaign-bounds construction; absent, the legacy detector-frame [6,70]
    mask is used and the published Fig 2/3 reproduce unchanged.
    """
    sfm = getattr(args, "sel_source_frame_mask", None)
    md = getattr(args, "sel_mass_domain", None)
    key = (tuple(sorted(sizes)), seed,
           None if sfm is None else tuple(sfm),
           None if md is None else tuple(md))
    if key not in _SEL_GMM_CACHE:
        _SEL_GMM_CACHE[key] = _build_selection_gmm(
            sizes, seed,
            source_frame_mask=None if sfm is None else tuple(sfm),
            mass_domain=None if md is None else tuple(md))
    return _SEL_GMM_CACHE[key]


def per_event_inv_neff_method1(coord, idx_pe):
    log_w = log_p_pop(pe['m1_ev'], pe['m2_ev'], pe['z_ev'], coord)
    log_w += (
        -jnp.log(ddL_of_z(pe['z_ev'], pe['dL_ev'], H0Planck, Om0Planck))
        - 2.0 * jnp.log1p(pe['z_ev'])
        - 2.0 * jnp.log(pe['dL_ev'])
    )
    log_w = log_w.reshape((Nobs, nsamp))[:, idx_pe]
    log_w = jnp.where(jnp.isfinite(log_w), log_w, -1e30)
    log_num = 2.0 * logsumexp(log_w, axis=-1)
    log_den = logsumexp(2.0 * log_w, axis=-1)
    Neff_i = jnp.clip(jnp.exp(jnp.clip(log_num - log_den, a_min=-100, a_max=100)),
                      a_min=1.0)
    return float(jnp.sum(1.0 / Neff_i))


def per_event_inv_neff_pool(coord, pool_obj, idx_pop):
    m1p = pool_obj['m1s_pop'][idx_pop]
    m2p = pool_obj['m2s_pop'][idx_pop]
    zp  = pool_obj['zs_pop'][idx_pop]
    log_pop = log_p_pop(m1p, m2p, zp, coord)
    raw = log_pop + pool_obj['logJ'][idx_pop] - pool_obj['log_q'][idx_pop]
    log_w = jnp.where(jnp.isfinite(raw), raw, -jnp.inf)
    log_w_i = log_w[None, :] + pool_obj['logp_E_N'][:, idx_pop]
    log_num = 2.0 * logsumexp(log_w_i, axis=1)
    log_den = logsumexp(2.0 * log_w_i, axis=1)
    Neff_i = jnp.exp(log_num - log_den)
    return float(jnp.sum(1.0 / jnp.clip(Neff_i, a_min=1.0)))


def pe_full_variance(coord):
    log_w = log_p_pop(pe['m1_ev'], pe['m2_ev'], pe['z_ev'], coord)
    log_w += (
        -jnp.log(ddL_of_z(pe['z_ev'], pe['dL_ev'], H0Planck, Om0Planck))
        - 2.0 * jnp.log1p(pe['z_ev'])
        - 2.0 * jnp.log(pe['dL_ev'])
    )
    log_w = jnp.where(jnp.isfinite(log_w), log_w, -jnp.inf).reshape((Nobs, nsamp))
    logu = log_w - logsumexp(log_w, axis=1, keepdims=True)
    sum_u2 = jnp.exp(logsumexp(2.0 * logu, axis=1))
    V_i = (nsamp * sum_u2 - 1.0) / (nsamp - 1.0)
    return float(jnp.sum(V_i))


def pool_full_variance(coord, pool_obj):
    M = pool_obj['nsamp_pop']
    log_w = (
        log_p_pop(pool_obj['m1s_pop'], pool_obj['m2s_pop'], pool_obj['zs_pop'], coord)
        + pool_obj['logJ']
        - pool_obj['log_q']
    )
    log_w = jnp.where(jnp.isfinite(log_w), log_w, -jnp.inf)
    log_w_i = log_w[None, :] + pool_obj['logp_E_N']
    logu = log_w_i - logsumexp(log_w_i, axis=1, keepdims=True)
    D_k = jnp.sum(jnp.exp(logu), axis=0)
    V = (M * jnp.sum(D_k ** 2) - Nobs ** 2) / (M - 1.0)
    return float(V)


def pe_variance_sub(coord, idx_pe):
    N = int(idx_pe.shape[0])
    log_w = log_p_pop(pe['m1_ev'], pe['m2_ev'], pe['z_ev'], coord)
    log_w += (
        -jnp.log(ddL_of_z(pe['z_ev'], pe['dL_ev'], H0Planck, Om0Planck))
        - 2.0 * jnp.log1p(pe['z_ev'])
        - 2.0 * jnp.log(pe['dL_ev'])
    )
    log_w = jnp.where(jnp.isfinite(log_w), log_w, -jnp.inf).reshape((Nobs, nsamp))[:, idx_pe]
    logu = log_w - logsumexp(log_w, axis=1, keepdims=True)
    sum_u2 = jnp.exp(logsumexp(2.0 * logu, axis=1))
    V_i = (N * sum_u2 - 1.0) / (N - 1.0)
    return float(jnp.sum(V_i))


def pool_variance_sub(coord, pool_obj, idx_pop):
    M = int(idx_pop.shape[0])
    log_w = (
        log_p_pop(pool_obj['m1s_pop'][idx_pop], pool_obj['m2s_pop'][idx_pop],
                  pool_obj['zs_pop'][idx_pop], coord)
        + pool_obj['logJ'][idx_pop]
        - pool_obj['log_q'][idx_pop]
    )
    log_w = jnp.where(jnp.isfinite(log_w), log_w, -jnp.inf)
    log_w_i = log_w[None, :] + pool_obj['logp_E_N'][:, idx_pop]
    logu = log_w_i - logsumexp(log_w_i, axis=1, keepdims=True)
    D_k = jnp.sum(jnp.exp(logu), axis=0)
    V = (M * jnp.sum(D_k ** 2) - Nobs ** 2) / (M - 1.0)
    return float(V)


# ==========================================================
# 9. Load dynesty samplers
# ==========================================================
# Dummy callables must exist in the global namespace: NestedSampler.restore()
# unpickles references to them.
def prior_transform(u): return u
def loglike_method_1(theta): return 0.0
def loglike_method_3(theta): return 0.0
def loglike_method_4(theta): return 0.0


def _method_pattern(method):
    """Per-method filename pattern, with --method-patterns overriding --sampler-pattern."""
    for spec in (args.method_patterns or []):
        if not spec or "=" not in spec:
            continue
        m, pat = spec.split("=", 1)
        if m.strip() == method:
            return pat
    return args.sampler_pattern


def _sampler_path(method):
    return args.indir / _method_pattern(method).format(
        run=args.run, i=args.catalog, method=method)


def _restore(path: Path, *, required=True):
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"{path}\nAdjust --indir / --catalog / --sampler-pattern / "
                f"--method-patterns.")
        print(f"[samplers] {path.name} not found; skipping")
        return None
    return dynesty.NestedSampler.restore(str(path))


def _post_from_sampler(s):
    res = s.results
    w = np.exp(res['logwt'] - res['logz'][-1])
    return resample_equal(res.samples, w)


print(f"[samplers] catalog {args.catalog:03d}, indir = {args.indir}")
sampler1 = _restore(_sampler_path("met1"), required=True)
post1 = _post_from_sampler(sampler1)
print(f"[samplers] met1 loaded: {post1.shape}  ({_sampler_path('met1').name})")

sampler3 = _restore(_sampler_path("met3"), required=False)
post3 = _post_from_sampler(sampler3) if sampler3 is not None else None
if post3 is not None:
    print(f"[samplers] met3 loaded: {post3.shape}  ({_sampler_path('met3').name})")

sampler4 = _restore(_sampler_path("met4"), required=False)
post4 = _post_from_sampler(sampler4) if sampler4 is not None else None
if post4 is not None:
    print(f"[samplers] met4 loaded: {post4.shape}  ({_sampler_path('met4').name})")

for _p, _nm in ((post1, "met1"), (post3, "met3"), (post4, "met4")):
    if _p is not None and _p.shape[1] != ndims:
        raise SystemExit(
            f"[samplers] {_nm}: {_p.shape[1]} columns but the prior box for "
            f"run={args.run} has {ndims} ({labels}). Column order must be {labels}.")

print("[samplers] *** CONSISTENCY: the PE samples "
      f"({args.pe_m1det.name}, ...) must be the 69 events of catalog "
      f"{args.catalog:03d}. Nothing checks this. ***")


# ==========================================================
# 10. Variance vs N_samp plot
# ==========================================================
def make_variance_vs_Nsamp_plot():
    print("\n[variance] Building variance-vs-N_samp plot (empirical subsampling)...")
    n_test = args.variance_n_coords
    rng = np.random.default_rng(args.seed)

    methods = [("PE", post1, C[0], 's', 'pe', nsamp)]
    if post3 is not None:
        methods.append(("Ensemble", post3, C[1], 'o', 'pool', pool['nsamp_pop'] // Nobs))
    if post4 is not None:
        methods.append(("Mixture", post4, C[2], '^', 'pool', pool['nsamp_pop'] // Nobs))

    coord_sets = []
    for name, post, color, marker, kind, ceiling in methods:
        idx = rng.choice(post.shape[0], size=min(n_test, post.shape[0]), replace=False)
        coord_sets.append((name, jnp.asarray(post[idx]), color, marker, kind, ceiling))

    N_grid = np.unique(np.logspace(np.log10(10), np.log10(args.variance_nsamp_max),
                                   args.variance_nsamp_points).astype(int))
    N_grid = N_grid[N_grid >= 10]

    # ---- load or compute sweep ----
    cache_path = args.variance_results_cache
    if cache_path is not None and cache_path.exists():
        print(f"[variance] loading cached results from {cache_path}")
        d = np.load(cache_path, allow_pickle=False)
        results = {}
        _scalar_keys = {"Vf_med", "Vf_lo", "Vf_hi", "ceiling"}
        for name, *_ in coord_sets:
            results[name] = {
                k: d[f"{name}_{k}"].item() if k in _scalar_keys else d[f"{name}_{k}"]
                for k in ("x", "med", "lo", "hi", "Vf_med", "Vf_lo", "Vf_hi", "ceiling")
            }
    else:
        results = {name: {"x": [], "med": [], "lo": [], "hi": []}
                   for name, *_ in coord_sets}

        for N in tqdm(N_grid, desc="N_samp sweep"):
            for name, coords, _, _, kind, _ in coord_sets:
                if kind == 'pe':
                    if int(N) > nsamp:
                        continue
                    idx_pe = jnp.asarray(rng.choice(nsamp, size=int(N), replace=False))
                    vals = [pe_variance_sub(c, idx_pe) for c in coords]
                else:
                    M = int(N) * Nobs
                    if M > pool['nsamp_pop']:
                        continue
                    idx_pop = jnp.asarray(rng.choice(pool['nsamp_pop'], size=M, replace=False))
                    vals = [pool_variance_sub(c, pool, idx_pop) for c in coords]
                arr = np.asarray(vals)
                results[name]["x"].append(N)
                results[name]["med"].append(np.nanmedian(arr))
                results[name]["lo"].append(np.nanpercentile(arr, 16))
                results[name]["hi"].append(np.nanpercentile(arr, 84))

        # full-N variance for the analytical 1/N overlay
        for name, coords, _, _, kind, ceiling in coord_sets:
            Vf = np.array([pe_full_variance(c) if kind == 'pe'
                           else pool_full_variance(c, pool)
                           for c in tqdm(coords, desc=f"{name} V_full")])
            results[name]["Vf_med"] = np.nanmedian(Vf)
            results[name]["Vf_lo"]  = np.nanpercentile(Vf, 16)
            results[name]["Vf_hi"]  = np.nanpercentile(Vf, 84)
            results[name]["ceiling"] = ceiling

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_dict = {}
            for name in results:
                for k in ("x", "med", "lo", "hi", "Vf_med", "Vf_lo", "Vf_hi", "ceiling"):
                    save_dict[f"{name}_{k}"] = np.atleast_1d(np.array(results[name][k]))
            np.savez_compressed(cache_path, **save_dict)
            print(f"[variance] saved results cache -> {cache_path}")

    # ---- plot ----
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for name, _, color, marker, _, _ in coord_sets:
        x    = np.array(results[name]["x"])
        med  = np.array(results[name]["med"])
        lo   = np.array(results[name]["lo"])
        hi   = np.array(results[name]["hi"])

        err = np.vstack([med - lo, hi - med])
        ax.errorbar(x, med, yerr=err, marker=marker, ls='-', color=color, lw=2,
                    capsize=3, elinewidth=1.2, label=rf'$\mathrm{{{name}}}$')

        # analytical 1/N band
        Vf_med = float(results[name]["Vf_med"])
        Vf_lo  = float(results[name]["Vf_lo"])
        Vf_hi  = float(results[name]["Vf_hi"])
        ceil_N = int(results[name]["ceiling"])
        Ns_an  = np.logspace(np.log10(max(x.min(), 10)), np.log10(x.max()), 300)
        ax.plot(Ns_an, Vf_med * ceil_N / Ns_an,
                color=color, ls='--', lw=1.5, alpha=0.85)
        ax.fill_between(Ns_an, Vf_lo * ceil_N / Ns_an, Vf_hi * ceil_N / Ns_an,
                        color=color, alpha=0.15)

    ax.axvline(nsamp, color='k', ls='--', lw=1.5, alpha=0.8,
               label=rf'$\mathrm{{PE\ ceiling}}\ (N={nsamp})$')
    ax.set_xscale('log')
    ax.grid(True, which='both', ls=':', alpha=0.3)
    ax.axhline(1.0, color='0.5', ls='--', lw=1.0, alpha=0.5)
    ax.set_yscale('log')
    ax.set_xlabel(r'$N_{\rm samp}$')
    ax.set_ylabel(r'$\sigma^2_{\ln\mathcal{L}_{\rm pe}}$')
    ax.legend(loc='lower left')
    plt.tight_layout()
    out = args.outdir / f"{args.run}_variance_Nsamp.pdf"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[variance] saved -> {out}")


# ==========================================================
# 10b. Variance vs N_samp -- analytical 1/N law
# ==========================================================
def make_variance_analytical_plot():
    print("\n[variance-analytic] Building analytical 1/N variance plot...")
    rng = np.random.default_rng(args.seed)
    n_test = args.variance_n_coords

    methods = [("PE", post1, C[0], 's', 'pe', nsamp, False)]
    if post3 is not None:
        methods.append(("Ensemble", post3, C[1], 'o', 'pool', pool['nsamp_pop'] // Nobs, True))
    if post4 is not None:
        methods.append(("Mixture", post4, C[2], '^', 'pool', pool['nsamp_pop'] // Nobs, True))

    N_grid = np.unique(np.logspace(np.log10(10), np.log10(args.variance_nsamp_max),
                                   args.variance_nsamp_points).astype(int))
    fig, ax = plt.subplots(figsize=(10, 7))

    for name, post, color, marker, kind, ceiling, extrap in methods:
        idx = rng.choice(post.shape[0], size=min(n_test, post.shape[0]), replace=False)
        coords = jnp.asarray(post[idx])
        Vf = np.array([pe_full_variance(c) if kind == 'pe' else pool_full_variance(c, pool)
                       for c in tqdm(coords, desc=f"{name} V_full")])
        med = np.nanmedian(Vf)
        lo  = np.nanpercentile(Vf, 16)
        hi  = np.nanpercentile(Vf, 84)
        Ns = N_grid if extrap else N_grid[N_grid <= ceiling]
        med_curve, lo_curve, hi_curve = med * ceiling / Ns, lo * ceiling / Ns, hi * ceiling / Ns
        ax.plot(Ns, med_curve, color=color, lw=2, marker=marker, ms=5,
                label=rf'$\mathrm{{{name}}}$')
        ax.fill_between(Ns, lo_curve, hi_curve, color=color, alpha=0.2)
        ax.plot([ceiling], [med], color=color, marker='*', ms=14, mec='k', zorder=6)

    ax.axvline(nsamp, color='k', ls=':', lw=2, alpha=0.7,
               label=fr'$\mathrm{{PE\ ceiling}}\ (N={nsamp})$')
    ax.axhline(1.0, color='0.5', ls='--', lw=1.0, alpha=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$N_{\rm samp}$')
    ax.set_ylabel(r'$\sigma^2_{\ln\mathcal{L}_{\rm pe}}$')
    ax.grid(True, which='both', ls=':', alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()
    out = args.outdir / f"{args.run}_variance_analytic.pdf"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[variance-analytic] saved -> {out}")


# ==========================================================
# 10c. Term (B): bootstrap fit-variance floor
# ==========================================================
# The estimator is unbiased, at fixed fit, for the SMOOTHED integral
#     L[phat_i](Lambda) = int phat_i(theta) p_pop(theta|Lambda)/pi(theta) dtheta,
# not for the true L[p_i].  Its error therefore decomposes as
#     MSE = (A) E_D[v_i]/N_samp        <- conditional MC, the 1/N curves above
#         + (B) Var_D[ ln L[phat_i] ]  <- fit variance, THIS SECTION
#         + b^2                        <- smoothing bias (needs the PE-difference probe)
# (A) vanishes as N_samp -> inf; (B) does not.  It is set by N_PE and is the
# floor the 1/N curves descend towards.  Estimated by bootstrap-resampling the
# EXISTING PE samples -- no parameter estimation is rerun.
K_PER_EVENT_FITVAR = 7      # must match _build_pool
N_INIT_FITVAR      = 5      # must match _build_pool
REG_COVAR_FITVAR   = 1e-6   # must match _build_pool


def _gmm_to_jax(g):
    """sklearn GaussianMixture -> the arrays exact_log_gmm() expects."""
    logdet = -2.0 * np.sum(
        np.log(np.diagonal(g.precisions_cholesky_, axis1=-2, axis2=-1)), axis=-1
    )
    return (jnp.asarray(g.weights_,    dtype=jnp.float64),
            jnp.asarray(g.means_,      dtype=jnp.float64),
            jnp.asarray(g.precisions_, dtype=jnp.float64),
            jnp.asarray(logdet,        dtype=jnp.float64))


def _pe_detframe(e):
    """Event e's PE samples in the detector frame the GMMs are fitted in."""
    m1d = np.asarray(pe['m1det_ev']).reshape(Nobs, nsamp)[e]
    m2d = np.asarray(pe['m2det_ev']).reshape(Nobs, nsamp)[e]
    dLd = np.asarray(pe['dL_ev']).reshape(Nobs, nsamp)[e]
    return np.column_stack([m1d, m2d, dLd])


def _eval_pool_detframe(pool_obj, M):
    """
    Detector-frame coordinates of the first M samples of the given evaluation
    pool.  The pool stores source-frame quantities; m1det = m1src (1+z),
    m2det = m2src (1+z), dL = dL.  Reconstructing rather than re-drawing keeps
    the evaluation pool IDENTICAL across bootstraps -- this is the
    common-random-numbers trick that makes the stage-2 MC noise common-mode
    so it largely cancels from the spread over b.
    """
    z = np.asarray(pool_obj['zs_pop'][:M])
    return np.column_stack([
        np.asarray(pool_obj['m1s_pop'][:M]) * (1.0 + z),
        np.asarray(pool_obj['m2s_pop'][:M]) * (1.0 + z),
        np.asarray(pool_obj['dLs_pop'][:M]),
    ])


def fitvar_moment_check(n_events=5, n_draw=400_000):
    """
    Validation of the whole density path, free of charge.

    At an EM fixed point the responsibilities satisfy sum_k r_jk = 1, and the
    M-step gives a_k = (1/N) sum_j r_jk and mu_k = sum_j r_jk theta_j / sum_j r_jk.
    Hence sum_k a_k mu_k = (1/N) sum_j theta_j EXACTLY: a converged GMM reproduces
    the first two moments of its training data.  If this fails, either EM has not
    converged or exact_log_gmm / the frame handling is wrong -- and every bootstrap
    number downstream is worthless.  Checked analytically (via a_k, mu_k, Sigma_k)
    and by sampling.
    """
    print("\n[fitvar] EM moment identity check "
          "(GMM must reproduce training-data mean & covariance)...")
    rng = np.random.default_rng(args.seed)
    ok = True
    for e in range(min(n_events, Nobs)):
        X = _pe_detframe(e)
        g = GaussianMixture(n_components=K_PER_EVENT_FITVAR, covariance_type='full',
                            reg_covar=REG_COVAR_FITVAR, n_init=N_INIT_FITVAR,
                            random_state=args.seed).fit(X)

        a, mu, S = g.weights_, g.means_, g.covariances_
        mu_an = a @ mu                                        # analytic GMM mean
        C_an = np.einsum('k,kij->ij', a, S) \
             + np.einsum('k,ki,kj->ij', a, mu - mu_an, mu - mu_an)

        Xs, _ = g.sample(n_draw)                              # sampling path
        mu_s, C_s = Xs.mean(0), np.cov(Xs.T)

        d_mu = np.abs(mu_an - X.mean(0)) / np.abs(X.mean(0))
        d_C  = np.abs(C_an - np.cov(X.T)) / np.abs(np.cov(X.T))
        d_s  = np.abs(mu_s - X.mean(0)) / np.abs(X.mean(0))
        bad = (d_mu.max() > 1e-5) or (d_C.max() > 1e-3)
        ok &= not bad
        print(f"  event {e:3d}: |dmean|_max = {d_mu.max():.2e}  "
              f"|dcov|_max = {d_C.max():.2e}  "
              f"(sampled mean: {d_s.max():.2e})  {'FAIL' if bad else 'ok'}")
    print(f"[fitvar] moment check: {'PASSED' if ok else '*** FAILED ***'}\n")
    return ok


# ----------------------------------------------------------------------
# Defensive evaluation proposal q* = (1-f) q_E + f q_broad
# ----------------------------------------------------------------------
# The shared Ensemble pool built in section 7 uses q_E = (1/Nobs) sum_k phat_k
# as its proposal.  q_E's weights are unbounded exactly where the population
# model has support and an individual event's PE does not -- the m_min edge --
# which is precisely where common random numbers fail to cancel the stage-2
# MC noise (N_eff << M).  q* bounds the weights (defensive importance
# sampling; Hesterberg 1995, Owen & Zhou 2000) at the cost of a (1-f)^-1
# inflation in the bulk.  L[phat_i] does not depend on q at all -- q only
# ever appears inside the estimator of L[phat_i], and cancels in expectation
# -- so q* changes only the noise of this measurement, never the quantity
# being measured.  It is built and used ONLY here, never for the (A) curves.

def _fit_qE_components(seed):
    """
    Per-event K=7 full-data GMMs defining q_E = (1/Nobs) sum_e phat_e.
    Matches _build_pool's per-event fit exactly (same settings, same data),
    so this q_E is identical to the production Ensemble proposal -- refit
    here only because _build_pool does not expose the per-component arrays.
    """
    print("[fitvar] fitting q_E per-event GMMs (matches production Ensemble)...")
    gmms = [GaussianMixture(
                n_components=K_PER_EVENT_FITVAR, covariance_type='full',
                reg_covar=REG_COVAR_FITVAR, n_init=N_INIT_FITVAR,
                random_state=seed).fit(_pe_detframe(e))
            for e in tqdm(range(Nobs), desc="q_E per-event GMMs")]
    wts    = jnp.asarray(np.stack([g.weights_    for g in gmms]), dtype=jnp.float64)
    mus    = jnp.asarray(np.stack([g.means_      for g in gmms]), dtype=jnp.float64)
    prec   = jnp.asarray(np.stack([g.precisions_ for g in gmms]), dtype=jnp.float64)
    logdet = jnp.asarray(np.stack([
        -2.0 * np.sum(np.log(np.diagonal(g.precisions_cholesky_, axis1=-2, axis2=-1)), axis=-1)
        for g in gmms]), dtype=jnp.float64)
    return gmms, wts, mus, prec, logdet


def _log_qE(x_j, wts, mus, prec, logdet):
    """log q_E(x) = log[ (1/Nobs) sum_e phat_e(x) ], matching pool['log_q']."""
    logp_e = jnp.stack([exact_log_gmm(x_j, wts[e], mus[e], prec[e], logdet[e])
                        for e in range(Nobs)])
    return logsumexp(logp_e, axis=0) - jnp.log(Nobs)


def _fit_q_broad(inflate, seed):
    """
    K=3 full-covariance GMM on the pooled PE samples of ALL events, with
    covariances inflated by `inflate` so it retains support in the tails
    where individual q_E components vanish. Cheap, closed-form: the fit is
    done once on the un-inflated data, then the precision/logdet are rescaled
    analytically (Sigma -> c*Sigma  =>  Prec -> Prec/c, logdet -> logdet + D ln c).
    """
    print(f"[fitvar] fitting q_broad (K=3, pooled PE samples, "
          f"covariance x{inflate})...")
    X_all = np.vstack([_pe_detframe(e) for e in range(Nobs)])
    g = GaussianMixture(n_components=3, covariance_type='full',
                        reg_covar=REG_COVAR_FITVAR, n_init=N_INIT_FITVAR,
                        random_state=seed).fit(X_all)
    D = X_all.shape[1]
    logdet0 = -2.0 * np.sum(
        np.log(np.diagonal(g.precisions_cholesky_, axis1=-2, axis2=-1)), axis=-1)
    return dict(
        weights=jnp.asarray(g.weights_,          dtype=jnp.float64),
        means=jnp.asarray(g.means_,               dtype=jnp.float64),
        prec=jnp.asarray(g.precisions_ / inflate,  dtype=jnp.float64),
        logdet=jnp.asarray(logdet0 + D * np.log(inflate), dtype=jnp.float64),
        weights_np=g.weights_, means_np=g.means_, cov_np=g.covariances_ * inflate,
    )


def _build_qstar_pool(M, f, inflate, seed):
    """
    Defensive evaluation pool for term (B) only.  Draws (1-f)*M points from
    q_E (same per-event-GMM mechanism as _build_pool) and f*M from q_broad,
    then evaluates log q* = logaddexp(log(1-f) + log q_E, log(f) + log q_broad)
    at EVERY point regardless of which component it came from -- a mixture
    density must be evaluated as the mixture, not as its originating
    component, or the importance weights are simply wrong.
    """
    print(f"[fitvar] building defensive q* pool: M={M}, f={f}, "
          f"broad covariance inflation={inflate}")
    rng = np.random.default_rng(seed)

    gmms_qE, wts_qE, mus_qE, prec_qE, logdet_qE = _fit_qE_components(seed)
    qb = _fit_q_broad(inflate, seed)

    M_qE = int(round((1.0 - f) * M))
    M_qb = M - M_qE

    per_event = M_qE // Nobs
    leftover  = M_qE - per_event * Nobs
    sizes = np.full(Nobs, per_event)
    sizes[:leftover] += 1
    chunks = [gmms_qE[e].sample(sizes[e])[0] for e in range(Nobs) if sizes[e] > 0]
    X_qE = np.vstack(chunks) if chunks else np.zeros((0, 3))

    comp = rng.choice(3, size=M_qb, p=qb['weights_np'])
    X_qb = np.zeros((M_qb, 3))
    for k in range(3):
        sel = comp == k
        n_k = int(sel.sum())
        if n_k:
            X_qb[sel] = rng.multivariate_normal(qb['means_np'][k], qb['cov_np'][k], size=n_k)

    X = np.vstack([X_qE, X_qb])
    X_j = jnp.asarray(X)

    log_qE_x = np.asarray(_log_qE(X_j, wts_qE, mus_qE, prec_qE, logdet_qE))
    log_qb_x = np.asarray(exact_log_gmm(X_j, qb['weights'], qb['means'], qb['prec'], qb['logdet']))
    log_qstar = np.logaddexp(np.log(1.0 - f) + log_qE_x, np.log(f) + log_qb_x)

    m1det, m2det, dL_p = X[:, 0], X[:, 1], X[:, 2]
    zs = np.asarray(z_of_dL(jnp.asarray(dL_p), H0Planck, Om0Planck))
    m1s, m2s = m1det / (1.0 + zs), m2det / (1.0 + zs)
    logJ = np.asarray(
        -jnp.log(ddL_of_z(jnp.asarray(zs), jnp.asarray(dL_p), H0Planck, Om0Planck))
        - 2.0 * jnp.log1p(jnp.asarray(zs)) - 2.0 * jnp.log(jnp.asarray(dL_p))
    )
    valid = (
        (m1s >= 6.0) & (m1s <= 70.0) &
        (m2s >= 6.0) & (m2s <= m1s) &
        (zs  >= 0.0) & (zs  <= 1.9) &
        (dL_p >= 50.0) & np.isfinite(logJ)
    )
    print(f"[fitvar] q* pool valid: {valid.sum()}/{valid.size} "
          f"({100.0 * valid.mean():.1f}%)")

    # The M-scan (Task 4) slices the first-M PREFIX of this pool at every scan
    # point.  X was built as [q_E samples][q_broad samples]: without shuffling,
    # every prefix shorter than the (validity-thinned) q_E block would contain
    # NO defensive samples at all, silently undoing the whole point of q* for
    # every M except the very last (full) one. Permute once, consistently
    # across every array, so any prefix is a representative mixed draw.
    perm = rng.permutation(int(valid.sum()))

    return dict(
        m1s_pop=jnp.asarray(m1s[valid][perm]), m2s_pop=jnp.asarray(m2s[valid][perm]),
        zs_pop=jnp.asarray(zs[valid][perm]),   dLs_pop=jnp.asarray(dL_p[valid][perm]),
        log_q=jnp.asarray(log_qstar[valid][perm]), logJ=jnp.asarray(logJ[valid][perm]),
        nsamp_pop=int(valid.sum()),
        gmms_qE=gmms_qE,
    )


@jax.jit
def _lnL_chunk(log_h_CM, lp_BM):
    """log_h_CM (C, M), lp_BM (B, M) -> (C, B) unnormalised log-integrals."""
    return logsumexp(log_h_CM[:, None, :] + lp_BM[None, :, :], axis=-1)


def _log_h(coords, M, pool_obj):
    """
    log[ p_pop(theta_j|Lambda) J_j / q(theta_j) ] at the first M samples of
    the given evaluation pool.  This is the factor shared by every event and
    every bootstrap; all the Lambda-dependence lives here, which is what
    makes the sweep affordable.
    """
    m1 = pool_obj['m1s_pop'][:M]; m2 = pool_obj['m2s_pop'][:M]; z = pool_obj['zs_pop'][:M]
    base = pool_obj['logJ'][:M] - pool_obj['log_q'][:M]
    out = np.zeros((len(coords), M))
    for c, coord in enumerate(tqdm(coords, desc="log_h(Lambda)")):
        lw = log_p_pop(m1, m2, z, coord) + base
        out[c] = np.asarray(jnp.where(jnp.isfinite(lw), lw, -jnp.inf))
    return out


def _fitvar_bootstrap_and_scan(coords, Ms, qstar_pool, n_boot, train_frac,
                                vary_seed, n_jobs, seed, chunk, gmms_qE):
    """
    Term (B), event-outer, with the 1/M convergence MEASURED at a scan of
    pool sizes instead of assumed from a two-point Richardson extrapolation.

    For each event: fit n_boot bootstrap GMMs (parallel over b), evaluate all
    of them on the SAME fixed M_max-sample q* pool -> a (n_boot, M_max)
    float32 block (100 MB at n_boot=50, M_max=500000 -- not the 6.9 GB of
    the old (n_boot, Nobs, M_max) cache).  The smaller M's in the scan are
    free: just slice the first M columns of the block and of log_h.  Training
    sets are independent across events, so summing per-event variances is
    exactly right -- there is no cross-event covariance at the fit level.

    Returns
    -------
    V_of_M    : (n_M, n_coords)       summed over events
    V_iC_of_M : (n_M, n_coords, Nobs) per-event breakdown
    n_train   : int
    """
    n_train = int(round(train_frac * nsamp))
    M_max = Ms[-1]
    print(f"[fitvar] event-outer bootstrap: n_boot={n_boot}, n_train={n_train}/{nsamp} "
          f"per event, M_max={M_max}, Ms={list(Ms)}, "
          f"EM seed {'varying' if vary_seed else 'fixed'} "
          f"(WARM-STARTED from production fits)")
    if n_train != nsamp:
        print(f"[fitvar] WARNING: train_frac={train_frac} => B is inflated by "
              f"{nsamp / n_train:.2f}x relative to fitting on all samples. "
              f"Only use this if PRODUCTION also trains on a subset.")

    X_eval_j = jnp.asarray(_eval_pool_detframe(qstar_pool, M_max))
    log_h_full = _log_h(coords, M_max, qstar_pool)     # (n_coords, M_max), shared
    Xs = [_pe_detframe(e) for e in range(Nobs)]

    def _one(e, b):
        rs = np.random.default_rng([seed, e, b])   # reproducible + parallel-safe
        idx = rs.integers(0, nsamp, size=n_train)   # WITH replacement
        em_seed = int(rs.integers(0, 2**31 - 1)) if vary_seed else seed
        g0 = gmms_qE[e]
        return b, GaussianMixture(
            n_components=K_PER_EVENT_FITVAR, covariance_type='full',
            reg_covar=REG_COVAR_FITVAR, n_init=1,
            weights_init=g0.weights_, means_init=g0.means_, precisions_init=g0.precisions_,
            random_state=em_seed).fit(Xs[e][idx])

    n_M = len(Ms)
    V_of_M    = np.zeros((n_M, len(coords)))
    V_iC_of_M = np.zeros((n_M, len(coords), Nobs))

    for e in tqdm(range(Nobs), desc="events (fit-variance, event-outer)"):
        if n_jobs != 1:
            from joblib import Parallel, delayed
            fits = Parallel(n_jobs=n_jobs)(delayed(_one)(e, b) for b in range(n_boot))
        else:
            fits = [_one(e, b) for b in range(n_boot)]
        fits.sort(key=lambda t: t[0])

        logp_block = np.zeros((n_boot, M_max), dtype=np.float32)
        for b, g in fits:
            logp_block[b] = np.asarray(
                exact_log_gmm(X_eval_j, *_gmm_to_jax(g)), dtype=np.float32)

        for m_idx, M in enumerate(Ms):
            lnL = np.zeros((n_boot, len(coords)))
            lp_M = jnp.asarray(logp_block[:, :M], dtype=jnp.float64)
            for c0 in range(0, len(coords), chunk):
                c1 = min(c0 + chunk, len(coords))
                lnL[:, c0:c1] = np.asarray(
                    _lnL_chunk(jnp.asarray(log_h_full[c0:c1, :M]), lp_M)
                ).T - np.log(M)
            V_e = np.var(lnL, axis=0, ddof=1)          # (n_coords,)
            V_of_M[m_idx]           += V_e
            V_iC_of_M[m_idx, :, e]   = V_e

        del logp_block

    return V_of_M, V_iC_of_M, n_train


def _fitvar_coords():
    """
    Lambda draws for the (B) sweep.

    Default 'posterior': the SAME ensemble the (A) curves are evaluated on, so the
    floor and the curves are directly comparable.  (Sampling the prior box instead
    answers a different question and must not be overlaid on the (A) curves.)
    """
    rng = np.random.default_rng(args.seed)
    n_c = args.fitvar_n_coords
    if args.fitvar_coord_source == "posterior":
        post = post4 if post4 is not None else (post3 if post3 is not None else post1)
        src = "met4" if post4 is not None else ("met3" if post3 is not None else "met1")
        idx = rng.choice(post.shape[0], size=min(n_c, post.shape[0]), replace=False)
        print(f"[fitvar] Lambda draws: {len(idx)} from the {src} hyper-posterior")
        return [jnp.asarray(post[i]) for i in idx], np.asarray(post[idx])
    samples = {lab: rng.uniform(lo, hi, n_c)
               for lab, lo, hi in zip(labels, lower_bound, upper_bound)}
    print(f"[fitvar] Lambda draws: {n_c} from the PRIOR box "
          f"(not comparable with the posterior-averaged (A) curves)")
    arr = np.column_stack([samples[lab] for lab in labels])
    return [jnp.asarray(arr[i]) for i in range(n_c)], arr


def compute_fit_variance():
    """
    Bootstrap estimate of term (B), with the 1/M convergence MEASURED rather
    than assumed: Vhat(M) = B + c/M, fit by OLS over a scan of pool sizes
    (--fitvar-m-scan). The intercept is B; the slope c is the residual
    conditional-MC contamination that survives common random numbers. If the
    intercept is not resolved above its own standard error, the pool is too
    small and the caller must report an upper limit instead of a number.
    Uses the defensive q* evaluation pool (Task 3), not the shared Ensemble
    pool, and the event-outer bootstrap (Task 2) to bound memory.
    """
    cache = args.fitvar_results_cache
    if cache is not None and cache.exists():
        print(f"[fitvar] loading cached results from {cache}")
        d = np.load(cache, allow_pickle=False)
        out = {k: d[k] for k in d.files}
        out["n_train"] = int(out["n_train"])
        out["M_max"] = int(out["M_max"])
        return out

    M_max = int(args.fitvar_pool_size)
    qstar_pool = _build_qstar_pool(M_max, args.fitvar_defensive_frac,
                                    args.fitvar_broad_inflate, args.seed)
    M_max = min(M_max, qstar_pool['nsamp_pop'])

    Ms = sorted({int(round(frac * M_max)) for frac in args.fitvar_m_scan
                if frac * M_max >= 500})
    if len(Ms) < 3:
        raise SystemExit(f"[fitvar] need >= 3 pool sizes in --fitvar-m-scan; "
                         f"got {Ms} from M_max={M_max}. Build a bigger pool "
                         f"(--fitvar-pool-size) or a bigger --n-pool.")

    coords, coord_arr = _fitvar_coords()

    V_of_M, V_iC_of_M, n_train = _fitvar_bootstrap_and_scan(
        coords, Ms, qstar_pool, args.fitvar_nboot, args.fitvar_train_frac,
        args.fitvar_vary_seed, args.fitvar_n_jobs, args.seed, args.fitvar_coord_chunk,
        qstar_pool['gmms_qE'])

    # ---- regress V(M) = B + c*(1/M), per Lambda -----------------------------
    x = 1.0 / np.asarray(Ms, dtype=float)
    X = np.vstack([np.ones_like(x), x]).T
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ X.T @ V_of_M                     # (2, n_coords)
    B_hat, c_hat = coef[0], coef[1]

    resid = V_of_M - X @ coef
    dof = max(len(Ms) - 2, 1)
    s2 = np.sum(resid ** 2, axis=0) / dof
    B_err = np.sqrt(np.clip(XtX_inv[0, 0] * s2, 0.0, None))

    # PE's total error on the SAME Lambda draws -- the reference the floor is
    # measured against.  rho = B / sigma^2_PE is the smoothing reduction factor.
    print("[fitvar] PE variance on the same Lambda draws (reference)...")
    V_pe = np.array([pe_full_variance(c) for c in tqdm(coords, desc="PE V_full")])

    res = dict(Ms=np.asarray(Ms), V_of_M=V_of_M, V_iC_of_M=V_iC_of_M,
               B_hat=B_hat, B_err=B_err, c_hat=c_hat, V_pe=V_pe,
               coords=coord_arr, M_max=M_max, n_train=n_train)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **{k: np.atleast_1d(np.asarray(v))
                                      for k, v in res.items()})
        print(f"[fitvar] saved results cache -> {cache}")
    return res


def make_fitvar_plot():
    """
    Three panels; crash-proof against an unresolved (empty/noise-dominated)
    floor -- the code must produce a figure and a verdict in every case.

    Left   -- the budget: (A) 1/N curves, the (B) floor as a BAND (16-84
              across Lambda -- (B) is a function of Lambda, not a constant)
              or, if unresolved, a dash-dot 2-sigma UPPER LIMIT, plus the
              (A)+(B) total per GMM method.  PE gets no floor: its estimator
              is the empirical measure, so its (A) curve IS its total error.
    Middle -- the convergence: Vhat(M) against 1/M, with the OLS line and the
              intercept marked.  A line through the origin means the floor is
              unresolved.  Read this panel BEFORE believing the left one.
    Right  -- the trade: B(Lambda) against sigma^2_PE(Lambda), log-log, with
              the diagonal, coloured by m_min.  Points below the diagonal are
              the variance reduction bought by smoothing.
    """
    print("\n[fitvar] Building variance-budget plot...")
    fv = compute_fit_variance()
    Ms     = np.asarray(fv["Ms"])
    V_of_M = np.asarray(fv["V_of_M"])
    B_hat  = np.asarray(fv["B_hat"])
    B_err  = np.asarray(fv["B_err"])
    c_hat  = np.asarray(fv["c_hat"])
    V_pe   = np.asarray(fv["V_pe"])

    finite_B = B_hat[np.isfinite(B_hat)]
    finite_e = B_err[np.isfinite(B_err)]
    snr = (np.nanmedian(finite_B) / max(np.nanmedian(finite_e), 1e-300)
           if finite_B.size and finite_e.size else 0.0)
    resolved = bool(finite_B.size) and (np.nanmedian(finite_B) > 0) and (snr > args.fitvar_min_snr)
    b_med = float(np.nanmedian(B_hat)) if resolved else float("nan")
    b_lo  = float(np.nanpercentile(B_hat, 16)) if resolved else float("nan")
    b_hi  = float(np.nanpercentile(B_hat, 84)) if resolved else float("nan")
    ul_terms = B_hat + 2.0 * B_err
    b_ul = float(np.nanmedian(ul_terms)) if np.isfinite(ul_terms).any() else float("nan")

    fig, (ax, axm, ax2) = plt.subplots(1, 3, figsize=(21, 6.5), constrained_layout=True)

    # ================= left: the budget =================
    rng = np.random.default_rng(args.seed)
    methods = [("PE", post1, C[0], 's', 'pe', nsamp, False)]
    if post3 is not None:
        methods.append(("Ensemble", post3, C[1], 'o', 'pool',
                        pool['nsamp_pop'] // Nobs, True))
    if post4 is not None:
        methods.append(("Mixture", post4, C[2], '^', 'pool',
                        pool['nsamp_pop'] // Nobs, True))
    N_grid = np.unique(np.logspace(np.log10(10), np.log10(args.variance_nsamp_max),
                                   args.variance_nsamp_points).astype(int))
    pe_ceiling, crossings = None, {}

    for name, post, color, marker, kind, ceiling, extrap in methods:
        idx = rng.choice(post.shape[0],
                         size=min(args.variance_n_coords, post.shape[0]), replace=False)
        cds = jnp.asarray(post[idx])
        Vf = np.array([pe_full_variance(c) if kind == 'pe' else pool_full_variance(c, pool)
                       for c in tqdm(cds, desc=f"{name} V_full")])
        med = np.nanmedian(Vf); lo = np.nanpercentile(Vf, 16); hi = np.nanpercentile(Vf, 84)
        Ns = N_grid if extrap else N_grid[N_grid <= ceiling]
        A = med * ceiling / Ns
        ax.plot(Ns, A, color=color, lw=2, marker=marker, ms=5,
                label=rf'$\mathrm{{{name}}}$: (A)')
        ax.fill_between(Ns, lo * ceiling / Ns, hi * ceiling / Ns, color=color, alpha=0.15)
        if kind == 'pe':
            pe_ceiling = med
        elif resolved and b_med > 0:
            ax.plot(Ns, A + b_med, color=color, lw=2.4, ls='--', alpha=0.95,
                    label=rf'$\mathrm{{{name}}}$: (A)+(B)')
            crossings[name] = med * ceiling / b_med

    if resolved:
        ax.axhline(b_med, color='k', ls='-', lw=2.0, alpha=0.9,
                   label=r'$\mathrm{(B)\ fit\ variance}$')
        ax.axhspan(b_lo, b_hi, color='k', alpha=0.12)
    elif np.isfinite(b_ul):
        ax.axhline(b_ul, color='k', ls='-.', lw=2.0, alpha=0.9,
                   label=r'$\mathrm{(B)}$: $2\sigma$ upper limit')
        ax.axhspan(0, b_ul, color='k', alpha=0.07)
    ax.axvline(nsamp, color='k', ls=':', lw=2, alpha=0.7, label=fr'$N_{{\rm PE}}={nsamp}$')
    ax.axhline(1.0, color='0.5', ls='--', lw=1.0, alpha=0.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$N_{\rm samp}$'); ax.set_ylabel(r'$\sigma^2_{\ln\hat{\mathcal{L}}}$')
    ax.grid(True, which='both', ls=':', alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)

    # ================= middle: convergence in 1/M =================
    x = 1.0 / Ms.astype(float)
    Vq_valid = np.isfinite(V_of_M) & (V_of_M != 0)
    if Vq_valid.any():
        qs = [16, 50, 84]
        Vq = np.nanpercentile(V_of_M, qs, axis=1)
        axm.errorbar(x, Vq[1], yerr=[Vq[1] - Vq[0], Vq[2] - Vq[1]],
                     marker='o', color=C[3], lw=2, capsize=4,
                     label=r'$\hat{V}(M)$, median over $\Lambda$')
    bm = float(np.nanmedian(B_hat)) if np.isfinite(B_hat).any() else 0.0
    cm = float(np.nanmedian(c_hat)) if np.isfinite(c_hat).any() else 0.0
    xx = np.linspace(0, x.max() * 1.05, 50)
    axm.plot(xx, bm + cm * xx, 'k--', lw=1.5, label=r'$\hat{V}=B+c/M$')
    axm.axhline(0.0, color='0.6', lw=1.0)
    axm.plot([0], [bm], marker='*', ms=18, color='k', zorder=5,
             label=rf'$B={bm:.3g}$' if resolved else 'intercept (unresolved)')
    axm.set_xlim(left=0)
    axm.set_xlabel(r'$1/M$')
    axm.set_ylabel(r'$\hat{V}(M)$  [measured bootstrap spread]')
    axm.set_title('unresolved: line passes through origin' if not resolved
                  else 'resolved: nonzero intercept', fontsize=13)
    axm.grid(True, ls=':', alpha=0.3)
    axm.legend(fontsize=11)

    # ================= right: the trade =================
    good = np.isfinite(V_pe) & np.isfinite(B_hat) & (V_pe > 0)
    if good.any():
        yy = np.clip(B_hat[good], 1e-12, None)
        try:
            ci = labels.index('m_min')
            sc = ax2.scatter(V_pe[good], yy, c=np.asarray(fv["coords"])[good, ci],
                             cmap='viridis', s=18, alpha=0.75, edgecolors='none')
            plt.colorbar(sc, ax=ax2, label=r'$m_{\min}$')
        except (ValueError, IndexError):
            ax2.scatter(V_pe[good], yy, color=C[2], s=18, alpha=0.7, edgecolors='none')
        lim = [max(min(np.nanmin(V_pe[good]), np.nanmin(yy)) * 0.5, 1e-8),
               max(np.nanmax(V_pe[good]), np.nanmax(yy)) * 2]
        ax2.plot(lim, lim, 'k--', lw=1.2, alpha=0.7,
                 label=r'$\mathrm{(B)}=\sigma^2_{\rm PE}$')
        ax2.set_xlim(lim); ax2.set_ylim(lim)
        ax2.legend(loc='upper left', fontsize=11)
    else:
        ax2.text(0.5, 0.5, 'no finite (B), PE pairs', ha='center', va='center',
                 transform=ax2.transAxes)
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel(r'$\sigma^2_{\rm PE}(\Lambda)$')
    ax2.set_ylabel(r'$\mathrm{(B)}(\Lambda)$')
    ax2.grid(True, which='both', ls=':', alpha=0.3)

    out = args.outdir / f"{args.run}_variance_budget.pdf"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"[fitvar] saved -> {out}")

    # ================= the numbers =================
    print("\n" + "=" * 70)
    print(f"  TERM (B)   [Lambda from {args.fitvar_coord_source}, "
          f"n_boot={args.fitvar_nboot}, M_max={fv['M_max']}, N_train={fv['n_train']}]")
    print("=" * 70)
    print(f"  pool sizes scanned          : {list(Ms)}")
    print(f"  V(M) median at each M       : "
          + ", ".join(f"{v:.3g}" for v in np.nanmedian(V_of_M, axis=1)))
    print(f"  slope c (residual MC)       : {cm:.4g}")
    print(f"  intercept B                 : {bm:.4g} "
          f"+/- {float(np.nanmedian(B_err)) if np.isfinite(B_err).any() else float('nan'):.4g}"
          f"   (SNR {snr:.1f})")
    if not resolved:
        print()
        print("  *** THE FLOOR IS NOT RESOLVED. ***")
        print("  V(M) is consistent with pure 1/M noise: the intercept is not")
        print("  significant, so the bootstrap spread you measured is residual")
        print("  conditional-MC error, NOT fit variance.  This is an upper limit:")
        print(f"      (B) < {b_ul:.3g}  (2 sigma)" if np.isfinite(b_ul)
              else "      (B) upper limit could not be computed (no finite draws)")
        print("  To resolve it: (i) enlarge the pool (--n-pool, --fitvar-pool-size);")
        print("  (ii) check the defensive fraction/inflation (--fitvar-defensive-frac,")
        print("       --fitvar-broad-inflate); (iii) confirm jax x64 is enabled.")
        print()
        if pe_ceiling is not None and np.isfinite(b_ul):
            print(f"  NOTE: even the upper limit is {b_ul / pe_ceiling:.2g} x the PE")
            print(f"  ceiling ({pe_ceiling:.3g}).  If that ratio is << 1 the paper's")
            print("  claim survives on the upper limit alone, which may be enough.")
    else:
        rho = B_hat / np.clip(V_pe, 1e-300, None)
        print(f"  (B) median [16-84%]         : {b_med:.4g}  [{b_lo:.4g}, {b_hi:.4g}]")
        if pe_ceiling is not None:
            print(f"  PE total error at N_PE      : {pe_ceiling:.4g}")
        print(f"  rho = (B)/sigma^2_PE        : {np.nanmedian(rho):.3g}   "
              "<-- smoothing variance-reduction factor")
        for name, nc in crossings.items():
            print(f"  N_cross ({name:8s})        : {nc:.3g}")
        print(f"  (B) below threshold 1?      : "
              f"{'YES' if np.isfinite(b_hi) and b_hi < 1.0 else 'NO'}")
    print("  Selection fit variance (N_obs^2 * Var_D[ln xi]) NOT included.")
    print("=" * 70 + "\n")

    V_iC_of_M = np.asarray(fv["V_iC_of_M"])   # (n_M, n_coords, Nobs)
    per_ev = np.nanmedian(V_iC_of_M[-1], axis=0)          # at M_max, median over Lambda
    if np.isfinite(per_ev).any() and per_ev.sum() > 0:
        order = np.argsort(per_ev)[::-1]
        print("  Events dominating the floor (median over Lambda, at M_max):")
        for e in order[:5]:
            print(f"    event {e:3d}:  {per_ev[e]:.4g}   "
                  f"({100 * per_ev[e] / max(per_ev.sum(), 1e-30):.1f}% of (B))")
        print()


# ==========================================================
# 11. N_eff vs alpha & gamma plot
# ==========================================================
def make_neff_alpha_gamma_plot():
    print(f"\n[neff-scan] Building N_eff vs {args.xi_params[0]}/{args.xi_params[1]} plot...")
    THRESH = 4 * Nobs
    sizes = sorted(args.xi_gmm_sizes)

    cache_path = args.neff_results_cache
    cached = None
    if cache_path is not None and cache_path.exists():
        try:
            print(f"[neff-scan] loading cached results from {cache_path}")
            d = np.load(cache_path, allow_pickle=False)
            samples  = {lab: d[f"samples_{lab}"] for lab in labels}
            neff_inj = d["neff_inj"]
            neff_gmm = {s: d[f"neff_gmm_{s}"] for s in sizes}
            cached = True
        except KeyError as e:
            print(f"[neff-scan] cache is stale/incompatible ({e}); recomputing")

    if cached is None:
        N_SAMP = args.neff_n_prior_samples
        rng = np.random.default_rng(args.seed)

        samples = {lab: rng.uniform(lo, hi, N_SAMP)
                   for lab, lo, hi in zip(labels, lower_bound, upper_bound)}
        coords = [jnp.array([samples[lab][i] for lab in labels]) for i in range(N_SAMP)]

        neff_inj = np.array([float(selection_neff_method1(c))
                              for c in tqdm(coords, desc="N_eff (reweighting injections)")])

        sel = _get_selection_gmm(sizes, seed=args.seed)
        neff_gmm = {}
        for s in sizes:
            n_use = min(int(s), sel['nfull'])
            neff_gmm[s] = np.array([selection_neff_gmm(c, sel, n_use)
                                    for c in tqdm(coords, desc=f"N_eff GMM {s // 1000}k")])

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_dict = {f"samples_{lab}": samples[lab] for lab in labels}
            save_dict["neff_inj"] = neff_inj
            for s in sizes:
                save_dict[f"neff_gmm_{s}"] = neff_gmm[s]
            np.savez_compressed(cache_path, **save_dict)
            print(f"[neff-scan] saved results cache -> {cache_path}")

    bounds = dict(zip(labels, zip(lower_bound, upper_bound)))
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), constrained_layout=True)
    SCATTER_KW = dict(s=10, alpha=0.45)
    gmm_colors = ["C1", "C2", "C3", "C4"]

    panel_cfg = [
        (axes[i], samples[p], _XI_LATEX.get(p, p),
         (bounds[p][0] - 0.3, bounds[p][1] + 0.3))
        for i, p in enumerate(args.xi_params)
    ]

    for ax, xvals, xlabel, xlim in panel_cfg:
        ax.scatter(xvals, neff_inj, color="C0",
                   label=r"$\mathrm{reweighting\ injections}$", **SCATTER_KW)
        for color, s in zip(gmm_colors, sizes):
            ax.scatter(xvals, neff_gmm[s], color=color,
                       label=rf"$\mathrm{{GMM}}\ (N_{{\mathrm{{samp}}}}={_tex_int(s)})$",
                       **SCATTER_KW)
        ax.axhline(THRESH, color="black", ls="--", lw=1.2,
                   label=r"$\mathrm{Threshold}$", zorder=5)
        ax.set_yscale("log")
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$N_{\rm eff}$")
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
        # No explicit fontsize: Fig 3 (make_xi_scan_plot) passes none, so both
        # legends now inherit the same seaborn "talk" rcParam and match in size.
        ax.legend(framealpha=0.8, loc="lower right")
        ax.tick_params(labelsize=13)

    out = args.outdir / f"{args.run}_neff_alpha_gamma.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[neff-scan] saved -> {out}")


# ==========================================================
# 11b. xi projection scans + xi ratio
# ==========================================================
_XI_LATEX = {
    'gamma': r'$\gamma$', 'm_min': r'$m_{\min}$', 'm_max': r'$m_{\max}$',
    'alpha': r'$\alpha$', 'beta': r'$\beta$', 'mu': r'$\mu$',
    'sigma': r'$\sigma$', 'f1': r'$\lambda$',
    'dm_min': r'$\delta m_{\min}$', 'dm_max': r'$\delta m_{\max}$',
}
_XI_SCAN_CACHE = {}


def _binned_median(x, y, nbins=18, mincount=8):
    x = np.asarray(x); y = np.asarray(y)
    edges = np.linspace(np.min(x), np.max(x), nbins + 1)
    b = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    cx, cy = [], []
    for k in range(nbins):
        m = b == k
        if m.sum() >= mincount and np.isfinite(y[m]).any():
            cx.append(0.5 * (edges[k] + edges[k + 1]))
            cy.append(np.nanmedian(y[m]))
    return np.array(cx), np.array(cy)


def _tex_int(n):
    return f"{n:,}".replace(",", r"{,}")


def _xi_prior_scan():
    if _XI_SCAN_CACHE:
        return _XI_SCAN_CACHE

    sizes = sorted(args.xi_gmm_sizes)
    cache_path = args.xi_results_cache
    if cache_path is not None and cache_path.exists():
        try:
            print(f"[xi-scan] loading cached results from {cache_path}")
            d = np.load(cache_path, allow_pickle=False)
            samples = {lab: d[f"samples_{lab}"] for lab in labels}
            xi_inj  = d["xi_inj"]
            xi_gmm  = {s: d[f"xi_gmm_{s}"] for s in sizes}
            nsub    = {s: int(d[f"nsub_{s}"]) for s in sizes}
            _XI_SCAN_CACHE.update(samples=samples, xi_inj=xi_inj, xi_gmm=xi_gmm,
                                  sizes=sizes, nsub=nsub)
            return _XI_SCAN_CACHE
        except KeyError as e:
            print(f"[xi-scan] cache is stale/incompatible ({e}); recomputing")

    N_SAMP = args.neff_n_prior_samples
    rng = np.random.default_rng(args.seed)
    samples = {lab: rng.uniform(lo, hi, N_SAMP)
               for lab, lo, hi in zip(labels, lower_bound, upper_bound)}
    coords = [jnp.array([samples[lab][i] for lab in labels]) for i in range(N_SAMP)]

    xi_inj = np.array([selection_xi_method1(c) for c in tqdm(coords, desc="xi inj")])

    sel = _get_selection_gmm(sizes, seed=args.seed)
    xi_gmm, nsub = {}, {}
    for s in sizes:
        n_use = min(int(s), sel['nfull'])
        nsub[s] = n_use
        xi_gmm[s] = np.array([selection_xi_gmm(c, sel, n_use)
                              for c in tqdm(coords, desc=f"xi GMM {s // 1000}k")])

    _XI_SCAN_CACHE.update(samples=samples, xi_inj=xi_inj, xi_gmm=xi_gmm,
                          sizes=sizes, nsub=nsub)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {f"samples_{lab}": samples[lab] for lab in labels}
        save_dict["xi_inj"] = xi_inj
        for s in sizes:
            save_dict[f"xi_gmm_{s}"] = xi_gmm[s]
            save_dict[f"nsub_{s}"] = np.array(nsub[s])
        np.savez_compressed(cache_path, **save_dict)
        print(f"[xi-scan] saved results cache -> {cache_path}")

    return _XI_SCAN_CACHE


def make_xi_scan_plot():
    print(f"\n[xi-scan] Building xi vs {tuple(args.xi_params)} projection...")
    d = _xi_prior_scan()
    gmm_colors = [C[1], C[2]]
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), constrained_layout=True)
    KW = dict(s=10, alpha=0.4, edgecolors='none')

    for ax, pname in zip(axes, args.xi_params):
        x = d['samples'][pname]
        ax.scatter(x, d['xi_inj'], color=C[0],
                   label=r"$\mathrm{reweighting\ injections}$", **KW)
        for s, col in zip(d['sizes'], gmm_colors):
            if s in d['xi_gmm']:
                ax.scatter(x, d['xi_gmm'][s], color=col,
                           # Label the REQUESTED size s, matching make_neff_alpha_gamma_plot,
                           # rather than n_use = min(s, nfull). Identical whenever the
                           # selection GMM yields at least s valid draws (the case for the
                           # published 100k/300k figures: nsub == 100000/300000 exactly),
                           # and keeps the two legends consistent when it does not.
                           label=rf"$\mathrm{{GMM}}\ (N_{{\mathrm{{samp}}}}={_tex_int(s)})$",
                           **KW)
        ax.set_yscale('log')
        ax.set_xlabel(_XI_LATEX.get(pname, pname))
        ax.set_ylabel(r'$\xi(\Lambda)$')
        ax.grid(True, which='both', ls=':', lw=0.5, alpha=0.4)
        ax.legend(framealpha=0.8, loc='best')
        ax.tick_params(labelsize=13)

    out = args.outdir / f"{args.run}_xi_scan.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[xi-scan] saved -> {out}")


def _xi_ratio_regression_and_plot(xi_inj, xi_gmm_ref, samples, tag, gmm_size):
    """r = xi_inj/xi_GMM at every prior draw; regress log10(r) against each
    hyperparameter (referee demand M1(b)). A flat scatter at r=1 (after the
    fix) or at some r=const!=1 (before the fix) confirms a pure normalisation
    offset; any resolved slope means the offset is Lambda-dependent, which
    would be a second, worse problem (see CLAUDE_CODE_TASK_selection.md Task 5)."""
    ratio = np.asarray(xi_inj) / np.asarray(xi_gmm_ref)
    finite = np.isfinite(ratio) & (ratio > 0)
    log10_r = np.log10(ratio[finite])

    print(f"\n[xi-ratio:{tag}] N={int(finite.sum())}/{len(ratio)} finite  "
          f"(GMM size {gmm_size:,}), median r={np.median(ratio[finite]):.4g}  "
          f"log10(r): mean={np.mean(log10_r):.4f} std={np.std(log10_r):.4f}")

    ncols = 2
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.4 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    fits = {}
    for ax, lab in zip(axes, labels):
        x = samples[lab][finite]
        ax.scatter(x, log10_r, s=8, alpha=0.35, edgecolors='none', color=C[0])
        ax.axhline(np.mean(log10_r), color='k', ls=':', lw=1, alpha=0.5)
        if np.ptp(x) > 0:
            res = linregress(x, log10_r)
            fits[lab] = res
            xx = np.array([x.min(), x.max()])
            ax.plot(xx, res.intercept + res.slope * xx, color=C[2], lw=2,
                    label=rf"slope$={res.slope:.2e}\pm{res.stderr:.1e}$")
            ax.legend(fontsize=8, loc='best')
        else:
            fits[lab] = None  # fixed (not scanned) in this study
            ax.text(0.05, 0.9, "fixed (not scanned)", transform=ax.transAxes,
                    fontsize=8, va='top')
        ax.set_xlabel(_XI_LATEX.get(lab, lab))
        ax.set_ylabel(r'$\log_{10}(\xi_{\mathrm{inj}}/\xi_{\mathrm{GMM}})$')
        ax.grid(True, ls=':', lw=0.5, alpha=0.4)
    for ax in axes[len(labels):]:
        ax.set_visible(False)

    out = args.outdir / f"{args.run}_xi_ratio_{tag}.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[xi-ratio:{tag}] saved -> {out}")

    print(f"[xi-ratio:{tag}] regression of log10(xi_inj/xi_GMM) against each hyperparameter:")
    for lab, res in fits.items():
        if res is None:
            print(f"    {lab:8s} fixed in this study (not scanned) -- skipped")
            continue
        flag = "" if abs(res.slope) < 3 * res.stderr else "  <-- NONZERO at >3sigma"
        print(f"    {lab:8s} slope={res.slope: .4e}  stderr={res.stderr:.4e}  "
              f"r={res.rvalue: .3f}  p={res.pvalue:.3g}{flag}")
    return fits, log10_r


def make_xi_ratio_plot():
    print("\n[xi-ratio] Building xi_inj/xi_GMM verification figure (referee comment M1(b))...")
    d = _xi_prior_scan()
    gmm_size = max(d['sizes'])
    xi_gmm_ref = d['xi_gmm'][gmm_size]

    # ---- AFTER: current (fixed) code, both estimators through log_pdraw_marginal ----
    _xi_ratio_regression_and_plot(d['xi_inj'], xi_gmm_ref, d['samples'],
                                   tag="after", gmm_size=gmm_size)

    # ---- BEFORE: optional pre-fix cache (same prior draws, old buggy xi_inj) ----
    if args.xi_ratio_before_cache is not None and args.xi_ratio_before_cache.exists():
        b = np.load(args.xi_ratio_before_cache, allow_pickle=False)
        gmm_key = f"xi_gmm_{gmm_size}"
        if gmm_key not in b.files or len(b["xi_inj"]) != len(d['xi_inj']):
            print(f"[xi-ratio] WARNING: --xi-ratio-before-cache is incompatible with the "
                  f"current prior draws (size mismatch or missing {gmm_key}); "
                  f"skipping the before-fix panel.")
        else:
            samples_before = {lab: b[f"samples_{lab}"] for lab in labels}
            _xi_ratio_regression_and_plot(b["xi_inj"], b[gmm_key], samples_before,
                                           tag="before", gmm_size=gmm_size)
    else:
        print("[xi-ratio] no --xi-ratio-before-cache given (or file missing); "
              "only the after-fix figure was produced.")


# ==========================================================
# 12. Main
# ==========================================================
if args.fitvar_moment_check:
    fitvar_moment_check()
if args.make_variance_plot:
    make_variance_vs_Nsamp_plot()
if args.make_variance_analytic_plot:
    make_variance_analytical_plot()
if args.make_fitvar_plot:
    make_fitvar_plot()
if args.make_neff_scan_plot:
    make_neff_alpha_gamma_plot()
if args.make_xi_scan_plot:
    make_xi_scan_plot()
if args.make_xi_ratio_plot:
    make_xi_ratio_plot()
print("\nDone.")
