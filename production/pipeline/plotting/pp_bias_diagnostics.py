#!/usr/bin/env python
"""
pp_bias_diagnostics.py

Extracts weighted PIT values and posterior-median bias from dynesty .h5
nested-sampling results across a multi-catalog simulation study, then
produces:
  (1) a population-level PP-plot (calibration diagnostic), and
  (2) a per-parameter boxplot of bias across catalogs,
both overlaid for three inference methods (met1/PE, met3/Ensemble,
met4/Mixture).

Usage
-----
Step 1 — extract PIT/bias from h5 files into a CSV (slow step, run once):

    python pp_bias_diagnostics.py extract \
        --data-dir /path/to/catalog_h5_dir \
        --pattern "catalog_{i}_{method}.h5" \
        --n-catalogs 50 \
        --methods met1 met3 met4 \
        --out results.csv

Step 2 — make plots from the CSV (fast, iterate on styling):

    python pp_bias_diagnostics.py plot \
        --csv results.csv \
        --outdir figures/ \
        --use-tex

Both steps can be chained in one call with `python pp_bias_diagnostics.py all ...`.

Confidence bands and catalog overlap
------------------------------------
The 50 catalogs are drawn 69-from-276 without replacement from a shared
event pool, so any two catalogs share on average 69^2/276 ~ 17.25 events
and their PIT values are positively correlated (latent correlation
rho ~ 69/276 = 0.25 under the equal-information-per-event approximation).
The default band mode ("overlap") therefore replaces the independent
binomial funnel with a Monte Carlo null: each pool event carries an iid
N(0,1) influence contribution eps_j, each catalog statistic is
z_c = sum_{j in c} eps_j / sqrt(69), and PIT_c = Phi(z_c).  Membership is
either redrawn each realization (unconditional null) or fixed to the
actual catalog composition via --membership.  Pointwise 1/2/3-sigma
envelopes are taken as quantiles of the simulated ECDFs.  At p = 0.5 this
inflates the ECDF variance by ~8.9x (band-width ~3x, N_eff ~ 5.6 of 50).
Use --band-mode independent to recover the old binomial funnel, and
--show-independent-outline to overlay its 3-sigma outline for comparison.

For the pooled (all-parameter) PP-plot, the same shared-membership null
is used with one independent influence field per parameter; within-catalog
cross-parameter posterior correlations can optionally be folded in with
--pooled-param-corr estimate (probit-PIT correlation estimated from a
reference method; see caveats printed at runtime).

H5 file assumptions
--------------------
Each file is a dynesty NestedSampler checkpoint (saved via sampler.save() and
loaded via NestedSampler.restore()).  The results dict must contain samples
with 8 columns ordered [alpha, beta, gamma, m_min, m_max, mu, sigma, f1].
Dummy log-likelihood and prior_transform functions are provided in the global
namespace to satisfy pickle during restore().
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri  # standard normal CDF / inverse CDF

try:
    from dynesty import NestedSampler
    from dynesty import utils as dyutils
except ImportError:  # dynesty only needed for the extract step
    NestedSampler = None
    dyutils = None

# MDC catalog design: 50 catalogs of 69 events drawn without replacement
# from a shared pool of 276 events (Ray et al. MDC).
DEFAULT_N_POOL = 276
DEFAULT_N_EVENTS = 69

# ── Dummy functions required by dynesty.NestedSampler.restore() ──────────────
# The restore() call uses pickle, which needs these in the global namespace.
def loglike_method_1(coord): return 0.0
def loglike_method_3(coord): return 0.0
def loglike_method_4(coord): return 0.0
def prior_transform(theta): return theta

PARAM_NAMES = ["gamma", "m_min", "m_max", "alpha", "beta", "mu", "sigma", "f1"]
PARAM_LABELS_TEX = {
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "gamma": r"$\gamma$",
    "m_min": r"$m_\mathrm{min}$",
    "m_max": r"$m_\mathrm{max}$",
    "mu": r"$\mu$",
    "sigma": r"$\sigma$",
    "f1": r"$\lambda$",
}
TRUE_VALUES = {
    "alpha": 4.00,
    "beta": 1.50,
    "gamma": 3.00,
    "m_min": 6.00,
    "m_max": 70.00,
    "mu": 35.00,
    "sigma": 4.00,
    "f1": 0.04,
}
METHOD_LABELS = {
    "met1": "PE",
    "met3": "Ensemble",
    "met4": "Mixture",
    # *_boxfix = --pe-bounds-pct 0.02 99.98 (box at the PE sample edge)
    "met3_boxfix": "Ensemble (boxfix)",
    "met4_boxfix": "Mixture (boxfix)",
}
# Colorblind-safe palette (seaborn 'colorblind'): blue, orange, green
METHOD_COLORS = {
    "met1": "#0173B2",
    "met3": "#DE8F05",
    "met4": "#029E73",
    "met3_boxfix": "#D55E00",
    "met4_boxfix": "#CC78BC",
}


def load_equal_weight_samples(h5_path):
    """Load dynesty checkpoint and return equal-weight posterior samples."""
    if NestedSampler is None:
        raise ImportError("dynesty is required for the extract step")
    sampler = NestedSampler.restore(str(h5_path))
    results = sampler.results
    weights = np.exp(results.logwt - results.logz[-1])
    samples = dyutils.resample_equal(results.samples, weights)
    return samples


def pit(samples_1d, true_value):
    """Empirical CDF of equal-weight posterior samples evaluated at true_value."""
    return float(np.mean(samples_1d <= true_value))


def _get_pattern(args, method):
    """Return the filename pattern for a given method, checking
    --method-patterns first, then falling back to --pattern."""
    mp = getattr(args, "method_patterns", None) or {}
    return mp.get(method, args.pattern)


def extract(args):
    data_dir = Path(args.data_dir)
    rows = []
    missing = []

    for i in range(args.n_catalogs):
        for method in args.methods:
            pat = _get_pattern(args, method)
            fname = pat.format(i=i, method=method)
            fpath = data_dir / fname
            if not fpath.exists():
                missing.append(str(fpath))
                continue

            try:
                samples = load_equal_weight_samples(fpath)
            except Exception as e:
                print(f"[WARN] failed to load {fpath}: {e}", file=sys.stderr)
                continue

            if samples.shape[1] != len(args.param_names):
                print(
                    f"[WARN] {fpath}: expected {len(args.param_names)} dims, "
                    f"got {samples.shape[1]}; skipping",
                    file=sys.stderr,
                )
                continue

            for j, pname in enumerate(args.param_names):
                true_val = args.true_values[pname]
                col = samples[:, j]
                pit_val = pit(col, true_val)
                med = float(np.median(col))
                mean = float(np.mean(col))
                bias = mean - true_val  # bias defined via posterior mean
                                         # (the expectation value of the estimator),
                                         # not the median

                lo, hi = np.percentile(col, [5, 95])
                ci_half_width = max((hi - lo) / 2.0, 1e-300)  # guard div-by-zero
                post_std = max(float(np.std(col)), 1e-300)  # guard div-by-zero
                pull = bias / post_std  # bias in units of posterior sigma

                rows.append(
                    {
                        "catalog": i,
                        "method": method,
                        "parameter": pname,
                        "pit": pit_val,
                        "posterior_median": med,
                        "posterior_mean": mean,
                        "true_value": true_val,
                        "bias": bias,
                        "ci_lower_90": float(lo),
                        "ci_upper_90": float(hi),
                        "ci_half_width_90": float(ci_half_width),
                        "posterior_std": post_std,
                        "pull": float(pull),
                    }
                )

    if missing:
        print(
            f"[WARN] {len(missing)} expected file(s) not found "
            f"(showing up to 5): {missing[:5]}",
            file=sys.stderr,
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    return df


def setup_style(use_tex, latex_engine=True):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if use_tex and latex_engine:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Computer Modern Roman"]
    elif use_tex:
        # --mathtext: same TeX-styled labels, rendered by matplotlib's own
        # mathtext instead of a real LaTeX run.  Needed on hosts whose
        # texlive + CM Type1 fonts drop the minus glyph -- there, usetex
        # renders "-1" ticks and "a - b" axis labels with the minus MISSING
        # (verified on citlogin0 with texlive 2025: every negative tick came
        # out as a bare digit), which silently turns a bias/pull plot into a
        # figure whose sign cannot be read.
        plt.rcParams["text.usetex"] = False
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "cm"
    sns.set_context("talk", font_scale=1.3)
    sns.set_palette("colorblind")
    # Shared axis-label / legend sizes across all diagnostic scripts (see run_all_diagnostics.py).
    import os
    matplotlib.rcParams["axes.labelsize"] = float(os.environ.get("AXIS_LABEL_FONTSIZE", 20))
    matplotlib.rcParams["legend.fontsize"] = float(os.environ.get("LEGEND_FONTSIZE", 16))
    return plt, sns


def binomial_sigma_bands(grid, n, sigmas=(1, 2, 3)):
    """Pointwise binomial confidence bands around the diagonal for a PP-plot.

    At each credibility level p, the number of catalogs with PIT <= p is
    Binomial(n, p) under perfect calibration. Returns, for each requested
    sigma level, the (lower, upper) envelope of the fraction p_hat = k/n
    using the normal approximation to the binomial, i.e. the same
    1/2/3-sigma funnel-shaped bands used in the original bilby PP-plot
    validation figures (e.g. Romero-Shaw et al. 2020; Ashton et al. 2019).
    """
    bands = {}
    for s in sigmas:
        half_width = s * np.sqrt(grid * (1 - grid) / n)
        lower = np.clip(grid - half_width, 0, 1)
        upper = np.clip(grid + half_width, 0, 1)
        bands[s] = (lower, upper)
    return bands


# ── Overlap-aware null bands (shared 276-event pool) ─────────────────────────

def load_membership(path, n_pool):
    """Load the actual catalog composition: an (n_catalogs, n_events) integer
    array of pool indices. Supports .npy, .json (list of lists), and
    whitespace/comma-delimited text (one catalog per row)."""
    path = Path(path)
    if path.suffix == ".npy":
        mem = np.load(path)
    elif path.suffix == ".json":
        import json
        with open(path) as f:
            mem = np.asarray(json.load(f))
    else:
        mem = np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None)
    mem = np.asarray(mem, dtype=int)
    if mem.ndim != 2:
        raise ValueError(f"membership must be 2D (n_catalogs, n_events); got {mem.shape}")
    if mem.min() < 0 or mem.max() >= n_pool:
        raise ValueError(
            f"membership indices must lie in [0, {n_pool}); got "
            f"[{mem.min()}, {mem.max()}]"
        )
    return mem


def _nearest_pd_cholesky(corr):
    """Symmetrize, clip eigenvalues, renormalize to unit diagonal, Cholesky."""
    C = np.asarray(corr, dtype=float)
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-6, None)
    C = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    return np.linalg.cholesky(C)


def simulate_overlap_null_bands(
    grid,
    n_catalogs,
    n_events=DEFAULT_N_EVENTS,
    n_pool=DEFAULT_N_POOL,
    n_params=1,
    n_null=20000,
    membership=None,
    param_corr=None,
    sigmas=(1, 2, 3),
    seed=0,
    return_diagnostics=False,
):
    """Pointwise 1/2/3-sigma PP-plot bands under the correlated null induced
    by drawing catalogs without replacement from a shared event pool.

    Null model (first-order / equal-Fisher-information approximation): each
    pool event j carries an iid N(0,1) influence contribution eps_j; the
    catalog-level standardized statistic is z_c = sum_{j in c} eps_j /
    sqrt(n_events) and PIT_c = Phi(z_c), so PITs are marginally Uniform(0,1)
    but pairwise correlated with latent correlation overlap/n_events
    (mean n_events/n_pool = 0.25 for 69-from-276). For the pooled plot
    (n_params > 1) each parameter gets its own influence field sharing the
    same catalog membership; an optional (n_params x n_params) latent
    correlation matrix `param_corr` models within-catalog cross-parameter
    posterior correlation.

    If `membership` is given (actual catalog composition, shape
    (n_catalogs, n_events)), the null conditions on that fixed design;
    otherwise membership is redrawn each realization (unconditional null).

    Returns bands dict {sigma: (lower, upper)} on `grid`; if
    return_diagnostics, also a dict with variance-inflation factors and
    effective sample sizes at several credibility levels.
    """
    rng = np.random.default_rng(seed)
    grid = np.asarray(grid)
    L = _nearest_pd_cholesky(param_corr) if param_corr is not None else None

    if membership is not None:
        membership = np.asarray(membership, dtype=int)
        if membership.shape[0] != n_catalogs:
            print(
                f"[WARN] membership has {membership.shape[0]} catalogs; "
                f"overriding n_catalogs={n_catalogs}",
                file=sys.stderr,
            )
            n_catalogs = membership.shape[0]
        n_events = membership.shape[1]

    n_stat = n_catalogs * n_params
    # keep the gathered eps array around ~2e7 elements per batch
    batch = max(1, int(2e7 // max(n_catalogs * n_events * n_params, 1)))
    ecdfs = np.empty((n_null, grid.size), dtype=np.float32)

    done = 0
    while done < n_null:
        b = min(batch, n_null - done)
        eps = rng.standard_normal((b, n_pool, n_params))
        if L is not None:
            eps = eps @ L.T
        if membership is None:
            u = rng.random((b, n_catalogs, n_pool))
            idx = np.argpartition(u, n_events - 1, axis=2)[:, :, :n_events]
        else:
            idx = np.broadcast_to(
                membership[None, :, :], (b, n_catalogs, n_events)
            )
        # gather per-event contributions: (b, n_cat, n_events, n_params)
        gathered = eps[np.arange(b)[:, None, None], idx, :]
        z = gathered.sum(axis=2) / np.sqrt(n_events)  # (b, n_cat, n_params)
        pit = ndtr(z).reshape(b, n_stat)
        ecdfs[done : done + b] = (
            (pit[:, :, None] <= grid[None, None, :]).mean(axis=1)
        )
        done += b

    bands = {}
    for s in sigmas:
        lo_q, hi_q = ndtr(-float(s)), ndtr(float(s))
        lower = np.quantile(ecdfs, lo_q, axis=0)
        upper = np.quantile(ecdfs, hi_q, axis=0)
        bands[s] = (np.clip(lower, 0, 1), np.clip(upper, 0, 1))

    if not return_diagnostics:
        return bands

    diag_ps = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    cols = np.searchsorted(grid, diag_ps)
    cols = np.clip(cols, 0, grid.size - 1)
    var_mc = ecdfs[:, cols].var(axis=0).astype(float)
    var_indep = diag_ps * (1 - diag_ps) / n_stat
    with np.errstate(divide="ignore", invalid="ignore"):
        inflation = np.where(var_indep > 0, var_mc / var_indep, np.nan)
    diagnostics = {
        "p": diag_ps,
        "var_mc": var_mc,
        "var_indep": var_indep,
        "inflation": inflation,
        "n_eff": n_stat / inflation,
        "n_stat": n_stat,
        "mean_overlap": n_events**2 / n_pool,
        "rho_latent": n_events / n_pool,
    }
    return bands, diagnostics


def estimate_param_corr_from_pits(df, method, param_names):
    """Estimate the within-catalog cross-parameter latent correlation from
    the observed PITs of a reference method via a probit transform.

    Caveats: assumes the reference method's PIT marginals are approximately
    uniform (miscalibration contaminates the estimate), and with only
    ~50 catalogs the 8x8 correlation matrix is noisy. Use as a sensitivity
    check on the pooled bands, not as ground truth.
    """
    sub = df[df["method"] == method]
    piv = sub.pivot_table(index="catalog", columns="parameter", values="pit")
    piv = piv[param_names]
    z = ndtri(np.clip(piv.values, 1e-3, 1 - 1e-3))
    C = np.corrcoef(z, rowvar=False)
    return C


def print_band_diagnostics(diag, tag):
    print(f"\n[{tag}] overlap-corrected null-band diagnostics")
    print(
        f"  mean pairwise catalog overlap = {diag['mean_overlap']:.2f} events, "
        f"latent rho = {diag['rho_latent']:.3f}, "
        f"nominal N = {diag['n_stat']}"
    )
    print("   p     Var_null    Var_indep   inflation  band-widen   N_eff")
    for p, vm, vi, f, ne in zip(
        diag["p"], diag["var_mc"], diag["var_indep"],
        diag["inflation"], diag["n_eff"],
    ):
        print(
            f"  {p:4.2f}  {vm:.6f}   {vi:.6f}    {f:6.2f}     x{np.sqrt(f):4.2f}"
            f"     {ne:6.1f}"
        )


def _plot_sigma_bands(ax, grid, bands, use_tex, sigmas=(1, 2, 3),
                       base_alpha=0.25, label=True):
    """Draw nested 1/2/3-sigma bands (darkest=1sigma, lightest=3sigma) from a
    precomputed {sigma: (lower, upper)} dict."""
    # draw widest (outermost) first so narrower bands layer on top
    for s in sorted(sigmas, reverse=True):
        lower, upper = bands[s]
        alpha = base_alpha * (1.0 - 0.22 * (s - 1))  # 1sigma darkest, 3sigma lightest
        lbl = None
        if label:
            lbl = rf"${s}\sigma$" if use_tex else f"{s}-sigma"
        ax.fill_between(
            grid, lower, upper, color="gray", alpha=alpha, lw=0, label=lbl,
        )


def make_pp_plot(df, methods, outdir, plt, use_tex, grid, bands,
                 indep_outline_n=None, mode="combined"):
    fig, ax = plt.subplots(figsize=(7, 7))

    for method in methods:
        sub = df[df["method"] == method]
        pits = np.sort(sub["pit"].values)
        n = len(pits)
        if n == 0:
            continue
        ecdf = np.searchsorted(pits, grid, side="right") / n
        label = METHOD_LABELS.get(method, method)
        ax.plot(grid, ecdf, lw=2.2, color=METHOD_COLORS.get(method), label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Ideal calibration")
    _plot_sigma_bands(ax, grid, bands, use_tex, sigmas=(1, 2, 3))

    # optional dashed outline of the (too-narrow) independent binomial
    # 3-sigma funnel, for visual comparison with the overlap-corrected band
    if indep_outline_n is not None:
        ib = binomial_sigma_bands(grid, indep_outline_n, sigmas=(3,))
        lo, hi = ib[3]
        lbl = (
            r"$3\sigma$ (independent, too narrow)"
            if use_tex
            else "3-sigma (independent, too narrow)"
        )
        ax.plot(grid, lo, color="gray", lw=1.0, ls=":", label=lbl)
        ax.plot(grid, hi, color="gray", lw=1.0, ls=":")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    xlabel = r"Credibility level $p$" if use_tex else "Credibility level p"
    ylabel = (
        r"Fraction of catalogs with PIT $\leq p$"
        if use_tex
        else "Fraction of catalogs with PIT <= p"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()

    out_path = Path(outdir) / "pp_plot_combined.pdf"
    fig.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close(fig)


def make_pp_plot_per_param(df, methods, param_names, outdir, plt, use_tex,
                           grid, bands):
    n_params = len(param_names)
    ncols = 4
    nrows = int(np.ceil(n_params / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, pname in zip(axes, param_names):
        # same null for every parameter panel (same n_catalogs, same
        # catalog-overlap structure), so the bands are shared
        _plot_sigma_bands(ax, grid, bands, use_tex, sigmas=(1, 2, 3), label=False)

        for method in methods:
            sub = df[(df["method"] == method) & (df["parameter"] == pname)]
            pits = np.sort(sub["pit"].values)
            n = len(pits)
            if n == 0:
                continue
            ecdf = np.searchsorted(pits, grid, side="right") / n
            ax.plot(
                grid, ecdf, lw=1.8, color=METHOD_COLORS.get(method),
                label=METHOD_LABELS.get(method, method),
            )
        ax.plot([0, 1], [0, 1], "k--", lw=1.0)
        label = PARAM_LABELS_TEX.get(pname, pname) if use_tex else pname
        ax.set_title(label, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

    for ax in axes[n_params:]:
        ax.axis("off")

    fig.tight_layout()
    out_path = Path(outdir) / "pp_plot_per_parameter.pdf"
    fig.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close(fig)


def make_bias_boxplot(df, methods, param_names, outdir, plt, sns, use_tex):
    n_params = len(param_names)
    ncols = 4
    nrows = int(np.ceil(n_params / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    hue_order = [METHOD_LABELS.get(m, m) for m in methods]
    palette = {METHOD_LABELS.get(m, m): METHOD_COLORS.get(m) for m in methods}

    plot_df = df[df["parameter"].isin(param_names)].copy()
    plot_df["method_label"] = plot_df["method"].map(
        lambda m: METHOD_LABELS.get(m, m)
    )

    for ax, pname in zip(axes, param_names):
        sub = plot_df[plot_df["parameter"] == pname]

        sns.boxplot(
            data=sub,
            x="method_label",
            y="bias",
            hue="method_label",
            order=hue_order,
            hue_order=hue_order,
            palette=palette,
            legend=False,
            ax=ax,
            showfliers=False,
            linewidth=1.2,
            width=0.6,
        )
        ax.axhline(0, color="black", lw=1.0, ls="--", zorder=0)

        label = PARAM_LABELS_TEX.get(pname, pname) if use_tex else pname
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if len(methods) > 3:
            # method names are in the legend; rotated ticks overlap past 3
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        else:
            ax.tick_params(axis="x", rotation=30)

    # shared y-label on leftmost panels
    for i in range(nrows):
        ylabel = (
            r"Bias ($\hat{\theta}_\mathrm{mean} - \theta_\mathrm{true}$)"
            if use_tex
            else "Bias (mean - true)"
        )
        axes[i * ncols].set_ylabel(ylabel)

    # hide unused panels
    for ax in axes[n_params:]:
        ax.axis("off")

    # shared legend from colored patches
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=METHOD_COLORS.get(m), label=METHOD_LABELS.get(m, m))
        for m in methods
    ]
    if len(methods) > 3:
        fig.legend(handles=handles, loc="lower center", frameon=False,
                   bbox_to_anchor=(0.5, 1.0), ncol=len(methods))
        fig.tight_layout(rect=[0, 0, 1.0, 0.95])
    else:
        fig.legend(handles=handles, loc="upper right", frameon=False,
                   bbox_to_anchor=(0.98, 0.98))
        fig.tight_layout(rect=[0, 0, 0.95, 1.0])
    out_path = Path(outdir) / "bias_boxplot.pdf"
    # bbox_inches: with 4+ methods the legend is placed OUTSIDE the axes and
    # would otherwise be cropped off the page.
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def make_pull_boxplot(df, methods, param_names, outdir, plt, sns, use_tex):
    """Single-panel boxplot of standardized bias ('pull'):
    (mean - true) / posterior_std. Dimensionless and in units of the
    posterior's own sigma, so all 8 parameters can share one y-axis
    without scale issues.
    """
    fig, ax = plt.subplots(figsize=(12 + 1.2 * max(0, len(methods) - 3), 6))

    plot_df = df[df["parameter"].isin(param_names)].copy()
    plot_df["method_label"] = plot_df["method"].map(
        lambda m: METHOD_LABELS.get(m, m)
    )
    plot_df["param_label"] = plot_df["parameter"].map(
        lambda p: PARAM_LABELS_TEX.get(p, p) if use_tex else p
    )

    order = [
        PARAM_LABELS_TEX.get(p, p) if use_tex else p for p in param_names
    ]
    hue_order = [METHOD_LABELS.get(m, m) for m in methods]
    palette = {METHOD_LABELS.get(m, m): METHOD_COLORS.get(m) for m in methods}

    sns.boxplot(
        data=plot_df,
        x="param_label",
        y="pull",
        hue="method_label",
        order=order,
        hue_order=hue_order,
        palette=palette,
        ax=ax,
        showfliers=False,
        linewidth=1.2,
    )
    ax.axhline(0, color="black", lw=1.0, ls="--", zorder=0)

    # reference bands at +/-1, +/-2, +/-3 sigma: pull is now literally in
    # units of the posterior's own standard deviation, so these bands mean
    # exactly what they say (1-sigma, 2-sigma, 3-sigma), unlike a
    # CI-half-width normalization which would need a 1.645 factor for a
    # Gaussian posterior.
    for s, alpha in zip((3, 2, 1), (0.06, 0.10, 0.16)):
        ax.axhspan(-s, s, color="gray", alpha=alpha, lw=0, zorder=-1)

    ylabel = (
        r"$(\bar{\theta}_\mathrm{mean} - \theta_\mathrm{true}) \,/\, \sigma_\mathrm{posterior}$"
        if use_tex
        else "Pull ((mean - true) / posterior sigma)"
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    if len(methods) > 3:
        # in-axes it covers the right-hand parameters once there are 4+ boxes
        ax.legend(title="", loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  frameon=False)
    else:
        ax.legend(title="", loc="upper right", frameon=False)
    fig.tight_layout()

    out_path = Path(outdir) / "pull_boxplot.pdf"
    # bbox_inches: with 4+ methods the legend is placed OUTSIDE the axes and
    # would otherwise be cropped off the page.
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot(args):
    df = pd.read_csv(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mathtext = getattr(args, "mathtext", False)
    plt, sns = setup_style(
        args.use_tex or mathtext, latex_engine=not mathtext
    )
    args.use_tex = args.use_tex or mathtext  # gates the TeX-styled labels
    methods = args.methods
    param_names = args.param_names

    grid = np.linspace(0, 1, 200)
    n_catalogs = df["catalog"].nunique()
    n_pooled = min(
        (df["method"] == m).sum() for m in methods if (df["method"] == m).sum() > 0
    )

    membership = None
    if getattr(args, "membership", None):
        membership = load_membership(args.membership, args.n_pool)
        print(f"Using fixed catalog membership from {args.membership} "
              f"(shape {membership.shape})")

    if args.band_mode == "overlap":
        # per-parameter null: 50 correlated catalogs, one parameter
        bands_pp, diag_pp = simulate_overlap_null_bands(
            grid, n_catalogs,
            n_events=args.n_events, n_pool=args.n_pool,
            n_params=1, n_null=args.n_null,
            membership=membership, seed=args.band_seed,
            return_diagnostics=True,
        )
        print_band_diagnostics(diag_pp, "per-parameter PP")

        # pooled null: shared membership across len(param_names) fields
        param_corr = None
        if args.pooled_param_corr == "estimate":
            param_corr = estimate_param_corr_from_pits(
                df, args.param_corr_method, param_names
            )
            print(
                f"\n[pooled PP] within-catalog cross-parameter latent "
                f"correlation estimated from method "
                f"'{args.param_corr_method}' probit-PITs "
                f"(noisy at ~{n_catalogs} catalogs; assumes that method is "
                f"approximately calibrated):"
            )
            with np.printoptions(precision=2, suppress=True):
                print(param_corr)
        bands_pooled, diag_pooled = simulate_overlap_null_bands(
            grid, n_catalogs,
            n_events=args.n_events, n_pool=args.n_pool,
            n_params=len(param_names), n_null=args.n_null,
            membership=membership, param_corr=param_corr,
            seed=args.band_seed + 1,
            return_diagnostics=True,
        )
        print_band_diagnostics(diag_pooled, "pooled PP")
        if args.pooled_param_corr == "none":
            print(
                "  note: pooled bands treat the 8 parameters as independent "
                "given the catalog;\n  within-catalog posterior correlations "
                "would widen them further (--pooled-param-corr estimate)."
            )
    else:
        bands_pp = binomial_sigma_bands(grid, n_catalogs)
        bands_pooled = binomial_sigma_bands(grid, n_pooled)
        print(
            "[WARN] band-mode 'independent' ignores the 69-from-276 catalog "
            "overlap; bands are ~2.5-3x too narrow.",
            file=sys.stderr,
        )

    indep_outline_n = n_pooled if args.show_independent_outline else None

    make_pp_plot(
        df, methods, outdir, plt, args.use_tex, grid, bands_pooled,
        indep_outline_n=indep_outline_n, mode="combined",
    )
    if args.per_param_pp:
        make_pp_plot_per_param(
            df, methods, param_names, outdir, plt, args.use_tex, grid, bands_pp
        )
    make_bias_boxplot(df, methods, param_names, outdir, plt, sns, args.use_tex)
    make_pull_boxplot(df, methods, param_names, outdir, plt, sns, args.use_tex)


def _add_band_args(p):
    p.add_argument(
        "--band-mode",
        choices=["overlap", "independent"],
        default="overlap",
        help="'overlap' (default): Monte Carlo null bands accounting for the "
        "shared 276-event pool; 'independent': legacy binomial funnel.",
    )
    p.add_argument(
        "--n-pool", type=int, default=DEFAULT_N_POOL,
        help="Size of the shared event pool (default 276)",
    )
    p.add_argument(
        "--n-events", type=int, default=DEFAULT_N_EVENTS,
        help="Events per catalog (default 69)",
    )
    p.add_argument(
        "--n-null", type=int, default=20000,
        help="Number of null realizations for the overlap bands",
    )
    p.add_argument("--band-seed", type=int, default=0)
    p.add_argument(
        "--membership", type=str, default=None,
        help="Optional path (.npy/.json/.csv/.txt) with the actual "
        "(n_catalogs, n_events) pool indices; conditions the null on the "
        "realized catalog design instead of redrawing memberships.",
    )
    p.add_argument(
        "--pooled-param-corr",
        choices=["none", "estimate"],
        default="none",
        help="Fold within-catalog cross-parameter posterior correlation "
        "into the pooled-PP null (estimated from probit-PITs of "
        "--param-corr-method).",
    )
    p.add_argument("--param-corr-method", type=str, default="met4")
    p.add_argument(
        "--show-independent-outline",
        action="store_true",
        help="Overlay the legacy independent 3-sigma binomial outline on the "
        "combined PP-plot for comparison.",
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common_true_vals = " ".join(f"{k}={v}" for k, v in TRUE_VALUES.items())

    # extract
    p_ext = sub.add_parser("extract", help="Extract PIT/bias from h5 files into a CSV")
    p_ext.add_argument("--data-dir", required=True, type=str)
    p_ext.add_argument(
        "--pattern",
        type=str,
        default="catalog_{i}_{method}.h5",
        help="Filename pattern with {i} and {method} placeholders",
    )
    p_ext.add_argument("--n-catalogs", type=int, default=50)
    p_ext.add_argument(
        "--methods", nargs="+", default=["met1", "met3", "met4"]
    )
    p_ext.add_argument(
        "--param-names",
        nargs="+",
        default=PARAM_NAMES,
        help="Column order of the 'samples' array in the h5 files",
    )
    p_ext.add_argument(
        "--true-values",
        type=str,
        default=None,
        help=f"Override true values as 'key=val key=val ...'. Default: {common_true_vals}",
    )
    p_ext.add_argument("--out", type=str, default="pit_bias_results.csv")
    p_ext.add_argument(
        "--method-patterns",
        nargs="+",
        default=None,
        help="Per-method pattern overrides as 'method=pattern ...', "
        "e.g. met3=mdcnm_cat{i:03d}_{method}.h5. Falls back to --pattern.",
    )
    p_ext.set_defaults(func=extract)

    # plot
    p_plot = sub.add_parser("plot", help="Make PP-plot and bias boxplot from CSV")
    p_plot.add_argument("--csv", required=True, type=str)
    p_plot.add_argument("--outdir", type=str, default="figures")
    p_plot.add_argument(
        "--methods", nargs="+", default=["met1", "met3", "met4"]
    )
    p_plot.add_argument("--param-names", nargs="+", default=PARAM_NAMES)
    p_plot.add_argument("--use-tex", action="store_true")
    p_plot.add_argument(
        "--mathtext",
        action="store_true",
        help="TeX-styled labels via matplotlib mathtext instead of a real "
        "LaTeX run. Use where text.usetex drops the minus glyph (see "
        "setup_style).",
    )
    p_plot.add_argument(
        "--per-param-pp",
        action="store_true",
        help="Also produce an 8-panel per-parameter PP-plot grid",
    )
    _add_band_args(p_plot)
    p_plot.set_defaults(func=plot)

    # all (chain extract -> plot)
    p_all = sub.add_parser("all", help="Run extract then plot")
    p_all.add_argument("--data-dir", required=True, type=str)
    p_all.add_argument("--pattern", type=str, default="catalog_{i}_{method}.h5")
    p_all.add_argument("--n-catalogs", type=int, default=50)
    p_all.add_argument("--methods", nargs="+", default=["met1", "met3", "met4"])
    p_all.add_argument("--param-names", nargs="+", default=PARAM_NAMES)
    p_all.add_argument("--true-values", type=str, default=None)
    p_all.add_argument("--out", type=str, default="pit_bias_results.csv")
    p_all.add_argument("--method-patterns", nargs="+", default=None)
    p_all.add_argument("--outdir", type=str, default="figures")
    p_all.add_argument("--use-tex", action="store_true")
    p_all.add_argument("--mathtext", action="store_true")
    p_all.add_argument("--per-param-pp", action="store_true")
    _add_band_args(p_all)

    def run_all(args):
        df = extract(args)
        args.csv = args.out
        plot(args)

    p_all.set_defaults(func=run_all)

    return parser


def parse_true_values_override(s):
    out = dict(TRUE_VALUES)
    if s:
        for tok in s.split():
            k, v = tok.split("=")
            out[k] = float(v)
    return out


def parse_method_patterns(lst):
    """Parse ['met3=mdcnm_cat{i:03d}_{method}.h5', ...] into a dict."""
    out = {}
    if lst:
        for tok in lst:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def main():
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "true_values"):
        args.true_values = parse_true_values_override(args.true_values)
    if hasattr(args, "method_patterns"):
        args.method_patterns = parse_method_patterns(args.method_patterns)
    args.func(args)


if __name__ == "__main__":
    main()