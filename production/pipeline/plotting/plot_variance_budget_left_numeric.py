"""
Left-panel-only variance-budget plot, with term (A) taken from the EMPIRICAL
subsampling sweep (same numerical method/data as mdc_variance_Nsamp.pdf,
make_diagnostic_plots.py:make_variance_vs_Nsamp_plot()) instead of the
analytic 1/N scaling used in the original 3-panel budget plot.

Pure post-processing of two existing NPZ caches -- no pool rebuild, no
dynesty loading, no JAX. Runs in seconds.

Inputs:
    --variance-cache   NPZ from make_variance_vs_Nsamp_plot() (term A, empirical).
                        Default: results/data/mdc_variance_results.npz
    --fitvar-cache     NPZ from compute_fit_variance() (term B, bootstrap floor).
                        Default: cache/fitvar_cat000_full.npz
Output:
    {outdir}/{run}_variance_budget_left_numeric.pdf
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

p = argparse.ArgumentParser()
p.add_argument("--run", default="mdc")
p.add_argument("--outdir", type=Path, default=Path("fitvar_full_outputs"))
p.add_argument("--variance-cache", type=Path,
                default=Path("results/data/mdc_variance_results.npz"))
p.add_argument("--fitvar-cache", type=Path,
                default=Path("cache/fitvar_cat000_full.npz"))
p.add_argument("--fitvar-min-snr", type=float, default=3.0)
p.add_argument("--band-mode", choices=("fit", "fit-line", "analytic", "none"),
               default="fit-line",
               help="overlay drawn on each method: 'fit-line' = dashed measured power law "
                    "sigma^2 = A*N^b fitted to each median, no shading, no legend entry "
                    "(default); 'fit' = the same dashed line plus a shaded band from "
                    "independent fits to the 16/84 envelopes and the exponent in the legend; "
                    "'analytic' = the 1/N law V_full*N_ceiling/N that mdc_variance_Nsamp.pdf "
                    "uses, anchored at the sampling ceiling; 'none' = empirical points only. "
                    "The cached medians follow N^-0.85..-0.91, not N^-1, so 'analytic' "
                    "overshoots by up to ~2.5x below N~200 (see SOURCES.md, Fig 4)")
p.add_argument("--sigma-band", choices=("jackknife", "ols", "none"),
               default="jackknife",
               help="Statistical 1-sigma band on the (B) floor, drawn hatched and "
                    "distinct from the shaded 16-84%% Lambda spread. 'jackknife' = "
                    "leave-one-event-out SE from V_iC_of_M (recommended); 'ols' = the "
                    "stored B_err, which is scatter about the V(M) regression only and "
                    "understates it ~23x because the 4 pool sizes are NESTED prefixes "
                    "of one pool fitted with dof=2; 'none' = previous behaviour.")
p.add_argument("--band-legend", action="store_true",
               help="Add legend entries naming the two spans (16-84%% over Lambda, "
                    "and the +-1sigma statistical band). Off by default -- the "
                    "paper caption describes them instead.")
p.add_argument("--legend-fontsize", type=float, default=15.0,
               help="Legend font size (was hard-coded 11).")
args = p.parse_args()
args.outdir.mkdir(parents=True, exist_ok=True)

sns.set_context('talk', font_scale=1.3)
sns.set_style('ticks')
sns.set_palette('colorblind')
C = sns.color_palette('colorblind')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['cmr10', 'Computer Modern Roman', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['axes.formatter.use_mathtext'] = True
matplotlib.rcParams['axes.unicode_minus'] = False

# ================= term (A): empirical subsampling sweep =================
dv = np.load(args.variance_cache, allow_pickle=False)
methods = [("PE", C[0], 's'), ("Ensemble", C[1], 'o'), ("Mixture", C[2], '^')]

# ================= term (B): bootstrap fit-variance floor =================
fv = np.load(args.fitvar_cache, allow_pickle=False)
B_hat = np.asarray(fv["B_hat"])
B_err = np.asarray(fv["B_err"])
n_train = int(np.asarray(fv["n_train"]).item())  # == PE ceiling (N_train PE samples)

finite_B = B_hat[np.isfinite(B_hat)]
finite_e = B_err[np.isfinite(B_err)]
snr = (np.nanmedian(finite_B) / max(np.nanmedian(finite_e), 1e-300)
       if finite_B.size and finite_e.size else 0.0)
resolved = bool(finite_B.size) and (np.nanmedian(finite_B) > 0) and (snr > args.fitvar_min_snr)
b_med = float(np.nanmedian(B_hat)) if resolved else float("nan")
b_lo = float(np.nanpercentile(B_hat, 16)) if resolved else float("nan")
b_hi = float(np.nanpercentile(B_hat, 84)) if resolved else float("nan")
ul_terms = B_hat + 2.0 * B_err
b_ul = float(np.nanmedian(ul_terms)) if np.isfinite(ul_terms).any() else float("nan")

# The shaded span drawn below is the 16-84%% SPREAD OF B ACROSS LAMBDA -- how the
# floor varies over hyperparameter space -- NOT an uncertainty on it. The two are
# nearly the same width here (band [0.154, 0.260] vs +-1sigma [0.167, 0.240]) for
# entirely different reasons, which is why the 1sigma is hatched, not shaded.
def _jackknife_B_se(fv_):
    """Leave-one-event-out SE on the OLS intercept B, from the cached per-event
    breakdown. Per-event training sets are independent, so V_of_M is a plain sum
    over events (verified: sum(V_iC_of_M, axis=2) == V_of_M to 1.7e-16) and the
    events are the resamplable unit."""
    Vi = np.asarray(fv_["V_iC_of_M"])            # (n_M, n_coords, Nobs)
    x = 1.0 / np.asarray(fv_["Ms"], dtype=float)
    X = np.vstack([np.ones_like(x), x]).T
    Xi = np.linalg.inv(X.T @ X)
    n_e = Vi.shape[2]
    tot = Vi.sum(axis=2)
    Bj = np.empty((n_e, tot.shape[1]))
    for e in range(n_e):
        Bj[e] = (Xi @ X.T @ ((tot - Vi[:, :, e]) * n_e / (n_e - 1.0)))[0]
    return np.sqrt((n_e - 1.0) / n_e * np.sum((Bj - Bj.mean(0)) ** 2, axis=0))

b_se, se_tag = float("nan"), ""
if args.sigma_band == "jackknife" and "V_iC_of_M" in fv.files:
    b_se = float(np.nanmedian(_jackknife_B_se(fv)))
    se_tag = r"event\ jackknife"
elif args.sigma_band == "ols":
    b_se = float(np.nanmedian(finite_e)) if finite_e.size else float("nan")
    se_tag = r"OLS\ intercept"

# ================= plot =================
fig, ax = plt.subplots(1, 1, figsize=(8, 6.5), constrained_layout=True)

drew_band = False
slopes = {}
for name, color, marker in methods:
    if f"{name}_x" not in dv.files:
        continue
    x = np.asarray(dv[f"{name}_x"])
    med = np.asarray(dv[f"{name}_med"])
    lo = np.asarray(dv[f"{name}_lo"])
    hi = np.asarray(dv[f"{name}_hi"])
    order = np.argsort(x)
    x, med, lo, hi = x[order], med[order], lo[order], hi[order]
    err = np.vstack([med - lo, hi - med])

    Ns_an = np.logspace(np.log10(max(x.min(), 10)), np.log10(x.max()), 300)
    label = rf'$\mathrm{{{name}}}$'

    if args.band_mode in ("fit", "fit-line"):
        # Measured power law: fit ln(sigma^2) = ln A + b ln N to the median. In
        # 'fit' mode the 16/84 envelopes get their own independent fits, shaded
        # between -- that band is the same coord-to-coord spread the error bars
        # already show, so 'fit-line' draws the median law alone.
        b_med_slope, lnA_med = np.polyfit(np.log(x), np.log(med), 1)
        slopes[name] = b_med_slope
        ax.plot(Ns_an, np.exp(lnA_med) * Ns_an ** b_med_slope,
                color=color, ls='--', lw=1.5, alpha=0.85, zorder=1)
        if args.band_mode == "fit":
            b_lo_slope, lnA_lo = np.polyfit(np.log(x), np.log(lo), 1)
            b_hi_slope, lnA_hi = np.polyfit(np.log(x), np.log(hi), 1)
            ax.fill_between(Ns_an,
                            np.exp(lnA_lo) * Ns_an ** b_lo_slope,
                            np.exp(lnA_hi) * Ns_an ** b_hi_slope,
                            color=color, alpha=0.15, lw=0, zorder=0)
            drew_band = True
            label = rf'$\mathrm{{{name}}}\ (b={b_med_slope:.2f})$'
    elif args.band_mode == "analytic" and f"{name}_Vf_med" in dv.files:
        # 1/N law anchored at the sampling ceiling, as make_variance_vs_Nsamp_plot()
        # draws it: V(N) = V_full * N_ceiling / N.
        vf_med = float(np.asarray(dv[f"{name}_Vf_med"]).item())
        vf_lo = float(np.asarray(dv[f"{name}_Vf_lo"]).item())
        vf_hi = float(np.asarray(dv[f"{name}_Vf_hi"]).item())
        ceil_N = int(np.asarray(dv[f"{name}_ceiling"]).item())
        ax.plot(Ns_an, vf_med * ceil_N / Ns_an,
                color=color, ls='--', lw=1.5, alpha=0.85, zorder=1)
        ax.fill_between(Ns_an, vf_lo * ceil_N / Ns_an, vf_hi * ceil_N / Ns_an,
                        color=color, alpha=0.15, lw=0, zorder=0)
        drew_band = True

    ax.errorbar(x, med, yerr=err, marker=marker, ls='-', color=color, lw=2,
                ms=5, capsize=3, elinewidth=1.2, label=label)

if resolved:
    ax.axhline(b_med, color='k', ls='-', lw=2.0, alpha=0.9,
               label=r'$\mathrm{fit\ variance}$')
    ax.axhspan(b_lo, b_hi, color='k', alpha=0.12)
    if np.isfinite(b_se) and b_se > 0:
        ax.axhspan(b_med - b_se, b_med + b_se, facecolor='none',
                   edgecolor='k', hatch='///', lw=0.0, alpha=0.5, zorder=2)
elif np.isfinite(b_ul):
    ax.axhline(b_ul, color='k', ls='-.', lw=2.0, alpha=0.9,
               label=r'$2\sigma$ upper limit')
    ax.axhspan(0, b_ul, color='k', alpha=0.07)

ax.axvline(n_train, color='k', ls=':', lw=2, alpha=0.7,
           label=fr'$N_{{\rm PE}}={n_train}$')
ax.axhline(1.0, color='0.5', ls='--', lw=1.0, alpha=0.5)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$N_{\rm samp}$'); ax.set_ylabel(r'$\sigma^2_{\ln\hat{\mathcal{L}}}$')
ax.grid(True, which='both', ls=':', alpha=0.3)
if drew_band:
    proxy = (r'$\mathrm{fit}\ \propto N_{\rm samp}^{\,b}$' if args.band_mode == "fit"
             else r'$\mathrm{predicted}\ \propto 1/N_{\rm samp}$')
    ax.plot([], [], color='0.35', ls='--', lw=1.5, label=proxy)
from matplotlib.patches import Patch
# Band legend entries are OFF by default: both spans are described in the paper
# caption instead. --band-legend restores them.
_extra = []
if resolved and args.band_legend:
    _extra.append(Patch(facecolor='k', alpha=0.12,
                        label=r'$16\!-\!84\%\ \mathrm{over}\ \Lambda$'))
    if np.isfinite(b_se) and b_se > 0:
        _extra.append(Patch(facecolor='none', edgecolor='k', hatch='///',
                            label=rf'$\pm1\sigma\ (\mathrm{{{se_tag}}})$'))
_h, _l = ax.get_legend_handles_labels()
ax.legend(_h + _extra, _l + [a.get_label() for a in _extra],
          loc='upper right', fontsize=args.legend_fontsize)

out = args.outdir / f"{args.run}_variance_budget_left_numeric.pdf"
fig.savefig(out, dpi=200)
plt.close(fig)
print(f"saved -> {out}")
print(f"(B) resolved={resolved}  b_med={b_med:.4g}  b_ul={b_ul:.4g}  snr={snr:.2f}")
if np.isfinite(b_se) and b_se > 0:
    print(f"(B) 1-sigma [{args.sigma_band}] = {b_se:.4g} -> [{b_med - b_se:.4g}, {b_med + b_se:.4g}]"
          f"   | Lambda spread 16-84 = [{b_lo:.4g}, {b_hi:.4g}]"
          f"   | snr on this SE = {b_med / b_se:.2f} (printed snr uses B_err)")
if slopes:
    print("fitted slopes b (sigma^2 ~ N^b; -1 == exact 1/N): "
          + ", ".join(f"{k}={v:+.3f}" for k, v in slopes.items()))
