"""
Fig-4-style variance budget with BOTH fit-variance floors drawn:

  * the published floor, B = 0.0605, now labelled an UPPER BOUND -- it contains
    EM optimiser instability (basin switching between bootstrap draws) on top of
    the genuine finite-sample fit variance;
  * the warm-started floor, B = 0.0224, obtained by initialising every bootstrap
    EM from the full-data fit g_0 with n_init=1, which removes basin switching.

Derived from plot_variance_budget_left_numeric.py (the generator SOURCES.md
records as reproducing fig04_variance.pdf byte-for-byte). That file is NOT
modified -- this is a separate variant so fig04 stays reproducible.

Term (A) curves, the published floor, its 16-84% Lambda spread and its jackknife
1-sigma band are all read from the same two caches fig04 uses. The warm-start
value is passed in (--warmstart-B), because that run left no results cache on
disk; point --warmstart-cache at an NPZ to read it properly instead.

Usage (the fig04 caches):
  python plot_fig04_warmstart.py \
    --fitvar-cache   sim_cat6_sharp_w8/paper_diagnostics/fitvar_cache_cat000.npz \
    --variance-cache sim_cat6_sharp_w8/paper_diagnostics/variance_cache_3method_cat000_2e4.npz
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

PROD = Path('/home/manchun.yeung/population/simon/popnflow/production')
HERE = Path(__file__).resolve().parent

p = argparse.ArgumentParser()
p.add_argument("--run", default="simcat6")
p.add_argument("--outdir", type=Path, default=HERE / "figs")
p.add_argument("--variance-cache", type=Path,
               default=PROD / "sim_cat6_sharp_w8/paper_diagnostics/variance_cache_3method_cat000_2e4.npz")
p.add_argument("--fitvar-cache", type=Path,
               default=PROD / "sim_cat6_sharp_w8/paper_diagnostics/fitvar_cache_cat000.npz")
p.add_argument("--warmstart-B", type=float, default=0.02237,
               help="warm-started floor; default transcribed from fitvar_summary.md")
p.add_argument("--warmstart-cache", type=Path, default=None,
               help="NPZ from a warm-started compute_fit_variance(); overrides --warmstart-B")
p.add_argument("--legend-fontsize", type=float, default=13.0)
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

# ================= term (B): the published floor =========================
fv = np.load(args.fitvar_cache, allow_pickle=False)
B_hat = np.asarray(fv["B_hat"])
n_train = int(np.asarray(fv["n_train"]).item())
b_med = float(np.nanmedian(B_hat))
b_lo = float(np.nanpercentile(B_hat, 16))
b_hi = float(np.nanpercentile(B_hat, 84))


def _jackknife_B_se(fv_):
    """Leave-one-event-out SE on the OLS intercept B (same as fig04's)."""
    Vi = np.asarray(fv_["V_iC_of_M"])
    x = 1.0 / np.asarray(fv_["Ms"], dtype=float)
    X = np.vstack([np.ones_like(x), x]).T
    Xi = np.linalg.inv(X.T @ X)
    n_e = Vi.shape[2]
    tot = Vi.sum(axis=2)
    Bj = np.empty((n_e, tot.shape[1]))
    for e in range(n_e):
        Bj[e] = (Xi @ X.T @ ((tot - Vi[:, :, e]) * n_e / (n_e - 1.0)))[0]
    return np.sqrt((n_e - 1.0) / n_e * np.sum((Bj - Bj.mean(0)) ** 2, axis=0))


b_se = float(np.nanmedian(_jackknife_B_se(fv))) if "V_iC_of_M" in fv.files else np.nan

# ================= term (B): the warm-started floor ======================
if args.warmstart_cache is not None:
    wf = np.load(args.warmstart_cache, allow_pickle=False)
    w_med = float(np.nanmedian(np.asarray(wf["B_hat"])))
    w_lo = float(np.nanpercentile(np.asarray(wf["B_hat"]), 16))
    w_hi = float(np.nanpercentile(np.asarray(wf["B_hat"]), 84))
    w_src = "measured"
else:
    w_med, w_lo, w_hi, w_src = args.warmstart_B, np.nan, np.nan, "fitvar_summary.md"

# ================= plot =================
fig, ax = plt.subplots(1, 1, figsize=(8.6, 6.8), constrained_layout=True)

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
    slope, lnA = np.polyfit(np.log(x), np.log(med), 1)
    ax.plot(Ns_an, np.exp(lnA) * Ns_an ** slope, color=color, ls='--', lw=1.5,
            alpha=0.85, zorder=1)
    ax.errorbar(x, med, yerr=err, marker=marker, ls='-', color=color, lw=2,
                ms=5, capsize=3, elinewidth=1.2, label=rf'$\mathrm{{{name}}}$')

# published floor -> upper bound
ax.axhline(b_med, color='k', ls='-', lw=2.0, alpha=0.9,
           label=rf'$\mathrm{{upper\ bound}}:\ B={b_med:.4f}$')
ax.axhspan(b_lo, b_hi, color='k', alpha=0.12)
if np.isfinite(b_se) and b_se > 0:
    ax.axhspan(b_med - b_se, b_med + b_se, facecolor='none', edgecolor='k',
               hatch='///', lw=0.0, alpha=0.5, zorder=2)

# warm-started floor
WS = C[3]
ax.axhline(w_med, color=WS, ls='-', lw=2.4, alpha=0.95,
           label=rf'$\mathrm{{warm\ start}}:\ B={w_med:.4f}$')
if np.isfinite(w_lo):
    ax.axhspan(w_lo, w_hi, color=WS, alpha=0.12)

# the gap between them
ax.annotate("", xy=(1.5e1, w_med), xytext=(1.5e1, b_med),
            arrowprops=dict(arrowstyle="<->", color='0.25', lw=1.6))
ax.text(1.75e1, np.sqrt(b_med * w_med), rf'$-{(1 - w_med / b_med) * 100:.0f}\%$',
        ha="left", va="center", fontsize=14, color='0.15')

ax.axvline(n_train, color='k', ls=':', lw=2, alpha=0.7,
           label=fr'$N_{{\rm PE}}={n_train}$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$N_{\rm samp}$')
ax.set_ylabel(r'$\sigma^2_{\ln\hat{\mathcal{L}}}$')
ax.grid(True, which='both', ls=':', alpha=0.3)

extra = [Patch(facecolor='k', alpha=0.12,
               label=r'$16\!-\!84\%\ \mathrm{over}\ \Lambda$'),
         Patch(facecolor='none', edgecolor='k', hatch='///',
               label=r'$\pm1\sigma\ (\mathrm{event\ jackknife})$')]
h, l = ax.get_legend_handles_labels()
ax.legend(h + extra, l + [a.get_label() for a in extra],
          loc='upper right', fontsize=args.legend_fontsize)

out = args.outdir / f"{args.run}_fig04_warmstart.pdf"
fig.savefig(out, dpi=200)
fig.savefig(str(out).replace(".pdf", ".png"), dpi=170)
plt.close(fig)
print(f"upper bound  B = {b_med:.5f}   (measured; 16-84% [{b_lo:.4f}, {b_hi:.4f}], "
      f"1sigma {b_se:.4f})")
print(f"warm start   B = {w_med:.5f}   ({w_src})")
print(f"reduction      = {(1 - w_med / b_med) * 100:.1f}%")
print(f"saved -> {out}")
