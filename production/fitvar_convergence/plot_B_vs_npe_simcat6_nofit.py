#!/usr/bin/env python
"""B vs N_PE for sim_cat6_sharp_w8 only, with NO fitted curve.

Same data and styling as fitvar_npe_scan/simcat6/plot_B_vs_npe_solo.py, minus
the fitted B_inf + kappa/N_PE curve. The grey dashed line is NOT a fit: it is
the pure 1/N_PE reference anchored to the largest-N_PE point, kept so the plot
is readable on its own (pass --no-reference to drop it too).

Pure post-processing of the saved fitvar_ntrain*.npz caches -- no GMM fitting,
no bootstrap, no pool rebuild.
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter

PROD = '/home/manchun.yeung/population/simon/popnflow/production'
HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument('--cache-dir',
                default=os.path.join(PROD, 'fitvar_npe_scan/simcat6/cache'))
ap.add_argument('--out', default=os.path.join(HERE, 'figs/B_vs_npe_simcat6_nofit.pdf'))
ap.add_argument('--no-reference', action='store_true',
                help='also drop the grey dashed pure-1/N_PE reference line')
cli = ap.parse_args()

# usetex drops minus signs on this host -- mathtext only.
plt.rcParams.update({
    "text.usetex": False, "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12.5,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
    "axes.edgecolor": "#5a5a5a", "axes.linewidth": 0.9,
    "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
    "legend.frameon": False, "figure.dpi": 140,
})
SEQ = LinearSegmentedColormap.from_list("npe", ["#bcd9f2", "#4e94cf", "#0b3d6b"])
INK, REF = "#1b1b1b", "#9a9a9a"

runs = []
for f in sorted(glob.glob(os.path.join(cli.cache_dir, "fitvar_ntrain*.npz"))):
    d = np.load(f)
    runs.append(dict(n=int(np.ravel(d["n_train"])[0]),
                     ViC=np.squeeze(d["V_iC_of_M"])[-1],
                     B=np.ravel(d["B_hat"])))
if not runs:
    raise SystemExit(f"no fitvar_ntrain*.npz under {cli.cache_dir}")
runs.sort(key=lambda r: r["n"])

N = np.array([r["n"] for r in runs], float)
Bmed = np.array([np.median(r["B"]) for r in runs])
Blo = np.array([np.percentile(r["B"], 16) for r in runs])
Bhi = np.array([np.percentile(r["B"], 84) for r in runs])
Berr = np.array([np.median(np.sqrt(np.sum(2.0 * r["ViC"] ** 2 / 49, axis=-1)))
                 for r in runs])
cols = [SEQ(t) for t in np.linspace(0.15, 1.0, len(N))]

print(f"sim_cat6: {len(N)} runs, N_PE = {list(N.astype(int))}")
for n, b, e in zip(N, Bmed, Berr):
    print(f"  N_PE={int(n):5d}  B={b:.4f} +- {e:.4f}   N*B={n * b:8.1f}")

fig, ax = plt.subplots(figsize=(6.2, 5.2))

if not cli.no_reference:
    xs = np.logspace(np.log10(N.min() * 0.8), np.log10(N.max() * 1.2), 60)
    ax.plot(xs, Bmed[-1] * (N[-1] / xs), "--", color=REF, lw=1.6,
            label="pure $1/N_{\\rm PE}$ (reference)")

ax.fill_between(N, Blo, Bhi, color="#bcd9f2", alpha=0.4, lw=0,
                label="16-84% over $\\Lambda$")
ax.errorbar(N, Bmed, yerr=Berr, fmt="none", ecolor=INK, elinewidth=1.3,
            capsize=3.5, zorder=3)
for x, y, c in zip(N, Bmed, cols):
    ax.plot(x, y, "o", ms=9, color=c, mec="white", mew=1.4, zorder=4)
ax.plot([], [], "o", ms=9, color=SEQ(0.85), mec="white", label="measured $B$")

ax.set_xscale("log")
ax.set_yscale("log")
show = list(N.astype(int))
ax.xaxis.set_major_locator(FixedLocator(show))
ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in show]))
ax.xaxis.set_minor_formatter(NullFormatter())
plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel("$N_{\\rm PE}$   (posterior samples per event)")
ax.set_ylabel("$B = {\\rm Var}_D[\\ln L]$   (fit variance)")
ax.set_title("sim_cat6_sharp_w8", color=INK)
ax.legend(loc="lower left", fontsize=9)

fig.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
fig.savefig(cli.out, bbox_inches="tight")
fig.savefig(cli.out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
print(f"saved -> {cli.out}")
print(f"saved -> {cli.out.replace('.pdf', '.png')}")
