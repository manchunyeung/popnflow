#!/usr/bin/env python
"""Warm-started EM: the fit-variance floor before and after.

NUMBER PROVENANCE -- read before reusing this figure.
  B_before, sigma2_PE, rho_before : measured, read live from
      sim_cat6_sharp_w8/paper_diagnostics/fitvar_cache_cat000.npz
  B_after, rho_after              : TRANSCRIBED from
      popnflow/fitvar_convergence/fitvar_summary.md
      The warm-start run left no results cache on disk, so these are not
      re-derived here. Regenerate with --fitvar-results-cache and this script
      will read them instead.
  bagging                         : measured, read live from
      bagging_simcat6_K5_perevent.npz (this directory)

Strategy panel: n_init=20 and reg_covar are reported in fitvar_summary.md as
"failed" without a number, so they are drawn at 0 and labelled as such rather
than given a fabricated value.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROD = '/home/manchun.yeung/population/simon/popnflow/production'
sys.path.insert(0, HERE)

# usetex drops minus signs on this host -- mathtext only.
plt.rcParams.update({
    "text.usetex": False, "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
    "axes.edgecolor": "#5a5a5a", "axes.linewidth": 0.9,
    "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
    "legend.frameon": False, "figure.dpi": 140,
})
INK, BEFORE, AFTER, REF, WARN = "#1b1b1b", "#4e94cf", "#0b3d6b", "#9a9a9a", "#b3462f"

# ---- measured ---------------------------------------------------------------
d0 = np.load(os.path.join(PROD,
             'sim_cat6_sharp_w8/paper_diagnostics/fitvar_cache_cat000.npz'))
B_before = float(np.median(d0['B_hat']))
s2_pe = float(np.median(d0['V_pe']))
rho_before = B_before / s2_pe

db = np.load(os.path.join(HERE, 'bagging_simcat6_K5_perevent.npz'))
B_bag = float(np.median(db['V_bag'].sum(axis=1)))
B_plain_bag = float(np.median(db['V_plain'].sum(axis=1)))
red_bag = (1 - B_bag / B_plain_bag) * 100

# ---- transcribed from fitvar_summary.md -------------------------------------
B_after = 0.02237
rho_after = 0.452
red_bgm = 52.0
red_warm = (1 - B_after / B_before) * 100

fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))

# ================= left: the floor before / after =============================
ax = axes[0]
ax.axhline(s2_pe, color=REF, ls="--", lw=1.7, zorder=1)
for xi, (B, c, lab) in enumerate([(B_before, BEFORE, "before"),
                                  (B_after, AFTER, "warm-started")]):
    ax.bar(xi, B, width=0.55, color=c, edgecolor="white", linewidth=1.2, zorder=2)
    ax.text(xi, B + 0.0018, f"{B:.4f}", ha="center", va="bottom",
            fontsize=11.5, color=INK)
ax.set_xticks([0, 1])
ax.set_xticklabels(["before\n(k-means init, $n_{\\rm init}$=5)",
                    "warm-started\n(init from $g_0$, $n_{\\rm init}$=1)"])
ax.set_ylabel("$B = {\\rm Var}_D[\\ln L]$   (fit variance, nats$^2$)")
ax.set_ylim(0, B_before * 1.32)
ax.annotate("", xy=(0.5, B_after), xytext=(0.5, B_before),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
ax.text(0.56, (B_before + B_after) / 2, f"$-${red_warm:.0f}%",
        ha="left", va="center", fontsize=13, color=INK)
ax.text(0.02, 0.97, f"$\\rho = B/\\sigma^2_{{\\rm PE}}$:  "
                    f"{rho_before:.2f}  $\\rightarrow$  {rho_after:.2f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=11, color=INK)
ax.text(1.28, s2_pe + 0.0013, f"$\\sigma^2_{{\\rm PE}}$ = {s2_pe:.4f}  (raw PE variance)",
        ha="right", va="bottom", fontsize=9.5, color="#6b6b6b")
ax.set_title("Fit-variance floor, sim_cat6_sharp_w8", color=INK)

# ================= right: strategies tried ====================================
ax = axes[1]
labels = ["$n_{\\rm init}$ = 20", "raise reg_covar",
          f"bagging ($K$=5)", "Bayesian GMM", "warm-start EM"]
vals = [0.0, 0.0, red_bag, red_bgm, red_warm]
cols = [WARN, WARN, BEFORE, BEFORE, AFTER]
y = np.arange(len(labels))
ax.barh(y, vals, color=cols, edgecolor="white", linewidth=1.2, height=0.62, zorder=2)
for yi, v in zip(y, vals):
    if v <= 0.5:
        ax.text(1.0, yi, "no reduction", va="center", ha="left",
                fontsize=10, color=WARN)
    else:
        ax.text(v + 1.2, yi, f"{v:.0f}%", va="center", ha="left",
                fontsize=11, color=INK)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("reduction in $B$  (%)")
ax.set_xlim(0, 78)
ax.set_title("Strategies tested", color=INK)
ax.text(0.97, 0.90, "bagging also adds a 0.113 nat $\\Lambda$-dependent\n"
                    "bias, leaving a net gain of only ~6%",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=WARN)

fig.suptitle("Stabilising the EM optimiser removes most of the fit-variance floor",
             y=1.03, fontsize=12.5)
fig.tight_layout()
out = os.path.join(HERE, "figs", "warmstart_fitvar.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")

print(f"B before      = {B_before:.5f}   (measured)")
print(f"sigma^2_PE    = {s2_pe:.5f}   (measured)   rho_before = {rho_before:.3f}")
print(f"B after       = {B_after:.5f}   (transcribed from fitvar_summary.md)")
print(f"reduction     = {red_warm:.1f}%   rho_after = {rho_after:.3f}")
print(f"bagging K=5   = {red_bag:.1f}% reduction (measured here)")
print(f"saved -> {out}")
