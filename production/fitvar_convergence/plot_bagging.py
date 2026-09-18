#!/usr/bin/env python
"""Bagging result on sim_cat6: variance reduction, and what it costs in bias.

Left  -- B for the plain vs bagged estimator at the per-event K the inference
         actually uses, with the 16-84% spread over Lambda.
Right -- the honest accounting. Bagging removes variance but introduces a
         Lambda-dependent shift; counting bias^2 against the variance saved
         turns a 27.6% variance cut into a ~6% net gain.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# usetex drops minus signs on this host -- mathtext only.
plt.rcParams.update({
    "text.usetex": False, "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
    "axes.edgecolor": "#5a5a5a", "axes.linewidth": 0.9,
    "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
    "legend.frameon": False, "figure.dpi": 140,
})
INK, PLAIN, BAG, BIAS = "#1b1b1b", "#4e94cf", "#0b3d6b", "#b3462f"

d = np.load(os.path.join(HERE, 'bagging_simcat6_K5_perevent.npz'))
Vp = d['V_plain'].sum(axis=1)          # (nC,) summed over events
Vb = d['V_bag'].sum(axis=1)
Bp, Bb = np.median(Vp), np.median(Vb)
shift = (d['lnL_ref_bag'] - d['lnL_ref_plain']).sum(axis=0)
bias_dep = np.std(shift, ddof=1)       # Lambda-dependent part
Kc = d['Kcomp']

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))

# ---- left: the variance -----------------------------------------------------
ax = axes[0]
x = [0, 1]
for xi, (B, V, c, lab) in enumerate([
        (Bp, Vp, PLAIN, "plain\n(current estimator)"),
        (Bb, Vb, BAG, "bagged\n(5-member ensemble)")]):
    lo, hi = np.percentile(V, 16), np.percentile(V, 84)
    ax.bar(xi, B, width=0.55, color=c, edgecolor="white", linewidth=1.2, zorder=2)
    ax.errorbar(xi, B, yerr=[[B - lo], [hi - B]], fmt="none", ecolor=INK,
                elinewidth=1.4, capsize=5, zorder=3)
    ax.text(xi, hi + 0.003, f"{B:.4f}", ha="center", va="bottom",
            fontsize=11, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(["plain\n(current estimator)", "bagged\n($K_{\\rm bag}$ = 5)"])
ax.set_ylabel("$B = {\\rm Var}_D[\\ln L]$   (fit variance)")
ax.set_ylim(0, max(np.percentile(Vp, 84), np.percentile(Vb, 84)) * 1.28)
ax.annotate("", xy=(1, Bb), xytext=(1, Bp),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
ax.text(1.32, (Bp + Bb) / 2, f"$-${(1 - Bb / Bp) * 100:.1f}%",
        ha="left", va="center", fontsize=12, color=INK)
ax.set_title(f"Variance reduction  (per-event $K$ = {Kc.min()}-{Kc.max()}, "
             f"median {int(np.median(Kc))})", color=INK)
ax.plot([], [], " ", label="error bar: 16-84% over $\\Lambda$")
ax.legend(loc="upper right", fontsize=9)

# ---- right: what it costs ---------------------------------------------------
ax = axes[1]
b2 = bias_dep ** 2
ax.bar(0, Bp, width=0.55, color=PLAIN, edgecolor="white", linewidth=1.2,
       label="variance", zorder=2)
ax.bar(1, Bb, width=0.55, color=BAG, edgecolor="white", linewidth=1.2, zorder=2)
ax.bar(1, b2, bottom=Bb, width=0.55, color=BIAS, edgecolor="white",
       linewidth=1.2, label="(induced $\\Lambda$-dependent bias)$^2$", zorder=2)
ax.text(0, Bp + 0.0025, f"{Bp:.4f}", ha="center", va="bottom", fontsize=11, color=INK)
ax.text(1, Bb + b2 + 0.0025, f"{Bb + b2:.4f}", ha="center", va="bottom",
        fontsize=11, color=INK)
ax.set_xticks([0, 1])
ax.set_xticklabels(["plain", "bagged\n($K_{\\rm bag}$ = 5)"])
ax.set_ylabel("variance + bias$^2$   (nats$^2$)")
ax.set_ylim(0, max(Bp, Bb + b2) * 1.30)
ax.set_title(f"Counting the bias: net $-${(1 - (Bb + b2) / Bp) * 100:.0f}%, "
             f"not $-${(1 - Bb / Bp) * 100:.0f}%", color=INK)
ax.legend(loc="upper right", fontsize=9)

fig.suptitle("Bagging the per-event GMM, sim_cat6_sharp_w8 "
             "($N_{\\rm PE}$ = 5000, 69 events)", y=1.02, fontsize=12.5)
fig.tight_layout()
out = os.path.join(HERE, "figs", "bagging_simcat6.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")

print(f"plain  B = {Bp:.4f}")
print(f"bagged B = {Bb:.4f}   ({(1 - Bb / Bp) * 100:.1f}% reduction)")
print(f"Lambda-dependent bias = {bias_dep:.4f} nats, bias^2 = {b2:.4f}")
print(f"variance+bias^2: {Bp:.4f} -> {Bb + b2:.4f} "
      f"({(1 - (Bb + b2) / Bp) * 100:.1f}% net)")
print(f"saved -> {out}")
