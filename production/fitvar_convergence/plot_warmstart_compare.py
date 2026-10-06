#!/usr/bin/env python
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "text.usetex": False, "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
    "axes.edgecolor": "#5a5a5a", "axes.linewidth": 0.9,
    "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
    "legend.frameon": False, "figure.dpi": 140,
})
INK, PLAIN, WARM, BIAS = "#1b1b1b", "#4e94cf", "#2ca02c", "#b3462f"

d = np.load(os.path.join(HERE, 'warmstart_simcat6_69.npz'))
Vp = d['V_plain'].sum(axis=1)          # (nC,) summed over events
Vw = d['V_warm'].sum(axis=1)
Bp, Bw = np.median(Vp), np.median(Vw)
shift = (d['lnL_ref_warm'] - d['lnL_ref_plain']).sum(axis=0)
bias_dep = np.std(shift, ddof=1)       # Lambda-dependent part
Kc = d['Kcomp']

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))

# ---- left: the variance -----------------------------------------------------
ax = axes[0]
x = [0, 1]
for xi, (B, V, c, lab) in enumerate([
        (Bp, Vp, PLAIN, "plain\n(current estimator)"),
        (Bw, Vw, WARM, "warm_start")]):
    lo, hi = np.percentile(V, 16), np.percentile(V, 84)
    ax.bar(xi, B, width=0.55, color=c, edgecolor="white", linewidth=1.2, zorder=2)
    ax.errorbar(xi, B, yerr=[[B - lo], [hi - B]], fmt="none", ecolor=INK,
                elinewidth=1.4, capsize=5, zorder=3)
    ax.text(xi, hi + 0.003, f"{B:.4f}", ha="center", va="bottom",
            fontsize=11, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(["plain\n(current estimator)", "warm_start\n(diagnostic)"])
ax.set_ylabel("$B = {\\rm Var}_D[\\ln L]$   (fit variance)")
ax.set_ylim(0, max(np.percentile(Vp, 84), np.percentile(Vw, 84)) * 1.28)
ax.annotate("", xy=(1, Bw), xytext=(1, Bp),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
ax.text(1.32, (Bp + Bw) / 2, f"$-${(1 - Bw / Bp) * 100:.1f}%",
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
ax.bar(1, Bw, width=0.55, color=WARM, edgecolor="white", linewidth=1.2, zorder=2)
ax.bar(1, b2, bottom=Bw, width=0.55, color=BIAS, edgecolor="white",
       linewidth=1.2, label="(induced $\\Lambda$-dependent bias)$^2$", zorder=2)
ax.text(0, Bp + 0.0025, f"{Bp:.4f}", ha="center", va="bottom", fontsize=11, color=INK)
ax.text(1, Bw + b2 + 0.0025, f"{Bw + b2:.4f}", ha="center", va="bottom",
        fontsize=11, color=INK)
ax.set_xticks([0, 1])
ax.set_xticklabels(["plain", "warm_start\n(diagnostic)"])
ax.set_ylabel("variance + bias$^2$   (nats$^2$)")
ax.set_ylim(0, max(Bp, Bw + b2) * 1.30)
ax.set_title(f"Counting the bias: net $-${(1 - (Bw + b2) / Bp) * 100:.0f}%, "
             f"not $-${(1 - Bw / Bp) * 100:.0f}%", color=INK)
ax.legend(loc="upper right", fontsize=9)

fig.suptitle("Warm-starting the per-event GMM, sim_cat6_sharp_w8 "
             "($N_{\\rm PE}$ = 5000, 69 events)", y=1.02, fontsize=12.5)
fig.tight_layout()
out = os.path.join(HERE, "figs", "mdc_variace_warmstart_compare.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")

print(f"saved -> {out}")
