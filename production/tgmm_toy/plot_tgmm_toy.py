#!/usr/bin/env python3
"""Tables + figures for tgmm_toy.py output.  python plot_tgmm_toy.py out"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tgmm_toy import LAMS, METHODS, XG

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "text.usetex": False})
COL = dict(true="#0b0b0b", gmm="#2a78d6", box="#eb6834",
           gmm_trunc="#1baf7a", tgmm="#6250d6")
LAB = dict(gmm="GMM (unbounded)", box="GMM + box (production)",
           gmm_trunc="GMM, post-hoc truncated", tgmm="TGMM (trunc.-aware fit)")
x = XG[::4]
inD = (x >= -1) & (x <= 1)
dx = x[1] - x[0]

# ── Part 1 ───────────────────────────────────────────────────────────────
p1 = json.load(open(out / "part1.json"))
cases = list(dict.fromkeys(r["case"] for r in p1))
i3 = [np.argmin(np.abs(LAMS - v)) for v in (-3, 3)]
print("PART 1  (mean over seeds; err = [logZ(lam)-logZ(0)] - same for truth)")
hdr = f"{'case':26s} {'K':>4s} {'K_tg':>4s} {'leak':>8s} {'boxcut':>8s} {'true_out':>8s}"
print(hdr + "".join(f" {m+':L1':>13s} {m+':e(+3)':>14s}" for m in METHODS))
for c in cases:
    rs = [r for r in p1 if r["case"] == c]
    f = lambda k: np.mean([r[k] for r in rs])
    line = (f"{c:26s} {f('K'):4.1f} {f('K_tgmm'):4.1f} {f('leak'):8.4f} "
            f"{f('box_removed_in_D'):8.5f} {f('true_mass_outside_box'):8.5f}")
    for m in METHODS:
        e = np.array([r[f"err_{m}"] for r in rs])
        line += f" {f('L1_'+m):13.4f} {e[:, i3[1]].mean():+8.4f}±{e[:, i3[1]].std():.3f}"
    print(line)

fig, ax = plt.subplots(3, len(cases), figsize=(3.0 * len(cases), 7.6),
                       gridspec_kw=dict(height_ratios=[1.2, 1, 1]))
for j, c in enumerate(cases):
    rs = [r for r in p1 if r["case"] == c]
    r0 = rs[0]
    pt = np.array(r0["dens_true"])
    a = ax[0, j]
    a.axvspan(-1.6, -1, color="#00000010", lw=0); a.axvspan(1, 1.6, color="#00000010", lw=0)
    a.plot(x, pt, color=COL["true"], lw=2, ls="--", label="true posterior")
    for m in ("gmm", "box", "tgmm"):
        a.plot(x, r0[f"dens_{m}"], color=COL[m], lw=1.4, label=LAB[m])
    a.set_xlim(-1.4, 1.4); a.set_title(f"{c}\nK_GMM={r0['K']}, K_TGMM={r0['K_tgmm']}", fontsize=8.5)
    if j == 0: a.set_ylabel("density (seed 0)")
    # ratio inside D, every density renormalised on D
    a = ax[1, j]
    for m in ("gmm", "tgmm"):
        R = []
        for r in rs:
            g = np.where(inD, np.array(r[f"dens_{m}"]), 0); g /= g.sum() * dx
            R.append(np.where(inD & (pt > 1e-3 * pt.max()), g / np.maximum(pt, 1e-300), np.nan))
        R = np.array(R)
        a.fill_between(x, np.nanpercentile(R, 10, 0), np.nanpercentile(R, 90, 0), color=COL[m], alpha=.2, lw=0)
        a.plot(x, np.nanmedian(R, 0), color=COL[m], lw=1.4,
               label="GMM / box / post-hoc trunc." if m == "gmm" else LAB[m])
    a.axhline(1, color="#888", lw=.8); a.set_xlim(-1.05, 1.05); a.set_ylim(0.6, 1.3)
    if j == 0: a.set_ylabel("fit / truth inside [-1,1]\n(renormalised on D; 10-90% over seeds)")
    a = ax[2, j]
    for m in METHODS:
        e = np.array([r[f"err_{m}"] for r in rs])
        a.plot(LAMS, e.mean(0), color=COL[m], lw=1.4, ls=":" if m == "gmm_trunc" else "-", label=LAB[m])
    a.axhline(0, color="#888", lw=.8); a.set_xlabel(r"population tilt $\lambda$")
    if j == 0: a.set_ylabel(r"$\Delta\log Z(\lambda)$ error vs truth")
ax[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
ax[1, 0].legend(fontsize=7, frameon=False, loc="lower center")
ax[2, 0].legend(fontsize=7, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(out / "fig1_single_posteriors.png", dpi=150)

# ── Part 2 ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
print("\nPART 2  shift of lambda posterior mean vs EXACT, in units of exact posterior sd")
for a, tag, title in zip(ax, ("popD", "popwide"),
                         ("population support = PE domain [-1,1]",
                          "population on [-1.5,1.5], PE prior on [-1,1]")):
    f = out / f"part2_{tag}.json"
    if not f.exists():
        continue
    S = json.load(open(f))
    ex = np.array([s["exact"] for s in S])     # (ncat, 3) mean, sd, median
    print(f"  [{tag}] exact: mean sd {ex[:,1].mean():.3f}; mean(lam_hat - 2) = {np.mean(ex[:,0]-2):+.3f}")
    for i, m in enumerate(METHODS):
        v = np.array([s[m] for s in S])
        z = (v[:, 0] - ex[:, 0]) / ex[:, 1]
        print(f"     {m:10s} shift = {z.mean():+.3f} ± {z.std()/np.sqrt(len(z)):.3f} sd  "
              f"(range {z.min():+.2f}..{z.max():+.2f}); width ratio {np.mean(v[:,1]/ex[:,1]):.3f}")
        a.scatter(np.full(len(z), i) + np.random.default_rng(i).uniform(-.15, .15, len(z)), z,
                  s=14, color=COL[m], alpha=.7, edgecolor="none")
        a.errorbar(i + .3, z.mean(), z.std() / np.sqrt(len(z)), fmt="o", color=COL[m], ms=5)
    a.axhline(0, color="#888", lw=.8)
    a.set_xticks(range(4)); a.set_xticklabels(["GMM", "GMM+box\n(prod.)", "GMM\npost-hoc", "TGMM"])
    a.set_title(title, fontsize=9)
ax[0].set_ylabel(r"($\hat\lambda_{\rm method}-\hat\lambda_{\rm exact}$) / $\sigma_{\rm exact}$")
fig.tight_layout(); fig.savefig(out / "fig2_hierarchical_lambda.png", dpi=150)
print("\nfigures written to", out)
