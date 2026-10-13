#!/usr/bin/env python3
"""Per-case comparison of GMM / box / post-hoc trunc / TGMM against the truth.
    python plot_one_case.py out "peak at edge TN(1, 0.3)" [seed]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tgmm_toy import METHODS, NSAMP, XG, truth_cases

out = Path(sys.argv[1]); case = sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COL = dict(gmm="#2a78d6", box="#eb6834", gmm_trunc="#1baf7a", tgmm="#6250d6")
LAB = dict(gmm="GMM (unbounded)", box="GMM + 0.02/99.98 box (production)",
           gmm_trunc="GMM, post-hoc truncated", tgmm="TGMM (truncation-aware fit)")
x = XG[::4]
dist = truth_cases()[case]
p1 = [r for r in json.load(open(out / "part1.json")) if r["case"] == case]
r0 = next(r for r in p1 if r["seed"] == seed)
samp = dist.rvs(NSAMP, random_state=np.random.default_rng(seed))   # same draw as the fit
pt = np.where((x >= -1) & (x <= 1), dist.pdf(x), 0)
Ft = dist.cdf(x)


def cdf(d):
    d = np.asarray(d, float)
    c = np.concatenate([[0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(x))])
    return c / c[-1]


def ren(d):  # renormalise on its own support, for density comparison
    d = np.asarray(d, float); return d / (d.sum() * (x[1] - x[0]))


fig = plt.figure(figsize=(12, 7.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=.38, wspace=.28)

# (a) densities + samples
a = fig.add_subplot(gs[0, :2])
a.axvspan(1, 1.3, color="#00000010", lw=0); a.axvspan(-1.3, -1, color="#00000010", lw=0)
a.hist(samp, bins=80, range=(-1, 1), density=True, color="#cfcfca", label=f"posterior samples (N={NSAMP})")
a.plot(x, pt, "k--", lw=2, label="true density")
for m in METHODS:
    a.plot(x, ren(r0[f"dens_{m}"]) if m != "gmm" else r0["dens_gmm"], color=COL[m], lw=1.4,
           ls=":" if m == "gmm_trunc" else "-", label=LAB[m])
a.set_xlim(-0.6, 1.25); a.set_ylabel("density")
a.set_title(f"{case}, seed {seed}:  K_GMM = {r0['K']},  K_TGMM = {r0['K_tgmm']}; "
            f"box = [{r0['lo']:.3f}, {r0['hi']:.4f}]", fontsize=9.5)
a.legend(frameon=False, fontsize=8, loc="upper left")

# (b) zoom on the edge
a = fig.add_subplot(gs[0, 2])
a.axvspan(1, 1.3, color="#00000010", lw=0)
a.hist(samp, bins=np.linspace(0.8, 1.0, 21), density=False,
       weights=np.full(NSAMP, 1 / (NSAMP * 0.01)), color="#cfcfca")
a.plot(x, pt, "k--", lw=2)
for m in METHODS:
    a.plot(x, ren(r0[f"dens_{m}"]) if m != "gmm" else r0["dens_gmm"], color=COL[m], lw=1.4,
           ls=":" if m == "gmm_trunc" else "-")
a.axvline(r0["hi"], color=COL["box"], lw=.8, ls="--")
a.text(r0["hi"], 0.15, " box edge\n (last sample)", color=COL["box"], fontsize=7.5, ha="right")
lk = float(np.sum(np.asarray(r0["dens_gmm"])[x > 1]) * (x[1] - x[0]))
a.text(1.01, 1.5, f"GMM leak\n{100*lk:.2f}% of mass", fontsize=7.5, color=COL["gmm"])
a.set_xlim(0.8, 1.1); a.set_ylim(0, None); a.set_title("zoom on the x = 1 boundary", fontsize=9.5)

# (c) CDF residual
a = fig.add_subplot(gs[1, :2])
crit = 1.358 / np.sqrt(NSAMP)
a.axhspan(-crit, crit, color="#00000008", lw=0)
ecdf = np.searchsorted(np.sort(samp), x, side="right") / NSAMP
a.plot(x, ecdf - Ft, color="#9a9a94", lw=.9, label="empirical CDF of the samples (noise floor)")
for m in METHODS:
    a.plot(x, cdf(r0[f"dens_{m}"]) - Ft, color=COL[m], lw=1.4, ls=":" if m == "gmm_trunc" else "-",
           label=LAB[m])
a.axhline(0, color="k", lw=.6)
a.text(-0.58, crit * 1.08, f"one-sample KS 5% bound ±{crit:.3f}", fontsize=7.5, color="#666")
a.set_xlim(-0.6, 1.1); a.set_xlabel("x"); a.set_ylabel(r"$F_{\rm fit}(x) - F_{\rm true}(x)$")
a.set_title("CDF residual (KS distance = max |residual|)", fontsize=9.5)
a.legend(frameon=False, fontsize=7.5, loc="lower left")

# (d) KS D vs truth over all seeds
a = fig.add_subplot(gs[1, 2])
rng = np.random.default_rng(0)
emp = []
for r in p1:
    s = dist.rvs(NSAMP, random_state=np.random.default_rng(r["seed"]))
    emp.append(np.abs(np.searchsorted(np.sort(s), x, side="right") / NSAMP - Ft).max())
cols = [("samples\n(ECDF)", emp, "#9a9a94")]
for m in METHODS:
    cols.append(({"gmm": "GMM", "box": "box\n(prod.)", "gmm_trunc": "post-hoc\ntrunc", "tgmm": "TGMM"}[m],
                 [np.abs(cdf(r[f"dens_{m}"]) - Ft).max() for r in p1], COL[m]))
for i, (lab, v, c) in enumerate(cols):
    a.scatter(i + rng.uniform(-.15, .15, len(v)), v, s=16, color=c, alpha=.75, edgecolor="none")
    a.plot([i - .25, i + .25], [np.mean(v)] * 2, color=c, lw=2)
a.axhline(crit, color="#666", lw=.8, ls="--"); a.text(4.4, crit, "KS 5%", fontsize=7.5, va="bottom", ha="right", color="#666")
a.set_xticks(range(len(cols))); a.set_xticklabels([c[0] for c in cols], fontsize=7.5)
a.set_ylabel(r"KS distance to truth  $\sup|F-F_{\rm true}|$"); a.set_ylim(0, None)
a.set_title(f"all {len(p1)} seeds", fontsize=9.5)

fn = out / f"case_{''.join(ch for ch in case if ch.isalnum())}_seed{seed}.png"
fig.savefig(fn, dpi=150, bbox_inches="tight"); print(fn)
