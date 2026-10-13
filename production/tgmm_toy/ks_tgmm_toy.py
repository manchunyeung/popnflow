#!/usr/bin/env python3
"""KS comparison of the fitted per-event densities in part1.json.

For each case/seed/method:
  D_true    sup |F_fit - F_true| on the grid (exact, no sampling noise)
  D_ho, p   one-sample KS of 5000 FRESH truth draws vs F_fit (held out, so
            not biased by fitting on the same samples)
F_fit is the CDF of the density as used: gmm over the whole line (leak shows
up as F(-1) > 0), box normalised on its box, gmm_trunc/tgmm on [-1,1].
Also D_edge = sup |F_fit - F_true| restricted to the outer 5% of the domain.

    python ks_tgmm_toy.py out
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from tgmm_toy import METHODS, NSAMP, XG, truth_cases

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
p1 = json.load(open(out / "part1.json"))
x = XG[::4]
cases = truth_cases()
edge = (np.abs(x) > 0.9) & (np.abs(x) <= 1.0)


def cdf_from(d):
    d = np.asarray(d, float)
    c = np.concatenate([[0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(x))])
    return c / c[-1]


rows = {}
for r in p1:
    dist = cases[r["case"]]
    Ft = dist.cdf(x)
    ho = np.sort(dist.rvs(NSAMP, random_state=np.random.default_rng(10_000 + r["seed"])))
    for m in METHODS:
        F = cdf_from(r[f"dens_{m}"])
        Dt = np.abs(F - Ft).max()
        De = np.abs(F - Ft)[edge].max()
        res = stats.kstest(ho, lambda v: np.interp(v, x, F))
        rows.setdefault((r["case"], m), []).append((Dt, De, res.statistic, res.pvalue))
    # reference: KS of the FIT samples' own empirical CDF vs truth (noise floor)

crit = 1.358 / np.sqrt(NSAMP)
print(f"KS 5% critical D for N={NSAMP}: {crit:.4f}\n")
print(f"{'case':26s} {'method':10s} {'D_true':>8s} {'D_edge':>8s} {'D_heldout':>9s} "
      f"{'median p':>9s} {'frac p<0.05':>11s}")
for c in cases:
    for m in METHODS:
        a = np.array(rows[(c, m)])
        print(f"{c:26s} {m:10s} {a[:,0].mean():8.4f} {a[:,1].mean():8.4f} "
              f"{a[:,2].mean():9.4f} {np.median(a[:,3]):9.3f} {np.mean(a[:,3]<0.05):11.2f}")
    print()
json.dump({f"{c}|{m}": v for (c, m), v in rows.items()}, open(out / "ks.json", "w"))
