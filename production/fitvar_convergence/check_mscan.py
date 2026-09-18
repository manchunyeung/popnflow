#!/usr/bin/env python
"""How much of the quoted B is MEASURED at the evaluation-pool size M, and how
much is the 1/M extrapolation?

B is the intercept of Vhat(M) = B + c/M over the --fitvar-m-scan points. The
extrapolation is only a small correction if c/M_max << B. This prints, for each
published run, Vhat at every scan point and the size of the correction.
"""
import os

import numpy as np

PROD = '/home/manchun.yeung/population/simon/popnflow/production'

RUNS = {
    'MDC  (fitvar_npe_scan, n_train=3343)':
        os.path.join(PROD, 'fitvar_npe_scan/cache/fitvar_ntrain3343.npz'),
    'sim_cat6 PAPER (paper_diagnostics)':
        os.path.join(PROD, 'sim_cat6_sharp_w8/paper_diagnostics/fitvar_cache_cat000.npz'),
    'sim_cat6 rescan (pool 200k, n_train=5000)':
        os.path.join(PROD, 'fitvar_npe_scan/simcat6/cache/fitvar_ntrain5000.npz'),
}

for label, path in RUNS.items():
    if not os.path.exists(path):
        print(f"\n=== {label}: MISSING ({path}) ===")
        continue
    d = np.load(path, allow_pickle=False)
    Ms = d['Ms']
    V = d['V_of_M']                       # (n_M, n_coords)
    x = 1.0 / np.asarray(Ms, float)
    X = np.vstack([np.ones_like(x), x]).T
    coef = np.linalg.lstsq(X, V, rcond=None)[0]
    B, c = coef[0], coef[1]
    Bm, cm = np.median(B), np.median(c)

    print(f"\n=== {label} ===")
    print(f"  n_train={int(np.atleast_1d(d['n_train'])[0])}  M_max={int(Ms[-1])}  "
          f"n_coords={V.shape[1]}")
    print(f"  {'M':>9} {'Vhat(M)':>10} {'Vhat/B':>8}")
    for M, v in zip(Ms, np.median(V, axis=1)):
        print(f"  {M:9d} {v:10.4f} {v / Bm:8.3f}")
    print(f"  intercept B = {Bm:.4f}   slope c = {cm:.4g}")
    print(f"  c/M_max = {cm / Ms[-1]:.4f}  =  {cm / Ms[-1] / Bm * 100:.1f}% of B "
          f"  <- the MC contamination removed by the extrapolation at the LARGEST M")
    print(f"  c/M_min = {cm / Ms[0]:.4f}  =  {cm / Ms[0] / Bm * 100:.1f}% of B "
          f"  <- at the SMALLEST scan point")
