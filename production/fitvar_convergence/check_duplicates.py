#!/usr/bin/env python
"""How many of each event's 3343 'PE samples' are actually distinct points?

If the MDC posterior samples were produced by resampling weighted nested-sampling
output WITH replacement, the nominal 3343 hides a much smaller number of distinct
locations. The GMM would then be pinned by those distinct points, and neither
bootstrapping nor subsampling the file could ever recover 1/N_PE scaling --
which is exactly the pattern run_arms.py measured.
"""
import os

import numpy as np

PROD = '/home/manchun.yeung/population/simon/popnflow/production'
Nobs = 69

m1 = np.loadtxt(os.path.join(PROD, 'input_data/mdc_m1det_69rand.txt'))
m2 = np.loadtxt(os.path.join(PROD, 'input_data/mdc_m2det_69rand.txt'))
dL = np.loadtxt(os.path.join(PROD, 'input_data/mdc_dL_69rand.txt'))
nsamp = m1.shape[0] // Nobs
print(f"{Nobs} events x {nsamp} samples")

X = np.column_stack([m1, m2, dL]).reshape(Nobs, nsamp, 3)

pub = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'published_summary.npz'))
p_e, V_e = pub['p_e'], pub['V_full']

frac, ess = np.zeros(Nobs), np.zeros(Nobs)
for e in range(Nobs):
    u, counts = np.unique(X[e], axis=0, return_counts=True)
    frac[e] = len(u) / nsamp
    # Kish effective sample size of the duplicate-multiplicity distribution
    ess[e] = counts.sum() ** 2 / np.sum(counts ** 2)

print(f"\ndistinct fraction over the {Nobs} events: "
      f"min {frac.min():.4f}  median {np.median(frac):.4f}  max {frac.max():.4f}")
print(f"n_distinct:       min {int(frac.min() * nsamp)}  "
      f"median {int(np.median(frac) * nsamp)}  max {int(frac.max() * nsamp)}")

good = np.isfinite(p_e) & (V_e > 0)
if frac.std() > 1e-12:
    print(f"\ncorr(distinct fraction, p_e)        = "
          f"{np.corrcoef(frac[good], p_e[good])[0, 1]:+.3f}")
    print(f"corr(distinct fraction, log V_e)    = "
          f"{np.corrcoef(frac[good], np.log(V_e[good]))[0, 1]:+.3f}")

order = np.argsort(-V_e)
print(f"\n{'ev':>4} {'V_e':>10} {'p_e':>7} {'distinct':>9} {'frac':>7} {'kish ESS':>9}")
for e in order[:15]:
    print(f"{e:4d} {V_e[e]:10.5f} {p_e[e]:+7.3f} {int(frac[e] * nsamp):9d} "
          f"{frac[e]:7.4f} {ess[e]:9.1f}")

print("\nlowest distinct fraction:")
for e in np.argsort(frac)[:10]:
    print(f"  ev {e:3d}: {int(frac[e] * nsamp):5d} distinct ({frac[e]:.4f}), "
          f"p_e={p_e[e]:+.3f}, V_e={V_e[e]:.5f}")
