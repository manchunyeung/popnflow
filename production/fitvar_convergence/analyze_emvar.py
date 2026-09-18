#!/usr/bin/env python
"""Is the N_PE-independent floor EM optimisation noise?

'emseed' holds the DATA fixed and varies only the EM random_state, so whatever
variance it shows cannot depend on N_PE.  The EM training log-likelihood spread
(lower_bound_) says directly whether replicates land in different optima.
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--npz', default=os.path.join(HERE, 'emvar.npz'))
cli = ap.parse_args()

d = np.load(cli.npz)
events = d['events']
arms = [k[2:] for k in d.files if k.startswith('V_')]
print(f"events={len(events)}  R={int(d['R'])}  n_init={int(d['n_init'])}  arms={arms}")

pub = np.load(os.path.join(HERE, 'published_summary.npz'))
p_e, V_pub = pub['p_e'], pub['V_full']

print("\n=== aggregate, summed over events, median over Lambda ===")
tot = {}
for a in arms:
    tot[a] = np.median(d[f'V_{a}'].sum(axis=1))
    print(f"  {a:7s}: V = {tot[a]:.4f}")
print("  reference: published B(N_PE=3343) = 0.1945; "
      "two-component B_inf = 0.173")
if 'emseed' in tot:
    print(f"\n  => EM-only variance is {tot['emseed'] / 0.1945 * 100:.0f}% of the "
          f"published B(3343), and {tot['emseed'] / 0.173 * 100:.0f}% of B_inf.")

print("\n=== per-event ===")
hdr = f"{'ev':>4} {'p_e(pub)':>9} {'V_pub':>9} " + " ".join(f"{'V_' + a:>10}" for a in arms)
print(hdr)
Vmed = {a: np.median(d[f'V_{a}'], axis=0) for a in arms}
order = np.argsort(-Vmed[arms[0]])
for i in order:
    e = events[i]
    print(f"{e:4d} {p_e[e]:+9.3f} {V_pub[e]:9.5f} "
          + " ".join(f"{Vmed[a][i]:10.5f}" for a in arms))

# ---- did EM actually land in different optima? ------------------------------
print("\n=== EM training log-likelihood spread at FIXED data (emseed arm) ===")
print("If EM always found the same optimum, sd(lower_bound_) would be ~0.")
if 'LB_emseed' in d.files:
    LB = d['LB_emseed']                      # (nE, R)
    sd = LB.std(axis=1, ddof=1)
    rng_ = LB.max(axis=1) - LB.min(axis=1)
    nuniq = np.array([len(np.unique(np.round(LB[i], 6))) for i in range(LB.shape[0])])
    print(f"{'ev':>4} {'sd(LB)':>10} {'range(LB)':>11} {'n_distinct_optima':>18} "
          f"{'V_emseed':>10}")
    for i in order:
        print(f"{events[i]:4d} {sd[i]:10.2e} {rng_[i]:11.2e} {nuniq[i]:18d} "
              f"{Vmed['emseed'][i]:10.5f}")
    good = sd > 0
    if good.sum() > 2:
        print(f"\ncorr(sd(LB), V_emseed) = "
              f"{np.corrcoef(sd[good], Vmed['emseed'][good])[0, 1]:+.3f}")
