#!/usr/bin/env python
"""Fresh-sample convergence curve: does V_e(N) keep falling above N_PE=3343?"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--npz', default=os.path.join(HERE, 'fresh_full.npz'))
cli = ap.parse_args()

d = np.load(cli.npz)
events, N_list, V = d['events'], d['N_list'], d['V']      # V: (nE, nN, nM, nC)
print(f"events={list(events)}  R={int(d['R'])}  K_true={int(d['K_true'])}  "
      f"inflate={float(d['inflate'])}")

pub = np.load(os.path.join(HERE, 'published_summary.npz'))
p_pub = pub['p_e']

lnN = np.log(N_list.astype(float))
print(f"\n{'N_PE':>8} | " + " ".join(f"{'ev%d' % e:>11}" for e in events))
med = np.median(V[:, :, -1, :], axis=2)                   # (nE, nN) at M_max
for iN, N in enumerate(N_list):
    print(f"{N:8d} | " + " ".join(f"{med[ie, iN]:11.4e}" for ie in range(len(events))))

print("\n=== exponents from the FRESH arm ===")
print(f"{'ev':>4} {'p_fresh(all)':>13} {'p_fresh(N<=3343)':>17} {'p_fresh(N>=3343)':>17} "
      f"{'p_bootstrap(pub)':>17}")
lo = N_list <= 3343
hi = N_list >= 3343
for ie, e in enumerate(events):
    y = med[ie]
    p_all = np.polyfit(lnN, np.log(y), 1)[0]
    p_lo = np.polyfit(lnN[lo], np.log(y[lo]), 1)[0]
    p_hi = np.polyfit(lnN[hi], np.log(y[hi]), 1)[0]
    print(f"{e:4d} {p_all:13.3f} {p_lo:17.3f} {p_hi:17.3f} {p_pub[e]:17.3f}")

print("\n=== N*V, flat iff exactly 1/N ===")
print(f"{'N_PE':>8} | " + " ".join(f"{'ev%d' % e:>11}" for e in events))
for iN, N in enumerate(N_list):
    print(f"{N:8d} | " + " ".join(f"{N * med[ie, iN]:11.4e}" for ie in range(len(events))))
