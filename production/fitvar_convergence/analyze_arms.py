#!/usr/bin/env python
"""Compare the with-replacement (published) and without-replacement arms."""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument('--npz', default=os.path.join(HERE, 'arms_full.npz'))
cli = ap.parse_args()

d = np.load(cli.npz)
events, N_list, Ms, nsamp, R = d['events'], d['N_list'], d['Ms'], int(d['nsamp']), int(d['R'])
print(f"events={len(events)}  N_list={list(N_list)}  Ms={list(Ms)}  nsamp={nsamp}  R={R}")


def richardson(V):
    """V has shape (len(Ms), nC, nE) with Ms = [M/2, M]; return the 1/M intercept."""
    return 2.0 * V[1] - V[0]


rows = {}
for arm in ('boot', 'srs'):
    for N in N_list:
        V = d[f'V_{arm}_{N}']                       # (nM, nC, nE)
        raw = V[-1].sum(axis=1)                     # at M_max, summed over events -> (nC,)
        rich = richardson(V).sum(axis=1)
        fpc = (nsamp - 1.0) / (nsamp - N) if arm == 'srs' else 1.0
        rows[(arm, N)] = dict(
            raw=np.median(raw) * fpc,
            rich=np.median(rich) * fpc,
            per_event_raw=np.median(V[-1], axis=0) * fpc,      # (nE,)
            fpc=fpc,
        )

print("\n=== aggregate B(N), summed over events, median over Lambda ===")
print(f"{'N':>7} | {'boot raw':>10} {'boot rich':>10} | {'srs raw':>10} {'srs rich':>10} | "
      f"{'FPC':>5} | {'srs/boot':>8}")
for N in N_list:
    b, s = rows[('boot', N)], rows[('srs', N)]
    print(f"{N:7d} | {b['raw']:10.4f} {b['rich']:10.4f} | {s['raw']:10.4f} {s['rich']:10.4f} | "
          f"{s['fpc']:5.3f} | {s['raw'] / b['raw']:8.3f}")

print("\n=== power-law exponents over the measured range ===")
lnN = np.log(N_list.astype(float))
for key in ('raw', 'rich'):
    for arm in ('boot', 'srs'):
        y = np.array([rows[(arm, N)][key] for N in N_list])
        if np.all(y > 0):
            p = np.polyfit(lnN, np.log(y), 1)[0]
            print(f"  {arm:5s} ({key:4s}): B ~ N^({p:+.3f})")
        else:
            print(f"  {arm:5s} ({key:4s}): non-positive values, skipped")

# The FPC correction reaches 2.0 at N = nsamp/2, so repeat the fit on the
# low-N end where it is <=1.34 and therefore cannot be driving the answer.
sel = N_list <= nsamp // 4
print(f"\n=== same, restricted to N <= {nsamp // 4} (FPC <= "
      f"{(nsamp - 1.0) / (nsamp - N_list[sel].max()):.3f}; correction cannot drive it) ===")
for arm in ('boot', 'srs'):
    y = np.array([rows[(arm, N)]['raw'] for N in N_list[sel]])
    y_unc = np.array([rows[(arm, N)]['raw'] / rows[(arm, N)]['fpc'] for N in N_list[sel]])
    p = np.polyfit(lnN[sel], np.log(y), 1)[0]
    p_unc = np.polyfit(lnN[sel], np.log(y_unc), 1)[0]
    print(f"  {arm:5s}: B ~ N^({p:+.3f})   [uncorrected: N^({p_unc:+.3f})]")

# ---- per-event exponents ----------------------------------------------------
print("\n=== per-event exponents (raw, at M_max) ===")
pe = {}
for arm in ('boot', 'srs'):
    Y = np.vstack([rows[(arm, N)]['per_event_raw'] for N in N_list])   # (nN, nE)
    good = np.all(Y > 0, axis=0)
    p = np.full(Y.shape[1], np.nan)
    p[good] = np.polyfit(lnN, np.log(Y[:, good]), 1)[0]
    pe[arm] = p

V_ref = rows[('boot', N_list[-1])]['per_event_raw']
order = np.argsort(-V_ref)
print(f"{'ev':>4} {'V_e(boot,Nmax)':>15} {'p_boot':>8} {'p_srs':>8}   shift")
for e in order[:20]:
    print(f"{events[e]:4d} {V_ref[e]:15.6f} {pe['boot'][e]:+8.3f} {pe['srs'][e]:+8.3f}   "
          f"{pe['srs'][e] - pe['boot'][e]:+.3f}")

w = np.clip(V_ref, 0, None)
w = w / w.sum()
for arm in ('boot', 'srs'):
    g = np.isfinite(pe[arm])
    print(f"\n{arm}: median p_e = {np.nanmedian(pe[arm]):+.3f}, "
          f"contribution-weighted mean = {np.sum(w[g] * pe[arm][g]) / np.sum(w[g]):+.3f}")

np.savez(os.path.join(HERE, 'arms_analysis.npz'),
         N_list=N_list, events=events,
         p_boot=pe['boot'], p_srs=pe['srs'],
         B_boot=np.array([rows[('boot', N)]['raw'] for N in N_list]),
         B_srs=np.array([rows[('srs', N)]['raw'] for N in N_list]))
print(f"\nsaved -> {os.path.join(HERE, 'arms_analysis.npz')}")
