#!/usr/bin/env python
"""Post-process the seven published n_train caches: headline B(N), per-event
exponents, and the distinct-atom (bootstrap granularity) prediction."""
import numpy as np, glob, re, os

CDIR = '/home/manchun.yeung/population/simon/popnflow/production/fitvar_npe_scan/cache'
HERE = os.path.dirname(os.path.abspath(__file__))

runs = {}
for f in sorted(glob.glob(os.path.join(CDIR, 'fitvar_ntrain*.npz'))):
    m = re.search(r'ntrain(\d+)\.npz$', f)
    if not m:
        continue
    runs[int(m.group(1))] = np.load(f, allow_pickle=False)

Ns = np.array(sorted(runs))
print("n_train runs found:", list(Ns))


def intercept(V_of_M, Ms):
    """OLS of V on 1/M -> intercept (the M->inf fit variance)."""
    x = 1.0 / np.asarray(Ms, float)
    X = np.vstack([np.ones_like(x), x]).T
    return np.linalg.lstsq(X, V_of_M, rcond=None)[0][0]


B, Be = {}, {}
for n in Ns:
    d = runs[n]
    Ms = d['Ms']
    B[n] = np.median(intercept(d['V_of_M'], Ms))
    V_iC = d['V_iC_of_M']                       # (n_M, n_coords, Nobs)
    nM, nC, Nobs = V_iC.shape
    b_iC = intercept(V_iC.reshape(nM, -1), Ms).reshape(nC, Nobs)
    Be[n] = np.median(b_iC, axis=0)

print("\n  N_PE      B        N*B")
for n in Ns:
    print(f"{n:6d}  {B[n]:.4f}  {n * B[n]:8.1f}")

lg = np.polyfit(np.log(Ns), np.log([B[n] for n in Ns]), 1)
print(f"\naggregate power law: B ~ N^({lg[0]:+.3f})")

Bmat = np.vstack([Be[n] for n in Ns])           # (nN, Nobs)
p_e = np.polyfit(np.log(Ns), np.log(np.clip(Bmat, 1e-300, None)), 1)[0]
V_full = Be[Ns[-1]]
w = V_full / V_full.sum()
order = np.argsort(-V_full)

print(f"\nper-event exponent p_e: median {np.median(p_e):+.3f}, "
      f"contribution-weighted mean {np.sum(w * p_e):+.3f}")
print(f"frac of B carried by events with p_e > -0.5: "
      f"{V_full[p_e > -0.5].sum() / V_full.sum():.3f}  (n={np.sum(p_e > -0.5)})")
print(f"corr(log V_e, p_e) = {np.corrcoef(np.log(V_full), p_e)[0, 1]:+.3f}")
print("\ntop-15 contributors (event, V_e at N=3343, p_e):")
for e in order[:15]:
    print(f"  ev {e:3d}   V_e={V_full[e]:.5f}  frac={w[e]:.3f}  p_e={p_e[e]:+.3f}")

np.savez(os.path.join(HERE, 'published_summary.npz'),
         Ns=Ns, B=np.array([B[n] for n in Ns]), Bmat=Bmat, p_e=p_e, V_full=V_full)

# ---- bootstrap granularity prediction --------------------------------------
n0 = 3343.0
m_eff = n0 * (1.0 - np.exp(-Ns / n0))
print("\nbootstrap distinct-atom effective size (m_eff = n(1-e^{-m/n})):")
for n, me in zip(Ns, m_eff):
    print(f"  nominal {n:6d} -> distinct {me:8.1f}   ({me / n:.3f} of nominal)")
slope_eff = np.polyfit(np.log(Ns), np.log(m_eff), 1)[0]
print(f"\nIf V ~ 1/m_eff exactly, the exponent in NOMINAL m would be {-slope_eff:+.3f}")
print(f"Measured {lg[0]:+.3f}; pure iid -1.000")
print(f"=> distinct-atom deficit explains "
      f"{(-slope_eff - (-1.0)) / (lg[0] - (-1.0)) * 100:.0f}% of the departure from -1.")
