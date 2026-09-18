#!/usr/bin/env python
"""
Layer 1 uses n_train = N_PE, but WITH replacement. Is that the right thing?

The production density estimate p_hat_e is fitted on all N_PE = 3343 DISTINCT
samples. Each bootstrap replicate is also of size 3343, but drawn with
replacement, so it contains only n(1-e^{-1}) ~ 2113 distinct points -- a 1.58x
shortfall in distinct sample size relative to the fit whose variance we are
trying to quote.

Standard bootstrap theory says this is exactly right: Var_boot(n) is consistent
for Var(n) for smooth functionals, and the resample is *supposed* to match the
original size. But the events that dominate B were shown (see README) to be
non-asymptotic and EM-multimodal, which is the regime where that consistency is
not guaranteed. So measure the bias directly rather than assume it.

Test: compare ln L_e(Lambda) of the PRODUCTION fit (all distinct samples)
against the distribution of ln L_e(Lambda) over the bootstrap replicates.

  - If the bootstrap is faithful, the production value sits in the middle of the
    replicate distribution (z ~ 0) and the replicate mean matches it.
  - A systematic offset means the replicates are not representative of the fit
    whose uncertainty B is supposed to describe.

Also reports 'srs_n' : replicates of size n drawn WITHOUT replacement, which is
degenerate (it returns the full sample every time) -- included only to make
explicit that there is no without-replacement alternative at m = n. That is the
structural reason the bootstrap must be used here.
"""
import argparse
import os
import sys
import time

import numpy as np

PROD = '/home/manchun.yeung/population/simon/popnflow/production'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def prior_transform(u): return u
def loglike_method_1(theta): return 0.0
def loglike_method_3(theta): return 0.0
def loglike_method_4(theta): return 0.0


def _fit_gmm(Xsub, k, reg, n_init, seed):
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(n_components=k, covariance_type='full', reg_covar=reg,
                           n_init=n_init, random_state=seed).fit(Xsub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', type=str, default='all')
    ap.add_argument('--R', type=int, default=60)
    ap.add_argument('--M', type=int, default=150000)
    ap.add_argument('--n-coords', type=int, default=50)
    ap.add_argument('--n-jobs', type=int, default=30)
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--dataset', type=str, default='simcat6',
                    help="'simcat6' (the paper's catalog) or 'mdc' (retired).")
    ap.add_argument('--out', type=str, default=os.path.join(HERE, 'bootbias.npz'))
    cli = ap.parse_args()

    from _dataset import build_argv, check_nsamp
    scratch = os.path.join(HERE, 'scratch')
    os.makedirs(scratch, exist_ok=True)
    sys.argv = build_argv(cli.dataset, scratch, cli.n_coords, cli.seed)
    sys.path.insert(0, PROD)
    import make_diagnostic_plots as MD
    import jax.numpy as jnp
    from joblib import Parallel, delayed

    nsamp, Nobs = MD.nsamp, MD.Nobs
    check_nsamp(cli.dataset, nsamp)
    events = (list(range(Nobs)) if cli.events == 'all'
              else [int(x) for x in cli.events.split(',')])

    qstar = MD._build_qstar_pool(cli.M, MD.args.fitvar_defensive_frac,
                                 MD.args.fitvar_broad_inflate, cli.seed)
    M_max = min(cli.M, qstar['nsamp_pop'])
    coords, coord_arr = MD._fitvar_coords()
    nC = len(coords)
    X_eval = jnp.asarray(MD._eval_pool_detframe(qstar, M_max))
    log_h = MD._log_h(coords, M_max, qstar)

    def lnL_of(gmms):
        block = np.stack([np.asarray(MD.exact_log_gmm(X_eval, *MD._gmm_to_jax(g)),
                                     dtype=np.float32) for g in gmms])
        lp = jnp.asarray(block, dtype=jnp.float64)
        out = np.zeros((len(gmms), nC))
        for c0 in range(0, nC, 8):
            c1 = min(c0 + 8, nC)
            out[:, c0:c1] = np.asarray(
                MD._lnL_chunk(jnp.asarray(log_h[c0:c1]), lp)).T - np.log(M_max)
        return out

    lnL_prod = np.zeros((len(events), nC))
    lnL_boot = np.zeros((len(events), cli.R, nC))
    n_distinct = np.zeros((len(events), cli.R))

    t0 = time.time()
    for ie, e in enumerate(events):
        X = MD._pe_detframe(e)
        # production fit: ALL distinct samples, exactly as _fit_qE_components does
        g_prod = _fit_gmm(X, MD.K_PER_EVENT_FITVAR, MD.REG_COVAR_FITVAR,
                          MD.N_INIT_FITVAR, cli.seed)
        lnL_prod[ie] = lnL_of([g_prod])[0]

        subs = []
        for b in range(cli.R):
            rs = np.random.default_rng([cli.seed, e, b])
            idx = rs.integers(0, nsamp, size=nsamp)      # production layer-1 draw
            n_distinct[ie, b] = len(np.unique(idx))
            subs.append(X[idx])
        gms = Parallel(n_jobs=cli.n_jobs)(
            delayed(_fit_gmm)(s, MD.K_PER_EVENT_FITVAR, MD.REG_COVAR_FITVAR,
                              MD.N_INIT_FITVAR, cli.seed) for s in subs)
        lnL_boot[ie] = lnL_of(gms)
        if (ie + 1) % 10 == 0 or ie == len(events) - 1:
            print(f"[bootbias] {ie + 1}/{len(events)}  {(time.time() - t0) / 60:.1f} min",
                  flush=True)

    mu = lnL_boot.mean(axis=1)                    # (nE, nC)
    sd = lnL_boot.std(axis=1, ddof=1)
    z = (lnL_prod - mu) / np.where(sd > 0, sd, np.nan)
    off = lnL_prod - mu

    print(f"\ndistinct points per bootstrap replicate: "
          f"{n_distinct.mean():.0f} / {nsamp} = {n_distinct.mean() / nsamp:.3f}")
    print(f"(analytic n(1-e^-1) = {nsamp * (1 - np.exp(-1)):.0f})")
    print("\n=== production ln L vs the bootstrap replicate distribution ===")
    print(f"  median z over (event, Lambda) : {np.nanmedian(z):+.3f}")
    print(f"  mean   z                      : {np.nanmean(z):+.3f}")
    print(f"  frac of (event,Lambda) with z > 0 : {np.nanmean(z > 0):.3f}")
    print(f"  median offset  lnL_prod - mean(lnL_boot) : {np.median(off):+.4f}")
    print(f"\n  SUM over events of offset (median over Lambda): "
          f"{np.median(off.sum(axis=0)):+.4f}")
    print(f"  SUM over events of Var_b (median over Lambda) = B : "
          f"{np.median((sd ** 2).sum(axis=0)):.4f}")

    print("\n=== per-event, largest |z| ===")
    zz = np.nanmedian(z, axis=1)
    for i in np.argsort(-np.abs(zz))[:12]:
        print(f"  ev {events[i]:3d}  z={zz[i]:+7.3f}  offset={np.median(off[i]):+.4f}  "
              f"sd={np.median(sd[i]):.4f}")

    np.savez(cli.out, events=np.array(events), lnL_prod=lnL_prod,
             lnL_boot=lnL_boot, n_distinct=n_distinct)
    print(f"\n[bootbias] saved -> {cli.out}")


if __name__ == '__main__':
    main()
