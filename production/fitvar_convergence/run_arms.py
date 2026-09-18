#!/usr/bin/env python
"""
Is the term-(B) "N_PE-independent floor" real, or an artefact of bootstrapping
one fixed 3343-sample pool?

The published scan (production/fitvar_npe_scan) varies N_PE by drawing
n_train indices WITH REPLACEMENT from each event's fixed nsamp=3343 PE samples.
That measures Var under resampling from the empirical measure p_hat_n, not under
a genuinely longer/shorter PE run.  Two things go wrong as n_train -> nsamp:

  1. granularity: a with-replacement resample of size m from n atoms contains
     only n(1-e^{-m/n}) DISTINCT atoms -- 63% at m=n.  A real PE run of m
     samples has m distinct ones.  So the bootstrap systematically
     under-resolves the density, and the deficit GROWS with m/n, flattening
     the measured curve.
  2. it is the classic failure mode for statistics driven by extreme order
     statistics -- which is exactly what the non-scaling events are
     (broad, high-mass, tail-extrapolation-dominated).

This script runs the SAME estimator with a second arm that has neither defect:

  arm 'boot' : idx = rng.integers(0, nsamp, N)          (published; with repl.)
  arm 'srs'  : idx = rng.permutation(nsamp)[:N]         (without replacement)

'srs' draws N genuinely DISTINCT iid samples from the true posterior -- exactly
what a PE run of length N would deliver.  Its only bias is the finite-population
correction: for a linearisable statistic,

    Var_srs(N) = Var_iid(N) * (n - N) / (n - 1),

so the iid-equivalent is Var_srs * (n-1)/(n-N).  The correction is exact to
first order, which is why the arm is capped at N <= n/2 (factor <= 2).

To linear order the two arms must AGREE after the correction.  If they do not,
the linearisation fails -- and that is precisely the extreme-value regime where
the bootstrap is known to be inconsistent, i.e. the floor would be an artefact.

Everything else (Lambda draws, q* evaluation pool, GMM settings, seed, the
common-random-numbers evaluation) is held identical to the published run.
"""
import argparse
import os
import sys
import time

import numpy as np

PROD = '/home/manchun.yeung/population/simon/popnflow/production'
HERE = os.path.dirname(os.path.abspath(__file__))

# The dynesty checkpoints were pickled while make_diagnostic_plots.py was
# __main__, so restoring them looks these up in OUR __main__. They are pure
# stubs there too (the samplers are restored only for their stored samples).
def prior_transform(u): return u
def loglike_method_1(theta): return 0.0
def loglike_method_3(theta): return 0.0
def loglike_method_4(theta): return 0.0


def _fit_gmm(Xsub, k, reg, n_init, seed):
    """Worker: fit one GMM. Module-level and closure-free so loky can pickle it;
    the density evaluation stays in the parent, exactly as production does."""
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(n_components=k, covariance_type='full',
                           reg_covar=reg, n_init=n_init,
                           random_state=seed).fit(Xsub)


def _draw_idx(arm, nsamp, N, seed, e, r):
    rs = np.random.default_rng([seed, e, N, r, 0 if arm == 'boot' else 1])
    if arm == 'boot':
        return rs.integers(0, nsamp, size=N)        # with replacement (published)
    return rs.permutation(nsamp)[:N]                # without replacement (new arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', type=str, default='all',
                    help="'all', or comma-separated event indices.")
    ap.add_argument('--N-list', type=str, default='209,418,836,1671')
    ap.add_argument('--R', type=int, default=50, help='replicates per (event, N, arm)')
    ap.add_argument('--M', type=int, default=200000, help='q* evaluation pool size')
    ap.add_argument('--n-coords', type=int, default=60)
    ap.add_argument('--n-jobs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--out', type=str, default=os.path.join(HERE, 'arms.npz'))
    cli = ap.parse_args()

    # ---- import the production module with its own CLI ---------------------
    sys.argv = [
        'make_diagnostic_plots.py',
        '--run', 'mdc', '--catalog', '0',
        '--indir', os.path.join(PROD, 'results/data'),
        '--outdir', os.path.join(HERE, 'scratch'),
        '--pe-m1det', os.path.join(PROD, 'input_data/mdc_m1det_69rand.txt'),
        '--pe-m2det', os.path.join(PROD, 'input_data/mdc_m2det_69rand.txt'),
        '--pe-dL', os.path.join(PROD, 'input_data/mdc_dL_69rand.txt'),
        '--injection-file', os.path.join(PROD, 'input_data/endo3_bbhpop-LIGO-T2100113-v12.hdf5'),
        '--pool-cache', os.path.join(PROD, 'cache/pool_1M.npz'),
        '--fitvar-n-coords', str(cli.n_coords),
        '--seed', str(cli.seed),
        # The module refuses to import with no task; the moment check is the
        # cheapest one AND it validates the very density-evaluation path we reuse.
        '--fitvar-moment-check',
    ]
    os.makedirs(os.path.join(HERE, 'scratch'), exist_ok=True)
    sys.path.insert(0, PROD)
    import make_diagnostic_plots as MD
    import jax.numpy as jnp
    from sklearn.mixture import GaussianMixture
    from joblib import Parallel, delayed

    nsamp, Nobs = MD.nsamp, MD.Nobs
    print(f"[arms] nsamp={nsamp} Nobs={Nobs}")

    events = (list(range(Nobs)) if cli.events == 'all'
              else [int(x) for x in cli.events.split(',')])
    N_list = [int(x) for x in cli.N_list.split(',')]
    for N in N_list:
        if N > nsamp // 2:
            raise SystemExit(f"N={N} exceeds nsamp/2={nsamp // 2}: the FPC "
                             f"correction is only trustworthy to first order below that.")

    # ---- the evaluation machinery, identical to production ------------------
    qstar = MD._build_qstar_pool(cli.M, MD.args.fitvar_defensive_frac,
                                 MD.args.fitvar_broad_inflate, cli.seed)
    M_max = min(cli.M, qstar['nsamp_pop'])
    Ms = [M_max // 2, M_max]          # 2-point 1/M check; contamination is common-mode
    coords, coord_arr = MD._fitvar_coords()
    nC = len(coords)

    X_eval = jnp.asarray(MD._eval_pool_detframe(qstar, M_max))
    log_h_full = MD._log_h(coords, M_max, qstar)          # (nC, M_max)
    Xs = {e: MD._pe_detframe(e) for e in events}

    arms = ['boot', 'srs']
    # V[arm][N] -> (len(Ms), nC, n_events)
    V = {a: {N: np.zeros((len(Ms), nC, len(events))) for N in N_list} for a in arms}

    t0 = time.time()
    for ie, e in enumerate(events):
        for N in N_list:
            for arm in arms:
                subs = [Xs[e][_draw_idx(arm, nsamp, N, cli.seed, e, r)]
                        for r in range(cli.R)]
                gms = Parallel(n_jobs=cli.n_jobs)(
                    delayed(_fit_gmm)(s, MD.K_PER_EVENT_FITVAR, MD.REG_COVAR_FITVAR,
                                      MD.N_INIT_FITVAR, cli.seed) for s in subs)
                block = np.stack([
                    np.asarray(MD.exact_log_gmm(X_eval, *MD._gmm_to_jax(g)),
                               dtype=np.float32) for g in gms])   # (R, M_max)
                for mi, M in enumerate(Ms):
                    lp = jnp.asarray(block[:, :M], dtype=jnp.float64)
                    lnL = np.zeros((cli.R, nC))
                    for c0 in range(0, nC, 8):
                        c1 = min(c0 + 8, nC)
                        lnL[:, c0:c1] = np.asarray(
                            MD._lnL_chunk(jnp.asarray(log_h_full[c0:c1, :M]), lp)
                        ).T - np.log(M)
                    V[arm][N][mi, :, ie] = np.var(lnL, axis=0, ddof=1)
        el = time.time() - t0
        print(f"[arms] event {e} ({ie + 1}/{len(events)}) done, "
              f"{el / (ie + 1):.1f}s/event, ETA {el / (ie + 1) * (len(events) - ie - 1) / 60:.1f} min",
              flush=True)

    np.savez(cli.out,
             events=np.array(events), N_list=np.array(N_list), Ms=np.array(Ms),
             nsamp=nsamp, R=cli.R, coord_arr=coord_arr,
             **{f'V_{a}_{N}': V[a][N] for a in arms for N in N_list})
    print(f"[arms] saved -> {cli.out}")


if __name__ == '__main__':
    main()
