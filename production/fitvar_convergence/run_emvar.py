#!/usr/bin/env python
"""
Is the N_PE-independent part of term (B) just EM optimisation noise?

run_arms.py ruled out the bootstrap as the cause of the floor: with-replacement
and without-replacement resampling agree to 0.1% at N/n = 0.06 and neither
scales as 1/N.  So the floor is in the estimator, not the resampling.

The natural N-independent mechanism is EM itself.  Every bootstrap replicate
re-runs EM from a data-dependent k-means init, so if the K=7 likelihood surface
for an event has several comparable optima, replicates land in DIFFERENT ones.
The resulting spread in ln L is optimisation noise, and it does NOT shrink as
N_PE grows -- it is a property of the surface, not the sample size.

Decisive measurement: hold the DATA completely fixed (all nsamp samples, every
replicate) and vary only the EM random_state.  Any variance that survives is
pure optimisation noise.

  arm 'emseed' : same data every replicate, EM seed varies   -> V_EM
  arm 'both'   : data resampled AND EM seed varies           -> upper bound

Compare V_EM against the published two-component fit B = B_inf + kappa/N_PE,
which had B_inf = 0.173 at n_init=5.  If V_EM lands there, the floor is EM.

--n-init sweeps the obvious lever: n_init picks the best of several EM runs by
training likelihood, so raising it should suppress exactly this variance.
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
    g = GaussianMixture(n_components=k, covariance_type='full', reg_covar=reg,
                        n_init=n_init, random_state=seed).fit(Xsub)
    return g, float(g.lower_bound_), int(g.n_iter_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', type=str, default='all')
    ap.add_argument('--arms', type=str, default='emseed,both')
    ap.add_argument('--R', type=int, default=60)
    ap.add_argument('--M', type=int, default=150000)
    ap.add_argument('--n-coords', type=int, default=50)
    ap.add_argument('--n-init', type=int, default=-1,
                    help='EM restarts; -1 = production value (N_INIT_FITVAR).')
    ap.add_argument('--n-jobs', type=int, default=30)
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--dataset', type=str, default='simcat6',
                    help="'simcat6' (the paper's catalog) or 'mdc' (retired).")
    ap.add_argument('--out', type=str, default=os.path.join(HERE, 'emvar.npz'))
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
    n_init = MD.N_INIT_FITVAR if cli.n_init < 0 else cli.n_init
    events = (list(range(Nobs)) if cli.events == 'all'
              else [int(x) for x in cli.events.split(',')])
    arms = cli.arms.split(',')
    print(f"[emvar] nsamp={nsamp} n_init={n_init} arms={arms} R={cli.R}")

    qstar = MD._build_qstar_pool(cli.M, MD.args.fitvar_defensive_frac,
                                 MD.args.fitvar_broad_inflate, cli.seed)
    M_max = min(cli.M, qstar['nsamp_pop'])
    coords, coord_arr = MD._fitvar_coords()
    nC = len(coords)
    X_eval = jnp.asarray(MD._eval_pool_detframe(qstar, M_max))
    log_h_full = MD._log_h(coords, M_max, qstar)
    Xs = {e: MD._pe_detframe(e) for e in events}

    V = {a: np.zeros((nC, len(events))) for a in arms}
    LB = {a: np.zeros((len(events), cli.R)) for a in arms}   # EM training loglik
    t0 = time.time()
    for ie, e in enumerate(events):
        for arm in arms:
            subs, seeds = [], []
            for r in range(cli.R):
                rs = np.random.default_rng([cli.seed, e, r, hash(arm) % 1000])
                if arm == 'emseed':
                    subs.append(Xs[e])                       # DATA FIXED
                else:
                    subs.append(Xs[e][rs.integers(0, nsamp, size=nsamp)])
                seeds.append(int(rs.integers(0, 2**31 - 1)))  # EM SEED VARIES
            res = Parallel(n_jobs=cli.n_jobs)(
                delayed(_fit_gmm)(s, MD.K_PER_EVENT_FITVAR, MD.REG_COVAR_FITVAR,
                                  n_init, sd) for s, sd in zip(subs, seeds))
            LB[arm][ie] = [r[1] for r in res]
            block = np.stack([
                np.asarray(MD.exact_log_gmm(X_eval, *MD._gmm_to_jax(g)),
                           dtype=np.float32) for g, _, _ in res])
            lp = jnp.asarray(block, dtype=jnp.float64)
            lnL = np.zeros((cli.R, nC))
            for c0 in range(0, nC, 8):
                c1 = min(c0 + 8, nC)
                lnL[:, c0:c1] = np.asarray(
                    MD._lnL_chunk(jnp.asarray(log_h_full[c0:c1]), lp)).T - np.log(M_max)
            V[arm][:, ie] = np.var(lnL, axis=0, ddof=1)
        if (ie + 1) % 10 == 0 or ie == len(events) - 1:
            print(f"[emvar] {ie + 1}/{len(events)} events, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)

    print("\n=== summed over events, median over Lambda ===")
    for arm in arms:
        print(f"  {arm:7s}: V = {np.median(V[arm].sum(axis=1)):.4f}")
    print("\n(published: B(N=3343) = 0.1945, two-component B_inf = 0.173)")

    np.savez(cli.out, events=np.array(events), n_init=n_init, R=cli.R,
             **{f'V_{a}': V[a] for a in arms}, **{f'LB_{a}': LB[a] for a in arms})
    print(f"[emvar] saved -> {cli.out}")


if __name__ == '__main__':
    main()
