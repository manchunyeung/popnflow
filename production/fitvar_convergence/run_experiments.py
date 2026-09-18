#!/usr/bin/env python
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


def _fit_gmm_plain(Xsub, k, reg, n_init, seed):
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(n_components=k, covariance_type='full', reg_covar=reg,
                           n_init=n_init, random_state=seed).fit(Xsub)

def _fit_gmm_warm(Xsub, k, reg, seed, g0):
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(n_components=k, covariance_type='full', reg_covar=reg,
                           n_init=1, random_state=seed,
                           weights_init=g0.weights_,
                           means_init=g0.means_,
                           precisions_init=g0.precisions_).fit(Xsub)

def trans_mc(X):
    m1, m2, dL = X[:, 0], X[:, 1], X[:, 2]
    Mc = (m1 * m2)**0.6 / (m1 + m2)**0.2
    return np.column_stack([Mc, m2 / m1, np.log(dL)])

def jac_mc(X):
    m1, m2, dL = X[:, 0], X[:, 1], X[:, 2]
    Mc = (m1 * m2)**0.6 / (m1 + m2)**0.2
    return np.log(Mc) - 2 * np.log(m1) - np.log(dL)

def trans_log(X):
    return np.log(X)

def jac_log(X):
    return -np.sum(np.log(X), axis=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', type=str, default='all')
    ap.add_argument('--R', type=int, default=50)
    ap.add_argument('--M', type=int, default=150000)
    ap.add_argument('--n-coords', type=int, default=50)
    ap.add_argument('--n-jobs', type=int, default=30)
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--dataset', type=str, default='simcat6')
    ap.add_argument('--out', type=str, default=os.path.join(HERE, 'experiments.npz'))
    cli = ap.parse_args()

    from _dataset import build_argv, check_nsamp, per_event_K
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
    Kcomp = per_event_K(cli.dataset)
    print(f"[exp] R={cli.R} nsamp={nsamp} events={len(events)}")

    qstar = MD._build_qstar_pool(cli.M, MD.args.fitvar_defensive_frac,
                                 MD.args.fitvar_broad_inflate, cli.seed)
    M_max = min(cli.M, qstar['nsamp_pop'])
    coords, coord_arr = MD._fitvar_coords()
    nC = len(coords)
    X_eval = jnp.asarray(MD._eval_pool_detframe(qstar, M_max))
    log_h = MD._log_h(coords, M_max, qstar)
    X_eval_np = np.asarray(X_eval)

    # Pre-transform eval pools
    X_eval_mc = jnp.asarray(trans_mc(X_eval_np))
    jac_eval_mc = jnp.asarray(jac_mc(X_eval_np), dtype=jnp.float32)
    X_eval_log = jnp.asarray(trans_log(X_eval_np))
    jac_eval_log = jnp.asarray(jac_log(X_eval_np), dtype=jnp.float32)

    def logp_of(g, eval_pts, jac=None):
        lp = np.asarray(MD.exact_log_gmm(eval_pts, *MD._gmm_to_jax(g)), dtype=np.float32)
        if jac is not None:
            lp += np.asarray(jac)
        return lp

    def lnL_from(block):
        lp = jnp.asarray(block, dtype=jnp.float64)
        out = np.zeros((block.shape[0], nC))
        for c0 in range(0, nC, 8):
            c1 = min(c0 + 8, nC)
            out[:, c0:c1] = np.asarray(
                MD._lnL_chunk(jnp.asarray(log_h[c0:c1]), lp)).T - np.log(M_max)
        return out

    arms = ['baseline', 'n_init_20', 'reg_covar_1e3', 'warm_start', 'reparam_mc', 'reparam_log']
    V = {a: np.zeros((nC, len(events))) for a in arms}

    t0 = time.time()
    for ie, e in enumerate(events):
        X = MD._pe_detframe(e)
        k = int(Kcomp[e])

        # Reference fit for warm-start
        g0 = _fit_gmm_plain(X, k, MD.REG_COVAR_FITVAR, MD.N_INIT_FITVAR, cli.seed)

        tasks_base, tasks_n20, tasks_reg, tasks_warm, tasks_mc, tasks_log = [], [], [], [], [], []
        
        for b in range(cli.R):
            rs = np.random.default_rng([cli.seed, e, b])
            idx = rs.integers(0, nsamp, size=nsamp)
            Db = X[idx]
            
            tasks_base.append(delayed(_fit_gmm_plain)(Db, k, MD.REG_COVAR_FITVAR, MD.N_INIT_FITVAR, cli.seed))
            tasks_n20.append(delayed(_fit_gmm_plain)(Db, k, MD.REG_COVAR_FITVAR, 20, cli.seed))
            tasks_reg.append(delayed(_fit_gmm_plain)(Db, k, 1e-3, MD.N_INIT_FITVAR, cli.seed))
            tasks_warm.append(delayed(_fit_gmm_warm)(Db, k, MD.REG_COVAR_FITVAR, cli.seed, g0))
            tasks_mc.append(delayed(_fit_gmm_plain)(trans_mc(Db), k, MD.REG_COVAR_FITVAR, MD.N_INIT_FITVAR, cli.seed))
            tasks_log.append(delayed(_fit_gmm_plain)(trans_log(Db), k, MD.REG_COVAR_FITVAR, MD.N_INIT_FITVAR, cli.seed))

        all_tasks = tasks_base + tasks_n20 + tasks_reg + tasks_warm + tasks_mc + tasks_log
        fits = Parallel(n_jobs=cli.n_jobs)(all_tasks)

        blk = {a: np.zeros((cli.R, M_max), dtype=np.float32) for a in arms}
        for b in range(cli.R):
            blk['baseline'][b]      = logp_of(fits[0*cli.R + b], X_eval)
            blk['n_init_20'][b]     = logp_of(fits[1*cli.R + b], X_eval)
            blk['reg_covar_1e3'][b] = logp_of(fits[2*cli.R + b], X_eval)
            blk['warm_start'][b]    = logp_of(fits[3*cli.R + b], X_eval)
            blk['reparam_mc'][b]    = logp_of(fits[4*cli.R + b], X_eval_mc, jac_eval_mc)
            blk['reparam_log'][b]   = logp_of(fits[5*cli.R + b], X_eval_log, jac_eval_log)

        for arm in arms:
            V[arm][:, ie] = np.var(lnL_from(blk[arm]), axis=0, ddof=1)

        if (ie + 1) % 5 == 0 or ie == len(events) - 1:
            el = (time.time() - t0) / 60
            print(f"[{ie + 1}/{len(events)}] {el:.1f} min  ETA {el / (ie + 1) * (len(events) - ie - 1):.1f} min", flush=True)

    print("\n=== B, summed over events, median over Lambda ===")
    B_base = np.median(V['baseline'].sum(axis=1))
    print(f"  baseline      : B = {B_base:.4f}")
    for arm in arms[1:]:
        B_arm = np.median(V[arm].sum(axis=1))
        print(f"  {arm:13s} : B = {B_arm:.4f}  ({B_arm / B_base:.3f}x, {(B_arm/B_base - 1)*100:+.1f}%)")

    np.savez(cli.out, events=np.array(events), R=cli.R, coord_arr=coord_arr,
             **{f'V_{a}': V[a] for a in arms})
    print(f"\n[exp] saved -> {cli.out}")

if __name__ == '__main__':
    main()
