#!/usr/bin/env python
import os, sys, time, argparse
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
                           weights_init=g0.weights_, means_init=g0.means_, precisions_init=g0.precisions_).fit(Xsub)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', type=str, default='0,1,2,39,47')
    ap.add_argument('--R', type=int, default=50)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--M', type=int, default=150000)
    cli = ap.parse_args()

    from _dataset import build_argv, check_nsamp, per_event_K
    sys.argv = build_argv('simcat6', os.path.join(HERE, 'scratch'), 50, 777)
    sys.path.insert(0, PROD)
    import make_diagnostic_plots as MD
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp as jlse
    from joblib import Parallel, delayed

    events = [int(x) for x in cli.events.split(',')]
    Kcomp = per_event_K('simcat6')
    
    qstar = MD._build_qstar_pool(cli.M, 0.1, 4.0, 777)
    M_max = min(cli.M, qstar['nsamp_pop'])
    coords, _ = MD._fitvar_coords()
    nC = len(coords)
    X_eval = jnp.asarray(MD._eval_pool_detframe(qstar, M_max))
    log_h = MD._log_h(coords, M_max, qstar)

    def logp_of(g): return np.asarray(MD.exact_log_gmm(X_eval, *MD._gmm_to_jax(g)), dtype=np.float32)

    def lnL_from(block):
        lp = jnp.asarray(block, dtype=jnp.float64)
        out = np.zeros((block.shape[0], nC))
        for c0 in range(0, nC, 8):
            c1 = min(c0 + 8, nC)
            out[:, c0:c1] = np.asarray(MD._lnL_chunk(jnp.asarray(log_h[c0:c1]), lp)).T - np.log(M_max)
        return out

    V_plain = np.zeros((nC, len(events)))
    V_warm = np.zeros((nC, len(events)))
    V_bag_warm = np.zeros((nC, len(events)))

    for ie, e in enumerate(events):
        X = MD._pe_detframe(e)
        k = int(Kcomp[e])
        g0 = _fit_gmm_plain(X, k, 1e-6, 5, 777)
        
        tasks_plain, tasks_warm, tasks_bag = [], [], []
        
        for b in range(cli.R):
            rs = np.random.default_rng([777, e, b])
            Db = X[rs.integers(0, 5000, size=5000)]
            tasks_plain.append(delayed(_fit_gmm_plain)(Db, k, 1e-6, 5, 777))
            tasks_warm.append(delayed(_fit_gmm_warm)(Db, k, 1e-6, 777, g0))
            
            rb = np.random.default_rng([777, e, b, 1])
            for _ in range(cli.K):
                D_inner = Db[rb.integers(0, 5000, size=5000)]
                tasks_bag.append(delayed(_fit_gmm_warm)(D_inner, k, 1e-6, 777, g0))

        fits_plain = Parallel(n_jobs=30)(tasks_plain)
        fits_warm = Parallel(n_jobs=30)(tasks_warm)
        fits_bag = Parallel(n_jobs=30)(tasks_bag)

        blk_plain = np.zeros((cli.R, M_max), dtype=np.float32)
        blk_warm = np.zeros((cli.R, M_max), dtype=np.float32)
        blk_bag_warm = np.zeros((cli.R, M_max), dtype=np.float32)

        for b in range(cli.R):
            blk_plain[b] = logp_of(fits_plain[b])
            blk_warm[b] = logp_of(fits_warm[b])
            members = np.stack([logp_of(fits_bag[b*cli.K + i]) for i in range(cli.K)])
            blk_bag_warm[b] = np.asarray(jlse(jnp.asarray(members), axis=0) - np.log(cli.K), dtype=np.float32)

        V_plain[:, ie] = np.var(lnL_from(blk_plain), axis=0, ddof=1)
        V_warm[:, ie] = np.var(lnL_from(blk_warm), axis=0, ddof=1)
        V_bag_warm[:, ie] = np.var(lnL_from(blk_bag_warm), axis=0, ddof=1)
        print(f"Event {e} done")

    print("\n=== B, summed over 5 events ===")
    Bp = np.median(V_plain.sum(axis=1))
    Bw = np.median(V_warm.sum(axis=1))
    Bbw = np.median(V_bag_warm.sum(axis=1))
    print(f"  plain             : {Bp:.4f}")
    print(f"  warm_start        : {Bw:.4f}  ({Bw/Bp:.3f}x)")
    print(f"  bagged+warm_start : {Bbw:.4f}  ({Bbw/Bp:.3f}x)")

if __name__ == '__main__':
    main()
