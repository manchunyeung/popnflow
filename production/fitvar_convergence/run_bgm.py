#!/usr/bin/env python
import os, sys, argparse
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

def _fit_bgm(Xsub, k, reg, seed):
    from sklearn.mixture import BayesianGaussianMixture
    # BGM uses Dirichlet process prior. We can give it a high Kmax and it will select.
    # To match standard GMM closely, we can set weight_concentration_prior.
    return BayesianGaussianMixture(n_components=10, covariance_type='full', reg_covar=reg,
                                   n_init=5, random_state=seed, 
                                   weight_concentration_prior=0.01).fit(Xsub)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', type=str, default='0,1,2,39,47')
    ap.add_argument('--R', type=int, default=50)
    cli = ap.parse_args()

    from _dataset import build_argv, per_event_K
    sys.argv = build_argv('simcat6', os.path.join(HERE, 'scratch'), 50, 777)
    sys.path.insert(0, PROD)
    import make_diagnostic_plots as MD
    import jax.numpy as jnp
    from joblib import Parallel, delayed

    events = [int(x) for x in cli.events.split(',')]
    Kcomp = per_event_K('simcat6')
    
    qstar = MD._build_qstar_pool(150000, 0.1, 4.0, 777)
    M_max = min(150000, qstar['nsamp_pop'])
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
    V_bgm = np.zeros((nC, len(events)))

    for ie, e in enumerate(events):
        X = MD._pe_detframe(e)
        k = int(Kcomp[e])
        
        tasks_plain, tasks_bgm = [], []
        
        for b in range(cli.R):
            rs = np.random.default_rng([777, e, b])
            Db = X[rs.integers(0, 5000, size=5000)]
            tasks_plain.append(delayed(_fit_gmm_plain)(Db, k, 1e-6, 5, 777))
            tasks_bgm.append(delayed(_fit_bgm)(Db, k, 1e-6, 777))

        fits_plain = Parallel(n_jobs=30)(tasks_plain)
        fits_bgm = Parallel(n_jobs=30)(tasks_bgm)

        blk_plain = np.zeros((cli.R, M_max), dtype=np.float32)
        blk_bgm = np.zeros((cli.R, M_max), dtype=np.float32)

        for b in range(cli.R):
            blk_plain[b] = logp_of(fits_plain[b])
            blk_bgm[b] = logp_of(fits_bgm[b])

        V_plain[:, ie] = np.var(lnL_from(blk_plain), axis=0, ddof=1)
        V_bgm[:, ie] = np.var(lnL_from(blk_bgm), axis=0, ddof=1)
        print(f"Event {e} done")

    print("\n=== B, summed over 5 events ===")
    Bp = np.median(V_plain.sum(axis=1))
    Bbgm = np.median(V_bgm.sum(axis=1))
    print(f"  plain : {Bp:.4f}")
    print(f"  bgm   : {Bbgm:.4f}  ({Bbgm/Bp:.3f}x)")

if __name__ == '__main__':
    main()
