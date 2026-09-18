#!/usr/bin/env python
import os, sys
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
    from _dataset import build_argv, per_event_K
    sys.argv = build_argv('simcat6', os.path.join(HERE, 'scratch'), 50, 777)
    sys.path.insert(0, PROD)
    import make_diagnostic_plots as MD
    import jax.numpy as jnp
    from joblib import Parallel, delayed

    events = list(range(69))
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

    R = 50
    V_plain = np.zeros((nC, len(events)))
    V_warm = np.zeros((nC, len(events)))
    lnL_plain = np.zeros((len(events), R, nC))
    lnL_warm = np.zeros((len(events), R, nC))

    for ie, e in enumerate(events):
        X = MD._pe_detframe(e)
        k = int(Kcomp[e])
        g0 = _fit_gmm_plain(X, k, 1e-6, 10, 777)
        
        tasks_plain, tasks_warm = [], []
        
        for b in range(R):
            rs = np.random.default_rng([777, e, b])
            Db = X[rs.integers(0, 5000, size=5000)]
            tasks_plain.append(delayed(_fit_gmm_plain)(Db, k, 1e-6, 5, 777))
            tasks_warm.append(delayed(_fit_gmm_warm)(Db, k, 1e-6, 777, g0))

        fits_plain = Parallel(n_jobs=30)(tasks_plain)
        fits_warm = Parallel(n_jobs=30)(tasks_warm)

        blk_plain = np.zeros((R, M_max), dtype=np.float32)
        blk_warm = np.zeros((R, M_max), dtype=np.float32)

        for b in range(R):
            blk_plain[b] = logp_of(fits_plain[b])
            blk_warm[b] = logp_of(fits_warm[b])

        lp = lnL_from(blk_plain)
        lw = lnL_from(blk_warm)
        lnL_plain[ie] = lp
        lnL_warm[ie] = lw
        V_plain[:, ie] = np.var(lp, axis=0, ddof=1)
        V_warm[:, ie] = np.var(lw, axis=0, ddof=1)
        print(f"Event {e} done")

    # Shift is warm - plain, taking the mean over bootstraps first to get the expected value
    # Then summing over events
    lnL_ref_plain = lnL_plain.mean(axis=1) # (69, nC)
    lnL_ref_warm = lnL_warm.mean(axis=1)

    np.savez_compressed(
        'warmstart_simcat6_69.npz',
        V_plain=V_plain, V_warm=V_warm,
        lnL_ref_plain=lnL_ref_plain, lnL_ref_warm=lnL_ref_warm,
        Kcomp=Kcomp
    )
    print("Saved warmstart_simcat6_69.npz")

if __name__ == '__main__':
    main()
