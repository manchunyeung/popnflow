#!/usr/bin/env python
"""
Does term (B) converge as N_PE -> infinity?

No amount of resampling the real 3343 PE samples can answer this: the empirical
measure has only 3343 atoms, so nothing above N_PE = 3343 is reachable, and
below it the with-replacement bootstrap under-resolves the density (see
run_arms.py).  Here we build a controlled surrogate where fresh, genuinely
independent samples can be drawn at ANY N:

  * reference "truth" p_true = a rich K=20 GMM fitted to the event's full 3343
    PE samples (optionally covariance-inflated to probe heavier tails);
  * draw N fresh iid samples from p_true;
  * fit the PRODUCTION K=7 GMM to them (so the misspecification that production
    actually has -- K=7 approximating a richer density -- is preserved);
  * evaluate ln L(Lambda) on the same fixed q* pool and take the variance over
    R replicates.

This is the estimator's genuine sampling-variance curve.  M-estimator
asymptotics say Var ~ 1/N even under misspecification, so the question is
purely whether the events that carry B are anywhere near that regime, and
where the crossover is.

Caveat, stated up front: p_true is a GMM, so its tails are Gaussian mixtures.
If the real posterior has heavier tails than any K=20 GMM, this arm is
optimistic. --inflate probes sensitivity to exactly that.
"""
import argparse
import os
import sys
import time

import numpy as np

PROD = '/home/manchun.yeung/population/simon/popnflow/production'
HERE = os.path.dirname(os.path.abspath(__file__))


def prior_transform(u): return u
def loglike_method_1(theta): return 0.0
def loglike_method_3(theta): return 0.0
def loglike_method_4(theta): return 0.0


def _fit_gmm(Xsub, k, reg, n_init, seed):
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(n_components=k, covariance_type='full',
                           reg_covar=reg, n_init=n_init,
                           random_state=seed).fit(Xsub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', type=str, default='55,17,60,1')
    ap.add_argument('--N-list', type=str,
                    default='209,418,836,1671,3343,6686,13372,26744,53488,106976')
    ap.add_argument('--R', type=int, default=40)
    ap.add_argument('--M', type=int, default=100000)
    ap.add_argument('--n-coords', type=int, default=40)
    ap.add_argument('--n-jobs', type=int, default=30)
    ap.add_argument('--K-true', type=int, default=20)
    ap.add_argument('--inflate', type=float, default=1.0,
                    help='Covariance inflation of the reference truth (tail probe).')
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--out', type=str, default=os.path.join(HERE, 'fresh.npz'))
    cli = ap.parse_args()

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
        '--fitvar-moment-check',
    ]
    os.makedirs(os.path.join(HERE, 'scratch'), exist_ok=True)
    sys.path.insert(0, PROD)
    import make_diagnostic_plots as MD
    import jax.numpy as jnp
    from sklearn.mixture import GaussianMixture
    from joblib import Parallel, delayed

    events = [int(x) for x in cli.events.split(',')]
    N_list = [int(x) for x in cli.N_list.split(',')]

    qstar = MD._build_qstar_pool(cli.M, MD.args.fitvar_defensive_frac,
                                 MD.args.fitvar_broad_inflate, cli.seed)
    M_max = min(cli.M, qstar['nsamp_pop'])
    Ms = [M_max // 2, M_max]
    coords, coord_arr = MD._fitvar_coords()
    nC = len(coords)
    X_eval = jnp.asarray(MD._eval_pool_detframe(qstar, M_max))
    log_h_full = MD._log_h(coords, M_max, qstar)

    V = np.zeros((len(events), len(N_list), len(Ms), nC))
    t0 = time.time()
    for ie, e in enumerate(events):
        X = MD._pe_detframe(e)
        truth = GaussianMixture(n_components=cli.K_true, covariance_type='full',
                                reg_covar=MD.REG_COVAR_FITVAR, n_init=5,
                                random_state=cli.seed).fit(X)
        if cli.inflate != 1.0:
            truth.covariances_ = truth.covariances_ * cli.inflate
            from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
            truth.precisions_cholesky_ = _compute_precision_cholesky(
                truth.covariances_, 'full')
            truth.precisions_ = np.stack([pc @ pc.T for pc in truth.precisions_cholesky_])
        print(f"[fresh] event {e}: reference truth K={cli.K_true} fitted "
              f"(inflate={cli.inflate})", flush=True)

        for iN, N in enumerate(N_list):
            subs = []
            for r in range(cli.R):
                truth.random_state = int(np.random.default_rng(
                    [cli.seed, e, N, r]).integers(0, 2**31 - 1))
                subs.append(truth.sample(N)[0])          # FRESH iid draws
            gms = Parallel(n_jobs=cli.n_jobs)(
                delayed(_fit_gmm)(s, MD.K_PER_EVENT_FITVAR, MD.REG_COVAR_FITVAR,
                                  MD.N_INIT_FITVAR, cli.seed) for s in subs)
            block = np.stack([
                np.asarray(MD.exact_log_gmm(X_eval, *MD._gmm_to_jax(g)),
                           dtype=np.float32) for g in gms])
            for mi, M in enumerate(Ms):
                lp = jnp.asarray(block[:, :M], dtype=jnp.float64)
                lnL = np.zeros((cli.R, nC))
                for c0 in range(0, nC, 8):
                    c1 = min(c0 + 8, nC)
                    lnL[:, c0:c1] = np.asarray(
                        MD._lnL_chunk(jnp.asarray(log_h_full[c0:c1, :M]), lp)).T - np.log(M)
                V[ie, iN, mi] = np.var(lnL, axis=0, ddof=1)
            print(f"[fresh] ev {e} N={N:7d}  V(med over Lambda)={np.median(V[ie, iN, -1]):.5e}"
                  f"   [{(time.time() - t0) / 60:.1f} min]", flush=True)

    np.savez(cli.out, events=np.array(events), N_list=np.array(N_list),
             Ms=np.array(Ms), V=V, R=cli.R, K_true=cli.K_true, inflate=cli.inflate)
    print(f"[fresh] saved -> {cli.out}")


if __name__ == '__main__':
    main()
