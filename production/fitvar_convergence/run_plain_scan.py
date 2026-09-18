#!/usr/bin/env python
"""
The published B-vs-N_PE scan, redone with the per-event K the inference uses.

fitvar_npe_scan measured B with a uniform K_PER_EVENT_FITVAR = 7. But
sim_cat_inference.py LOADS pre-fitted GMMs from event_gmms.h5 and takes each
event's K from the loaded shape (--kfixed is recorded but explicitly moot in
that code path). Those K's are the BIC-argmin selection: 3..10, median 5, and
only 5 of 69 events are K=7. So the published B -- and its -0.80 exponent and
B_inf floor -- describe an estimator the analysis never uses, over-parameterised
on 59 of 69 events.

This reruns the same measurement, changing ONLY the component count:

  for each n_train in the scan:
    for b = 1..R:  D*_b = resample(X, n_train)   <- identical RNG stream to
                   fit ONE GMM with K = K_e         make_diagnostic_plots.py
    B(n_train) = median_Lambda sum_e Var_b[ln L_e]

Everything else -- the q* evaluation pool, the Lambda draws, reg_covar, n_init,
the seed, the detector-frame coordinates -- is held at the production values, so
the only difference from the published curve is K.

--k-mode fixed reproduces the published uniform-7 behaviour on the same pool and
Lambda draws, which is the controlled comparison: run both and the ratio
isolates the effect of K alone.
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
    ap.add_argument('--n-train-list', type=str,
                    default='312,625,1250,1875,2500,3750,5000',
                    help='matches the published sim_cat6 scan')
    ap.add_argument('--R', type=int, default=50)
    ap.add_argument('--M', type=int, default=150000)
    ap.add_argument('--n-coords', type=int, default=50)
    ap.add_argument('--n-jobs', type=int, default=32)
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--dataset', type=str, default='simcat6')
    ap.add_argument('--k-mode', choices=['per-event', 'fixed'], default='per-event')
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'plain_scan.npz'))
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
    n_trains = [int(x) for x in cli.n_train_list.split(',')]
    Kcomp = (per_event_K(cli.dataset) if cli.k_mode == 'per-event'
             else np.full(Nobs, MD.K_PER_EVENT_FITVAR, dtype=int))
    print(f"[scan] k_mode={cli.k_mode}  R={cli.R}  n_train={n_trains}")

    qstar = MD._build_qstar_pool(cli.M, MD.args.fitvar_defensive_frac,
                                 MD.args.fitvar_broad_inflate, cli.seed)
    M_max = min(cli.M, qstar['nsamp_pop'])
    coords, coord_arr = MD._fitvar_coords()
    nC = len(coords)
    X_eval = jnp.asarray(MD._eval_pool_detframe(qstar, M_max))
    log_h = MD._log_h(coords, M_max, qstar)

    def lnL_from(block):
        lp = jnp.asarray(block, dtype=jnp.float64)
        out = np.zeros((block.shape[0], nC))
        for c0 in range(0, nC, 8):
            c1 = min(c0 + 8, nC)
            out[:, c0:c1] = np.asarray(
                MD._lnL_chunk(jnp.asarray(log_h[c0:c1]), lp)).T - np.log(M_max)
        return out

    V = np.zeros((len(n_trains), nC, len(events)))
    t0 = time.time()
    for it, n_train in enumerate(n_trains):
        for ie, e in enumerate(events):
            X = MD._pe_detframe(e)
            subs = []
            for b in range(cli.R):
                # identical stream to make_diagnostic_plots._fitvar_bootstrap_and_scan
                rs = np.random.default_rng([cli.seed, e, b])
                subs.append(X[rs.integers(0, nsamp, size=n_train)])
            gms = Parallel(n_jobs=cli.n_jobs)(
                delayed(_fit_gmm)(s, int(Kcomp[e]), MD.REG_COVAR_FITVAR,
                                  MD.N_INIT_FITVAR, cli.seed) for s in subs)
            block = np.stack([
                np.asarray(MD.exact_log_gmm(X_eval, *MD._gmm_to_jax(g)),
                           dtype=np.float32) for g in gms])
            V[it, :, ie] = np.var(lnL_from(block), axis=0, ddof=1)
        B = np.median(V[it].sum(axis=1))
        print(f"[scan] n_train={n_train:5d}  B={B:.4f}  N*B={n_train * B:8.1f}  "
              f"[{(time.time() - t0) / 60:.1f} min]", flush=True)

    Bs = np.array([np.median(V[it].sum(axis=1)) for it in range(len(n_trains))])
    N = np.array(n_trains, float)
    p = np.polyfit(np.log(N), np.log(Bs), 1)[0]
    X2 = np.vstack([np.ones_like(N), 1.0 / N]).T
    Binf, kappa = np.linalg.lstsq(X2, Bs, rcond=None)[0]

    print(f"\n=== B vs N_PE, k_mode={cli.k_mode} ===")
    print(f"{'N_PE':>7} {'B':>10} {'N*B':>10}")
    for n, b in zip(n_trains, Bs):
        print(f"{n:7d} {b:10.4f} {n * b:10.1f}")
    print(f"\npower law : B ~ N_PE^({p:+.3f})     (published, uniform K=7: -0.80)")
    print(f"two-comp  : B_inf = {Binf:.4f}, kappa = {kappa:.1f}")
    print(f"            B_inf is {Binf / Bs[-1] * 100:.0f}% of B(N_PE={n_trains[-1]})"
          f"   (published: 47%)")

    np.savez(cli.out, n_trains=np.array(n_trains), V=V, Bs=Bs, Kcomp=Kcomp,
             events=np.array(events), k_mode=cli.k_mode, coord_arr=coord_arr)
    print(f"\n[scan] saved -> {cli.out}")


if __name__ == '__main__':
    main()
