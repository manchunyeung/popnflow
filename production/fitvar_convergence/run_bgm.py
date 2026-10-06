#!/usr/bin/env python
"""
Bayesian GMM as a deployable replacement for the per-event density estimate.

fitvar_summary.md reports BGM cutting B by ~52% -- but on a 5-EVENT SUBSET.
This measures it on all 69 events at the per-event K the inference actually
uses, with the same estimator and the same paired construction as run_bagging.py.

Unlike warm-starting, BGM needs no reference fit, so it IS deployable: the
Dirichlet-process prior shrinks redundant components toward zero weight, which
both regularises the fit and makes the retained structure less sensitive to
which basin the optimiser lands in.

Arms, all sharing the SAME outer draw stream as production (so 'plain'
reproduces the published estimator and the arms are paired):

  plain    GaussianMixture(K_e)                      <- current estimator
  bgm_k    BayesianGaussianMixture(K_e)              <- drop-in, same capacity
  bgm_cap  BayesianGaussianMixture(--cap)            <- generous cap, let the DP prune

bgm_cap is the more faithful test of the "DP zeroes out redundant components"
claim; bgm_k is the conservative like-for-like swap.

B is Var over PE-sample draws, so as in run_bagging.py each outer draw is a
resample of size N_PE standing in for a new PE run, and B = Var_b[ln L].

Also reports the bias shift on the REAL samples, split into the Lambda-
independent part (cancels in the hyper-posterior) and the Lambda-dependent part
(the only part that can move the inference) -- a variance reduction bought with
a large shift is not a win. Measured against the CURRENT estimator, not truth.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np

PROD = '/home/manchun.yeung/population/simon/popnflow/production'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def prior_transform(u): return u
def loglike_method_1(theta): return 0.0
def loglike_method_3(theta): return 0.0
def loglike_method_4(theta): return 0.0


def _fit(Xsub, kind, k, reg, n_init, seed, max_iter):
    """Worker: closure-free so loky can pickle it."""
    warnings.filterwarnings("ignore")
    if kind == 'plain':
        from sklearn.mixture import GaussianMixture
        g = GaussianMixture(n_components=k, covariance_type='full', reg_covar=reg,
                            n_init=n_init, max_iter=max_iter, random_state=seed).fit(Xsub)
        return g, k
    from sklearn.mixture import BayesianGaussianMixture
    g = BayesianGaussianMixture(
        n_components=k, covariance_type='full', reg_covar=reg, n_init=n_init,
        max_iter=max_iter, weight_concentration_prior_type='dirichlet_process',
        random_state=seed).fit(Xsub)
    # components the DP kept (weight above 1% of uniform)
    n_eff = int(np.sum(g.weights_ > 0.01 / k))
    return g, n_eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', type=str, default='all')
    ap.add_argument('--arms', type=str, default='plain,bgm_k,bgm_cap')
    ap.add_argument('--cap', type=int, default=15, help='n_components for bgm_cap')
    ap.add_argument('--R', type=int, default=50)
    ap.add_argument('--M', type=int, default=150000)
    ap.add_argument('--n-coords', type=int, default=50)
    ap.add_argument('--n-jobs', type=int, default=32)
    ap.add_argument('--max-iter', type=int, default=300)
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--dataset', type=str, default='simcat6')
    ap.add_argument('--k-mode', choices=['per-event', 'fixed'], default='per-event')
    ap.add_argument('--out', type=str, default=os.path.join(HERE, 'bgm_simcat6.npz'))
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
    arms = cli.arms.split(',')
    Kcomp = (per_event_K(cli.dataset) if cli.k_mode == 'per-event'
             else np.full(Nobs, MD.K_PER_EVENT_FITVAR, dtype=int))
    print(f"[bgm] arms={arms} cap={cli.cap} R={cli.R} k_mode={cli.k_mode}")

    qstar = MD._build_qstar_pool(cli.M, MD.args.fitvar_defensive_frac,
                                 MD.args.fitvar_broad_inflate, cli.seed)
    M_max = min(cli.M, qstar['nsamp_pop'])
    coords, coord_arr = MD._fitvar_coords()
    nC = len(coords)
    X_eval = jnp.asarray(MD._eval_pool_detframe(qstar, M_max))
    log_h = MD._log_h(coords, M_max, qstar)

    def logp_of(g):
        return np.asarray(MD.exact_log_gmm(X_eval, *MD._gmm_to_jax(g)), dtype=np.float32)

    def lnL_from(block):
        lp = jnp.asarray(block, dtype=jnp.float64)
        out = np.zeros((block.shape[0], nC))
        for c0 in range(0, nC, 8):
            c1 = min(c0 + 8, nC)
            out[:, c0:c1] = np.asarray(
                MD._lnL_chunk(jnp.asarray(log_h[c0:c1]), lp)).T - np.log(M_max)
        return out

    def ncomp(arm, e):
        return cli.cap if arm == 'bgm_cap' else int(Kcomp[e])

    V = {a: np.zeros((nC, len(events))) for a in arms}
    ref = {a: np.zeros((len(events), nC)) for a in arms}
    neff = {a: np.zeros(len(events)) for a in arms}

    t0 = time.time()
    for ie, e in enumerate(events):
        X = MD._pe_detframe(e)
        for arm in arms:
            kind = 'plain' if arm == 'plain' else 'bgm'
            k = ncomp(arm, e)
            # reference point estimate on the REAL samples
            g0, n0 = _fit(X, kind, k, MD.REG_COVAR_FITVAR, MD.N_INIT_FITVAR,
                          cli.seed, cli.max_iter)
            ref[arm][ie] = lnL_from(logp_of(g0)[None])[0]
            neff[arm][ie] = n0
            # nested outer draws -- identical stream to production
            subs = []
            for b in range(cli.R):
                rs = np.random.default_rng([cli.seed, e, b])
                subs.append(X[rs.integers(0, nsamp, size=nsamp)])
            fits = Parallel(n_jobs=cli.n_jobs)(
                delayed(_fit)(s, kind, k, MD.REG_COVAR_FITVAR, MD.N_INIT_FITVAR,
                              cli.seed, cli.max_iter) for s in subs)
            block = np.stack([logp_of(g) for g, _ in fits])
            V[arm][:, ie] = np.var(lnL_from(block), axis=0, ddof=1)
        if (ie + 1) % 5 == 0 or ie == len(events) - 1:
            el = (time.time() - t0) / 60
            print(f"[bgm] {ie + 1}/{len(events)}  {el:.1f} min  "
                  f"ETA {el / (ie + 1) * (len(events) - ie - 1):.1f} min", flush=True)

    Bp = np.median(V['plain'].sum(axis=1))
    print(f"\n=== B, summed over events, median over Lambda ===")
    print(f"  {'arm':9s} {'B':>9} {'reduction':>11} {'Lam-dep bias':>13} "
          f"{'bias/sqrt(B)':>13} {'var+bias^2':>11} {'net':>7}")
    print(f"  {'plain':9s} {Bp:9.4f} {'--':>11} {'--':>13} {'--':>13} "
          f"{Bp:11.4f} {'--':>7}")
    out = {}
    for arm in arms:
        if arm == 'plain':
            continue
        Ba = np.median(V[arm].sum(axis=1))
        shift = (ref[arm] - ref['plain']).sum(axis=0)
        bdep = float(np.std(shift, ddof=1))
        tot = Ba + bdep ** 2
        print(f"  {arm:9s} {Ba:9.4f} {(1 - Ba / Bp) * 100:10.1f}% {bdep:13.4f} "
              f"{bdep / np.sqrt(Bp):13.3f} {tot:11.4f} {(1 - tot / Bp) * 100:6.1f}%")
        out[arm] = (Ba, bdep, tot)
        print(f"  {'':9s} Lambda-independent part: {np.mean(shift):+.4f} nats "
              f"(cancels in the posterior)")
        print(f"  {'':9s} components kept (median over events): "
              f"{np.median(neff[arm]):.1f} of {ncomp(arm, events[0])}")

    print(f"\n  reference: fitvar_summary.md quotes BGM at ~52% (5-event subset)")
    np.savez(cli.out, events=np.array(events), Kcomp=Kcomp, cap=cli.cap,
             **{f'V_{a}': V[a] for a in arms},
             **{f'ref_{a}': ref[a] for a in arms},
             **{f'neff_{a}': neff[a] for a in arms})
    print(f"[bgm] saved -> {cli.out}")


if __name__ == '__main__':
    main()
