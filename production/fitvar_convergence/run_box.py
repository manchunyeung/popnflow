#!/usr/bin/env python
"""
Fit variance B with the PE-bounds box the inference actually applies.

sim_cat_inference.py masks each event's GMM log-density to -inf outside a
per-axis percentile box of that event's PE samples (production: 0.02-99.98,
i.e. ~the sample min/max for 5000 samples). make_diagnostic_plots.py applies NO
box, so the published B is the variance of an estimator the analysis does not use.

Arms, computed from ONE set of GMM fits per outer draw (the box is a mask on the
same density block, so the extra arms are almost free):

  nobox       no mask                              <- what fitvar has measured
  box_resamp  box from each outer draw's OWN samples  <- faithful: a repeat PE
                                                        run would set its box
                                                        from its own samples
  box_fixed   box from the real samples, held fixed   <- leaks full-data info
                                                        (like warm-start); a
                                                        lower reference

Same paired outer-draw stream as production, per-event K, same q* pool and
Lambda draws as the other runs in this directory.
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
    ap.add_argument('--pct', type=float, nargs=2, default=[0.02, 99.98])
    ap.add_argument('--R', type=int, default=50)
    ap.add_argument('--M', type=int, default=150000)
    ap.add_argument('--n-coords', type=int, default=50)
    ap.add_argument('--n-jobs', type=int, default=32)
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--dataset', type=str, default='simcat6')
    ap.add_argument('--k-mode', choices=['per-event', 'fixed'], default='per-event')
    ap.add_argument('--out', type=str, default=os.path.join(HERE, 'box_simcat6.npz'))
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
    Kcomp = (per_event_K(cli.dataset) if cli.k_mode == 'per-event'
             else np.full(Nobs, MD.K_PER_EVENT_FITVAR, dtype=int))
    lo_p, hi_p = cli.pct
    print(f"[box] pct={lo_p}-{hi_p}  R={cli.R}  k_mode={cli.k_mode}")

    qstar = MD._build_qstar_pool(cli.M, MD.args.fitvar_defensive_frac,
                                 MD.args.fitvar_broad_inflate, cli.seed)
    M_max = min(cli.M, qstar['nsamp_pop'])
    coords, coord_arr = MD._fitvar_coords()
    nC = len(coords)
    Xe_np = np.asarray(MD._eval_pool_detframe(qstar, M_max))   # (M, 3) det frame
    X_eval = jnp.asarray(Xe_np)
    log_h = MD._log_h(coords, M_max, qstar)

    def outside(lo, hi):
        return np.any((Xe_np < lo) | (Xe_np > hi), axis=1)       # (M,)

    def lnL_from(block):
        lp = jnp.asarray(block, dtype=jnp.float64)
        out = np.zeros((block.shape[0], nC))
        for c0 in range(0, nC, 8):
            c1 = min(c0 + 8, nC)
            out[:, c0:c1] = np.asarray(
                MD._lnL_chunk(jnp.asarray(log_h[c0:c1]), lp)).T - np.log(M_max)
        return out

    arms = ['nobox', 'box_resamp', 'box_fixed']
    V = {a: np.zeros((nC, len(events))) for a in arms}
    ref = {a: np.zeros((len(events), nC)) for a in arms}
    frac_out = np.zeros(len(events))

    t0 = time.time()
    for ie, e in enumerate(events):
        X = MD._pe_detframe(e)
        k = int(Kcomp[e])
        lo_f = np.percentile(X, lo_p, axis=0)
        hi_f = np.percentile(X, hi_p, axis=0)
        out_f = outside(lo_f, hi_f)
        frac_out[ie] = out_f.mean()

        # reference on the real samples (box from the real samples by definition)
        g0 = _fit_gmm(X, k, MD.REG_COVAR_FITVAR, MD.N_INIT_FITVAR, cli.seed)
        lp0 = np.asarray(MD.exact_log_gmm(X_eval, *MD._gmm_to_jax(g0)), dtype=np.float32)
        ref['nobox'][ie] = lnL_from(lp0[None])[0]
        boxed0 = np.where(out_f, -np.inf, lp0).astype(np.float32)
        ref['box_resamp'][ie] = ref['box_fixed'][ie] = lnL_from(boxed0[None])[0]

        subs = []
        for b in range(cli.R):
            rs = np.random.default_rng([cli.seed, e, b])           # production stream
            subs.append(X[rs.integers(0, nsamp, size=nsamp)])
        gms = Parallel(n_jobs=cli.n_jobs)(
            delayed(_fit_gmm)(s, k, MD.REG_COVAR_FITVAR, MD.N_INIT_FITVAR, cli.seed)
            for s in subs)
        block = np.stack([np.asarray(MD.exact_log_gmm(X_eval, *MD._gmm_to_jax(g)),
                                     dtype=np.float32) for g in gms])     # (R, M)

        blk_resamp = np.empty_like(block)
        for b, s in enumerate(subs):
            o = outside(np.percentile(s, lo_p, axis=0), np.percentile(s, hi_p, axis=0))
            blk_resamp[b] = np.where(o, -np.inf, block[b])
        blk_fixed = np.where(out_f[None, :], -np.inf, block).astype(np.float32)

        for arm, blk in (('nobox', block), ('box_resamp', blk_resamp),
                         ('box_fixed', blk_fixed)):
            V[arm][:, ie] = np.var(lnL_from(blk), axis=0, ddof=1)

        if (ie + 1) % 5 == 0 or ie == len(events) - 1:
            el = (time.time() - t0) / 60
            print(f"[box] {ie + 1}/{len(events)}  {el:.1f} min  "
                  f"ETA {el / (ie + 1) * (len(events) - ie - 1):.1f} min", flush=True)

    Bn = np.median(V['nobox'].sum(axis=1))
    print(f"\n=== B, summed over events, median over Lambda (box {lo_p}-{hi_p}) ===")
    print(f"  eval-pool fraction outside each event's box: median "
          f"{np.median(frac_out) * 100:.1f}%")
    for arm in arms:
        Ba = np.median(V[arm].sum(axis=1))
        lo16, hi84 = np.percentile(V[arm].sum(axis=1), [16, 84])
        print(f"  {arm:11s} B = {Ba:.4f}  [16-84: {lo16:.4f}, {hi84:.4f}]  "
              f"{(Ba / Bn - 1) * 100:+6.1f}% change vs nobox")
    sh = (ref['box_fixed'] - ref['nobox']).sum(axis=0)
    print(f"\n  box vs nobox on the real samples: Lambda-indep {np.mean(sh):+.4f} nats, "
          f"Lambda-dep sd {np.std(sh, ddof=1):.4f} nats")
    print("  (the box IS the production estimator, so this shift is the correction "
          "toward what the inference uses, not a cost)")

    np.savez(cli.out, events=np.array(events), Kcomp=Kcomp, pct=np.array(cli.pct),
             frac_out=frac_out, **{f'V_{a}': V[a] for a in arms},
             **{f'ref_{a}': ref[a] for a in arms})
    print(f"[box] saved -> {cli.out}")


if __name__ == '__main__':
    main()
