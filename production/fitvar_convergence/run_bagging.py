#!/usr/bin/env python
"""
Does bagging the per-event GMM lower the fit variance B on sim_cat6?

Bagging replaces the single per-event density estimate

    p_hat_e            = GMM fitted to the event's N_PE samples

with the ensemble average

    p_bar_e(theta)     = (1/K) sum_{k=1..K} GMM_k(theta),

each GMM_k fitted to its own bootstrap resample of those samples. Averaging
cancels the part of the estimator's noise that is independent across members --
EM local-optimum switching above all -- while leaving the part driven by the
finite sample itself.

HOW B MUST BE MEASURED FOR A BAGGED ESTIMATOR
---------------------------------------------
B is Var over PE-SAMPLE DRAWS, so the ensemble has to live INSIDE each draw:

  outer b = 1..R : D*_b = resample(X, N_PE)      <- stands in for a new PE run
    plain  : one GMM fitted to D*_b                        (published estimator)
    bagged : K GMMs, each fitted to resample(D*_b, N_PE),  (bagged estimator)
             densities averaged
  B = Var_b[ ln L ]

Reusing the existing n_boot fits as the ensemble instead -- the shortcut noted
in the fitvar-npe-scaling memory -- would average over the very draws whose
spread defines B, collapsing it artifactually. The nested form costs K+1 fits
per outer replicate and is the honest measurement.

The 'plain' arm shares the outer draw stream with the production estimator, so
it reproduces the published B and the two arms are paired.

Bagging also SHIFTS the estimate (the average of K mixtures is smoother than one
mixture), so this reports the bias shift alongside the variance, using the
production fit on the real samples as the reference. Variance reduction bought
with a large shift is not a win.
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
    ap.add_argument('--n-bag', type=int, default=5, help='ensemble size K')
    ap.add_argument('--R', type=int, default=50, help='outer draws (=n_boot)')
    ap.add_argument('--M', type=int, default=150000)
    ap.add_argument('--n-coords', type=int, default=50)
    ap.add_argument('--n-jobs', type=int, default=30)
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--dataset', type=str, default='simcat6')
    ap.add_argument('--k-mode', choices=['per-event', 'fixed'], default='per-event',
                    help="'per-event' uses the BIC-argmin K the inference actually "
                         "loads (3..10, median 5); 'fixed' uses the fitvar "
                         "diagnostic's uniform K_PER_EVENT_FITVAR=7.")
    ap.add_argument('--out', type=str, default=os.path.join(HERE, 'bagging.npz'))
    cli = ap.parse_args()

    from _dataset import build_argv, check_nsamp, per_event_K
    scratch = os.path.join(HERE, 'scratch')
    os.makedirs(scratch, exist_ok=True)
    sys.argv = build_argv(cli.dataset, scratch, cli.n_coords, cli.seed)
    sys.path.insert(0, PROD)
    import make_diagnostic_plots as MD
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp as jlse
    from joblib import Parallel, delayed

    nsamp, Nobs = MD.nsamp, MD.Nobs
    check_nsamp(cli.dataset, nsamp)
    events = (list(range(Nobs)) if cli.events == 'all'
              else [int(x) for x in cli.events.split(',')])
    K = cli.n_bag
    if cli.k_mode == 'per-event':
        Kcomp = per_event_K(cli.dataset)
    else:
        Kcomp = np.full(Nobs, MD.K_PER_EVENT_FITVAR, dtype=int)
    print(f"[bag] n_bag={K}  R={cli.R}  nsamp={nsamp}  events={len(events)}  "
          f"k_mode={cli.k_mode}")

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
        """block: (n, M_max) log-densities -> (n, nC) ln L."""
        lp = jnp.asarray(block, dtype=jnp.float64)
        out = np.zeros((block.shape[0], nC))
        for c0 in range(0, nC, 8):
            c1 = min(c0 + 8, nC)
            out[:, c0:c1] = np.asarray(
                MD._lnL_chunk(jnp.asarray(log_h[c0:c1]), lp)).T - np.log(M_max)
        return out

    V_plain = np.zeros((nC, len(events)))
    V_bag = np.zeros((nC, len(events)))
    lnL_ref_plain = np.zeros((len(events), nC))     # production fit, real samples
    lnL_ref_bag = np.zeros((len(events), nC))       # bagged fit, real samples
    mean_plain = np.zeros((len(events), nC))
    mean_bag = np.zeros((len(events), nC))

    t0 = time.time()
    for ie, e in enumerate(events):
        X = MD._pe_detframe(e)

        # ---- reference point estimates on the REAL samples ------------------
        g0 = _fit_gmm(X, int(Kcomp[e]), MD.REG_COVAR_FITVAR,
                      MD.N_INIT_FITVAR, cli.seed)
        lnL_ref_plain[ie] = lnL_from(logp_of(g0)[None])[0]

        rs0 = np.random.default_rng([cli.seed, e, 10_000])
        subs0 = [X[rs0.integers(0, nsamp, size=nsamp)] for _ in range(K)]
        gs0 = Parallel(n_jobs=cli.n_jobs)(
            delayed(_fit_gmm)(s, int(Kcomp[e]), MD.REG_COVAR_FITVAR,
                              MD.N_INIT_FITVAR, cli.seed) for s in subs0)
        bag0 = np.asarray(jlse(jnp.asarray(np.stack([logp_of(g) for g in gs0])),
                               axis=0) - np.log(K), dtype=np.float32)
        lnL_ref_bag[ie] = lnL_from(bag0[None])[0]

        # ---- nested bootstrap ------------------------------------------------
        # Build every fit for this event in ONE parallel batch: the plain fit on
        # each outer draw, plus the K inner fits inside it.
        tasks, outers = [], []
        for b in range(cli.R):
            rs = np.random.default_rng([cli.seed, e, b])
            Db = X[rs.integers(0, nsamp, size=nsamp)]      # outer = published stream
            outers.append(Db)
            tasks.append(Db)                                # plain fit on D*_b
            rb = np.random.default_rng([cli.seed, e, b, 1])
            for _ in range(K):
                tasks.append(Db[rb.integers(0, nsamp, size=nsamp)])   # inner

        fits = Parallel(n_jobs=cli.n_jobs)(
            delayed(_fit_gmm)(s, int(Kcomp[e]), MD.REG_COVAR_FITVAR,
                              MD.N_INIT_FITVAR, cli.seed) for s in tasks)

        blk_plain = np.zeros((cli.R, M_max), dtype=np.float32)
        blk_bag = np.zeros((cli.R, M_max), dtype=np.float32)
        step = K + 1
        for b in range(cli.R):
            blk_plain[b] = logp_of(fits[b * step])
            members = np.stack([logp_of(fits[b * step + 1 + k]) for k in range(K)])
            blk_bag[b] = np.asarray(
                jlse(jnp.asarray(members), axis=0) - np.log(K), dtype=np.float32)

        Lp, Lb = lnL_from(blk_plain), lnL_from(blk_bag)
        V_plain[:, ie] = np.var(Lp, axis=0, ddof=1)
        V_bag[:, ie] = np.var(Lb, axis=0, ddof=1)
        mean_plain[ie], mean_bag[ie] = Lp.mean(axis=0), Lb.mean(axis=0)

        if (ie + 1) % 5 == 0 or ie == len(events) - 1:
            el = (time.time() - t0) / 60
            print(f"[bag] {ie + 1}/{len(events)} events  {el:.1f} min  "
                  f"ETA {el / (ie + 1) * (len(events) - ie - 1):.1f} min", flush=True)

    Bp = np.median(V_plain.sum(axis=1))
    Bb = np.median(V_bag.sum(axis=1))
    print("\n=== B, summed over events, median over Lambda ===")
    print(f"  plain  (published estimator) : B = {Bp:.4f}")
    print(f"  bagged (K={K})               : B = {Bb:.4f}")
    print(f"  ratio bagged/plain           : {Bb / Bp:.3f}   "
          f"({(1 - Bb / Bp) * 100:+.1f}% change)")

    shift = (lnL_ref_bag - lnL_ref_plain).sum(axis=0)
    print(f"\n=== bias shift on the REAL samples (bagged - plain), summed over events ===")
    print(f"  median over Lambda : {np.median(shift):+.4f} nats")
    print(f"  16-84%             : [{np.percentile(shift, 16):+.4f}, "
          f"{np.percentile(shift, 84):+.4f}]")
    print(f"  for scale, sqrt(B_plain) = {np.sqrt(Bp):.4f} nats")
    # Only the Lambda-DEPENDENT part of the shift can move the hyper-posterior:
    # a constant offset in ln L(Lambda) cancels in the posterior shape.
    print(f"\n  Lambda-INDEPENDENT part (cancels in the posterior) : "
          f"{np.mean(shift):+.4f} nats")
    print(f"  Lambda-DEPENDENT part, sd over Lambda (this is what bites) : "
          f"{np.std(shift, ddof=1):.4f} nats")
    print(f"  ratio to sqrt(B_plain) : {np.std(shift, ddof=1) / np.sqrt(Bp):.3f}"
          f"   (<1 means the induced bias is below the noise it removes)")

    np.savez(cli.out, events=np.array(events), n_bag=K, R=cli.R,
             V_plain=V_plain, V_bag=V_bag, coord_arr=coord_arr,
             lnL_ref_plain=lnL_ref_plain, lnL_ref_bag=lnL_ref_bag,
             mean_plain=mean_plain, mean_bag=mean_bag)
    print(f"\n[bag] saved -> {cli.out}")


if __name__ == '__main__':
    main()
