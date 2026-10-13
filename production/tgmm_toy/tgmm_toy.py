#!/usr/bin/env python3
"""
tgmm_toy.py -- 1D toy: does our per-event GMM "leak", and does a truncated GMM
(TGMM, as in Mancarella's P2600539 App. D) change anything?

Posterior samples live on the domain D = [-1, 1]. Each sample set is fit with
the PRODUCTION recipe (sim_cat_fit_event_gmms.py: sklearn GaussianMixture,
full cov, reg_covar=1e-6, n_init=5, random_state=0, K = BIC argmin over 1..20)
and then turned into a per-event density four ways:

  gmm        plain GMM, unbounded (leaks outside D)
  box        PRODUCTION met3/met4: GMM * 1[x in [P0.02, P99.98] of the samples],
             NOT renormalised (sim_cat_inference.py --pe-bounds-pct 0.02 99.98)
  gmm_trunc  post-hoc truncation: same GMM restricted to D, renormalised
  tgmm       truncation-aware fit: mixture of truncated normals on D, every
             component normalised on D, parameters re-fit by maximum likelihood
             (initialised from the GMM, same K)

Part 1 (single posteriors): leak mass, box-removed mass, shape error, and the
  Lambda-dependence error of the per-event evidence
      Z(lam) = int g(x) p(x|lam) dx,  p(x|lam) ∝ exp(lam x) on D,
  which is all the hierarchical likelihood sees (constants drop).
Part 2 (hierarchical): pop p(x|lam) ∝ exp(lam x) on D, N_ev events with
  Gaussian noise, flat PE prior on D -> exact truncated-normal posteriors.
  Posterior on lam from each method vs the exact one, over many catalogs.
  Part 2b repeats with the population support WIDER than D (pop on
  [-1.5, 1.5], PE prior still on D), the case where leaked mass gets weight.

Usage:  python tgmm_toy.py --outdir out [--ncat 20 --nev 200 --njobs 100]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy import optimize, special, stats
from sklearn.mixture import GaussianMixture

A, B = -1.0, 1.0
NSAMP = 5000
PCT = (0.02, 99.98)
XG = np.linspace(-1.6, 1.6, 6401)          # quadrature grid, dx = 5e-4
DX = XG[1] - XG[0]
IN_D = (XG >= A) & (XG <= B)


# ── fitting ──────────────────────────────────────────────────────────────
def fit_gmm_bic(x, kmax=20, seed=0):
    """Production recipe. Returns the BIC-argmin fit and every K's fit (for
    initialising the TGMM scan)."""
    X = x[:, None]
    fits, bics = {}, {}
    for k in range(1, kmax + 1):
        g = GaussianMixture(k, covariance_type="full", reg_covar=1e-6,
                            n_init=5, random_state=seed).fit(X)
        bics[k] = g.bic(X)
        fits[k] = (g.weights_.copy(), g.means_[:, 0].copy(),
                   np.sqrt(g.covariances_[:, 0, 0]))
    kbest = min(bics, key=bics.get)
    return fits[kbest], fits


def _lz(mu, sd, a, b, xp=np, sp=special):
    """log(Phi((b-mu)/sd) - Phi((a-mu)/sd)), stable on either tail."""
    al, be = (a - mu) / sd, (b - mu) / sd
    hi = xp.where(al > 0, sp.log_ndtr(-al), sp.log_ndtr(be))
    lo = xp.where(al > 0, sp.log_ndtr(-be), sp.log_ndtr(al))
    return hi + xp.log1p(-xp.exp(xp.minimum(lo - hi, -1e-12)))


def _log_trunc_terms(x, logits, mu, lsd, a, b):
    sd = np.exp(lsd)
    logw = logits - special.logsumexp(logits)
    lc = stats.norm.logpdf(x[:, None], mu, sd) - _lz(mu, sd, a, b)
    return logw, lc


def _jax_nll(K, a, b):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.scipy import special as jsp

    def nll(p, x):
        logits, mu, sd = p[:K], p[K:2*K], jnp.exp(p[2*K:])
        logw = logits - jsp.logsumexp(logits)
        lc = (-0.5 * ((x[:, None] - mu) / sd) ** 2 - jnp.log(sd)
              - 0.5 * jnp.log(2 * jnp.pi) - _lz(mu, sd, a, b, jnp, jsp))
        return -jsp.logsumexp(logw + lc, axis=1).mean()
    return jax.jit(jax.value_and_grad(nll))


def fit_tgmm_K(x, w0, mu0, sd0, a=A, b=B):
    """Truncation-aware MLE: mixture of normals each truncated & normalised on
    [a, b]. Returns ((logits, mu, logsd), BIC)."""
    import jax.numpy as jnp
    K = len(w0)
    f = _jax_nll(K, a, b)
    xj = jnp.asarray(x)
    fun = lambda p: tuple(np.asarray(v, dtype=float) for v in f(jnp.asarray(p), xj))
    p0 = np.concatenate([np.log(w0), np.clip(mu0, a - 2.9, b + 2.9),
                         np.log(np.clip(sd0, 2e-3, 9.0))])
    bounds = [(None, None)] * K + [(a - 3, b + 3)] * K + [(np.log(1e-3), np.log(10))] * K
    r = optimize.minimize(fun, p0, jac=True, method="L-BFGS-B", bounds=bounds,
                          options=dict(maxiter=5000))
    bic = 2 * r.fun * len(x) + (3 * K - 1) * np.log(len(x))
    return (r.x[:K], r.x[K:2*K], r.x[2*K:]), bic


def fit_tgmm(x, fits, kmax=20, patience=2):
    """BIC scan for the TGMM, K ascending from 1, each initialised from the
    sklearn GMM at that K; stop after `patience` successive BIC rises."""
    best, prev, rises = None, np.inf, 0
    for k in range(1, kmax + 1):
        tg, bic = fit_tgmm_K(x, *fits[k])
        if best is None or bic < best[1]:
            best = (tg, bic, k)
        rises = rises + 1 if bic > prev else 0
        prev = bic
        if rises >= patience:
            break
    return best[0], best[2]


def densities(x, w, mu, sd, tg):
    """All four per-event densities on XG (unnormalised ones left as-is)."""
    g = (w[None, :] * stats.norm.pdf(XG[:, None], mu, sd)).sum(1)
    lo, hi = np.percentile(x, PCT)
    box = np.where((XG >= lo) & (XG <= hi), g, 0.0)
    gt = np.where(IN_D, g, 0.0)
    gt /= gt.sum() * DX
    logw, lc = _log_trunc_terms(XG, *tg, A, B)
    t = np.where(IN_D, np.exp(special.logsumexp(logw + lc, axis=1)), 0.0)
    return dict(gmm=g, box=box, gmm_trunc=gt, tgmm=t), (lo, hi)


METHODS = ("gmm", "box", "gmm_trunc", "tgmm")


# ── population / evidence ────────────────────────────────────────────────
LAMS = np.linspace(-6, 6, 241)


def log_pop(lam, lo=A, hi=B):
    """(n_lam, n_grid) log p(x|lam) ∝ exp(lam x) on [lo, hi]."""
    lam = np.atleast_1d(lam)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        lognorm = np.where(np.abs(lam) < 1e-9, np.log(hi - lo),
                           np.log(np.abs(np.expm1(lam * (hi - lo)))) + lam * lo - np.log(np.abs(lam)))
    lp = lam * XG[None, :] - lognorm
    return np.where((XG >= lo) & (XG <= hi), lp, -np.inf)


def logZ(dens, LP):
    """log int g(x) p(x|lam) dx for each lam (flat PE prior -> no division)."""
    with np.errstate(divide="ignore"):
        return special.logsumexp(np.log(dens)[None, :] + LP, axis=1) + np.log(DX)


# ── Part 1: single posteriors ────────────────────────────────────────────
def truth_cases():
    tn = lambda m, s: stats.truncnorm((A - m) / s, (B - m) / s, loc=m, scale=s)
    ramp = stats.beta(4, 1, loc=-1, scale=2)                 # ∝ (1+x)^3
    return {
        "interior  TN(0, 0.3)": tn(0.0, 0.3),
        "near edge TN(0.85, 0.2)": tn(0.85, 0.2),
        "peak at edge TN(1, 0.3)": tn(1.0, 0.3),
        "ramp ∝ (1+x)^3": ramp,
        "uniform U(-1, 1)": stats.uniform(-1, 2),
    }


def part1_one(name, dist, seed):
    x = dist.rvs(NSAMP, random_state=np.random.default_rng(seed))
    (w, mu, sd), fits = fit_gmm_bic(x)
    tg, ktg = fit_tgmm(x, fits)
    D, (lo, hi) = densities(x, w, mu, sd, tg)
    ptrue = np.where(IN_D, dist.pdf(XG), 0.0)
    LP = log_pop(LAMS)
    lz_true = logZ(ptrue, LP)
    out = dict(case=name, seed=seed, K=len(w), K_tgmm=ktg, lo=lo, hi=hi,
               leak=float(D["gmm"][~IN_D].sum() * DX),
               box_removed_in_D=float(D["gmm"][IN_D & ((XG < lo) | (XG > hi))].sum() * DX),
               true_mass_outside_box=float(dist.cdf(lo) + dist.sf(hi)))
    i0 = np.argmin(np.abs(LAMS))
    for m in METHODS:
        gD = np.where(IN_D, D[m], 0.0)
        gn = gD / (gD.sum() * DX)
        out[f"L1_{m}"] = float(np.abs(gn - ptrue).sum() * DX)
        lz = logZ(D[m], LP)
        err = (lz - lz[i0]) - (lz_true - lz_true[i0])
        out[f"err_{m}"] = err.tolist()
        out[f"dens_{m}"] = D[m][::4].tolist()
    out["dens_true"] = ptrue[::4].tolist()
    return out


# ── Part 2: hierarchical lambda ──────────────────────────────────────────
def part2_event(d, s, seed, pop_range):
    al, be = (A - d) / s, (B - d) / s
    post = stats.truncnorm(al, be, loc=d, scale=s)
    x = post.rvs(NSAMP, random_state=np.random.default_rng(seed))
    (w, mu, sd), fits = fit_gmm_bic(x)
    tg, ktg = fit_tgmm(x, fits)
    D, _ = densities(x, w, mu, sd, tg)
    LP = log_pop(LAMS, *pop_range)
    res = {m: logZ(D[m], LP) for m in METHODS}
    res["K"] = (len(w), ktg)
    res["exact"] = logZ(np.where(IN_D, post.pdf(XG), 0.0), LP)
    return res


def lam_summary(logL):
    p = np.exp(logL - logL.max()); p /= p.sum()
    mean = (p * LAMS).sum()
    sdv = np.sqrt((p * (LAMS - mean) ** 2).sum())
    cdf = np.cumsum(p)
    return mean, sdv, float(np.interp(0.5, cdf, LAMS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("out"))
    ap.add_argument("--nseed", type=int, default=10)
    ap.add_argument("--ncat", type=int, default=20)
    ap.add_argument("--nev", type=int, default=200)
    ap.add_argument("--lam-true", type=float, default=2.0)
    ap.add_argument("--njobs", type=int, default=100)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    cases = truth_cases()
    p1 = Parallel(n_jobs=args.njobs)(
        delayed(part1_one)(n, d, s) for n, d in cases.items() for s in range(args.nseed))
    json.dump(p1, open(args.outdir / "part1.json", "w"))
    print("[part1] done")

    rng = np.random.default_rng(12345)
    for tag, pop_range in (("popD", (A, B)), ("popwide", (-1.5, 1.5))):
        # draw true x from the population; the PE prior is flat on D, so an
        # event whose truth lies outside D is impossible under the PE model --
        # with the wide population we still only keep events with x in D,
        # mimicking PE prior bounds that enclose every real event.
        jobs, meta = [], []
        for c in range(args.ncat):
            lp = log_pop(args.lam_true, *pop_range)[0]
            pw = np.exp(lp - lp.max()) * IN_D; pw /= pw.sum()
            xt = rng.choice(XG, size=args.nev, p=pw)
            sig = rng.uniform(0.1, 0.5, args.nev)
            d = xt + sig * rng.standard_normal(args.nev)
            for e in range(args.nev):
                jobs.append(delayed(part2_event)(d[e], sig[e], c * 10000 + e, pop_range))
                meta.append(c)
        res = Parallel(n_jobs=args.njobs)(jobs)
        meta = np.array(meta)
        summ = []
        for c in range(args.ncat):
            idx = np.where(meta == c)[0]
            row = {}
            for m in METHODS + ("exact",):
                L = np.sum([res[i][m] for i in idx], axis=0)
                row[m] = lam_summary(L)
            summ.append(row)
        json.dump(summ, open(args.outdir / f"part2_{tag}.json", "w"))
        print(f"[part2 {tag}] done")


if __name__ == "__main__":
    main()
