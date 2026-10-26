#!/usr/bin/env python
"""Paper figure `corner_sanity.pdf` (fig:proof_of_concept).

Gaussian toy population, end to end: 10 events with m1 ~ N(35, 1) and
z ~ N(1, 0.1), each observed as 1000 Gaussian PE samples in (m1_det, dL)
(sd 5 Msun, 100 Mpc). The hyperparameters (mu, sigma, mu_z, sigma_z) are
inferred three ways with dynesty (nlive 1000) and overlaid:
  PE        sum over the PE samples directly                        (blue)
  POP       draw from the population, evaluate per-event K=7 GMMs   (orange)
  Ensemble  importance sampling from the mixture of all event GMMs   (green)
with the true values in black. Runtime: minutes (three nested-sampling runs).

Extracted from inference/sanity_test_methods.py (Colab "GWTC3_gmm_method1to3"),
which made the paper figure, with four changes:
  * --seed fixes the toy data and the samplers; the original was unseeded, so
    the published figure is one unrepeatable realisation.
  * The Ensemble proposal is drawn as x = mu + L^-T z (Prec = L L^T). The
    original drew mu + L^-1 z, whose covariance (L^T L)^-1 is not the GMM's
    while log q(x) is evaluated for the correct mixture -- the same bug that
    biased met3/met4 (fixed in sim_cat_inference.py 2026-08-21).
  * POP's z draws use the shifted stratification the original intended. Its
    make_eps() ignored its argument for even N, so m1 and z got the SAME
    normal quantiles: perfectly correlated population draws.
  * POP carries no Jacobian. Its samples come from the population itself, so
    the per-event likelihood is just the mean of p_e(x_det(theta)); the
    original multiplied by J = ddL/dz (1+z), which put its logZ ~97 nats
    (10 events x log J ~ 9.7) above PE/Ensemble and tilts it with z.
--paper-bug restores all three original behaviours.

    python fig_corner_sanity.py [--seed 0] [--paper-bug] [--out corner_sanity.pdf]
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, random
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm as norm_jax

import astropy.constants as const
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM, Planck15

# toy population and observation model
MU, SIGMA, MU_Z, SIGMA_Z = 35.0, 1.0, 1.0, 0.1
N_OBS, N_PE = 10, 1000
SIGMA_M1DET, SIGMA_DL = 5.0, 100.0
K_GMM = 7
NLIVE = 1000
LO = np.array([25.0, 0.1, 0.1, 0.01])   # prior box for (mu, sigma, mu_z, sigma_z)
HI = np.array([45.0, 10.0, 2.0, 0.5])
LABELS = [r"$\mu$", r"$\sigma$", r"$\mu_z$", r"$\sigma_z$"]
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73"}

# cosmology (Planck15), comoving distance on a z grid out to 15
H0, OM0 = Planck15.H0.value, Planck15.Om0
C_KMS = const.c.to("km/s").value
ZGRID = np.expm1(np.linspace(0.0, np.log(16.0), 10000))
R_GRID = jnp.asarray(FlatLambdaCDM(H0=H0, Om0=OM0).comoving_distance(ZGRID).to(u.Mpc).value)
ZGRID = jnp.asarray(ZGRID)


def dL_of_z(z):
    return (1 + z) * jnp.interp(z, ZGRID, R_GRID)


def z_of_dL(dL):
    return jnp.interp(dL, (1 + ZGRID) * R_GRID, ZGRID)


def ddL_dz(z, dL):
    return dL / (1 + z) + C_KMS * (1 + z) / (H0 * jnp.sqrt(OM0 * (1 + z) ** 3 + 1 - OM0))


def prior_transform(u_):
    return tuple(LO + u_ * (HI - LO))


def run_dynesty(loglike, seed):
    from dynesty import NestedSampler
    from dynesty.utils import resample_equal
    s = NestedSampler(loglike, prior_transform, 4, bound="multi", sample="rwalk", nlive=NLIVE,
                      rstate=np.random.default_rng(seed))
    s.run_nested(dlogz=0.1, print_progress=False)
    r = s.results
    print(f"    logZ = {r.logz[-1]:.2f} +- {r.logzerr[-1]:.2f}")
    return resample_equal(r.samples, np.exp(r["logwt"] - r["logz"][-1]),
                          rstate=np.random.default_rng(seed))


def safe(f):
    def ll(theta):
        v = float(f(*theta))
        return -np.inf if np.isnan(v) else v
    return ll


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("corner_sanity.pdf"))
    ap.add_argument("--seed", type=int, default=0, help="toy-data and sampler seed")
    ap.add_argument("--paper-bug", action="store_true",
                    help="reproduce the published bugs: Ensemble proposal mu + L^-1 z, "
                         "identical m1/z quantiles in POP, and POP's extra Jacobian")
    args = ap.parse_args()

    matplotlib.rcParams.update({
        "font.family": "Times New Roman", "font.sans-serif": ["Bitstream Vera Sans"],
        "text.usetex": False, "mathtext.fontset": "cm", "figure.figsize": (16.0, 10.0),
        "axes.unicode_minus": False,
    })
    sns.set_context("talk")
    sns.set_style("ticks")
    sns.set_palette("colorblind")

    # ---- toy catalog: true source-frame values -> noisy detector-frame PE samples
    rng = np.random.default_rng(args.seed)
    m1 = rng.normal(MU, SIGMA, N_OBS)
    z = rng.normal(MU_Z, SIGMA_Z, N_OBS)
    m1det_true, dL_true = (1 + z) * m1, np.asarray(dL_of_z(jnp.asarray(z)))
    m1det_pe = np.stack([rng.normal(m, SIGMA_M1DET, N_PE) for m in m1det_true])   # (E, N)
    dL_pe = np.stack([rng.normal(d, SIGMA_DL, N_PE) for d in dL_true])

    # ---- per-event GMMs in (m1_det, dL)
    from sklearn.mixture import GaussianMixture
    gmms = [GaussianMixture(n_components=K_GMM, covariance_type="full", reg_covar=1e-6,
                            n_init=10, random_state=0).fit(np.column_stack([m1det_pe[e], dL_pe[e]]))
            for e in range(N_OBS)]
    wts = jnp.asarray(np.stack([g.weights_ for g in gmms]))                 # (E, K)
    mus = jnp.asarray(np.stack([g.means_ for g in gmms]))                   # (E, K, D)
    precs = jnp.asarray(np.stack([g.precisions_ for g in gmms]))            # (E, K, D, D)
    logdets = jnp.asarray(-2.0 * np.log(np.diagonal(
        np.stack([g.precisions_cholesky_ for g in gmms]), axis1=-2, axis2=-1)).sum(-1))   # log|Sigma|

    @jit
    def log_gmm_per_event(X):
        """log p_e(X_n) for every event's GMM: (E, N)."""
        E, K, D = mus.shape
        diffs = X[:, None, :] - mus.reshape(E * K, D)[None]
        quad = jnp.einsum("nkd,kdj,nkj->nk", diffs, precs.reshape(E * K, D, D), diffs)
        log_comp = (-0.5 * (D * jnp.log(2 * jnp.pi) + logdets.reshape(E * K) + quad)).reshape(-1, E, K)
        return logsumexp(jnp.log(jnp.clip(wts, 1e-12, 1.0))[None] + log_comp, axis=-1).T

    posts = {}

    # ---- PE: sum over the PE samples directly
    m1det_flat, dL_flat = jnp.asarray(m1det_pe.ravel()), jnp.asarray(dL_pe.ravel())

    @jit
    def ll_pe(mu_m, sigma_m, mu_z, sigma_z):
        zz = z_of_dL(dL_flat)
        lw = (norm_jax.logpdf(m1det_flat / (1 + zz), mu_m, sigma_m) + norm_jax.logpdf(zz, mu_z, sigma_z)
              - jnp.log(ddL_dz(zz, dL_flat)) - jnp.log1p(zz))
        return jnp.sum(-jnp.log(N_PE) + logsumexp(lw.reshape(N_OBS, N_PE), axis=-1))

    print("PE")
    posts["PE"] = run_dynesty(safe(ll_pe), args.seed)

    # ---- POP: stratified population draws, evaluated under each event's GMM
    n_pop = 20000
    half = (jnp.arange(n_pop // 2) + 0.5) / n_pop
    eps_m1 = jnp.concatenate([jsp.special.ndtri(half), -jsp.special.ndtri(half)])
    if args.paper_bug:
        eps_z = eps_m1          # original: make_eps ignored its argument, so m1 and z were comonotone
    else:
        eps_z = jsp.special.ndtri(((jnp.arange(n_pop) + 0.5) / n_pop + 0.73) % 1.0)   # the intended shift

    @jit
    def ll_pop(mu_m, sigma_m, mu_z, sigma_z):
        m1p, zp = mu_m + sigma_m * eps_m1, mu_z + sigma_z * eps_z
        dLp = dL_of_z(zp)
        logp = log_gmm_per_event(jnp.column_stack([m1p * (1 + zp), dLp]))
        # Draws are from the population itself, so E[p_e(x_det(theta))] needs no
        # Jacobian. The original multiplied by J = ddL/dz (1+z): ~+9.7 nats per
        # event at z~1 (its logZ sat ~97 above PE/Ensemble), and z-dependent.
        logJ = jnp.log(ddL_dz(zp, dLp)) + jnp.log1p(zp) if args.paper_bug else jnp.zeros_like(zp)
        return jnp.sum(jnp.nan_to_num(logsumexp(logp + logJ[None], axis=1)) - jnp.log(n_pop))

    print("POP")
    posts["POP"] = run_dynesty(safe(ll_pop), args.seed + 1)

    # ---- Ensemble: importance sampling from q = mixture of all event GMMs
    n_q = 15000
    E, K, D = mus.shape
    w_q = wts.reshape(E * K) / E
    w_q = w_q / w_q.sum()
    mu_q, prec_q = mus.reshape(E * K, D), precs.reshape(E * K, D, D)
    L = jnp.linalg.cholesky(prec_q)                                # Prec = L L^T
    k1, k2 = random.split(random.PRNGKey(args.seed))
    comp = random.choice(k1, E * K, shape=(n_q,), p=w_q, replace=True)
    zn = random.normal(k2, shape=(n_q, D))[..., None]
    if args.paper_bug:
        y = jsp.linalg.solve_triangular(L[comp], zn, lower=True)                       # L^-1 z (wrong)
    else:
        y = jsp.linalg.solve_triangular(jnp.swapaxes(L[comp], -1, -2), zn, lower=False)   # L^-T z
    x = mu_q[comp] + y.squeeze(-1)
    diffs = x[:, None, :] - mu_q[None]
    quad = jnp.einsum("nkd,kdj,nkj->nk", diffs, prec_q, diffs)
    log_q = logsumexp(jnp.log(w_q)[None] - 0.5 * (D * jnp.log(2 * jnp.pi) + logdets.reshape(E * K)[None] + quad),
                      axis=1)
    zq = z_of_dL(x[:, 1])
    m1q = x[:, 0] / (1 + zq)
    logp_q = log_gmm_per_event(x)
    logJq = -jnp.log(ddL_dz(zq, x[:, 1])) - jnp.log1p(zq)

    @jit
    def ll_ens(mu_m, sigma_m, mu_z, sigma_z):
        lw = norm_jax.logpdf(m1q, mu_m, sigma_m) + norm_jax.logpdf(zq, mu_z, sigma_z) - log_q
        return jnp.sum(jnp.nan_to_num(logsumexp(lw[None] + logp_q + logJq[None], axis=1)) - jnp.log(n_q))

    print("Ensemble" + (" (paper-bug proposal)" if args.paper_bug else ""))
    posts["Ensemble"] = run_dynesty(safe(ll_ens), args.seed + 2)

    # ---- corner
    import corner
    truths = [MU, SIGMA, MU_Z, SIGMA_Z]
    fig = None
    for name, col in zip(posts, (CB["blue"], CB["orange"], CB["green"])):
        fig = corner.corner(posts[name], hist_kwargs={"density": True}, labels=LABELS,
                            truths=truths, color=col, **({"fig": fig} if fig is not None else {}))
    fig.legend(handles=[mlines.Line2D([], [], color=col, lw=2, label=name)
                        for name, col in zip(posts, (CB["blue"], CB["orange"], CB["green"]))],
               loc="upper right", fontsize=14)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {args.out}")
    for name, s in posts.items():
        print(f"{name:8s} medians: " + "  ".join(f"{v:.3g}" for v in np.median(s, 0)))


if __name__ == "__main__":
    main()
