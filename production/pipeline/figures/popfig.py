"""Shared machinery for the six posterior figures (corner, dR/dm1, R(z)) of the
sim_cat6 ("mdc") and GWTC-3 runs. Imported by fig_{mdc,gwtc3}_{corner_pos,Rm1,Rz}.py.

Extracted from plotting/make_plots_met4.py, which produced the paper versions.
Deliberate differences from that script, all documented in pipeline/README.md:

  * posterior resampling is SEEDED (resample_equal had no rstate, so the paper
    corners/rates are one unrepeatable draw; re-renders match in content, not
    pixels);
  * r(z) is evaluated at Planck15 Om0 directly instead of bilinearly
    interpolating a 1000-point Om0 grid, and everything runs in numpy float64
    rather than jax float32 -- both differences are far below plot resolution.

The rate normalisation is reproduced AS IS, including for the sim_cat6 run:
R0 = N_obs / (epsilon V T) with epsilon from the LVK endo3 injection set
(input_data/endo3_bbhpop-LIGO-T2100113-v12.hdf5, ifar > 1 in any pipeline),
and a constant spin-density factor that enters as +log(0.25) in the GWTC-3
pairing model and -log(0.25) in the sim_cat6 model. The "true" sim_cat6 curves
go through the same normaliser, so posterior vs truth is a like-for-like
comparison, but the absolute sim_cat6 rate scale is not the simulation's own.
"""
import __main__
import math
from pathlib import Path

import numpy as np

N_OBS = 69
ENDO3 = Path("input_data/endo3_bbhpop-LIGO-T2100113-v12.hdf5")

# ---------------------------------------------------------------- run configs
LATEX = {
    "gamma": r"$\gamma$", "m_min": r"$m_{\mathrm{min}}$", "m_max": r"$m_{\mathrm{max}}$",
    "alpha": r"$\alpha$", "dm_min": r"$\delta m_{\mathrm{min}}$",
    "dm_max": r"$\delta m_{\mathrm{max}}$", "beta": r"$\beta$", "mu": r"$\mu$",
    "sigma": r"$\sigma$", "f1": r"$\lambda$",
}
RUNS = {
    # sim_cat6_sharp_w8 catalog 0: met1 from inference/, met3/met4 from the
    # nlive=1000 PE-box + Cholesky-fix runs
    "mdc": dict(
        params=["gamma", "m_min", "m_max", "alpha", "beta", "mu", "sigma", "f1"],
        truths=dict(gamma=3.0, m_min=6.0, m_max=70.0, alpha=4.0, beta=1.5, mu=35.0,
                    sigma=4.0, f1=0.04),
        checkpoints={
            "met1": "sim_cat6_sharp_w8/inference/simcat_cat000_met1.h5",
            "met3": "sim_cat6_sharp_w8/inference_boxfix_cholfix_nlive1000/simcatnm_cat000_met3.h5",
            "met4": "sim_cat6_sharp_w8/inference_met4_nlive1000_boxfix_cholfix/simcat_cat000_met4.h5",
        },
        rm1_ylim=(1e-4, 10),
    ),
    # GWTC-3, symmetric pairing model, seed-69 Colab run of 2026-05-05
    "gwtc3": dict(
        params=["gamma", "m_min", "m_max", "alpha", "dm_min", "dm_max", "beta", "mu", "sigma", "f1"],
        truths=None,
        checkpoints={m: f"results/data/inference/gwtc3_69{m}.pkl" for m in ("met1", "met3", "met4")},
        rm1_ylim=(1e-2, 100),
    ),
}
METHOD_LABEL = {"met1": r"$\mathrm{PE}$", "met3": r"$\mathrm{Ensemble}$", "met4": r"$\mathrm{Mixture}$"}
THIN = {"met1": 10, "met3": 5, "met4": 5}   # posterior thinning for the rate draws


def setup_style():
    """make_plots_met4.py's style: real LaTeX, seaborn talk/ticks/colorblind."""
    import matplotlib
    matplotlib.use("Agg")
    import seaborn as sns
    matplotlib.rcParams.update({
        "text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm", "figure.figsize": (16.0, 10.0),
        "axes.unicode_minus": False, "axes.labelsize": 20,
    })
    sns.set_context("talk")
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    return sns.color_palette("colorblind")


def log_tex_formatter(y, pos):
    """usetex + PDF drops the math-mode minus in 10^{-3}; \\mbox{-} survives."""
    if y <= 0:
        return ""
    e = int(round(math.log10(y)))
    return r"$10^{\mbox{-}%d}$" % abs(e) if e < 0 else r"$10^{%d}$" % e


# ---------------------------------------------------------- posterior loading
def load_posterior(path, seed):
    """Equal-weight posterior samples from a dynesty checkpoint (.h5 or .pkl).

    The checkpoints pickle references to the likelihood/prior functions of the
    script that wrote them, looked up in __main__; stand-ins suffice to restore.
    """
    import dynesty
    from dynesty.utils import resample_equal
    for name in ("prior_transform", "loglike_method_1", "loglike_method_3", "loglike_method_4"):
        if not hasattr(__main__, name):
            setattr(__main__, name, lambda x: 0.0)
    res = dynesty.NestedSampler.restore(str(path)).results
    w = np.exp(res["logwt"] - res["logz"][-1])
    return resample_equal(res.samples, w, rstate=np.random.default_rng(seed))


def load_run(run, seed, overrides=None):
    paths = dict(RUNS[run]["checkpoints"], **(overrides or {}))
    return {m: load_posterior(p, seed + i) for i, (m, p) in enumerate(paths.items())}


# ----------------------------------------------------------------- cosmology
from astropy.cosmology import Planck15, FlatLambdaCDM   # noqa: E402
import astropy.units as u                              # noqa: E402
import astropy.constants as const                      # noqa: E402

H0 = Planck15.H0.value
OM0 = Planck15.Om0
C_KMS = const.c.to("km/s").value
ZGRID = np.expm1(np.linspace(0.0, np.log(3.0), 10000))          # 0 .. zMax = 2
R_GRID = FlatLambdaCDM(H0=H0, Om0=OM0).comoving_distance(ZGRID).to(u.Mpc).value
DV_GRID = C_KMS * R_GRID ** 2 / (H0 * np.sqrt(OM0 * (1 + ZGRID) ** 3 + 1 - OM0))
ZS_PLOT = np.linspace(0.001, 2.0, 300)


def z_of_dL(dL):
    return np.interp(dL, (1 + ZGRID) * R_GRID, ZGRID)


def dV_normed(z, gamma):
    """p(z) ~ dV_c/dz (1+z)^(gamma-1), normalised on [0, 2]."""
    dv = DV_GRID * (1 + ZGRID) ** (gamma - 1)
    return np.interp(z, ZGRID, dv / np.trapezoid(dv, ZGRID))


# ---------------------------------------------------------- population models
MASS = np.linspace(1, 150, 2000)
QGRID = np.linspace(1e-5, 1, 2000)
M1S = np.linspace(1.0, 100.0, 300)
DM = M1S[1] - M1S[0]
M1G, M2G = np.meshgrid(M1S, M1S, indexing="ij")


def _smooth_low(m, m_min, dm):
    with np.errstate(all="ignore"):
        s = 1.0 / (np.exp(dm / (m - m_min) + dm / (m - m_min - dm)) + 1.0)
    return np.where(m > m_min, np.where(m < m_min + dm, s, 1.0), 0.0)


def _smooth_high(m, m_max, dm):
    with np.errstate(all="ignore"):
        s = 1.0 / (np.exp(-dm / (m - m_max) - dm / (m - m_max + dm)) + 1.0)
    return np.where(m < m_max, np.where(m > m_max - dm, s, 1.0), 0.0)


def _pairing_pm(m, m_min, m_max, alpha, dm_min, dm_max, mu, sigma, f1):
    """Smoothed power law + Gaussian peak for one component mass (pairing model)."""
    pl = _smooth_low(MASS, m_min, dm_min) * MASS ** (-alpha) * _smooth_high(MASS, m_max, dm_max)
    pk = np.exp(-(MASS - mu) ** 2 / (2 * sigma ** 2))
    p = ((1 - f1) * np.interp(m, MASS, pl / np.trapezoid(pl, MASS))
         + f1 * np.interp(m, MASS, pk / np.trapezoid(pk, MASS)))
    return p


def _pairing_logfq(m1, m2, beta):
    pq = QGRID ** beta
    return np.log(np.interp(m2 / m1, QGRID, pq / np.trapezoid(pq, QGRID)))


def _pairing_logp_m1m2(m1, m2, m_min, m_max, alpha, dm_min, dm_max, beta, mu, sigma, f1):
    with np.errstate(divide="ignore"):
        lp = (np.log(_pairing_pm(m1, m_min, m_max, alpha, dm_min, dm_max, mu, sigma, f1))
              + np.log(_pairing_pm(m2, m_min, m_max, alpha, dm_min, dm_max, mu, sigma, f1))
              + _pairing_logfq(m1, m2, beta))
    return np.where(m2 < m1, lp, -np.inf)


def _plpeak_logp_m1m2(m1, m2, m_min, m_max, alpha, beta, mu, sigma, f1):
    """sim_cat6 sharp PL + peak: p(m1) p(q|m1) / m1."""
    a = -alpha
    p_pl = (1 + a) * m1 ** a / (m_max ** (1 + a) - m_min ** (1 + a))
    p_pl = np.where((m1 >= m_min) & (m1 <= m_max), p_pl, 0.0)
    p_pk = np.exp(-0.5 * (m1 - mu) ** 2 / sigma ** 2) / np.sqrt(2 * np.pi * sigma ** 2)
    q = m2 / m1
    p_q = (1 + beta) * q ** beta / (1 - (m_min / m1) ** (1 + beta))
    p_q = np.where(q * m1 < m_min, 0.0, p_q)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(f1 * p_pk + (1 - f1) * p_pl) + np.log(p_q) - np.log(m1)


def log_p_pop(run, m1, m2, z, theta):
    """Population density over source-frame (m1, m2, z), as used in R0."""
    gamma = theta[0]
    if run == "mdc":
        lp = _plpeak_logp_m1m2(m1, m2, *theta[1:]) + np.log(dV_normed(z, gamma))
        return np.where(m2 < m1, lp, -np.inf) - np.log(0.25)
    return np.log(0.25) + _pairing_logp_m1m2(m1, m2, *theta[1:]) + np.log(dV_normed(z, gamma))


def pm1(run, theta):
    """Normalised p(m1) on M1S for one hyperparameter sample (gamma first)."""
    lam = theta[1:]
    if run == "mdc":
        m_min, m_max, alpha, beta, mu, sigma, f1 = lam
        a = -alpha
        p_pl = (1 + a) * M1S ** a / (m_max ** (1 + a) - m_min ** (1 + a))
        p_pl = np.where((M1S >= m_min) & (M1S <= m_max), p_pl, 0.0)
        p_pk = np.exp(-0.5 * (M1S - mu) ** 2 / sigma ** 2) / np.sqrt(2 * np.pi * sigma ** 2)
        return f1 * p_pk + (1 - f1) * p_pl
    # pairing: marginalise the joint over m2 on the 300 x 300 grid
    from scipy.special import logsumexp
    lj = _pairing_logp_m1m2(M1G, M2G, *lam)
    lj -= logsumexp(lj) + np.log(DM * DM)
    return np.exp(logsumexp(lj, axis=1) + np.log(DM))


# --------------------------------------------------------- rate normalisation
class Selection:
    """Found endo3 injections, in source frame, for the epsilon in R0."""

    def __init__(self, path=ENDO3):
        import h5py
        with h5py.File(path, "r") as f:
            self.T_yr = f.attrs["analysis_time_s"] / (365.25 * 24 * 3600)
            self.n_draw = int(f.attrs["total_generated"])
            g = f["injections"]
            found = ((g["ifar_gstlal"][:] > 1) | (g["ifar_pycbc_bbh"][:] > 1)
                     | (g["ifar_pycbc_hyperbank"][:] > 1))
            self.z = z_of_dL(g["distance"][:][found])
            self.m1 = g["mass1"][:][found] / (1 + self.z)
            self.m2 = g["mass2"][:][found] / (1 + self.z)
            self.log_pdraw = np.log(g["sampling_pdf"][:][found])

    def R0(self, run, theta):
        """Local rate [Gpc^-3 yr^-1] = N_obs / (epsilon V_pop T)."""
        from scipy.special import logsumexp
        lw = log_p_pop(run, self.m1, self.m2, self.z, theta) - self.log_pdraw
        eps = np.exp(logsumexp(lw) - np.log(self.n_draw))
        v_pop = np.trapezoid(DV_GRID * (1 + ZGRID) ** (theta[0] - 1), ZGRID)
        return N_OBS / (eps * v_pop * 1e-9 * self.T_yr)


def rate_draws(run, posts, sel, want_m1=True):
    """Per method: thinned R0, dR/dm1 on M1S and R(z) on ZS_PLOT draws."""
    out = {}
    for m, s in posts.items():
        th = s[::THIN[m]]
        R0 = np.array([sel.R0(run, t) for t in th])
        d = {"Rz": R0[:, None] * (1 + ZS_PLOT[None, :]) ** th[:, :1]}
        if want_m1:
            d["Rm1"] = R0[:, None] * np.array([pm1(run, t) for t in th])
        out[m] = d
        print(f"[{m}] {len(th)} draws, median R0 = {np.median(R0):.1f} Gpc^-3 yr^-1")
    return out


def truth_curves(run, sel):
    t = RUNS[run]["truths"]
    if t is None:
        return None
    theta = np.array([t[p] for p in RUNS[run]["params"]])
    R0 = sel.R0(run, theta)
    print(f"true R0 = {R0:.2f} Gpc^-3 yr^-1")
    return {"Rm1": R0 * pm1(run, theta), "Rz": R0 * (1 + ZS_PLOT) ** theta[0]}


def trace_index(n, n_lines=100):
    """make_plots_met4.py's trace-line pick: legacy np.random.seed(42)."""
    return np.random.RandomState(42).choice(n, size=min(n_lines, n), replace=False)
