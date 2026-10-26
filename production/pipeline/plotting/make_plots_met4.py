import argparse
from pathlib import Path
import os
import re
import math
import jax
from jax import random, jit, vmap, grad
from jax import numpy as jnp
from jax.lax import cond
import astropy
import numpy as np
import healpy as hp
import h5py
import astropy.units as u
from astropy.cosmology import Planck15, FlatLambdaCDM, z_at_value
import astropy.constants as constants
from jax.scipy.special import logsumexp
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from jaxinterp2d import interp2d, CartesianGrid
import dynesty
from dynesty import utils as dyfunc
from dynesty.utils import resample_equal
import corner
import scipy.special as jsp 

# -----------------------------------------------------------------------------
# NumPy / dynesty pickle compatibility shim
# -----------------------------------------------------------------------------
# Older NumPy versions pickled BitGenerator/RandomState constructors with the
# *class object* instead of its string name. Newer NumPy's
# `numpy.random._pickle.__bit_generator_ctor` expects a string, so loading an
# old dynesty sampler raises:
#   ValueError: <class 'numpy.random._pcg64.PCG64'> is not a known BitGenerator
# We wrap the ctor to accept either form. Best-effort: if numpy internals have
# moved, just let the original error surface during restore.
# try:
#     import numpy.random._pickle as _np_pickle_mod

#     _orig_bg_ctor = getattr(_np_pickle_mod, "__bit_generator_ctor")
#     def _bg_ctor_compat(bit_generator="MT19937"):
#         if isinstance(bit_generator, type):
#             bit_generator = bit_generator.__name__
#         return _orig_bg_ctor(bit_generator)
#     setattr(_np_pickle_mod, "__bit_generator_ctor", _bg_ctor_compat)

#     if hasattr(_np_pickle_mod, "__randomstate_ctor"):
#         _orig_rs_ctor = getattr(_np_pickle_mod, "__randomstate_ctor")
#         def _rs_ctor_compat(bit_generator="MT19937"):
#             if isinstance(bit_generator, type):
#                 bit_generator = bit_generator.__name__
#             return _orig_rs_ctor(bit_generator)
#         setattr(_np_pickle_mod, "__randomstate_ctor", _rs_ctor_compat)
# except Exception:
#     pass
# -----------------------------------------------------------------------------

# ==========================================================
# 1. Setup & Configuration
# ==========================================================
def parse_args():
    p = argparse.ArgumentParser(description="Plotting from cached samplers")
    p.add_argument("--indir", type=Path, required=True, help="Directory with sampler outputs")
    p.add_argument("--outdir", type=Path, required=True, help="Directory for plots")
    p.add_argument("--run", type=str, required=True, default="gwtc3", help="gwtc3/mdc")
    p.add_argument("--model", type=str, default="pairing", help="lvk/pairing")
    p.add_argument("--seed", type=int, default=None,
                   help="Realization seed; auto-detected from --indir if omitted")
    p.add_argument(
        "--trace-method",
        type=str,
        default="mixture",
        choices=["pe", "proposal", "mixture", "none"],
        help="Which method's posterior draws to overlay as trace lines (default: mixture).",
    )
    return p.parse_args()

def _infer_seed(indir: Path) -> int:
    """Find the realization seed from met1 filenames already in indir."""
    candidates = set()
    for f in list(indir.glob("*method1.pkl")) + list(indir.glob("*met1.pkl")):
        m = re.search(r"(\d+)", f.stem)
        if m:
            candidates.add(int(m.group(1)))
    if len(candidates) == 1:
        return candidates.pop()
    if not candidates:
        raise FileNotFoundError(
            f"Could not auto-detect a seed from met1 files in {indir}; pass --seed explicitly."
        )
    raise ValueError(
        f"Multiple realizations found in {indir} ({sorted(candidates)}); pass --seed explicitly."
    )

args = parse_args()
args.outdir.mkdir(parents=True, exist_ok=True)
if args.seed is None:
    args.seed = _infer_seed(args.indir)
    print(f"Auto-detected seed: {args.seed}")

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Computer Modern Roman']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['figure.figsize'] = (16.0, 10.0)
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.labelsize'] = 20

sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('colorblind')
c = sns.color_palette('colorblind')

def _log_tex_formatter(y, pos):
    # matplotlib's usetex + PDF backend drops the math-mode minus glyph
    # (e.g. "$10^{-3}$") -- wrapping it in \mbox{-} renders correctly.
    if y <= 0:
        return ''
    exp = int(round(math.log10(y)))
    if exp < 0:
        return r'$10^{\mbox{-}%d}$' % abs(exp)
    return r'$10^{%d}$' % exp

jax.config.update("jax_enable_x64", False) 
jax.config.update('jax_default_matmul_precision', 'tensorfloat32') 

def _restore_dynesty_sampler(path: Path, *, required: bool = True):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing sampler file: {path}")
        return None
    return dynesty.NestedSampler.restore(str(path))

def _resolve_sampler_path(indir: Path, run: str, seed: int, method: int) -> Path:
    """Try the naming conventions this pipeline has used for cached samplers."""
    candidates = [
        indir / f"{run}_{seed}met{method}.pkl",
        indir / f"{run}_{seed}_met{method}.pkl",
        indir / f"{seed}method{method}.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

# ==========================================================
# 2. Cosmology & Physics Grids
# ==========================================================
H0Planck = Planck15.H0.value
Om0Planck = Planck15.Om0
speed_of_light = constants.c.to('km/s').value

zMax = 2.0
zgrid = jnp.expm1(np.linspace(np.log(1), np.log(zMax+1), 10000))
Om0grid = jnp.linspace(0,1,1000)

rs = []
for Om0 in tqdm(Om0grid, desc="Building Cosmo Grid"):
    cosmo = FlatLambdaCDM(H0=H0Planck,Om0=Om0)
    rs.append(cosmo.comoving_distance(zgrid).to(u.Mpc).value)

rs = jnp.asarray(rs).reshape(len(Om0grid),len(zgrid))

@jit
def E(z,Om0=Om0Planck):
    return jnp.sqrt(Om0*(1+z)**3 + (1.0-Om0))

@jit
def r_of_z(z,H0,Om0=Om0Planck):
    return interp2d(Om0,z,Om0grid,zgrid,rs)*(H0Planck/H0)

@jit
def dL_of_z(z,H0,Om0=Om0Planck):
    return (1+z)*r_of_z(z,H0,Om0)

@jit
def z_of_dL(dL,H0,Om0=Om0Planck):
    return jnp.interp(dL,dL_of_z(zgrid,H0,Om0),zgrid)

@jit
def dV_of_z(z,H0,Om0=Om0Planck):
    return speed_of_light*r_of_z(z,H0,Om0)**2/(H0*E(z,Om0))

@jit
def dV_of_z_normed(z,Om0,gamma):
    dV = dV_of_z(zgrid,H0Planck,Om0)*(1+zgrid)**(gamma-1)
    prob = dV/jnp.trapezoid(dV,zgrid)
    return jnp.interp(z,zgrid,prob)

@jit
def ddL_of_z(z,dL,H0,Om0=Om0Planck):
    return dL/(1+z) + speed_of_light*(1+z)/(H0*E(z,Om0))

# Redshift plotting grid
zs_plot = jnp.linspace(0.001, 2.0, 300)

@jit
def pz_from_gamma(gamma):
    return dV_of_z_normed(zs_plot, Om0Planck, gamma)

V_total_Mpc3 = jnp.trapezoid(dV_of_z(zgrid, H0Planck, Om0Planck), zgrid)

@jit
def get_R0_from_lambda(coord):
    # Unpack the posterior sample (coord)
    if args.run == "mdc":
        gamma, m_min, m_max, alpha, beta, mu, sigma, f1 = coord
        # Use MDC-specific population model
        log_det_weights = log_p_pop_lvk(
            m1sels, m2sels, zsels, gamma, 
            m_min, m_max, alpha, beta, mu, sigma, f1
        )
    else:
        gamma, m_min, m_max, alpha, dm_min, dm_max, beta, mu, sigma, f1 = coord
        log_det_weights = log_p_pop_pl_pl(
            m1sels, m2sels, zsels, gamma, 
            m_min, m_max, alpha, dm_min, dm_max, 
            beta, mu, sigma, f1
        )

    # Selection effects normalization (identical to your likelihood script)
    log_det_weights += - jnp.log(p_draw) 
    epsilon = jnp.exp(logsumexp(log_det_weights) - jnp.log(Ndraw))
    
    # Efficiency epsilon (dimensionless fraction of the population detected)
    dV_weighted = dV_of_z(zgrid, H0Planck, Om0Planck) * (1 + zgrid)**(gamma - 1)
    V_pop_Mpc3 = jnp.trapezoid(dV_weighted, zgrid)
    
    # VT = efficiency * Population_Volume * Time
    VT_Gpc3_yr = epsilon * (V_pop_Mpc3 * 1e-9) * Tobs
    
    # R0 [Gpc^-3 yr^-1] = Nobs / VT
    return 69 / VT_Gpc3_yr

# --- Add this near your Configuration section ---
INJECTION_FILE = 'input_data/endo3_bbhpop-LIGO-T2100113-v12.hdf5'
with h5py.File(INJECTION_FILE, 'r') as f:
    # Tobs in years
    Tobs = f.attrs['analysis_time_s'] / (365.25 * 24 * 3600)
    Ndraw = int(f.attrs['total_generated'])   # NEVER overwrite this

    m1detsels_all    = f['injections/mass1'][:]
    m2detsels_all    = f['injections/mass2'][:]
    dLsels_all       = f['injections/distance'][:]
    p_draw_all       = f['injections/sampling_pdf'][:]
    ifar_gstlal      = f['injections/ifar_gstlal'][:]
    ifar_pycbc_bbh   = f['injections/ifar_pycbc_bbh'][:]
    ifar_pycbc_broad = f['injections/ifar_pycbc_hyperbank'][:]

sels    = (ifar_gstlal > 1) | (ifar_pycbc_bbh > 1) | (ifar_pycbc_broad > 1)
N_found = int(sels.sum())

m1detsels = jnp.asarray(m1detsels_all[sels])
m2detsels = jnp.asarray(m2detsels_all[sels])
dLsels    = jnp.asarray(dLsels_all[sels])
p_draw    = jnp.asarray(p_draw_all[sels])

zsels  = z_of_dL(dLsels, H0Planck, Om0Planck)
m1sels = m1detsels / (1 + zsels)
m2sels = m2detsels / (1 + zsels)


# Number of events (e.g., 69 for GWTC-3)
Nobs = 69

# ==========================================================
# 3. Mass Population Models
# ==========================================================
mass = jnp.linspace(1, 150, 2000)
mass_ratio = jnp.linspace(1e-5, 1, 2000)
m1s = jnp.linspace(1.0, 100.0, 300, dtype=jnp.float64)
m2s = jnp.linspace(1.0, 100.0, 300, dtype=jnp.float64)
dm1 = m1s[1] - m1s[0]
dm2 = m2s[1] - m2s[0]
M1, M2 = jnp.meshgrid(m1s, m2s, indexing="ij")

def Sfilter_low(m,m_min,dm_min):
    def f(mm,deltaMM):
        return jnp.exp(deltaMM/mm + deltaMM/(mm-deltaMM))
    S_filter = 1./(f(m-m_min,dm_min) + 1.)
    S_filter = jnp.where(m<m_min+dm_min,S_filter,1.)
    S_filter = jnp.where(m>m_min,S_filter,0.)
    return S_filter

def Sfilter_high(m,m_max,dm_max):
    def f(mm,deltaMM):
        return jnp.exp(deltaMM/mm + deltaMM/(mm-deltaMM))
    S_filter = 1./(f(m-m_max,-dm_max) + 1.)
    S_filter = jnp.where(m>m_max-dm_max,S_filter,1.)
    S_filter = jnp.where(m<m_max,S_filter,0.)
    return S_filter

@jit
def logpm1_powerlaw(m1,m_min,m_max,alpha,dm_min,dm_max):
    pm1 = Sfilter_low(mass,m_min,dm_min)*mass**(-alpha)*Sfilter_high(mass,m_max,dm_max)
    pm1 = pm1/jnp.trapezoid(pm1,mass)
    return jnp.log(jnp.interp(m1,mass,pm1))

@jit
def logpm1_peak(m1,mu,sigma):
    pm1 =  jnp.exp(-(mass - mu)**2 / (2 * sigma ** 2))
    pm1 = pm1/jnp.trapezoid(pm1,mass)
    return jnp.log(jnp.interp(m1,mass,pm1))

@jit
def logpm1_powerlaw_powerlaw(m1,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,mu,sigma,f1):
    p1 = jnp.exp(logpm1_powerlaw(m1,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1))
    p2 = jnp.exp(logpm1_peak(m1,mu,sigma))
    pm1 = (1-f1)*p1 + f1*p2
    return jnp.log(pm1)

@jit
def logfq(m1,m2,beta):
    q = m2/m1
    pq = mass_ratio**beta
    pq = pq/jnp.trapezoid(pq,mass_ratio)
    log_pq = jnp.log(jnp.interp(q,mass_ratio,pq))
    return log_pq

@jit
def log_pm1m2(m1, m2, m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1):
    log_dNdm1 = logpm1_powerlaw_powerlaw(m1,m_min,m_max,alpha,dm_min,dm_max,mu,sigma,f1)
    log_dNdm2 = logpm1_powerlaw_powerlaw(m2,m_min,m_max,alpha,dm_min,dm_max,mu,sigma,f1)
    log_pq = logfq(m1,m2,beta)
    log_p = log_dNdm1 + log_dNdm2 + log_pq
    log_p = jnp.where(m2<m1, log_p, -jnp.inf)
    return log_p

# --- LVK / Pop functions ---
@jit
def log_p_pop_pl_pl(m1,m2,z,gamma, m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,beta,mu,sigma,f1):
    log_dNdm1 = logpm1_powerlaw_powerlaw(m1,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,mu,sigma,f1)
    log_dNdm2 = logpm1_powerlaw_powerlaw(m2,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,mu,sigma,f1)
    log_pq = logfq(m1,m2,beta)
    log_dvdz = jnp.log(dV_of_z_normed(z,Om0Planck,gamma))
    log_p_sz = np.log(0.25) 
    log_p = log_p_sz + log_dNdm1 + log_dNdm2 + log_pq + log_dvdz
    log_p = jnp.where(m2<m1, log_p, -jnp.inf)
    return log_p

# --- MDC 8-Parameter functions ---
@jit
def logpm1m2_plpeak_massratio(m1, m2, m_min_1, m_max_1, alpha_1, beta, mu, sigma, f):
    q = m2/m1
    alpha_1 = -alpha_1
    norm_pl = (m_max_1**(1. + alpha_1) - m_min_1**(1. + alpha_1))
    p_m1_pl = (1. + alpha_1) * m1**alpha_1 / norm_pl
    p_m1_pl = jnp.where(m1 > m_max_1, 0.0, p_m1_pl)
    p_m1_pl = jnp.where(m1 < m_min_1, 0.0, p_m1_pl)

    p_m1_peak = jnp.exp(-0.5 * (m1 - mu)**2 / sigma**2) / jnp.sqrt(2. * jnp.pi * sigma**2)
    p_m1 = f * p_m1_peak + (1. - f) * p_m1_pl

    q_min = m_min_1/m1
    denom = 1 - q_min**(1. + beta)
    p_q = (1. + beta) * q**beta / denom
    p_q = jnp.where(q*m1 < m_min_1, 0.0, p_q)

    return jnp.log(p_m1) + jnp.log(p_q) - jnp.log(m1)

@jit
def log_p_pop_lvk(m1,m2,z,gamma, m_min_1,m_max_1,alpha_1,beta,mu,sigma,f1):
    log_joint_mass = logpm1m2_plpeak_massratio(m1, m2, m_min_1, m_max_1, alpha_1, beta, mu, sigma, f1)
    log_pz = jnp.log(dV_of_z_normed(z,Om0Planck,gamma))
    log_p = log_joint_mass + log_pz
    log_p = jnp.where(m2<m1, log_p, -jnp.inf)
    return log_p - jnp.log(0.25)

# ==========================================================
# 4. Priors and Logic Switcher (MDC vs GWTC3)
# ==========================================================
gamma_low, gamma_high = 0, 10
m_min_1_low, m_min_1_high = 2, 10
m_max_1_low, m_max_1_high = 50, 100
alpha_1_low, alpha_1_high = 0, 6
beta_low, beta_high = 0, 6
mu_low, mu_high = 20, 50
sigma_low, sigma_high = 1, 10
f1_low, f1_high = 0, 1

if args.run == "mdc":
    labels = ['gamma', 'm_min_1', 'm_max_1', 'alpha_1', 'beta', 'mu', 'sigma', 'f1']
    lower_bound = np.array([gamma_low, m_min_1_low, m_max_1_low, alpha_1_low, beta_low, mu_low, sigma_low, f1_low])
    upper_bound = np.array([gamma_high, m_min_1_high, m_max_1_high, alpha_1_high, beta_high, mu_high, sigma_high, f1_high])

    @jit
    def pm1_from_lambda(lam):
        m_min_1, m_max_1, alpha_1, beta, mu, sigma, f1 = lam
        alpha_1 = -alpha_1
        norm_pl = (m_max_1**(1. + alpha_1) - m_min_1**(1. + alpha_1))
        p_m1_pl = (1. + alpha_1) * m1s**alpha_1 / norm_pl
        p_m1_pl = jnp.where((m1s >= m_min_1) & (m1s <= m_max_1), p_m1_pl, 0.0)
        p_m1_peak = jnp.exp(-0.5 * (m1s - mu)**2 / sigma**2) / jnp.sqrt(2. * jnp.pi * sigma**2)
        return f1 * p_m1_peak + (1. - f1) * p_m1_pl

    @jit
    def pm2_from_lambda(lam):
        m_min_1, m_max_1, alpha_1, beta, mu, sigma, f1 = lam
        log_p_m1_q = logpm1m2_plpeak_massratio(M1, M2, m_min_1, m_max_1, alpha_1, beta, mu, sigma, f1)
        log_p_m1_m2 = log_p_m1_q - jnp.log(M1) 
        log_p_m1_m2 = jnp.where((M2 <= M1) & (M2 >= m_min_1), log_p_m1_m2, -jnp.inf)
        log_norm = logsumexp(log_p_m1_m2) + jnp.log(dm1 * dm2)
        log_pjoint = log_p_m1_m2 - log_norm
        log_pm2 = logsumexp(log_pjoint, axis=0) + jnp.log(dm1)
        return jnp.exp(log_pm2)

else:
    dm_min_1_low, dm_min_1_high = 1, 100
    dm_max_1_low, dm_max_1_high = 1, 100
    labels = ['gamma', 'm_min_1', 'm_max_1', 'alpha_1', 'dm_min_1', 'dm_max_1', 'beta', 'mu', 'sigma', 'f1']
    lower_bound = np.array([gamma_low, m_min_1_low, m_max_1_low, alpha_1_low, dm_min_1_low, dm_max_1_low, beta_low, mu_low, sigma_low, f1_low])
    upper_bound = np.array([gamma_high, m_min_1_high, m_max_1_high, alpha_1_high, dm_min_1_high, dm_max_1_high, beta_high, mu_high, sigma_high, f1_high])

    @jit
    def pm1_from_lambda(lam):
        m_min, m_max, alpha, dm_min, dm_max, beta, mu, sigma, f1 = lam
        log_joint = log_pm1m2(M1, M2, m_min, m_max, alpha, dm_min, dm_max, beta, mu, sigma, f1)
        log_norm = logsumexp(log_joint) + jnp.log(dm1 * dm2)
        log_pjoint = log_joint - log_norm
        log_pm1 = logsumexp(log_pjoint, axis=1) + jnp.log(dm2)
        return jnp.exp(log_pm1)
        
    @jit
    def pm2_from_lambda(lam):
        m_min, m_max, alpha, dm_min, dm_max, beta, mu, sigma, f1 = lam
        log_joint = log_pm1m2(M1, M2, m_min, m_max, alpha, dm_min, dm_max, beta, mu, sigma, f1)
        log_norm = logsumexp(log_joint) + jnp.log(dm1 * dm2)
        log_pjoint = log_joint - log_norm
        log_pm2 = logsumexp(log_pjoint, axis=0) + jnp.log(dm1)
        return jnp.exp(log_pm2)

ndims = len(lower_bound)

latex_labels = {
    'gamma': r'$\gamma$',
    'm_min_1': r'$m_{\mathrm{min}}$',
    'm_max_1': r'$m_{\mathrm{max}}$',
    'alpha_1': r'$\alpha$',
    'dm_min_1': r'$\delta m_{\mathrm{min}}$',
    'dm_max_1': r'$\delta m_{\mathrm{max}}$',
    'beta': r'$\beta$',
    'mu': r'$\mu$',
    'sigma': r'$\sigma$',
    'f1': r'$\lambda$'
}
labels_latex = [latex_labels[l] for l in labels]

# ==========================================================
# 5. Likelihood & Prior Definitions 
# ==========================================================
def prior_transform(theta):
    transformed_params = [
        theta[i] * (upper_bound[i] - lower_bound[i]) + lower_bound[i]
        for i in range(len(theta))
    ]
    return tuple(transformed_params)

@jit
def likelihood_method_1(gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1):
    log_det_weights = log_p_pop_pl_pl(m1sels,m2sels,zsels,gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1)
    log_det_weights += - jnp.log(p_draw) - 2*jnp.log1p(zsels) - jnp.log(ddL_of_z(zsels,dLsels,H0Planck, Om0Planck))
    log_mu = logsumexp(log_det_weights) - jnp.log(Ndraw)
    log_s2 = logsumexp(2*log_det_weights) - 2.0*jnp.log(Ndraw)
    log_sigma2 = logdiffexp(log_s2, 2.0*log_mu - jnp.log(Ndraw))
    Neff = jnp.exp(2.0*log_mu - log_sigma2)
    ll = -jnp.inf
    ll = jnp.where((Neff <= 4 * Nobs), ll, 0)
    ll += -Nobs*log_mu + Nobs*(3 + Nobs)/(2*Neff)
    log_weights = log_p_pop_pl_pl(m1,m2,z,gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1)
    log_weights += - jnp.log(ddL_of_z(z,dL,H0Planck,Om0Planck)) - 2 * jnp.log1p(z) - 2*jnp.log(dL)
    log_weights = log_weights.reshape((Nobs,nsamp))
    ll += jnp.sum(-jnp.log(nsamp) + logsumexp(log_weights,axis=-1))
    return ll, Neff

def loglike_method_1(coord):
    if len(coord) == 8:
        gamma, m_min, m_max, alpha, beta, mu, sigma, f1 = coord
        dm_min, dm_max = 1.0, 1.0 
    else:
        gamma, m_min, m_max, alpha, dm_min, dm_max, beta, mu, sigma, f1 = coord
    ll, Neff = likelihood_method_1(gamma, m_min, m_max, alpha, dm_min, dm_max, beta, mu, sigma, f1)
    if np.isnan(ll) or (Neff < 4*Nobs):
        return -np.inf
    return ll, blob

@jit
def likelihood_method_3(gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1):
    log_det_weights = log_p_pop_pl_pl(m1sels,m2sels,zsels,gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1)
    log_det_weights += - 2*jnp.log1p(zsels) - jnp.log(ddL_of_z(zsels,dL_sam,H0Planck, Om0Planck))
    log_mu = logsumexp(log_det_weights) - jnp.log(Ndraw)
    log_s2 = logsumexp(2*log_det_weights) - 2.0*jnp.log(Ndraw)
    log_sigma2 = logdiffexp(log_s2, 2.0*log_mu - jnp.log(Ndraw))
    Neff = jnp.exp(2.0*log_mu - log_sigma2)
    ll = -jnp.inf
    ll = jnp.where((Neff <= 4 * Nobs), ll, 0)
    ll += -Nobs*log_mu + Nobs*(3 + Nobs)/(2*Neff)
    log_pop = log_p_pop_pl_pl(m1s_pop,m2s_pop,zs_pop,gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1)
    log_weights = log_pop - log_q
    per_event_log_like = jnp.nan_to_num(jsp.logsumexp(log_weights[None, :] + logp_E_N + logJ[None, :], axis=1)) - jnp.log(nsamp_pop)
    ll += jnp.sum(per_event_log_like)
    return ll, Neff

def loglike_method_3(coord):
    if len(coord) == 8:
        gamma, m_min, m_max, alpha, beta, mu, sigma, f1 = coord
        dm_min, dm_max = 1.0, 1.0
    else:
        gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1 = coord
    ll, Neff = likelihood_method_3(gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1)
    if np.isnan(ll) or (Neff < 4*Nobs):
        return -np.inf
    return ll, blob

def loglike_method_4(coord):
    if len(coord) == 8:
        gamma, m_min, m_max, alpha, beta, mu, sigma, f1 = coord
        dm_min, dm_max = 1.0, 1.0
    else:
        gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1 = coord
    ll, Neff = likelihood_method_3(gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1)
    if np.isnan(ll) or (Neff < 4*Nobs):
        return -np.inf
    return ll, blob

# ==========================================================
# 6. Load Dynesty Samples & Plot Posteriors
# ==========================================================
truths = None
if args.run == "mdc":
    true_params_dict = {
        'gamma': 3.0, 
        'm_min_1': 6.0, 
        'm_max_1': 70.0, 
        'alpha_1': 4.0, 
        'beta': 1.5, 
        'mu': 35.0, 
        'sigma': 4.0, 
        'f1': 0.04
    }
    truths = [true_params_dict[l] for l in labels]

# Load saved sampler met1
sampler_file = _resolve_sampler_path(args.indir, args.run, args.seed, 1)
sampler = _restore_dynesty_sampler(sampler_file, required=True)
dres = sampler.results
dweights = np.exp(dres['logwt'] - dres['logz'][-1])
dpostsamples = resample_equal(dres.samples, dweights)

# Corner Plot with Truths (all parameters, one realization)
fig1 = corner.corner(
    dpostsamples,
    labels=labels_latex,
    truths=truths,
    truth_color='black',
    hist_kwargs={'density': True},
    color=c[0],
    label_kwargs={'fontsize': 20}
)

# Get mass and redshift parameters, slice indexing accordingly
lambda_samples = jnp.asarray(dpostsamples[::10, 1:], dtype=jnp.float64)  # M1, M2 hyperparams
gamma_samples = jnp.asarray(dpostsamples[::10, 0], dtype=jnp.float64)    # Gamma

# pm1
pm1_all = jax.jit(vmap(pm1_from_lambda))(lambda_samples)
p5, p50, p95 = jnp.percentile(pm1_all, jnp.array([5, 50, 95]), axis=0)

# pm2
pm2_all = jax.jit(vmap(pm2_from_lambda))(lambda_samples)
p5_m2, p50_m2, p95_m2 = jnp.percentile(pm2_all, jnp.array([5, 50, 95]), axis=0)

# pz (redshift)
pz_all = jax.jit(vmap(pz_from_gamma))(gamma_samples)
p5_z, p50_z, p95_z = jnp.percentile(pz_all, jnp.array([5, 50, 95]), axis=0)

# Sample 100 random indices for the trace lines
np.random.seed(42)  
n_realizations = 100
idx_100 = np.random.choice(len(lambda_samples), size=n_realizations, replace=False)
print("Truths: ", truths)

@jit
def get_Rz_from_gamma(gamma, R0_sample):
    # zs_plot is your redshift grid (0 to 2.0)
    # R(z) = R0 * (1+z)^gamma
    return R0_sample * (1 + zs_plot)**gamma

R0_samples = jax.jit(vmap(get_R0_from_lambda))(jnp.asarray(dpostsamples[::10]))
dR_dm1_all = R0_samples[:, jnp.newaxis] * pm1_all
dR_dm2_all = R0_samples[:, jnp.newaxis] * pm2_all
p5_R, p50_R, p95_R = jnp.percentile(dR_dm1_all, jnp.array([5, 50, 95]), axis=0)

Rz_all = jax.jit(vmap(get_Rz_from_gamma))(gamma_samples, R0_samples)
p5_Rz, p50_Rz, p95_Rz = jnp.percentile(Rz_all, jnp.array([5, 50, 95]), axis=0)

# ==========================================================
# 7. Additional Samplers (Optional Sel/Events)
# ==========================================================
_met3 = _restore_dynesty_sampler(_resolve_sampler_path(args.indir, args.run, args.seed, 3), required=False)
if _met3 is not None:
    d3res = _met3.results
    d3weights = np.exp(d3res['logwt'] - d3res['logz'][-1])
    d3postsamples = resample_equal(d3res.samples, d3weights)
    
    lambda_samples3 = jnp.asarray(d3postsamples[::5, 1:], dtype=jnp.float64)
    gamma_samples3 = jnp.asarray(d3postsamples[::5, 0], dtype=jnp.float64)
    
    pm1_all3 = jax.jit(vmap(pm1_from_lambda))(lambda_samples3)
    p5_3, p50_3, p95_3 = jnp.percentile(pm1_all3, jnp.array([5, 50, 95]), axis=0)
    
    pm2_all3 = jax.jit(vmap(pm2_from_lambda))(lambda_samples3)
    p5_3_m2, p50_3_m2, p95_3_m2 = jnp.percentile(pm2_all3, jnp.array([5, 50, 95]), axis=0)

    pz_all3 = jax.jit(vmap(pz_from_gamma))(gamma_samples3)
    p5_3_z, p50_3_z, p95_3_z = jnp.percentile(pz_all3, jnp.array([5, 50, 95]), axis=0)

    # merger rate
    R0_samples3 = jax.jit(vmap(get_R0_from_lambda))(jnp.asarray(d3postsamples[::5]))
    dR_dm1_all3 = R0_samples3[:, jnp.newaxis] * pm1_all3
    dR_dm2_all3 = R0_samples3[:, jnp.newaxis] * pm2_all3
    p5_R3, p50_R3, p95_R3 = jnp.percentile(dR_dm1_all3, jnp.array([5, 50, 95]), axis=0)

    Rz_all3 = jax.jit(vmap(get_Rz_from_gamma))(gamma_samples3, R0_samples3)
    p5_Rz3, p50_Rz3, p95_Rz3 = jnp.percentile(Rz_all3, jnp.array([5, 50, 95]), axis=0)
    corner.corner(
        d3postsamples,
        labels=labels_latex,
        truths=truths,
        truth_color='black',
        hist_kwargs={'density': True},
        color=c[1],
        fig=fig1
    )

_met4 = _restore_dynesty_sampler(_resolve_sampler_path(args.indir, args.run, args.seed, 4), required=False)
if _met4 is not None:
    d4res = _met4.results
    d4weights = np.exp(d4res['logwt'] - d4res['logz'][-1])
    d4postsamples = resample_equal(d4res.samples, d4weights)

    lambda_samples4 = jnp.asarray(d4postsamples[::5, 1:], dtype=jnp.float64)
    gamma_samples4 = jnp.asarray(d4postsamples[::5, 0], dtype=jnp.float64)

    pm1_all4 = jax.jit(vmap(pm1_from_lambda))(lambda_samples4)
    p5_4, p50_4, p95_4 = jnp.percentile(pm1_all4, jnp.array([5, 50, 95]), axis=0)

    pm2_all4 = jax.jit(vmap(pm2_from_lambda))(lambda_samples4)
    p5_4_m2, p50_4_m2, p95_4_m2 = jnp.percentile(pm2_all4, jnp.array([5, 50, 95]), axis=0)

    pz_all4 = jax.jit(vmap(pz_from_gamma))(gamma_samples4)
    p5_4_z, p50_4_z, p95_4_z = jnp.percentile(pz_all4, jnp.array([5, 50, 95]), axis=0)

    # merger rate
    R0_samples4 = jax.jit(vmap(get_R0_from_lambda))(jnp.asarray(d4postsamples[::5]))
    dR_dm1_all4 = R0_samples4[:, jnp.newaxis] * pm1_all4
    dR_dm2_all4 = R0_samples4[:, jnp.newaxis] * pm2_all4
    p5_R4, p50_R4, p95_R4 = jnp.percentile(dR_dm1_all4, jnp.array([5, 50, 95]), axis=0)

    Rz_all4 = jax.jit(vmap(get_Rz_from_gamma))(gamma_samples4, R0_samples4)
    p5_Rz4, p50_Rz4, p95_Rz4 = jnp.percentile(Rz_all4, jnp.array([5, 50, 95]), axis=0)

    corner.corner(
        d4postsamples,
        labels=labels_latex,
        truths=truths,
        truth_color='black',
        hist_kwargs={'density': True},
        color=c[2],
        fig=fig1
    )

from matplotlib.lines import Line2D
legend_elements_full = [Line2D([0], [0], color=c[0], lw=2, label=r'$\mathrm{PE}$')]
if _met3 is not None:
    legend_elements_full.append(Line2D([0], [0], color=c[1], lw=2, label=r'$\mathrm{Ensemble}$'))
if _met4 is not None:
    legend_elements_full.append(Line2D([0], [0], color=c[2], lw=2, label=r'$\mathrm{Mixture}$'))
fig1.legend(handles=legend_elements_full, loc='upper right', bbox_to_anchor=(0.9, 0.9), fontsize=20)
fig1.savefig(args.outdir / f"{args.run}_{args.seed}_corner_all.pdf", dpi=200)
plt.close(fig1)

if args.run == "mdc":
    # The two subsets partition the 8 mdc hyperparameters exactly once each --
    # no parameter appears in both panels, none is left to corner_all alone.
    # Subset 1: gamma, beta, f1
    idx_sub1 = [0, 4, 7]
    # Subset 2: alpha, m_min, m_max, mu, sigma
    idx_sub2 = [3, 1, 2, 5, 6]
else:
    # Subset 1: gamma, beta, mu, sigma, f1
    idx_sub1 = [0, 6, 7, 8, 9]
    # Subset 2: alpha, m_min, m_max, dm_min, dm_max
    idx_sub2 = [3, 1, 2, 4, 5]

labels_sub1 = [latex_labels[labels[i]] for i in idx_sub1]
labels_sub2 = [latex_labels[labels[i]] for i in idx_sub2]

truths_sub1 = [truths[i] for i in idx_sub1] if truths else None
truths_sub2 = [truths[i] for i in idx_sub2] if truths else None
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color=c[0], lw=2, label='PE'),
    Line2D([0], [0], color=c[1], lw=2, label='Ensemble'),
    Line2D([0], [0], color=c[2], lw=2, label='Mixture'),
]
# --- Subset 1 Plot ---
fig_sub1 = corner.corner(
    dpostsamples[:, idx_sub1],
    labels=labels_sub1,
    truths=truths_sub1,
    truth_color='black',
    hist_kwargs={'density': True},
    color=c[0],
    label_kwargs={'fontsize': 20}
)
if _met3 is not None:
    corner.corner(
        d3postsamples[:, idx_sub1],
        color=c[1],
        hist_kwargs={'density': True},
        fig=fig_sub1
    )
if _met4 is not None:
    corner.corner(
        d4postsamples[:, idx_sub1],
        color=c[2],
        hist_kwargs={'density': True},
        fig=fig_sub1
    )
fig_sub1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.9, 0.9), fontsize=20)
fig_sub1.savefig(args.outdir / f"{args.run}_subcorner_pos1.png", dpi=200)
plt.close(fig_sub1)

# --- Subset 2 Plot ---
fig_sub2 = corner.corner(
    dpostsamples[:, idx_sub2],
    labels=labels_sub2,
    truths=truths_sub2,
    truth_color='black',
    hist_kwargs={'density': True},
    color=c[0],
    label_kwargs={'fontsize': 20}
)
if _met3 is not None:
    corner.corner(
        d3postsamples[:, idx_sub2],
        color=c[1],
        hist_kwargs={'density': True},
        fig=fig_sub2
    )
if _met4 is not None:
    corner.corner(
        d4postsamples[:, idx_sub2],
        color=c[2],
        hist_kwargs={'density': True},
        fig=fig_sub2
    )

# 4. Add the legend to the FIGURE, not the last axis
# You can adjust loc and bbox_to_anchor to place it in the empty top-right area
fig_sub2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.9, 0.9), fontsize=20)

fig_sub2.savefig(args.outdir / f"{args.run}_subcorner_pos2.png", dpi=200)
plt.close(fig_sub2)

# ==========================================================
# 8. Final Output Plots
# ==========================================================
# Calculate True MDC Values
true_pm1 = None
true_pm2 = None
true_pz  = None

if args.run == "mdc":
    true_coord = jnp.array([
        true_params_dict['gamma'], true_params_dict['m_min_1'], 
        true_params_dict['m_max_1'], true_params_dict['alpha_1'], 
        true_params_dict['beta'], true_params_dict['mu'], 
        true_params_dict['sigma'], true_params_dict['f1']
    ])
    
    # 1. Calculate the True R0 for these parameters
    true_R0 = get_R0_from_lambda(true_coord)
    
    # 2. Scale True p(m1) to dR/dm1
    true_pm1 = pm1_from_lambda(true_coord[1:]) * true_R0
    
    # 3. Scale True p(z) to R(z)
    # R(z) = R0 * (1+z)^gamma
    true_pz = true_R0 * (1 + zs_plot)**true_params_dict['gamma']
    
    print(f"True R0 for MDC: {true_R0:.2f} Gpc^-3 yr^-1")

# Select which method's draws to use for the trace lines
# trace_pm1 = trace_pm2 = trace_pz = None
# trace_color = c[0]

# if args.trace_method == "pe":
#     # trace_pm1, trace_pm2, trace_pz = pm1_all, pm2_all, pz_all
#     trace_pm1, trace_pz = p50_R, p50_Rz
#     trace_color = c[0]
# elif args.trace_method == "proposal" and _met3 is not None:
#     # trace_pm1, trace_pm2, trace_pz = pm1_all3, pm2_all3, pz_all3
#     trace_pm1, trace_pz = p50_R3, p50_Rz3
#     trace_color = c[1]
# elif args.trace_method == "mixture" and _met4 is not None:
#     # trace_pm1, trace_pm2, trace_pz = pm1_all4, pm2_all4, pz_all4
#     trace_pm1, trace_pz = p50_R4, p50_Rz4
#     trace_color = c[4]
# elif args.trace_method != "none":
#     print(f"Warning: trace method '{args.trace_method}' requested but sampler not available; skipping traces.")

trace_pm1 = trace_pm2 = trace_pz = None
trace_color = c[0]

if args.trace_method == "pe":
    # Use the full collection of realizations, NOT the median
    trace_pm1, trace_pm2, trace_pz = dR_dm1_all, dR_dm2_all, Rz_all
    trace_color = c[0]
elif args.trace_method == "proposal" and _met3 is not None:
    # Use Method 3 rate realizations
    trace_pm1, trace_pm2, trace_pz = dR_dm1_all3, dR_dm2_all3, Rz_all3
    trace_color = c[1]
elif args.trace_method == "mixture" and _met4 is not None:
    # Use Method 4 rate realizations
    trace_pm1, trace_pm2, trace_pz = dR_dm1_all4, dR_dm2_all4, Rz_all4
    trace_color = c[4]
elif args.trace_method != "none":
    print(f"Warning: trace method '{args.trace_method}' requested but sampler not available; skipping traces.")
    
# Make sure idx_100 is valid for the chosen array length
if trace_pm1 is not None:
    n_avail = len(trace_pm1)
    np.random.seed(42)
    idx_100 = np.random.choice(n_avail, size=min(n_realizations, n_avail), replace=False)

# Plot pm1
plt.figure(figsize=(10, 6))

# Plot the 100 trace lines ONLY for standard pop
# for i in idx_100:
#     plt.plot(m1s, pm1_all[i], color=c[0], alpha=0.1, lw=1)

if trace_pm1 is not None:
    for i in idx_100:
        plt.plot(m1s, trace_pm1[i], color=trace_color, alpha=0.1, lw=1)

# Plot percentiles and medians
plt.fill_between(m1s, p5_R, p95_R, color=c[0], alpha=0.3)
plt.plot(m1s, p50_R, color=c[0], lw=2, label=r'$\mathrm{PE}$')

if _met3 is not None:
    plt.fill_between(m1s, p5_R3, p95_R3, color=c[1], alpha=0.3)
    plt.plot(m1s, p50_R3, color=c[1], lw=2, label=r'$\mathrm{Ensemble}$')

if _met4 is not None:
    plt.fill_between(m1s, p5_R4, p95_R4, color=c[4], alpha=0.3)
    plt.plot(m1s, p50_R4, color=c[4], lw=2, label=r'$\mathrm{Mixture}$')

if true_pm1 is not None:
    plt.plot(m1s, true_pm1, color='black', linestyle='--', lw=2, label=r'$\mathrm{True~Distribution}$')

plt.xlim(5, 80)
if args.run=="gwtc3":
    plt.ylim(1e-2, 100)
elif args.run=="mdc":
    plt.ylim(1e-4, 10)
plt.yscale("log")
plt.gca().yaxis.set_major_formatter(FuncFormatter(_log_tex_formatter))
plt.ylabel(r"$dR/dm_1 \, [\mathrm{Gpc}^{\mbox{-}3} \, \mathrm{yr}^{\mbox{-}1} \, M_\odot^{\mbox{-}1}]$", fontsize=25)
plt.xlabel(r"$m_1 \, [M_\odot]$", fontsize=25)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig(args.outdir / f"{args.run}_Rm1_posteriors.pdf", dpi=300)
plt.close()

# Plot pm2
plt.figure(figsize=(10, 6))

# Plot the 100 trace lines ONLY for standard pop
# for i in idx_100:
#     plt.plot(m2s, pm2_all[i], color=c[0], alpha=0.1, lw=1)

# if trace_pm2 is not None:
#     for i in idx_100:
#         plt.plot(m2s, trace_pm2[i], color=trace_color, alpha=0.1, lw=1)

# plt.fill_between(m2s, p5_m2, p95_m2, color=c[0], alpha=0.3)
# plt.plot(m2s, p50_m2, color=c[0], lw=2, label=r'$\mathrm{PE}$')

# if _met3 is not None:
#     plt.fill_between(m2s, p5_3_m2, p95_3_m2, color=c[1], alpha=0.3)
#     plt.plot(m2s, p50_3_m2, color=c[1], lw=2, label=r'$\mathrm{Proposal}$')

# if _met4 is not None:
#     plt.fill_between(m2s, p5_4_m2, p95_4_m2, color=c[4], alpha=0.3)
#     plt.plot(m2s, p50_4_m2, color=c[4], lw=2, label=r'$\mathrm{Mixture}$')

# if true_pm2 is not None:
#     plt.plot(m2s, true_pm2, color='black', linestyle='--', lw=2, label=r'$\mathrm{True~Distribution}$')

# plt.xlim(4, 80)
# plt.ylim(1e-6, 1)
# plt.xlabel(r"$m_2 \, [M_\odot]$", fontsize=18)
# plt.ylabel(r"$p(m_2 \mid \lambda)$", fontsize=18)
# plt.yscale("log")
# plt.legend(fontsize=14)
# plt.tight_layout()
# plt.savefig(args.outdir / f"{args.run}_pm2_posteriors.pdf", dpi=300)
# plt.close()

# Plot pz (redshift)
plt.figure(figsize=(10, 6))

# Plot the 100 trace lines ONLY for standard pop
# for i in idx_100:
#     plt.plot(zs_plot, pz_all[i], color=c[0], alpha=0.1, lw=1)

if trace_pz is not None:
    for i in idx_100:
        plt.plot(zs_plot, trace_pz[i], color=trace_color, alpha=0.1, lw=1)

plt.fill_between(zs_plot, p5_Rz, p95_Rz, color=c[0], alpha=0.3)
plt.plot(zs_plot, p50_Rz, color=c[0], lw=2, label=r'$\mathrm{PE}$')

if _met3 is not None:
    plt.fill_between(zs_plot, p5_Rz3, p95_Rz4, color=c[1], alpha=0.3)
    plt.plot(zs_plot, p50_Rz3, color=c[1], lw=2, label=r'$\mathrm{Ensemble}$')

if _met4 is not None:
    plt.fill_between(zs_plot, p5_Rz4, p95_Rz4, color=c[4], alpha=0.3)
    plt.plot(zs_plot, p50_Rz4, color=c[4], lw=2, label=r'$\mathrm{Mixture}$')

if true_pz is not None:
    plt.plot(zs_plot, true_pz, color='black', linestyle='--', lw=2, label=r'$\mathrm{True~Distribution}$')

plt.xlim(0, 1.5)
plt.yscale("log") # Rates are often plotted on log scales
plt.gca().yaxis.set_major_formatter(FuncFormatter(_log_tex_formatter))
plt.xlabel(r"$z$", fontsize=25)
plt.ylabel(r"$\mathcal{R}(z) \, [\mathrm{Gpc}^{\mbox{-}3} \, \mathrm{yr}^{\mbox{-}1}]$", fontsize=25)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig(args.outdir / f"{args.run}_Rz_posteriors.pdf", dpi=300)
plt.close()
