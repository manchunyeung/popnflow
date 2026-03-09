import argparse
from pathlib import Path
import os
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
import seaborn as sns
from jaxinterp2d import interp2d, CartesianGrid
import dynesty
from dynesty import utils as dyfunc
from dynesty.utils import resample_equal
import corner
import scipy.special as jsp # Added to satisfy unpickled code

# ==========================================================
# 1. Setup & Configuration
# ==========================================================
def parse_args():
    p = argparse.ArgumentParser(description="Plotting from cached samplers")
    p.add_argument("--indir", type=Path, required=True, help="Directory with sampler outputs")
    p.add_argument("--outdir", type=Path, required=True, help="Directory for plots")
    p.add_argument("--run", type=str, required=True, default="gwtc3", help="gwtc3/mdc")
    p.add_argument("--model", type=str, default="pairing", help="lvk/pairing")
    return p.parse_args()

args = parse_args()
args.outdir.mkdir(parents=True, exist_ok=True)

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.sans-serif'] = ['Bitstream Vera Sans']
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['figure.figsize'] = (16.0, 10.0)
matplotlib.rcParams['axes.unicode_minus'] = False

sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('colorblind')
c=sns.color_palette('colorblind')

jax.config.update("jax_enable_x64", False) # JAX defaults to float32
jax.config.update('jax_default_matmul_precision', 'tensorfloat32') # Faster matmuls

def _restore_dynesty_sampler(path: Path, *, required: bool = True):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing sampler file: {path}")
        return None
    return dynesty.NestedSampler.restore(str(path))

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

    return jnp.log(p_m1) + jnp.log(p_q)

@jit
def log_p_pop_lvk(m1,m2,z,gamma, m_min_1,m_max_1,alpha_1,beta,mu,sigma,f1):
    log_joint_mass = logpm1m2_plpeak_massratio(m1, m2, m_min_1, m_max_1, alpha_1, beta, mu, sigma, f1)
    log_pz = jnp.log(dV_of_z_normed(z,Om0Planck,gamma))
    log_p = log_joint_mass + log_pz
    log_p = jnp.where(m2<m1, log_p, -jnp.inf)
    return log_p

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

# ==========================================================
# 5. Likelihood & Prior Definitions (Required for unpickling)
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
    return ll

@jit
def likelihood_method_1_sel(gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1):
    log_det_weights = log_p_pop_pl_pl(m1sels,m2sels,zsels,gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1)
    log_det_weights += - 2*jnp.log1p(zsels) - jnp.log(ddL_of_z(zsels,dL_sam,H0Planck, Om0Planck))
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

def loglike_method_1_sel(coord):
    if len(coord) == 8:
        gamma, m_min, m_max, alpha, beta, mu, sigma, f1 = coord
        dm_min, dm_max = 1.0, 1.0
    else:
        gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1 = coord
    ll, Neff = likelihood_method_1_sel(gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1)
    if np.isnan(ll) or (Neff < 4*Nobs):
        return -np.inf
    return ll

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
    return ll

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
sampler_file = args.indir / "mdc_777met1.h5"
sampler = _restore_dynesty_sampler(sampler_file, required=True)
dres = sampler.results
dweights = np.exp(dres['logwt'] - dres['logz'][-1])
dpostsamples = resample_equal(dres.samples, dweights)

# Corner Plot with Truths
fig1 = corner.corner(
    dpostsamples, 
    labels=labels, 
    truths=truths, 
    truth_color='black',
    hist_kwargs={'density': True}
)
plt.close(fig1)

# Marginalize pm1 and pm2
lambda_samples = jnp.asarray(dpostsamples[::10, 1:], dtype=jnp.float64)

# pm1
pm1_all = jax.jit(vmap(pm1_from_lambda))(lambda_samples)
p5, p50, p95 = jnp.percentile(pm1_all, jnp.array([5, 50, 95]), axis=0)

# pm2
pm2_all = jax.jit(vmap(pm2_from_lambda))(lambda_samples)
p5_m2, p50_m2, p95_m2 = jnp.percentile(pm2_all, jnp.array([5, 50, 95]), axis=0)

print(truths)

# ==========================================================
# 7. Additional Samplers (Optional Sel/Events)
# ==========================================================
_met1_sel = None
if _met1_sel is not None:
    d2res = _met1_sel.results
    d2weights = np.exp(d2res['logwt'] - d2res['logz'][-1])
    d2postsamples = resample_equal(d2res.samples, d2weights)
    lambda_samples2 = jnp.asarray(d2postsamples[::5, 1:], dtype=jnp.float64)
    pm1_all2 = jax.jit(vmap(pm1_from_lambda))(lambda_samples2)
    p5_2, p50_2, p95_2 = jnp.percentile(pm1_all2, jnp.array([5, 50, 95]), axis=0)
    pm2_all2 = jax.jit(vmap(pm2_from_lambda))(lambda_samples2)
    p5_2_m2, p50_2_m2, p95_2_m2 = jnp.percentile(pm2_all2, jnp.array([5, 50, 95]), axis=0)

_met3 = _restore_dynesty_sampler(args.indir / "mdc_777met3.h5", required=False)
if _met3 is not None:
    d3res = _met3.results
    d3weights = np.exp(d3res['logwt'] - d3res['logz'][-1])
    d3postsamples = resample_equal(d3res.samples, d3weights)
    lambda_samples3 = jnp.asarray(d3postsamples[::5, 1:], dtype=jnp.float64)
    pm1_all3 = jax.jit(vmap(pm1_from_lambda))(lambda_samples3)
    p5_3, p50_3, p95_3 = jnp.percentile(pm1_all3, jnp.array([5, 50, 95]), axis=0)
    pm2_all3 = jax.jit(vmap(pm2_from_lambda))(lambda_samples3)
    p5_3_m2, p50_3_m2, p95_3_m2 = jnp.percentile(pm2_all3, jnp.array([5, 50, 95]), axis=0)
    
    fig2 = corner.corner(
        d3postsamples, 
        labels=labels, 
        truths=truths,
        truth_color='black',
        hist_kwargs={'density': True}, 
        color='orange', 
        fig=fig1
    )
    fig2.savefig(args.outdir / f"{args.run}_corner_pos.pdf", dpi=200)
    plt.close(fig2)

# ==========================================================
# 8. Final Output Plots
# ==========================================================

# Calculate True MDC Values
true_pm1 = None
true_pm2 = None

if args.run == "mdc":
    true_params_dict = {
        'gamma': 3.0, 
        'm_min_1': 6.0, 
        'm_max_1': 70.0, 
        'alpha_1': 4., 
        'beta': 1.5, 
        'mu': 35.0, 
        'sigma': 4.0, 
        'f1': 0.04
    }
    # Extract just the mass parameters to match the lambda functions
    # (skipping gamma, which is index 0 in labels)
    true_lambda = jnp.array([true_params_dict[k] for k in labels[1:]], dtype=jnp.float64)
    true_pm1 = pm1_from_lambda(true_lambda)
    true_pm2 = pm2_from_lambda(true_lambda)

# Plot pm1
plt.figure(figsize=(10, 6)) # Increased figure size
plt.fill_between(m1s, p5, p95, alpha=0.3)
plt.plot(m1s, p50, lw=2, label=r'$\mathrm{standard~pop}$')
if _met1_sel is not None:
    plt.fill_between(m1s, p5_2, p95_2, alpha=0.3)
    plt.plot(m1s, p50_2, lw=2, label=r'$\mathrm{gmm~sel}$')
if _met3 is not None:
    plt.fill_between(m1s, p5_3, p95_3, alpha=0.3)
    plt.plot(m1s, p50_3, lw=2, label=r'$\mathrm{gmm~sel+events}$')

# Overlay the True Value line if it exists
if true_pm1 is not None:
    plt.plot(m1s, true_pm1, color='black', linestyle='--', lw=2, label=r'$\mathrm{True~Distribution}$')

plt.xlim(4, 80)
plt.ylim(1e-6, 1)
plt.xlabel(r"$m_1 \, [M_\odot]$", fontsize=18)
plt.ylabel(r"$p(m_1 \mid \lambda)$", fontsize=18)
plt.yscale("log")
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(args.outdir / f"{args.run}_pm1_posteriors.pdf", dpi=300)
plt.close()

# Plot pm2
plt.figure(figsize=(10, 6)) # Increased figure size
plt.fill_between(m2s, p5_m2, p95_m2, alpha=0.3)
plt.plot(m2s, p50_m2, lw=2, label=r'$\mathrm{standard~pop}$')
if _met1_sel is not None:
    plt.fill_between(m2s, p5_2_m2, p95_2_m2, alpha=0.3)
    plt.plot(m2s, p50_2_m2, lw=2, label=r'$\mathrm{gmm~sel}$')
if _met3 is not None:
    plt.fill_between(m2s, p5_3_m2, p95_3_m2, alpha=0.3)
    plt.plot(m2s, p50_3_m2, lw=2, label=r'$\mathrm{gmm~sel+events}$')

# Overlay the True Value line if it exists
if true_pm2 is not None:
    plt.plot(m2s, true_pm2, color='black', linestyle='--', lw=2, label=r'$\mathrm{True~Distribution}$')

plt.xlim(4, 80)
plt.ylim(1e-6, 1)
plt.xlabel(r"$m_2 \, [M_\odot]$", fontsize=18)
plt.ylabel(r"$p(m_2 \mid \lambda)$", fontsize=18)
plt.yscale("log")
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(args.outdir / f"{args.run}_pm2_posteriors.pdf", dpi=300)
plt.close()