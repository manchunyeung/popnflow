import argparse
from pathlib import Path
import os
import re

def parse_args():
    p = argparse.ArgumentParser(description="Plotting from cached samplers")
    p.add_argument("--indir", type=Path, required=True, help="Directory with sampler outputs")
    p.add_argument("--outdir", type=Path, required=True, help="Directory for plots")
    return p.parse_args()

args = parse_args()
args.outdir.mkdir(parents=True, exist_ok=True)

import dynesty

def _restore_dynesty_sampler(path: Path, *, required: bool = True):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing sampler file: {path}")
        return None
    return dynesty.NestedSampler.restore(str(path))

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

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
import jax.scipy as jsp
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.sans-serif'] = ['Bitstream Vera Sans']
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['figure.figsize'] = (16.0, 10.0)
matplotlib.rcParams['axes.unicode_minus'] = False

import seaborn as sns
sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('colorblind')
c=sns.color_palette('colorblind')

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'highest')

from jaxinterp2d import interp2d, CartesianGrid

H0Planck = Planck15.H0.value
Om0Planck = Planck15.Om0
speed_of_light = constants.c.to('km/s').value

zMax = 2.0
zgrid = jnp.expm1(np.linspace(np.log(1), np.log(zMax+1), 10000))
Om0grid = jnp.linspace(0,1,1000)

rs = []
for Om0 in tqdm(Om0grid):
    cosmo = FlatLambdaCDM(H0=H0Planck,Om0=Om0)
    rs.append(cosmo.comoving_distance(zgrid).to(u.Mpc).value)

rs = jnp.asarray(rs)
rs = rs.reshape(len(Om0grid),len(zgrid))

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
def ddL_of_z(z,dL,H0,Om0=Om0Planck):
    return dL/(1+z) + speed_of_light*(1+z)/(H0*E(z,Om0))

mass = jnp.linspace(1, 150, 2000)
mass_ratio =  jnp.linspace(1e-5, 1, 2000)

from jax.scipy.stats import norm as norm_jax

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
def log_p_pop_pl_pl(m1,m2,z,gamma, m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,beta,mu,sigma,f1):
    log_dNdm1 = logpm1_powerlaw_powerlaw(m1,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,mu,sigma,f1)
    log_dNdm2 = logpm1_powerlaw_powerlaw(m2,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,mu,sigma,f1)
    log_pq = logfq(m1,m2,beta)
    log_dvdz = jnp.log(dV_of_z_normed(z,Om0Planck,gamma))
    
    log_p_sz = np.log(0.25)

    log_p = log_p_sz + log_dNdm1 + log_dNdm2 + log_pq + log_dvdz
    log_p = jnp.where(m2<m1, log_p, -jnp.inf)
    return log_p

@jit
def fq(q,beta):
    pq = mass_ratio**beta
    pq = pq/jnp.trapezoid(pq,mass_ratio)
    log_pq = jnp.interp(q,mass_ratio,pq)
    return log_pq

@jit
def dV_of_z_normed(z,Om0,gamma):
    dV = dV_of_z(zgrid,H0Planck,Om0)*(1+zgrid)**(gamma-1)
    prob = dV/jnp.trapezoid(dV,zgrid)
    return jnp.interp(z,zgrid,prob)

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
    gamma, m_min, m_max, alpha, dm_min, dm_max, beta, mu, sigma, f1 = coord

    ll, Neff = likelihood_method_1(
        gamma, m_min, m_max, alpha, dm_min, dm_max,
        beta, mu, sigma, f1,
    )
    if np.isnan(ll):
        return -np.inf
    elif (Neff < 4*Nobs):
        return -np.inf
    else:
        return ll

# --- Restored Methods needed for dynesty unpickling ---

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
    gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1 = coord

    ll, Neff = likelihood_method_3(gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1)
    if np.isnan(ll):
        return -np.inf
    elif (Neff < 4*Nobs):
        return -np.inf
    else:
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

    per_event_log_like = jnp.nan_to_num(jsp.special.logsumexp(log_weights[None, :] + logp_E_N + logJ[None, :], axis=1)) - jnp.log(nsamp_pop)  

    ll += jnp.sum(per_event_log_like)
    return ll, Neff

def loglike_method_3(coord):
    gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1 = coord

    ll, Neff = likelihood_method_3(gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1)
    if np.isnan(ll):
        return -np.inf
    elif (Neff < 4*Nobs):
        return -np.inf
    else:
        return ll

# Parameter Bounds
gamma_low, gamma_high = 0, 10
m_min_1_low, m_min_1_high = 2, 10
m_max_1_low, m_max_1_high = 50, 100
alpha_1_low, alpha_1_high = 0, 6
dm_min_1_low, dm_min_1_high = 1, 100
dm_max_1_low, dm_max_1_high = 1, 100
beta_low, beta_high = 0, 6
mu_low, mu_high = 20, 50
sigma_low, sigma_high = 1, 10
f1_low, f1_high = 0, 1

lower_bound = np.array([gamma_low, m_min_1_low,m_max_1_low,alpha_1_low,dm_min_1_low,dm_max_1_low,beta_low,mu_low,sigma_low,f1_low])
upper_bound = np.array([gamma_high, m_min_1_high,m_max_1_high,alpha_1_high,dm_min_1_high,dm_max_1_high,beta_high,mu_high,sigma_high,f1_high,])

ndims = len(lower_bound)
nlive = 1000
labels = ['gamma', 'm_min_1','m_max_1','alpha_1','dm_min_1','dm_max_1','beta', 'mu','sigma','f1']

def prior_transform(theta):
    transformed_params = [
        theta[i] * (upper_bound[i] - lower_bound[i]) + lower_bound[i]
        for i in range(len(theta))
    ]
    return tuple(transformed_params)

# =========================================================
# Multi-Seed Extractor & Plotter (Replaces Single-File Logic)
# =========================================================

logZs = {
    'lvk_met1': [], 'lvk_met3': [],
    'pairing_met1': [], 'pairing_met3': []
}
logZerrs = {
    'lvk_met1': [], 'lvk_met3': [],
    'pairing_met1': [], 'pairing_met3': []
}
valid_seeds = []

# Find all h5 files to determine which seeds are available
all_files = list(args.indir.glob("*.h5"))
seeds = set()
for f in all_files:
    # Match patterns like lvk_10met1.h5 or pairing_42met3.h5
    match = re.search(r'(lvk|pairing)_(\d+)_?met[13]\.h5', f.name)
    if match:
        seeds.add(int(match.group(2)))

# Sort seeds for clean ordered plotting
seeds = sorted(list(seeds))

print(f"Discovered {len(seeds)} unique seeds in {args.indir}")

for seed in seeds:
    seed_data = {}
    missing_file = False
    
    # Check all four combinations for the current seed
    for prefix in ['lvk', 'pairing']:
        for met in ['met1', 'met3']:
            # Handle naming variations (e.g., lvk_10met1.h5 vs lvk_10_met1.h5)
            f_exact = args.indir / f"{prefix}_{seed}met{met[-1]}.h5"
            f_under = args.indir / f"{prefix}_{seed}_{met}.h5"
            
            f_path = f_exact if f_exact.exists() else f_under
            
            if not f_path.exists():
                missing_file = True
                break
                
            sampler = _restore_dynesty_sampler(f_path, required=False)
            if sampler is None:
                missing_file = True
                break
                
            res = sampler.results
            seed_data[f"{prefix}_{met}"] = {
                'z': res.logz[-1],
                'err': res.logzerr[-1]
            }
            
        if missing_file: 
            break
            
    # If all 4 files were successfully loaded for this seed, store them
    if not missing_file:
        valid_seeds.append(seed)
        for k in logZs.keys():
            logZs[k].append(seed_data[k]['z'])
            logZerrs[k].append(seed_data[k]['err'])

# Convert to numpy arrays
for k in logZs.keys():
    logZs[k] = np.array(logZs[k])
    logZerrs[k] = np.array(logZerrs[k])

print(logZs)

# =============================
# Plotting 4-Set LogZ Comparison (Exact Values)
# =============================
if len(valid_seeds) > 0:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [2, 1]})
    
    # --- Top Panel: Exact Absolute LogZ Values ---
    # We plot exact points on the lnZ x-axis. We use y-offsets to prevent overlap.
    ax1.plot(logZs['lvk_met1'], np.full_like(logZs['lvk_met1'], 3), 'o', color='blue', markersize=10, alpha=0.8, label='LVK met1')
    ax1.plot(logZs['lvk_met3'], np.full_like(logZs['lvk_met3'], 2), 's', color='cyan', markersize=10, alpha=0.8, label='LVK met3')
    ax1.plot(logZs['pairing_met1'], np.full_like(logZs['pairing_met1'], 1), '^', color='red', markersize=10, alpha=0.8, label='Pairing met1')
    ax1.plot(logZs['pairing_met3'], np.full_like(logZs['pairing_met3'], 0), 'D', color='orange', markersize=10, alpha=0.8, label='Pairing met3')
    
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(['Pairing met3', 'Pairing met1', 'LVK met3', 'LVK met1'])
    ax1.set_xlabel(r"$\ln \mathcal{Z}$")
    ax1.set_title("Exact Bayesian Evidence Values (3 Seeds)")
    ax1.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # --- Bottom Panel: Bayes Factors (LVK - Pairing) ---
    # Difference: lnZ(lvk) - lnZ(pairing) under the same method
    delta_met1 = logZs['lvk_met1'] - logZs['pairing_met1']
    delta_met3 = logZs['lvk_met3'] - logZs['pairing_met3']
    
    ax2.plot(delta_met1, np.full_like(delta_met1, 1), 'o', color='purple', markersize=10, alpha=0.8, label=r'met1 ($\ln \mathcal{Z}_{\rm LVK} - \ln \mathcal{Z}_{\rm Pairing}$)')
    ax2.plot(delta_met3, np.full_like(delta_met3, 0), 's', color='green', markersize=10, alpha=0.8, label=r'met3 ($\ln \mathcal{Z}_{\rm LVK} - \ln \mathcal{Z}_{\rm Pairing}$)')
    
    # Add a vertical line at 0 to see which pipeline is statistically favored
    ax2.axvline(0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['met3', 'met1'])
    ax2.set_xlabel(r"$\Delta \ln \mathcal{Z}$ (LVK - Pairing)")
    ax2.set_title("Exact Bayes Factor Differences")
    ax2.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(args.outdir / "logZ_4sets_exact.pdf", dpi=200)
    plt.close()
    
    print(f"Successfully plotted exact evidence values for {len(valid_seeds)} complete seeds.")
else:
    print("No seeds found with all 4 sets of files (lvk met1/met3, pairing met1/met3).")

# =============================
# End of production wrapper
# =============================