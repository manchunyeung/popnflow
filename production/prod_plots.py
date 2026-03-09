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
from dynesty.utils import resample_equal
import corner

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

# =========================================================
# Pop Models: LVK (9 params) vs PAIRING (10 params)
# =========================================================
@jit
def logpm1m2_plpeak_massratio(m1, m2, m_min_1, m_max_1, alpha_1, dm_min_1, beta, mu, sigma, f):
    q = m2/m1
    alpha_1 = -alpha_1
    norm_pl = (m_max_1**(1. + alpha_1) - m_min_1**(1. + alpha_1))
    p_m1_pl = (1. + alpha_1) * m1**alpha_1 / norm_pl
    p_m1_pl = jnp.where(m1 > m_max_1, 0.0, p_m1_pl)
    p_m1_pl = jnp.where(m1 < m_min_1, 0.0, p_m1_pl)

    p_m1_peak = jnp.exp(-0.5 * (m1 - mu)**2 / sigma**2) / jnp.sqrt(2. * jnp.pi * sigma**2)
    p_m1 = Sfilter_low(m1,m_min_1,dm_min_1)*(f * p_m1_peak + (1. - f) * p_m1_pl)

    q_min = m_min_1/m1
    denom = 1 - q_min**(1. + beta)
    p_q = Sfilter_low(q*m1,m_min_1,dm_min_1) * (1. + beta) * q**beta / denom
    p_q = jnp.where(q*m1 < m_min_1, 0.0, p_q)

    return jnp.log(p_m1) + jnp.log(p_q)

@jit
def log_p_pop_lvk(m1,m2,z,gamma, m_min_1,m_max_1,alpha_1,dm_min_1,beta,mu,sigma,f1):
    log_pm1m2 = logpm1m2_plpeak_massratio(m1, m2, m_min_1, m_max_1, alpha_1, dm_min_1, beta, mu, sigma, f1)
    log_pz = jnp.log(dV_of_z_normed(z,Om0Planck,gamma))
    log_p = log_pm1m2 + log_pz
    log_p = jnp.where(m2<m1, log_p, -jnp.inf)
    return log_p

@jit
def log_p_pop_pl_pl(m1,m2,z,gamma, m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,beta,mu,sigma,f1):
    log_dNdm1 = logpm1_powerlaw_powerlaw(m1,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,mu,sigma,f1)
    log_dNdm2 = logpm1_powerlaw_powerlaw(m2,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,mu,sigma,f1)
    q = m2/m1
    log_pq = logfq(m1,m2,beta)
    log_dvdz = jnp.log(dV_of_z_normed(z,Om0Planck,gamma))
    log_p_sz = np.log(0.25)
    log_p = log_p_sz + log_dNdm1 + log_dNdm2 + log_pq + log_dvdz
    log_p = jnp.where(m2<m1, log_p, -jnp.inf)
    return log_p

# Dummy functions for Dynesty to unpack likelihood calls properly (prevents UnpicklingError)
# Using `len(coord)` handles both 9 (LVK) and 10 (Pairing) params dynamically in the same run.
def loglike_method_1(coord):
    if len(coord) == 9:
        gamma, m_min, m_max, alpha, dm_min, beta, mu, sigma, f1 = coord
        ll, Neff = likelihood_lvk_met1(gamma, m_min, m_max, alpha, dm_min, beta, mu, sigma, f1)
    else:
        gamma, m_min, m_max, alpha, dm_min, dm_max, beta, mu, sigma, f1 = coord
        ll, Neff = likelihood_method_1(gamma, m_min, m_max, alpha, dm_min, dm_max, beta, mu, sigma, f1)
    return ll if (not np.isnan(ll)) and (Neff >= 4*Nobs) else -np.inf

def loglike_method_3(coord):
    if len(coord) == 9:
        gamma, m_min, m_max, alpha, dm_min, beta, mu, sigma, f1 = coord
        ll, Neff = likelihood_lvk_met3(gamma, m_min, m_max, alpha, dm_min, beta, mu, sigma, f1)
    else:
        gamma, m_min, m_max, alpha, dm_min, dm_max, beta, mu, sigma, f1 = coord
        ll, Neff = likelihood_method_3(gamma, m_min, m_max, alpha, dm_min, dm_max, beta, mu, sigma, f1)
    return ll if (not np.isnan(ll)) and (Neff >= 4*Nobs) else -np.inf

# Prior Transforms & bounds (not directly used in plotting, but required for unpickler)
def prior_transform(theta): pass
def likelihood_method_1(gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1): pass
def likelihood_lvk_met1(gamma, m_min, m_max, alpha, dm_min, beta, mu, sigma, f1): pass
def likelihood_method_3(gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1): pass
def likelihood_lvk_met3(gamma, m_min, m_max, alpha, dm_min, beta, mu, sigma, f1): pass

# =========================================================
# Marginal Plotting Definitions
# =========================================================

m1s = jnp.linspace(1.0, 100.0, 300, dtype=jnp.float64)
m2s = jnp.linspace(1.0, 100.0, 300, dtype=jnp.float64)
dm1 = m1s[1] - m1s[0]
dm2 = m2s[1] - m2s[0]
M1, M2 = jnp.meshgrid(m1s, m2s, indexing="ij")

# Marginal Extractors for LVK (9 Params)
@jit
def pm1_from_lambda_lvk(lam):
    gamma, m_min, m_max, alpha, dm_min, beta, mu, sigma, f1 = lam
    log_joint = logpm1m2_plpeak_massratio(M1, M2, m_min, m_max, alpha, dm_min, beta, mu, sigma, f1)
    log_norm = logsumexp(log_joint) + jnp.log(dm1 * dm2)
    log_pjoint = log_joint - log_norm
    log_pm1 = logsumexp(log_pjoint, axis=1) + jnp.log(dm2)
    return jnp.exp(log_pm1)

@jit
def pm2_from_lambda_lvk(lam):
    gamma, m_min, m_max, alpha, dm_min, beta, mu, sigma, f1 = lam
    log_joint = logpm1m2_plpeak_massratio(M1, M2, m_min, m_max, alpha, dm_min, beta, mu, sigma, f1)
    log_norm = logsumexp(log_joint) + jnp.log(dm1 * dm2)
    log_pjoint = log_joint - log_norm
    log_pm2 = logsumexp(log_pjoint, axis=0) + jnp.log(dm1)
    return jnp.exp(log_pm2)

# Marginal Extractors for PAIRING (10 Params)
@jit
def pm1_from_lambda_pairing(lam):
    gamma, m_min, m_max, alpha, dm_min, dm_max, beta, mu, sigma, f1 = lam
    log_joint = log_p_pop_pl_pl(M1, M2, zgrid[0], gamma, m_min, m_max, alpha, dm_min, dm_max, beta, mu, sigma, f1)
    log_norm = logsumexp(log_joint) + jnp.log(dm1 * dm2)
    log_pjoint = log_joint - log_norm
    log_pm1 = logsumexp(log_pjoint, axis=1) + jnp.log(dm2)
    return jnp.exp(log_pm1)

@jit
def pm2_from_lambda_pairing(lam):
    gamma, m_min, m_max, alpha, dm_min, dm_max, beta, mu, sigma, f1 = lam
    log_joint = log_p_pop_pl_pl(M1, M2, zgrid[0], gamma, m_min, m_max, alpha, dm_min, dm_max, beta, mu, sigma, f1)
    log_norm = logsumexp(log_joint) + jnp.log(dm1 * dm2)
    log_pjoint = log_joint - log_norm
    log_pm2 = logsumexp(log_pjoint, axis=0) + jnp.log(dm1)
    return jnp.exp(log_pm2)


# =========================================================
# Scan and Plot Complete Seeds 
# =========================================================

labels_lvk = ['gamma', 'm_min_1', 'm_max_1', 'alpha_1', 'dm_min_1', 'beta', 'mu', 'sigma', 'f1']
labels_pairing = ['gamma', 'm_min_1', 'm_max_1', 'alpha_1', 'dm_min_1', 'dm_max_1', 'beta', 'mu', 'sigma', 'f1']

# 1. Discover all seeds that possess all 4 files (lvk/pairing x met1/met3)
all_seeds = set()
for f in args.indir.glob("*.h5"):
    match = re.search(r'(lvk|pairing)_(\d+)_?met[13]\.h5', f.name)
    if match:
        all_seeds.add(int(match.group(2)))

complete_seeds = []
for seed in sorted(list(all_seeds)):
    # Standardize checking for files with or without underscores
    f_lvk1 = args.indir / f"lvk_{seed}met1.h5" if (args.indir / f"lvk_{seed}met1.h5").exists() else args.indir / f"lvk_{seed}_met1.h5"
    f_lvk3 = args.indir / f"lvk_{seed}met3.h5" if (args.indir / f"lvk_{seed}met3.h5").exists() else args.indir / f"lvk_{seed}_met3.h5"
    f_pair1 = args.indir / f"pairing_{seed}met1.h5" if (args.indir / f"pairing_{seed}met1.h5").exists() else args.indir / f"pairing_{seed}_met1.h5"
    f_pair3 = args.indir / f"pairing_{seed}met3.h5" if (args.indir / f"pairing_{seed}met3.h5").exists() else args.indir / f"pairing_{seed}_met3.h5"
    
    if f_lvk1.exists() and f_lvk3.exists() and f_pair1.exists() and f_pair3.exists():
        complete_seeds.append(seed)

print(f"Found {len(complete_seeds)} fully complete seeds. Generating plots...")

for seed in complete_seeds:
    print(f"\nProcessing seed: {seed}")
    
    for model in ['lvk', 'pairing']:
        print(f"  -> Plotting {model.upper()}...")
        
        # Select appropriate parameters for the current model
        f_met1 = args.indir / f"{model}_{seed}met1.h5" if (args.indir / f"{model}_{seed}met1.h5").exists() else args.indir / f"{model}_{seed}_met1.h5"
        f_met3 = args.indir / f"{model}_{seed}met3.h5" if (args.indir / f"{model}_{seed}met3.h5").exists() else args.indir / f"{model}_{seed}_met3.h5"
        
        if model == "lvk":
            labels_current = labels_lvk
            pm1_fn = jax.jit(vmap(pm1_from_lambda_lvk))
            pm2_fn = jax.jit(vmap(pm2_from_lambda_lvk))
        else:
            labels_current = labels_pairing
            pm1_fn = jax.jit(vmap(pm1_from_lambda_pairing))
            pm2_fn = jax.jit(vmap(pm2_from_lambda_pairing))

        # --- Load MET1 ---
        sampler1 = _restore_dynesty_sampler(f_met1, required=False)
        if sampler1 is None: continue
        dres1 = sampler1.results
        dweights1 = np.exp(dres1['logwt'] - dres1['logz'][-1])
        dpostsamples1 = resample_equal(dres1.samples, dweights1)

        # Do NOT slice [1:] so gamma is retained for the param wrappers
        lambda_samples1 = jnp.asarray(dpostsamples1[::10, :], dtype=jnp.float64)
        pm1_all1 = pm1_fn(lambda_samples1)
        p5_1, p50_1, p95_1 = jnp.percentile(pm1_all1, jnp.array([5, 50, 95]), axis=0)

        pm2_all1 = pm2_fn(lambda_samples1)
        p5_pm2_1, p50_pm2_1, p95_pm2_1 = jnp.percentile(pm2_all1, jnp.array([5, 50, 95]), axis=0)

        # --- Load MET3 ---
        sampler3 = _restore_dynesty_sampler(f_met3, required=False)
        if sampler3 is None: continue
        dres3 = sampler3.results
        dweights3 = np.exp(dres3['logwt'] - dres3['logz'][-1])
        dpostsamples3 = resample_equal(dres3.samples, dweights3)

        lambda_samples3 = jnp.asarray(dpostsamples3[::5, :], dtype=jnp.float64)
        pm1_all3 = pm1_fn(lambda_samples3)
        p5_3, p50_3, p95_3 = jnp.percentile(pm1_all3, jnp.array([5, 50, 95]), axis=0)

        pm2_all3 = pm2_fn(lambda_samples3)
        p5_pm2_3, p50_pm2_3, p95_pm2_3 = jnp.percentile(pm2_all3, jnp.array([5, 50, 95]), axis=0)

        # --- Plot Overlapping Corner Plots ---
        fig_corner = corner.corner(dpostsamples1, labels=labels_current, hist_kwargs={'density': True}, color='blue')
        corner.corner(dpostsamples3, labels=labels_current, hist_kwargs={'density': True}, color='orange', fig=fig_corner)
        
        import matplotlib.lines as mlines
        blue_line = mlines.Line2D([], [], color='blue', label='met1')
        orange_line = mlines.Line2D([], [], color='orange', label='met3')
        fig_corner.legend(handles=[blue_line, orange_line], loc='upper right', fontsize=15)
        
        fig_corner.savefig(args.outdir / f"{model}_{seed}_corner_pos.pdf", dpi=200)
        plt.close(fig_corner)

        # --- Plot pm1 ---
        fig_pm1 = plt.figure(figsize=(6.5, 3.5))
        plt.fill_between(m1s, p5_1, p95_1, alpha=0.3, color='blue')
        plt.plot(m1s, p50_1, lw=2, color='blue', label='met1 (standard pop)')

        plt.fill_between(m1s, p5_3, p95_3, alpha=0.3, color='orange')
        plt.plot(m1s, p50_3, lw=2, color='orange', label='met3 (gmm sel+events)')

        plt.xlim(4, 80)
        plt.ylim(1e-6, 1)
        plt.xlabel(r"$m_1\,[M_\odot]$")
        plt.ylabel(r"$p(m_1 \mid \lambda)$")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.outdir / f"{model}_{seed}_pm1_posteriors.pdf", dpi=200)
        plt.close(fig_pm1)

        # --- Plot pm2 ---
        fig_pm2 = plt.figure(figsize=(6.5, 3.5))
        plt.fill_between(m2s, p5_pm2_1, p95_pm2_1, alpha=0.3, color='blue')
        plt.plot(m2s, p50_pm2_1, lw=2, color='blue', label='met1 (standard pop)')

        plt.fill_between(m2s, p5_pm2_3, p95_pm2_3, alpha=0.3, color='orange')
        plt.plot(m2s, p50_pm2_3, lw=2, color='orange', label='met3 (gmm sel+events)')

        plt.xlim(4, 80)
        plt.ylim(1e-6, 1)
        plt.xlabel(r"$m_2\,[M_\odot]$")
        plt.ylabel(r"$p(m_2 \mid \lambda)$")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.outdir / f"{model}_{seed}_pm2_posteriors.pdf", dpi=200)
        plt.close(fig_pm2)

print("\nFinished plotting all runs.")