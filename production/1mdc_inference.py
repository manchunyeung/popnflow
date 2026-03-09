import argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="GWTC-3 population inference")
    p.add_argument("--live-points", type=int, default=1000)
    p.add_argument("--catalog", type=str, default="GWTC3")
    p.add_argument("--indir", type=Path, required=True, help="Input data directory")
    p.add_argument("--outdir", type=Path, required=True, help="Output directory")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--nsamp-pop", type=int, default=200000)
    p.add_argument("--run", type=str, default = "gwtc3")
    p.add_argument("--model", type=str, default = "pairing")
    return p.parse_args()

args = parse_args()
args.outdir.mkdir(parents=True, exist_ok=True)

print('indir', args.indir)

import os
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

jax.config.update("jax_enable_x64", False) # JAX defaults to float32
jax.config.update('jax_default_matmul_precision', 'tensorfloat32') # Faster matmuls

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

if args.run == 'mdc':
    dL = np.loadtxt(f'{args.indir}/mdc_dL.txt')
    m2det = np.loadtxt(f'{args.indir}/mdc_m2det.txt')
    m1det = np.loadtxt(f'{args.indir}/mdc_m1det.txt')
else:
    dL = np.loadtxt(f'{args.indir}/dL.txt')
    m2det = np.loadtxt(f'{args.indir}/m2det.txt')
    m1det = np.loadtxt(f'{args.indir}/m1det.txt')

dL = jnp.array(dL)
m2det = jnp.array(m2det)
m1det = jnp.array(m1det)
q = m2det/m1det

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
def logpm1_powerlaw_GP(m1,z,mu,sigma):
    pass

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

@jit
def logpm1m2_plpeak_massratio(
    m1, m2,
    m_min_1, m_max_1,
    alpha_1,
    beta, mu, sigma,
    f
):
    q = m2/m1
    alpha_1 = -alpha_1
    # --- p(m1): Power-law component ---
    norm_pl = (m_max_1**(1. + alpha_1) - m_min_1**(1. + alpha_1))
    p_m1_pl = (1. + alpha_1) * m1**alpha_1 / norm_pl

    # Mask out-of-range m1
    p_m1_pl = jnp.where(m1 > m_max_1, 0.0, p_m1_pl)
    p_m1_pl = jnp.where(m1 < m_min_1, 0.0, p_m1_pl)

    # --- p(m1): Peak component ---
    p_m1_peak = jnp.exp(-0.5 * (m1 - mu)**2 / sigma**2) / jnp.sqrt(2. * jnp.pi * sigma**2)

    # Mixture
    p_m1 = f * p_m1_peak + (1. - f) * p_m1_pl

    # --- p(q | m1): mass-ratio power law ---
    q_min = m_min_1/m1
    denom = 1 - q_min**(1. + beta)
    p_q = (1. + beta) * q**beta / denom

    # Enforce m2 >= m_min_1
    p_q = jnp.where(q*m1 < m_min_1, 0.0, p_q)

    # --- log joint ---
    return jnp.log(p_m1) + jnp.log(p_q) - jnp.log(m1)

@jit
def log_p_pop_lvk(m1,m2,z,gamma, m_min_1,m_max_1,alpha_1,beta,mu,sigma,f1):
    log_pm1m2 = logpm1m2_plpeak_massratio(m1, m2, m_min_1, m_max_1, alpha_1, beta, mu, sigma, f1)
    log_pz = jnp.log(dV_of_z_normed(z,Om0Planck,gamma))
    log_p = log_pm1m2 + log_pz
    log_p = jnp.where(m2<m1, log_p, -jnp.inf)
    return log_p

from functools import partial
import jax.scipy as jsp

# draw N observations from pop model (IN SOURCE FRAME)
Nobs = 69
nsamp = 3343
dL_ = dL.reshape(Nobs, nsamp)
m1det_ = m1det.reshape(Nobs, nsamp)
m2det_ = m2det.reshape(Nobs, nsamp)

injection_file = f"{args.indir}/endo3_bbhpop-LIGO-T2100113-v12.hdf5"

with h5py.File(injection_file, 'r') as f:
    Tobs = f.attrs['analysis_time_s']/(365.25*24*3600) # years
    Ndraw = f.attrs['total_generated']

    m1detsels = f['injections/mass1'][:]
    m2detsels = f['injections/mass2'][:]
    dLsels = f['injections/distance'][:]
    rasels = f['injections/right_ascension'][:]
    decsels = f['injections/declination'][:]

    p_draw = f['injections/sampling_pdf'][:]

    pastro_cwb = f['injections/pastro_cwb'][:]
    pastro_gstlal = f['injections/pastro_gstlal'][:]
    pastro_mbta = f['injections/pastro_mbta'][:]
    pastro_pycbc_bbh = f['injections/pastro_pycbc_bbh'][:]
    pastro_pycbc_broad = f['injections/pastro_pycbc_hyperbank'][:]

    ifar_cwb = f['injections/ifar_cwb'][:]
    ifar_gstlal = f['injections/ifar_gstlal'][:]
    ifar_mbta = f['injections/ifar_mbta'][:]
    ifar_pycbc_bbh = f['injections/ifar_pycbc_bbh'][:]
    ifar_pycbc_broad = f['injections/ifar_pycbc_hyperbank'][:]

selection_ifar = {
    'cwb': ifar_cwb > 1,
    'gstlal': ifar_gstlal > 1,
    'mbta': ifar_mbta > 1,
    'pycbc_bbh': ifar_pycbc_bbh > 1,
    'pycbc_broad': ifar_pycbc_broad > 1,
    'any': ((ifar_cwb > 1) | (ifar_gstlal > 1) | (ifar_mbta > 1) |
            (ifar_pycbc_bbh > 1) | (ifar_pycbc_broad > 1) ),
    'cbc': ((ifar_gstlal > 1) | (ifar_pycbc_bbh > 1) | (ifar_pycbc_broad > 1) ),
}

sels = selection_ifar['cbc']
m1detsels = jnp.array(m1detsels[sels])
m2detsels = jnp.array(m2detsels[sels])
dLsels = jnp.array(dLsels[sels])
rasels = jnp.array(rasels[sels])
decsels = jnp.array(decsels[sels])
p_draw = jnp.array(p_draw[sels])

Ndet = m1detsels.shape[0]

from sklearn.mixture import GaussianMixture
import numpy as np

@jax.jit
def gmm_logpdf_optimized(x_batch, weights, means, precisions, logdets):
    D = means.shape[-1]
    diffs = x_batch[:, None, :] - means[None, :, :]                
    quad = jnp.einsum('nkd,kdj,nkj->nk', diffs, precisions, diffs)  
    log_comp = -0.5 * (D*jnp.log(2*jnp.pi) + logdets + quad)       
    logw = jnp.log(jnp.clip(weights, 1e-12, 1.0))[None, :]         
    return jsp.special.logsumexp(logw + log_comp, axis=-1)         

def chirp_mass(m1, m2):
  return (m1*m2)**(3/5) / (m1+m2)**(1/5)

from scipy.special import logit, expit
def logitq(m1, m2):
  q = m2/m1
  return logit(q)

def inverse_transform(M, lq):
  q = expit(lq)
  m1 = M*(1+q)**(1/5)/q**(3/5)
  m2 = q * m1
  return m1, m2

lq = logitq(m1detsels, m2detsels)
X = np.column_stack([np.asarray(m1detsels), np.asarray(lq), np.asarray(dLsels)])

N = m1detsels.shape[0]
w = np.array(1/p_draw, dtype=np.float64)
prob = w / np.sum(w)

idx = np.random.choice(
    N,
    size=N,
    replace=True,
    p=prob
)

X_resam = X[idx]

m1_sels, lq_sels, dL_sels = X_resam[:, 0], X_resam[:, 1], X_resam[:, 2]
q_sels = expit(lq_sels)
m2_sels = m1_sels * q_sels

K = 7
gmm = GaussianMixture(
    n_components=K, covariance_type='full',
    random_state=args.seed,
).fit(X_resam)

new_x = gmm.sample(150000)

m1_sam, lq_sam, dL_sam = new_x[0][:, 0], new_x[0][:, 1], new_x[0][:, 2]
q_sam = expit(lq_sam)
m2_sam = m1_sam * q_sam

z_sam = z_of_dL(dL_sam, H0Planck)
m_src_min, m_src_max = 2.0, 100.0
z_max = 1.9

m_det_min = m_src_min * (1.0 + z_sam)
m_det_max = m_src_max * (1.0 + z_sam)

mask = (
    (dL_sam > 0.0) &
    (z_sam >= 0.0) & (z_sam <= z_max) &
    (m1_sam >= m_det_min) & (m1_sam <= m_det_max) &
    (m2_sam >= m_det_min) & (m2_sam <= m_det_max) &
    (m2_sam <= m1_sam)
)

m1_sam = m1_sam[mask]
m2_sam = m2_sam[mask]
dL_sam = dL_sam[mask]

lq_sam = logitq(m1_sam, m2_sam)
X_sam = np.column_stack([np.asarray(m1_sam), np.asarray(lq_sam), np.asarray(dL_sam)])
logX = gmm.score_samples(X_sam)

@jit
def logdiffexp(x, y):
    return x + jnp.log1p(jnp.exp(y-x))

jr = jax.random
rng = jr.PRNGKey(args.seed)
Nresamp = 150000

nsamp = 3343
Nobs= 69

zsels = z_of_dL(dLsels, H0Planck,Om0Planck)
m1sels = m1detsels/(1+zsels)
m2sels = m2detsels/(1+zsels)

z = z_of_dL(dL, H0Planck, Om0Planck)
m1 = m1det/(1+z)
m2 = m2det/(1+z)

W_tot = jnp.sum(1.0 / p_draw)
# Ndraw = m1sels.shape[0]

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

@jit
def likelihood_lvk_met1(gamma, m_min, m_max, alpha, beta, mu, sigma, f1):
    log_det_weights = log_p_pop_lvk(m1sels, m2sels, zsels, gamma, m_min, m_max, alpha, beta, mu, sigma, f1)
    log_det_weights += - jnp.log(p_draw)

    log_mu = logsumexp(log_det_weights) - jnp.log(Ndraw)
    log_s2 = logsumexp(2*log_det_weights) - 2.0*jnp.log(Ndraw)
    log_sigma2 = logdiffexp(log_s2, 2.0*log_mu - jnp.log(Ndraw))
    Neff = jnp.exp(2.0*log_mu - log_sigma2)

    ll = -jnp.inf
    ll = jnp.where((Neff <= 4 * Nobs), ll, 0.0)
    ll += -Nobs*log_mu + Nobs*(3 + Nobs)/(2*Neff)

    log_weights = log_p_pop_lvk(m1,m2,z,gamma,m_min,m_max,alpha,beta,mu,sigma,f1)
    log_weights += - jnp.log(ddL_of_z(z,dL,H0Planck,Om0Planck)) - 2 * jnp.log1p(z) - 2*jnp.log(dL)

    log_weights = log_weights.reshape((Nobs,nsamp))
    ll += jnp.sum(-jnp.log(nsamp) + logsumexp(log_weights,axis=-1))
    
    return ll, Neff

def loglike_method_1(coord):
    gamma, m_min, m_max, alpha, beta, mu, sigma, f1 = coord

    ll, Neff = likelihood_lvk_met1(
        gamma, m_min, m_max, alpha,
        beta, mu, sigma, f1,
    )
    
    if np.isnan(ll):
        return -np.inf
    elif (Neff < 4*Nobs):
        return -np.inf
    else:
        return ll

gamma_low = 0; gamma_high = 10
m_min_1_low = 2; m_min_1_high = 10
m_max_1_low = 50; m_max_1_high = 100
alpha_1_low = 0; alpha_1_high = 6
dm_min_1_low = 1; dm_min_1_high = 100
dm_max_1_low = 1; dm_max_1_high = 100
beta_low = 0; beta_high = 6
mu_low = 20; mu_high = 50
sigma_low = 1; sigma_high = 10
f1_low = 0; f1_high = 1

# if args.model == "lvk":
#     lower_bound = np.array([gamma_low, m_min_1_low, m_max_1_low, alpha_1_low, dm_min_1_low, beta_low, mu_low, sigma_low, f1_low])
#     upper_bound = np.array([gamma_high, m_min_1_high, m_max_1_high, alpha_1_high, dm_min_1_high, beta_high, mu_high, sigma_high, f1_high])
#     labels = ['gamma', 'm_min_1', 'm_max_1', 'alpha_1', 'dm_min_1', 'beta', 'mu', 'sigma', 'f1']
# else:
lower_bound = np.array([gamma_low, m_min_1_low, m_max_1_low, alpha_1_low, beta_low, mu_low, sigma_low, f1_low])
upper_bound = np.array([gamma_high, m_min_1_high, m_max_1_high, alpha_1_high, beta_high, mu_high, sigma_high, f1_high])
labels = ['gamma', 'm_min_1', 'm_max_1', 'alpha_1', 'beta', 'mu', 'sigma', 'f1']

ndims = len(lower_bound)
nlive = args.live_points

def prior_transform(theta):
    transformed_params = [
        theta[i] * (upper_bound[i] - lower_bound[i]) + lower_bound[i]
        for i in range(len(theta))
    ]
    return tuple(transformed_params)

from dynesty.utils import resample_equal
from dynesty import NestedSampler, DynamicNestedSampler

bound = 'multi'
sample = 'rwalk'

# dsampler = NestedSampler(loglike_method_1, prior_transform, ndims, bound=bound, sample=sample, nlive=nlive)
# dsampler.run_nested(dlogz=0.1)
# dsampler.save(f'{args.outdir}/{args.run}_{args.seed}met1.h5')
print('met1 saved')

K_candidates = range(1, 8) 
gmms = []
K = 7 

for e in range(Nobs):
    X_e = np.column_stack([np.asarray(m1det_[e]), np.asarray(m2det_[e]), np.asarray(dL_[e])])
    gmm = GaussianMixture(
        n_components=K, covariance_type='full',
        reg_covar=1e-6, n_init=10, random_state=args.seed
    ).fit(X_e)
    gmms.append(gmm)

wts_np   = np.stack([g.weights_ for g in gmms])       
means_np = np.stack([g.means_ for g in gmms])       
prec_np       = np.stack([g.precisions_ for g in gmms])        
prec_chol_np  = np.stack([g.precisions_cholesky_ for g in gmms])        

logdets_np = -2.0 * np.sum(np.log(np.diagonal(prec_chol_np, axis1=-2, axis2=-1)), axis=-1)  

wts       = jnp.asarray(wts_np, dtype=jnp.float64)   
mus       = jnp.asarray(means_np, dtype=jnp.float64)   
precisions= jnp.asarray(prec_np, dtype=jnp.float64)   
logdets   = jnp.asarray(logdets_np, dtype=jnp.float64)   

_event_kernel = lambda X, w, m, P, ld: gmm_logpdf_optimized(X, w, m, P, ld)
gmm_logpdf_per_event = jax.vmap(_event_kernel, in_axes=(None, 0, 0, 0, 0), out_axes=0)  

def swap12(x):
    return x.at[:, [0, 1]].set(x[:, [1, 0]])

def sort_masses(x):
    m1 = x[:, 0]; m2 = x[:, 1]
    hi = jnp.maximum(m1, m2)
    lo = jnp.minimum(m1, m2)
    return x.at[:, 0].set(hi).at[:, 1].set(lo)

def _log_gauss_with_precision(x_ND, mu_KD, Prec_KDD, logdetS_K):
    diffs = x_ND[:, None, :] - mu_KD[None, :, :]          
    quad  = jnp.einsum('nkd,kdj,nkj->nk', diffs, Prec_KDD, diffs)  
    D = x_ND.shape[1]
    return -0.5 * (D*jnp.log(2*jnp.pi) + logdetS_K[None, :] + quad)

def exact_log_gmm(x, w_K, mu_KD, Prec_KDD, logdetS_K):
    logw = jnp.log(jnp.clip(w_K, 1e-300, 1.0))  
    logN = _log_gauss_with_precision(x, mu_KD, Prec_KDD, logdetS_K)  
    return logsumexp(logw[None, :] + logN, axis=1)  

def log_gmm_symmetrized(x_sorted, w_K, mu_KD, Prec_KDD, logdetS_K):
    log_px      = exact_log_gmm(x_sorted, w_K, mu_KD, Prec_KDD, logdetS_K)
    log_px_swap = exact_log_gmm(swap12(x_sorted), w_K, mu_KD, Prec_KDD, logdetS_K)
    return logsumexp(jnp.stack([log_px, log_px_swap], axis=0), axis=0)  

def build_q_mixture(event_wts, event_mus, event_precs, event_logdets):
    E = len(event_wts)
    w_all    = jnp.concatenate(event_wts, axis=0)             
    mu_all   = jnp.concatenate(event_mus, axis=0)              
    Prec_all = jnp.concatenate(event_precs, axis=0)            
    ld_all   = jnp.concatenate(event_logdets, axis=0)          
    w_q = w_all / E
    w_q = w_q / jnp.sum(w_q)
    L_prec = jnp.linalg.cholesky(Prec_all)                     
    return w_q, mu_all, Prec_all, L_prec, ld_all

def sample_from_q(key, N, w_q, mu_all, L_prec):
    K_total, D = mu_all.shape
    key, sk1, sk2 = random.split(key, 3)
    comp_idx = random.choice(sk1, K_total, shape=(N,), p=w_q, replace=True)
    mu_ND    = mu_all[comp_idx]           
    L_NDD    = L_prec[comp_idx]           
    z        = random.normal(sk2, shape=(N, D))  
    y = jax.scipy.linalg.solve_triangular(L_NDD, z[..., None], lower=True).squeeze(-1)
    x = mu_ND + y
    return x

def log_q_ord(x_sorted, q_params):
    w_q, mu_all, Prec_all, _L_prec, ld_all = q_params
    return log_gmm_symmetrized(x_sorted, w_q, mu_all, Prec_all, ld_all)  

def sample_from_q_batched(key, batch_size, w_q, mu_all, L_prec):
    while True:
        key, subkey = random.split(key)
        yield key, sample_from_q(subkey, batch_size, w_q, mu_all, L_prec)

def prepare_guided_is_cache_batched(
    key, N_samples, batch_size, event_wts, event_mus, event_precs, event_logdets, support_fn
):
    q_params = build_q_mixture(event_wts, event_mus, event_precs, event_logdets)
    w_q, mu_all, Prec_all, L_prec, ld_all = q_params

    n_kept = 0
    xs, lqs, lpes = [], [], []
    sampler = sample_from_q_batched(key, batch_size, w_q, mu_all, L_prec)

    while n_kept < N_samples:
        key, x = next(sampler)              
        x_sorted = sort_masses(x)           

        mask = support_fn(x_sorted)
        if not jnp.any(mask):
            continue

        x_s = x_sorted[mask]                
        lq = log_gmm_symmetrized(x_s, w_q, mu_all, Prec_all, ld_all)                                   

        log_pe_list = []
        for w, mu, Prec, ld in zip(event_wts, event_mus, event_precs, event_logdets):
            log_pe_list.append(log_gmm_symmetrized(x_s, w, mu, Prec, ld))                               

        log_pe = jnp.stack(log_pe_list, axis=0)  

        xs.append(x_s)
        lqs.append(lq)
        lpes.append(log_pe)
        n_kept += x_s.shape[0]

    x_all   = jnp.concatenate(xs, axis=0)[:N_samples]
    lq_all  = jnp.concatenate(lqs, axis=0)[:N_samples]
    lpe_all = jnp.concatenate(lpes, axis=1)[:, :N_samples]

    return {"x_sorted": x_all, "log_q_ord": lq_all, "log_pe_ord": lpe_all, "q_params": q_params}

nsamp_pop = args.nsamp_pop
batch_size = 40000

key = random.PRNGKey(args.seed)

def support_mask(x, m2_min=1e-3, dL_min=1e-6, m1_max=jnp.inf, dL_max=jnp.inf):
    m1, m2, dL = x[:, 0], x[:, 1], x[:, 2]
    ok = (m1 >= m2) & (m2 >= m2_min) & (dL >= dL_min) & (m1 <= m1_max) & (dL <= dL_max)
    return ok

cache = prepare_guided_is_cache_batched(
    key, N_samples=150000, batch_size=batch_size,
    event_wts=wts, event_mus=mus, event_precs=precisions, event_logdets=logdets, support_fn=support_mask,
)

@jax.jit
def logpdf_sum_over_events_with_mega_eval(X, wts, mus, precisions, logdets):
    N, D = X.shape
    E, K = wts.shape
    EK = E * K
    M  = mus.reshape(EK, D)                 
    P  = precisions.reshape(EK, D, D)       
    LD = logdets.reshape(EK)                
    diffs = X[:, None, :] - M[None, :, :]                       
    quad  = jnp.einsum('nkd,kdj,nkj->nk', diffs, P, diffs)      
    log_comp = -0.5 * (D*jnp.log(2*jnp.pi) + LD + quad)         
    log_comp_NEK = log_comp.reshape(N, E, K)                    
    logw_NEK     = jnp.log(jnp.clip(wts, 1e-12, 1.0))[None,:,:] 
    logp_N_E = jsp.special.logsumexp(logw_NEK + log_comp_NEK, axis=-1)  
    logp_E_N = jnp.swapaxes(logp_N_E, 0, 1)                              
    total_logpdf = jnp.sum(logp_E_N)  
    return logp_E_N, total_logpdf

def logpdf_sum_chunked(X, wts, mus, precisions, logdets, chunk_size=40000):
    E, K = wts.shape
    N, D = X.shape

    def chunk_fn(X_chunk):
        diffs = X_chunk[:, None, None, :] - mus[None, :, :, :]   
        quad  = jnp.einsum('nekd,ekdf,nekf->nek', diffs, precisions, diffs)                                                        
        log_comp = -0.5 * (D*jnp.log(2*jnp.pi) + logdets[None,:,:] + quad)                                                        
        logw = jnp.log(jnp.clip(wts, 1e-12, 1.0))[None,:,:]
        logp = jsp.special.logsumexp(logw + log_comp, axis=-1)   
        return logp.T  

    chunks = []
    for i in range(0, N, chunk_size):
        chunks.append(chunk_fn(X[i:i+chunk_size]))

    logp_E_N = jnp.concatenate(chunks, axis=1)
    total = jnp.sum(logp_E_N)
    return logp_E_N, total

zsels = z_of_dL(dL_sam, H0Planck,Om0Planck)
m1sels = m1_sam/(1+zsels)
m2sels = m2_sam/(1+zsels)

x_sorted  = cache["x_sorted"]
m1dets_pop, m2dets_pop, dLs_pop = x_sorted[:, 0], x_sorted[:, 1], x_sorted[:, 2]

zs_pop = z_of_dL(dLs_pop, H0Planck)
m1s_pop = m1dets_pop/(1+zs_pop)
m2s_pop = m2dets_pop/(1+zs_pop)

X = jnp.column_stack([m1dets_pop, m2dets_pop, dLs_pop])
logp_E_N, total = logpdf_sum_chunked(X, wts, mus, precisions, logdets)

logJ = - jnp.log(ddL_of_z(zs_pop, dLs_pop, H0Planck, Om0Planck)) - 2*jnp.log1p(zs_pop) - 2*jnp.log(dLs_pop)
log_q = cache["log_q_ord"]
nsamp_pop = m1dets_pop.shape[0]

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

W_tot = jnp.sum(1.0 / p_draw)

@jit
def likelihood_lvk_met3(gamma, m_min, m_max, alpha, beta, mu, sigma, f1):
    log_det_weights = log_p_pop_lvk(m1sels, m2sels, zsels, gamma, m_min, m_max, alpha, beta, mu, sigma, f1)
    log_det_weights += - 2*jnp.log1p(zsels) - jnp.log(ddL_of_z(zsels,dL_sam,H0Planck, Om0Planck))

    N_gmm_sel = m1sels.shape[0] 
    log_norm_factor = jnp.log(W_tot) - jnp.log(Ndraw) - jnp.log(N_gmm_sel)
    log_mu = jsp.special.logsumexp(log_det_weights) + log_norm_factor
    Neff = jnp.exp(2.0 * jsp.special.logsumexp(log_det_weights) - jsp.special.logsumexp(2.0 * log_det_weights))

    ll = -jnp.inf
    ll = jnp.where((Neff <= 4 * Nobs), ll, 0.0)
    ll += -Nobs * log_mu + Nobs * (3 + Nobs) / (2 * Neff)

    log_pop = log_p_pop_lvk(m1s_pop,m2s_pop,zs_pop,gamma, m_min, m_max, alpha, beta, mu, sigma, f1)
    log_weights = log_pop - log_q

    per_event_log_like = jnp.nan_to_num(jsp.special.logsumexp(log_weights[None, :] + logp_E_N + logJ[None, :], axis=1)) - jnp.log(nsamp_pop)  
    ll += jnp.sum(per_event_log_like)
    return ll, Neff

def loglike_method_3(coord):
    gamma, m_min, m_max, alpha, beta, mu, sigma, f1 = coord

    ll, Neff = likelihood_lvk_met3(
        gamma, m_min, m_max, alpha,
        beta, mu, sigma, f1,
    )
    
    if np.isnan(ll):
        return -np.inf
    elif (Neff < 4*Nobs):
        return -np.inf
    else:
        return ll

d2sampler = NestedSampler(loglike_method_3, prior_transform, ndims, bound=bound, sample=sample, nlive=nlive)
d2sampler.run_nested(dlogz=0.1)
d2sampler.save(f'{args.outdir}/{args.run}_{args.seed}met3.h5')
