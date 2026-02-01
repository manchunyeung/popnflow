
import argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Plotting from cached samplers")
    p.add_argument("--indir", type=Path, required=True, help="Directory with sampler outputs")
    p.add_argument("--outdir", type=Path, required=True, help="Directory for plots")
    return p.parse_args()

args = parse_args()
args.outdir.mkdir(parents=True, exist_ok=True)

# Sampler outputs written by `run_gwtc3_inference.py` live in `--outdir` there,
# which should be passed as `--indir` here (e.g., met1.h5, met3.h5, ...).
def _restore_dynesty_sampler(path: Path, *, required: bool = True):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing sampler file: {path}")
        return None
    # dynesty expects a string path
    return dynesty.NestedSampler.restore(str(path))

# Commented out IPython magic to ensure Python compatibility.
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
# %matplotlib inline

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
    """
    Smoothed filter function

    See Eq. B5 in https://arxiv.org/pdf/2111.03634.pdf
    """
    def f(mm,deltaMM):
        return jnp.exp(deltaMM/mm + deltaMM/(mm-deltaMM))

    S_filter = 1./(f(m-m_min,dm_min) + 1.)
    S_filter = jnp.where(m<m_min+dm_min,S_filter,1.)
    S_filter = jnp.where(m>m_min,S_filter,0.)
    return S_filter

def Sfilter_high(m,m_max,dm_max):
    """
    Smoothed filter function

    See Eq. B5 in https://arxiv.org/pdf/2111.03634.pdf
    """
    def f(mm,deltaMM):
        return jnp.exp(deltaMM/mm + deltaMM/(mm-deltaMM))

    S_filter = 1./(f(m-m_max,-dm_max) + 1.)
    S_filter = jnp.where(m>m_max-dm_max,S_filter,1.)
    S_filter = jnp.where(m<m_max,S_filter,0.)
    return S_filter

@jit
def logpm1_powerlaw(m1,m_min,m_max,alpha,dm_min,dm_max):

    pm1 = Sfilter_low(mass,m_min,dm_min)*mass**(-alpha)*Sfilter_high(mass,m_max,dm_max)
    # jax.debug.print('{}', pm1)
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
    # beta=2
    q = m2/m1
    pq = mass_ratio**beta
    pq = pq/jnp.trapezoid(pq,mass_ratio)
    # jax.debug.print("mr: {}", mass_ratio)
    # jax.debug.print("pq: {}",(mass_ratio**beta))
    # jax.debug.print("trap:{}", jnp.trapezoid(pq, mass_ratio))
    # jax.debug.print("pqd: {}", pq)

    log_pq = jnp.log(jnp.interp(q,mass_ratio,pq))

    return log_pq

@jit
def log_p_pop_pl_pl(m1,m2,z,gamma, m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,beta,mu,sigma,f1):
    # start_time = time.time()
    log_dNdm1 = logpm1_powerlaw_powerlaw(m1,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,mu,sigma,f1)
    log_dNdm2 = logpm1_powerlaw_powerlaw(m2,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,mu,sigma,f1)
    q = m2/m1
    # log_pq = beta * jnp.log(q) - jnp.log(beta + 1)
    log_pq = logfq(m1,m2,beta)
    # jax.debug.print('m1{}', log_dNdm1)
    # jax.debug.print('q{}', log_fq)
    log_dvdz = jnp.log(dV_of_z_normed(z,Om0Planck,gamma))
    # log_dvdz = 1

    # jax.debug.print('dVdz result: {}', log_dNdm1)

    log_p_sz = np.log(0.25) # 1/2 for each spin dimension

    log_p = log_p_sz + log_dNdm1 + log_dNdm2 + log_pq + log_dvdz
    log_p = jnp.where(m2<m1, log_p, -jnp.inf)

    # end_time = time.time()
    # print('time0', end_time-start_time)
    # return log_p_sz + log_dNdm1 + log_dNdm2 + log_fq + log_dvdz
    return log_p

@jit
def fq(q,beta):
    # q = m2/m1
    pq = mass_ratio**beta
    pq = pq/jnp.trapezoid(pq,mass_ratio)

    # jax.debug.print("pq: {}",(mass_ratio**beta).max())
    # jax.debug.print("trap:{}", jnp.trapezoid(pq, mass_ratio))

    log_pq = jnp.interp(q,mass_ratio,pq)

    return log_pq

@jit
def dV_of_z_normed(z,Om0,gamma):
    dV = dV_of_z(zgrid,H0Planck,Om0)*(1+zgrid)**(gamma-1)
    prob = dV/jnp.trapezoid(dV,zgrid)
    return jnp.interp(z,zgrid,prob)

@jit
def likelihood_method_1(gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1):
    # Ndraw = m1sels.shape[0]

    # new_x = gmm_sample_jax(subkey, Nresamp, gmm_params)
    # m1_sam, m2_sam, dL_sam = new_x[:, 0], new_x[:, 1], new_x[:, 2]

    # z_sam = z_of_dL(dL_sam, H0Planck)
    # m_det_min = m_src_min * (1.0 + z_sam)
    # m_det_max = m_src_max * (1.0 + z_sam)

    # mask = (
    #     (dL_sam > 0.0) &
    #     (z_sam >= 0.0) & (z_sam <= z_max) &
    #     (m1_sam >= m_det_min) & (m1_sam <= m_det_max) &
    #     (m2_sam >= m_det_min) & (m2_sam <= m_det_max) &
    #     (m2_sam <= m1_sam)
    # )

    # # m1_sam = m1_sam[mask]
    # # m2_sam = m2_sam[mask]
    # # dL_sam = dL_sam[mask]

    # zsels = z_of_dL(dL_sam, H0Planck,Om0Planck)
    # m1sels = m1_sam/(1+zsels)
    # m2sels = m2_sam/(1+zsels)


    log_det_weights = log_p_pop_pl_pl(m1sels,m2sels,zsels,gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1)
    log_det_weights += - jnp.log(p_draw) - 2*jnp.log1p(zsels) - jnp.log(ddL_of_z(zsels,dLsels,H0Planck, Om0Planck))
    # log_det_weights += - 2*jnp.log1p(zsels) - jnp.log(ddL_of_z(zsels,dL_sam,H0Planck, Om0Planck))
    # log_det_weights = jnp.where(mask, log_det_weights, -jnp.inf)

    log_mu = logsumexp(log_det_weights) - jnp.log(Ndraw)
    log_s2 = logsumexp(2*log_det_weights) - 2.0*jnp.log(Ndraw)
    log_sigma2 = logdiffexp(log_s2, 2.0*log_mu - jnp.log(Ndraw))
    Neff = jnp.exp(2.0*log_mu - log_sigma2)

    ll = -jnp.inf
    ll = jnp.where((Neff <= 4 * Nobs), ll, 0)
    ll += -Nobs*log_mu + Nobs*(3 + Nobs)/(2*Neff)

    # ll, Neff = log_mu_selection(gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1, rng, Nresamp, qdet_params, pinj_params)
    # print(ll)
    # jax.debug.print('ll{}', ll)

    # ll = 0
    log_weights = log_p_pop_pl_pl(m1,m2,z,gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1)
    log_weights += - jnp.log(ddL_of_z(z,dL,H0Planck,Om0Planck)) - 2 * jnp.log1p(z) - 2*jnp.log(dL) # jacobian

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

import jax.numpy as jnp
import numpy as np

gamma_low = 0
gamma_high = 10

m_min_1_low = 2
m_min_1_high = 10

m_max_1_low = 50
m_max_1_high = 100

alpha_1_low = 0
alpha_1_high = 6

dm_min_1_low = 1
dm_min_1_high = 100

dm_max_1_low = 1
dm_max_1_high = 100

beta_low = 0
beta_high = 6

mu_low = 20
mu_high = 50

sigma_low = 1
sigma_high = 10

f1_low = 0
f1_high = 1

muz_lo = 0.1
muz_hi = 2.0

sigmaz_lo = 0.01
sigmaz_hi = 0.5

lower_bound = np.array([gamma_low, m_min_1_low,m_max_1_low,alpha_1_low,dm_min_1_low,dm_max_1_low,beta_low,mu_low,sigma_low,f1_low])
upper_bound = np.array([gamma_high, m_min_1_high,m_max_1_high,alpha_1_high,dm_min_1_high,dm_max_1_high,beta_high,mu_high,sigma_high,f1_high,])

#priors
ndims = len(lower_bound)
nlive = 1000


labels = ['gamma', 'm_min_1','m_max_1','alpha_1','dm_min_1','dm_max_1','beta', 'mu','sigma','f1']


def prior_transform(theta):
    transformed_params = [
        theta[i] * (upper_bound[i] - lower_bound[i]) + lower_bound[i]
        for i in range(len(theta))
    ]

    return tuple(transformed_params)

import dynesty
from dynesty import utils as dyfunc

# load saved sampler
sampler = _restore_dynesty_sampler(args.indir / "met1.h5", required=True)
dres = sampler.results

dlogZdynesty = dres.logz[-1]        # value of logZ
dlogZerrdynesty = dres.logzerr[-1]  # estimate of the statistcal uncertainty on logZ

from dynesty.utils import resample_equal
dweights = np.exp(dres['logwt'] - dres['logz'][-1])
dpostsamples = resample_equal(dres.samples, dweights)

import jax
jax.config.update("jax_enable_x64", True)

@jit
def log_pm1m2(m1, m2, m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1):
    log_dNdm1 = logpm1_powerlaw_powerlaw(m1,m_min,m_max,alpha,dm_min,dm_max,mu,sigma,f1)
    log_dNdm2 = logpm1_powerlaw_powerlaw(m2,m_min,m_max,alpha,dm_min,dm_max,mu,sigma,f1)
    q = m2/m1
    # log_pq = beta * jnp.log(q) - jnp.log(beta + 1)
    log_pq = logfq(m1,m2,beta)

    log_p = log_dNdm1 + log_dNdm2 + log_pq
    log_p = jnp.where(m2<m1, log_p, -jnp.inf)
    return log_p

m1s = jnp.linspace(1.0, 100.0, 300, dtype=jnp.float64)
m2s = jnp.linspace(1.0, 100.0, 300, dtype=jnp.float64)
dm1 = m1s[1] - m1s[0]
dm2 = m2s[1] - m2s[0]

M1, M2 = jnp.meshgrid(m1s, m2s, indexing="ij")

@jit
def pm1_from_lambda(lam):
    """
    lam = (m_min, m_max, alpha, dm_min, dm_max, beta, mu, sigma, f1)
    returns p(m1 | lam) evaluated on m1_grid
    """

    m_min, m_max, alpha, dm_min, dm_max, beta, mu, sigma, f1 = lam

    log_joint = log_pm1m2(
        M1, M2,
        m_min, m_max, alpha, dm_min, dm_max,
        beta, mu, sigma, f1
    )

    # exponentiate safely
    log_norm = logsumexp(log_joint) + jnp.log(dm1 * dm2)

    # normalized log joint
    log_pjoint = log_joint - log_norm

    # marginal p(m1) = ∫ p(m1,m2) dm2  -> log p(m1) via logsumexp over m2 axis
    log_pm1 = logsumexp(log_pjoint, axis=1) + jnp.log(dm2)

    pm1 = jnp.exp(log_pm1)
    return pm1

lambda_samples = jnp.asarray(dpostsamples[::5, 1:], dtype=jnp.float64)

pm1_all = jax.jit(vmap(pm1_from_lambda))(lambda_samples)
# shape: (Nposterior, Nm)
p5, p50, p95 = jnp.percentile(
    pm1_all,
    jnp.array([5, 50, 95]),
    axis=0
)

import matplotlib.pyplot as plt

plt.figure(figsize=(6.5, 3.5))

plt.fill_between(m1s, p5, p95, alpha=0.3)
plt.plot(m1s, p50, lw=2)
plt.xlim(4, 80)
plt.ylim(1e-6, 1)
plt.xlabel(r"$m_1\,[M_\odot]$")
plt.ylabel(r"$p(m_1 \mid \lambda)$")
plt.yscale("log")

plt.tight_layout()
plt.show()

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
    log_weights += - jnp.log(ddL_of_z(z,dL,H0Planck,Om0Planck)) - 2 * jnp.log1p(z) - 2*jnp.log(dL) # jacobian

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

_met1_sel = _restore_dynesty_sampler(args.indir / "met1_sel.h5", required=False)
if _met1_sel is not None:
    d2sampler = _met1_sel
    dres = d2sampler.results

    dlogZdynesty = dres.logz[-1]        # value of logZ
    dlogZerrdynesty = dres.logzerr[-1]  # estimate of the statistcal uncertainty on logZ

    from dynesty.utils import resample_equal
    dweights = np.exp(dres['logwt'] - dres['logz'][-1])
    d2postsamples = resample_equal(dres.samples, dweights)

    lambda_samples2 = jnp.asarray(d2postsamples[::5, 1:], dtype=jnp.float64)

    pm1_all2 = jax.jit(vmap(pm1_from_lambda))(lambda_samples2)
    # shape: (Nposterior, Nm)
    p5_2, p50_2, p95_2 = jnp.percentile(
        pm1_all2,
        jnp.array([5, 50, 95]),
        axis=0
    )

    lambda_samples2 = jnp.asarray(d2postsamples[:1000, 1:], dtype=jnp.float64)

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

    # ll = 0
    # start_time = time.time()
    log_pop = log_p_pop_pl_pl(m1s_pop,m2s_pop,zs_pop,gamma,m_min,m_max,alpha,dm_min,dm_max,beta,mu,sigma,f1)
    log_weights = log_pop - log_q
    # log p_event(X) for all events: (E,N)
    # logp_E_N = gmm_logpdf_per_event(X, wts, mus, precisions, logdets)

    # Same Jacobian as your KDE version (original units)
    per_event_log_like = jnp.nan_to_num(jsp.special.logsumexp(log_weights[None, :] + logp_E_N + logJ[None, :], axis=1)) - jnp.log(nsamp_pop)  # (E,)

    ll += jnp.sum(per_event_log_like)
    # end_time = time.time()
    # duration = end_time - start_time

    # jax.debug.print('duration{}', duration)
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

_met3 = _restore_dynesty_sampler(args.indir / "met3.h5", required=False)
if _met3 is not None:
    d3sampler = _met3
    dres = d3sampler.results

    dlogZdynesty = dres.logz[-1]        # value of logZ
    dlogZerrdynesty = dres.logzerr[-1]  # estimate of the statistcal uncertainty on logZ

    from dynesty.utils import resample_equal
    dweights = np.exp(dres['logwt'] - dres['logz'][-1])
    d3postsamples = resample_equal(dres.samples, dweights)

    lambda_samples3 = jnp.asarray(d3postsamples[::5, 1:], dtype=jnp.float64)

    pm1_all3 = jax.jit(vmap(pm1_from_lambda))(lambda_samples3)
    # shape: (Nposterior, Nm)
    p5_3, p50_3, p95_3 = jnp.percentile(
        pm1_all3,
        jnp.array([5, 50, 95]),
        axis=0
    )

# plt.figure(figsize=(6.5, 3.5))

plt.fill_between(m1s, p5, p95, alpha=0.3)
plt.plot(m1s, p50, lw=2, label='standard pop')

if _met1_sel is not None:
    plt.fill_between(m1s, p5_2, p95_2, alpha=0.3)
    plt.plot(m1s, p50_2, lw=2, label='gmm sel')

if _met3 is not None:
    plt.fill_between(m1s, p5_3, p95_3, alpha=0.3)
    plt.plot(m1s, p50_3, lw=2, label='gmm sel+events')

plt.xlim(4, 80)
plt.ylim(1e-6, 1)
plt.xlabel(r"$m_1\,[M_\odot]$")
plt.ylabel(r"$p(m_1 \mid \lambda)$")
plt.yscale("log")

plt.legend()

plt.tight_layout()
plt.savefig(args.outdir / "pm1_posteriors.png", dpi=200)
plt.close()

@jit
def pm2_from_lambda(lam):
    """
    lam = (m_min, m_max, alpha, dm_min, dm_max, beta, mu, sigma, f1)
    returns p(m1 | lam) evaluated on m1_grid
    """

    m_min, m_max, alpha, dm_min, dm_max, beta, mu, sigma, f1 = lam

    log_joint = log_pm1m2(
        M1, M2,
        m_min, m_max, alpha, dm_min, dm_max,
        beta, mu, sigma, f1
    )

    # exponentiate safely
    log_norm = logsumexp(log_joint) + jnp.log(dm1 * dm2)

    # normalized log joint
    log_pjoint = log_joint - log_norm

    # marginal p(m1) = ∫ p(m1,m2) dm2  -> log p(m1) via logsumexp over m2 axis
    log_pm1 = logsumexp(log_pjoint, axis=0) + jnp.log(dm1)

    pm1 = jnp.exp(log_pm1)
    return pm1

if _met3 is not None:
    lambda_samples3 = jnp.asarray(d3postsamples[:1000, 1:], dtype=jnp.float64)

pm2_all = jax.jit(vmap(pm2_from_lambda))(lambda_samples)
# shape: (Nposterior, Nm)
p5, p50, p95 = jnp.percentile(
    pm2_all,
    jnp.array([5, 50, 95]),
    axis=0
)

if _met1_sel is not None:
    pm2_all2 = jax.jit(vmap(pm2_from_lambda))(lambda_samples2)
    # shape: (Nposterior, Nm)
    p5_2, p50_2, p95_2 = jnp.percentile(
        pm2_all2,
        jnp.array([5, 50, 95]),
        axis=0
    )

if _met3 is not None:
    pm2_all3 = jax.jit(vmap(pm2_from_lambda))(lambda_samples3)
    # shape: (Nposterior, Nm)
    p5_3, p50_3, p95_3 = jnp.percentile(
        pm2_all3,
        jnp.array([5, 50, 95]),
        axis=0
    )

# plt.figure(figsize=(6.5, 3.5))

plt.fill_between(m2s, p5, p95, alpha=0.3)
plt.plot(m2s, p50, lw=2, label='standard pop')

if _met1_sel is not None:
    plt.fill_between(m2s, p5_2, p95_2, alpha=0.3)
    plt.plot(m2s, p50_2, lw=2, label='gmm sel')

if _met3 is not None:
    plt.fill_between(m2s, p5_3, p95_3, alpha=0.3)
    plt.plot(m2s, p50_3, lw=2, label='gmm sel+events')

plt.xlim(4, 80)
plt.ylim(1e-6, 1)
plt.xlabel(r"$m_2\,[M_\odot]$")
plt.ylabel(r"$p(m_2 \mid \lambda)$")
plt.yscale("log")

plt.legend()

plt.tight_layout()
plt.savefig(args.outdir / "pm2_posteriors.png", dpi=200)
plt.close()




# =============================
# End of production wrapper
# =============================
