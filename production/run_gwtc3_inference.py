
import argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="GWTC-3 population inference")
    p.add_argument("--catalog", type=str, default="GWTC3")
    p.add_argument("--indir", type=Path, required=True, help="Input data directory")
    p.add_argument("--outdir", type=Path, required=True, help="Output directory")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--nsamp-pop", type=int, default=200000)
    return p.parse_args()

args = parse_args()
args.outdir.mkdir(parents=True, exist_ok=True)

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

ra = np.loadtxt(f'{args.indir}/ra.txt')
dec = np.loadtxt(f'{args.indir}/dec.txt')
dL = np.loadtxt(f'{args.indir}/dL.txt')
m2det = np.loadtxt(f'{args.indir}/m2det.txt')
m1det = np.loadtxt(f'{args.indir}/m1det.txt')

ra = jnp.array(ra)
dec = jnp.array(dec)
dL = jnp.array(dL)
m2det = jnp.array(m2det)
m1det = jnp.array(m1det)
q = m2det/m1det

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

log_p_pop_pl_pl(35,30,.1, 3.48562259e+00, 2.02821055e+00, 9.81226509e+01, 3.87235476e+00,
 3.92984552e+01, 3.01971181e+01, 5.70909256e+00, 2.73591739e+01,
 8.53809384e+00, 6.53853494e-0)

from functools import partial
jr = jax.random
from functools import partial
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit

# draw N observations from pop model (IN SOURCE FRAME)

Nobs = 69

dL_ = dL.reshape(Nobs, 4096)
m1det_ = m1det.reshape(Nobs, 4096)
m2det_ = m2det.reshape(Nobs, 4096)

injection_file = "/content/drive/MyDrive/popnflow_goog/endo3_bbhpop-LIGO-T2100113-v12.hdf5"
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

selection = {
    'cwb': pastro_cwb > 0.5,
    'gstlal': pastro_gstlal > 0.5,
    'mbta': pastro_mbta > 0.5,
    'pycbc_bbh': pastro_pycbc_bbh > 0.5,
    'pycbc_broad': pastro_pycbc_broad > 0.5,
    'any': ((pastro_cwb > 0.5) | (pastro_gstlal > 0.5) | (pastro_mbta > 0.5) |
            (pastro_pycbc_bbh > 0.5) | (pastro_pycbc_broad > 0.5) ),
}

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

# Ndraw0 = m1detsels.shape[0]

sels = selection_ifar['cbc']
m1detsels = jnp.array(m1detsels[sels])
m2detsels = jnp.array(m2detsels[sels])
dLsels = jnp.array(dLsels[sels])
rasels = jnp.array(rasels[sels])
decsels = jnp.array(decsels[sels])
p_draw = jnp.array(p_draw[sels])

# Ndet = m1detsels.shape[0]

Ndraw

from sklearn.mixture import GaussianMixture
import numpy as np
import jax.scipy as jsp

@jax.jit
def gmm_logpdf_optimized(x_batch, weights, means, precisions, logdets):
    """
    x_batch:   (N, D)
    weights:   (K,)
    means:     (K, D)
    precisions:(K, D, D)
    logdets:   (K,)
    returns:   (N,)
    """
    D = means.shape[-1]
    diffs = x_batch[:, None, :] - means[None, :, :]                # (N,K,D)
    quad = jnp.einsum('nkd,kdj,nkj->nk', diffs, precisions, diffs)  # (N,K)
    log_comp = -0.5 * (D*jnp.log(2*jnp.pi) + logdets + quad)       # (N,K)
    logw = jnp.log(jnp.clip(weights, 1e-12, 1.0))[None, :]         # (1,K)
    return jsp.special.logsumexp(logw + log_comp, axis=-1)         # (N,)

def chirp_mass(m1, m2):
  return (m1*m2)**(3/5) / (m1+m2)**(1/5)

from scipy.special import logit
def logitq(m1, m2):
  q = m2/m1
  return logit(q)

lq = logitq(m1detsels, m2detsels)

X = np.column_stack([np.asarray(m1detsels), np.asarray(lq), np.asarray(dLsels)])

N = m1detsels.shape[0]

w = 1/p_draw

idx = rng_np.choice(
    N,
    size=N,
    replace=True,
    p=w / np.sum(w)
)

X_resam = X[idx]

m1_sels, lq_sels, dL_sels = X_resam[:, 0], X_resam[:, 1], X_resam[:, 2]

from scipy.special import expit

q_sels = expit(lq_sels)
m2_sels = m1_sels * q_sels

# plt.scatter(m1_sels, m2_sels)

plt.scatter(m2_sels, dL_sels)

X_resam.shape

from sklearn.mixture import GaussianMixture

# X = np.column_stack([np.asarray(m1detsels), np.asarray(m2detsels), np.asarray(dLsels)])  # (N_e, D)
K = 7

# sam_weights = 1/p_draw

gmm = GaussianMixture(
    n_components=K, covariance_type='full',
    random_state=args.seed,
).fit(X_resam)

new_x = gmm.sample(150000)

m1_sam, lq_sam, dL_sam = new_x[0][:, 0], new_x[0][:, 1], new_x[0][:, 2]

q_sam = expit(lq_sam)
m2_sam = m1_sam * q_sam

# plt.scatter(m1_sam, m2_sam)

# m1_max = 100

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

# q_sam = m2_sam/m1_sam
lq_sam = logitq(m1_sam, m2_sam)
X_sam = np.column_stack([np.asarray(m1_sam), np.asarray(lq_sam), np.asarray(dL_sam)])

logX = gmm.score_samples(X_sam)

# plt.scatter(m2_sam, dL_sam)

# plt.scatter(m1_sam, m2_sam)

@jit
def logdiffexp(x, y):
    return x + jnp.log1p(jnp.exp(y-x))

jr = jax.random
rng = jr.PRNGKey(args.seed)
Nresamp = 150000

#Inference method 1. use samples directly with pop models define via analytical parametric models above
from jax.scipy.stats import norm as norm_jax

nsamp = 4096
Nobs= 69

zsels = z_of_dL(dLsels, H0Planck,Om0Planck)
m1sels = m1detsels/(1+zsels)
m2sels = m2detsels/(1+zsels)

# zsels = z_of_dL(dL_sam, H0Planck,Om0Planck)
# m1sels = m1_sam/(1+zsels)
# m2sels = m2_sam/(1+zsels)

# eps = 1e-6
# qsels = m2sels/m1sels
# qsels = jnp.clip(qsels, eps, 1.0 - eps)
# log_q = logX -jnp.log(m1sels) - jnp.log(qsels) - jnp.log1p(-qsels)

z = z_of_dL(dL, H0Planck, Om0Planck)
m1 = m1det/(1+z)
m2 = m2det/(1+z)

Ndraw = m1sels.shape[0]

# @partial(jax.jit, static_argnames=("gmm_params",))
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

likelihood_method_1(3, 5, 80, 1.5, 2, 3, 1, 35, 5, 0.2)

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

#sampling

from dynesty.utils import resample_equal
from dynesty import NestedSampler, DynamicNestedSampler
import multiprocessing as multi

bound = 'multi'
sample = 'rwalk'
nprocesses = 1

dsampler = NestedSampler(loglike_method_1, prior_transform, ndims, bound=bound, sample=sample, nlive=200)
dsampler.run_nested(dlogz=0.1)

import corner

dres = dsampler.results

dlogZdynesty = dres.logz[-1]        # value of logZ
dlogZerrdynesty = dres.logzerr[-1]  # estimate of the statistcal uncertainty on logZ

# output marginal likelihood
print('Marginalised evidence (using dynamic sampler) is {} ± {}'.format(dlogZdynesty, dlogZerrdynesty))

# get the posterior samples
dweights = np.exp(dres['logwt'] - dres['logz'][-1])
dpostsamples = resample_equal(dres.samples, dweights)

print('Number of posterior samples (using dynamic sampler) is {}'.format(dpostsamples.shape[0]))

fig1 = corner.corner(dpostsamples,  hist_kwargs={'density': True},labels=labels)
plt.show()

from sklearn.mixture import GaussianMixture
import numpy as np
import jax.scipy as jsp

K_candidates = range(1, 8)  # e.g. try 1..10 components

gmms = []
best_Ks = []

for e in range(Nobs):
    X_e = np.column_stack([
        np.asarray(m1det_[e]),
        np.asarray(m2det_[e]),
        np.asarray(dL_[e])
    ])  # (N_e, D)

    best_gmm = None
    best_bic = np.inf

    for K in K_candidates:
        gmm = GaussianMixture(
            n_components=K,
            covariance_type='full',
            reg_covar=1e-6,
            n_init=5,         # you can reduce n_init here to save time
            random_state=args.seed,
        ).fit(X_e)

        bic = gmm.bic(X_e)  # or gmm.aic(X_e)

        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm

    gmms.append(best_gmm)
    best_Ks.append(best_gmm.n_components)

# print("Chosen K per event:", best_Ks)
K_max = max(best_Ks)
print('K_max', K_max)

K = 7  # try BIC/AIC later
gmms = []
for e in range(Nobs):
    X_e = np.column_stack([np.asarray(m1det_[e]), np.asarray(m2det_[e]), np.asarray(dL_[e])])  # (N_e, D)
    gmm = GaussianMixture(
        n_components=K, covariance_type='full',
        reg_covar=1e-6, n_init=10, random_state=args.seed
    ).fit(X_e)
    gmms.append(gmm)

# Stack params
wts_np   = np.stack([g.weights_              for g in gmms])       # (E,K)
means_np = np.stack([g.means_                for g in gmms])       # (E,K,D)

# Precisions (and their Cholesky) are directly provided by sklearn
prec_np       = np.stack([g.precisions_             for g in gmms])        # (E,K,D,D)
prec_chol_np  = np.stack([g.precisions_cholesky_    for g in gmms])        # (E,K,D,D)

# log|Σ| = -log|Precision| = -2 * sum(log diag(L_precision))
logdets_np = -2.0 * np.sum(
    np.log(np.diagonal(prec_chol_np, axis1=-2, axis2=-1)),
    axis=-1
)  # (E,K)

# → JAX float64
wts       = jnp.asarray(wts_np,      dtype=jnp.float64)   # (E,K)
mus       = jnp.asarray(means_np,    dtype=jnp.float64)   # (E,K,D)
precisions= jnp.asarray(prec_np,     dtype=jnp.float64)   # (E,K,D,D)
logdets   = jnp.asarray(logdets_np,  dtype=jnp.float64)   # (E,K)

_event_kernel = lambda X, w, m, P, ld: gmm_logpdf_optimized(X, w, m, P, ld)
gmm_logpdf_per_event = jax.vmap(_event_kernel, in_axes=(None, 0, 0, 0, 0), out_axes=0)  # (E,N)

def swap12(x):
    # x: (N, 3) columns [m1, m2, dL]
    return x.at[:, [0, 1]].set(x[:, [1, 0]])

def sort_masses(x):
    # Enforce m1>=m2 by sorting the first two columns; keep dL as-is.
    m1 = x[:, 0]; m2 = x[:, 1]
    hi = jnp.maximum(m1, m2)
    lo = jnp.minimum(m1, m2)
    return x.at[:, 0].set(hi).at[:, 1].set(lo)

def _log_gauss_with_precision(x_ND, mu_KD, Prec_KDD, logdetS_K):
    """
    x:     (N, D)
    mu:    (K, D)
    Prec:  (K, D, D)  (precision matrices)
    logdetS: (K,)     (log det of covariance)
    returns: (N, K)   log N(x | mu_k, S_k) using precisions
    """
    diffs = x_ND[:, None, :] - mu_KD[None, :, :]          # (N,K,D)
    quad  = jnp.einsum('nkd,kdj,nkj->nk', diffs, Prec_KDD, diffs)  # (N,K)
    D = x_ND.shape[1]
    return -0.5 * (D*jnp.log(2*jnp.pi) + logdetS_K[None, :] + quad)

def exact_log_gmm(x, w_K, mu_KD, Prec_KDD, logdetS_K):
    """
    Exact log pdf of a single GMM at x: (N,), using all components.
    """
    logw = jnp.log(jnp.clip(w_K, 1e-300, 1.0))  # (K,)
    logN = _log_gauss_with_precision(x, mu_KD, Prec_KDD, logdetS_K)  # (N,K)
    return logsumexp(logw[None, :] + logN, axis=1)  # (N,)

def log_gmm_symmetrized(x_sorted, w_K, mu_KD, Prec_KDD, logdetS_K):
    """
    Symmetrized GMM density on the ordered space (m1>=m2):
    p_ord(x) = p(x) + p(swap(x)).
    Returns log p_ord(x_sorted) as (N,).
    """
    log_px      = exact_log_gmm(x_sorted, w_K, mu_KD, Prec_KDD, logdetS_K)
    log_px_swap = exact_log_gmm(swap12(x_sorted), w_K, mu_KD, Prec_KDD, logdetS_K)
    return logsumexp(jnp.stack([log_px, log_px_swap], axis=0), axis=0)  # (N,)

# ---------------------------
# Build global proposal q = average of event posteriors
# ---------------------------

def build_q_mixture(event_wts, event_mus, event_precs, event_logdets):
    """
    Inputs are lists over events of arrays with shapes (K_e, ...).
    Returns concatenated mixture parameters for q (weights normalized).
    """
    E = len(event_wts)
    w_all    = jnp.concatenate(event_wts, axis=0)             # (K_total,)
    mu_all   = jnp.concatenate(event_mus, axis=0)              # (K_total, 3)
    Prec_all = jnp.concatenate(event_precs, axis=0)            # (K_total, 3, 3)
    ld_all   = jnp.concatenate(event_logdets, axis=0)          # (K_total,)

    # Simple average of posteriors: scale weights by 1/E and re-normalize (for safety)
    w_q = w_all / E
    w_q = w_q / jnp.sum(w_q)

    # Cholesky of precision for sampling: Prec = L @ L^T (L lower-tri)
    L_prec = jnp.linalg.cholesky(Prec_all)                     # (K_total, 3, 3)
    return w_q, mu_all, Prec_all, L_prec, ld_all

# ---------------------------
# Sampling from q and evaluating q_ord
# ---------------------------

def sample_from_q(key, N, w_q, mu_all, L_prec):
    """
    Vectorized sampling from the global mixture q using precision Cholesky.
    Returns x: (N,3)
    """
    K_total, D = mu_all.shape
    key, sk1, sk2 = random.split(key, 3)
    comp_idx = random.choice(sk1, K_total, shape=(N,), p=w_q, replace=True)
    mu_ND    = mu_all[comp_idx]           # (N, D)
    L_NDD    = L_prec[comp_idx]           # (N, D, D)
    z        = random.normal(sk2, shape=(N, D))  # (N, D)

    # Solve L y = z -> y = L^{-1} z
    # Use triangular_solve per-sample
    y = jax.scipy.linalg.solve_triangular(L_NDD, z[..., None], lower=True).squeeze(-1)
    x = mu_ND + y
    return x

def log_q_ord(x_sorted, q_params):
    w_q, mu_all, Prec_all, _L_prec, ld_all = q_params
    return log_gmm_symmetrized(x_sorted, w_q, mu_all, Prec_all, ld_all)  # (N,)

# ---------------------------
# CACHING STAGE (run once)
# ---------------------------

def prepare_guided_is_cache(key, N_samples, event_wts, event_mus, event_precs, event_logdets):
    """
    Build q, draw samples, sort masses, and cache:
      - x_sorted: (N,3)
      - log_q_ord: (N,)
      - log_pe_ord: (E,N)
    Everything here is hyperparameter-independent.
    """
    # 1) Build q from event posteriors
    q_params = build_q_mixture(event_wts, event_mus, event_precs, event_logdets)
    print(len(q_params))
    # 2) Sample once and sort masses
    x = sample_from_q(key, N_samples, q_params[0], q_params[1], q_params[3])   # ignore ld in sampler
    x_sorted = sort_masses(x)                           # (N,3)

    # 3) Cache log q_ord(x_i)
    lq = log_q_ord(x_sorted, q_params)                  # (N,)

    # 4) Cache per-event symmetrized log posterior at x_i
    #    (Python loop is fine; this is outside the likelihood and runs once)
    log_pe_list = []
    for w, mu, Prec, ld in zip(event_wts, event_mus, event_precs, event_logdets):
        log_pe_list.append(log_gmm_symmetrized(x_sorted, w, mu, Prec, ld))  # (N,)
    log_pe = jnp.stack(log_pe_list, axis=0)                                  # (E, N)

    cache = {
        "x_sorted": x_sorted,       # (N,3)
        "log_q_ord": lq,            # (N,)
        "log_pe_ord": log_pe,       # (E,N)
        "q_params": q_params,       # if you want to reuse proposal later
    }
    return cache

nsamp_pop = args.nsamp_pop

key = random.PRNGKey(args.seed)
cache = prepare_guided_is_cache(
    key, N_samples=nsamp_pop,
    event_wts=wts, event_mus=mus,
    event_precs=precisions, event_logdets=logdets
)

def support_mask(x, m2_min=1e-3, dL_min=1e-6, m1_max=jnp.inf, dL_max=jnp.inf):
    """
    x: (N,3) = [m1_sorted >= m2_sorted, m2_sorted, dL]
    Returns boolean mask for physical support S.
    """
    m1, m2, dL = x[:, 0], x[:, 1], x[:, 2]
    ok = (m1 >= m2) & (m2 >= m2_min) & (dL >= dL_min) & (m1 <= m1_max) & (dL <= dL_max)
    return ok


mask = support_mask(cache["x_sorted"])

cache['x_sorted'] = cache['x_sorted'][mask]
cache['log_q_ord'] = cache['log_q_ord'][mask]

import jax
import jax.numpy as jnp
import jax.scipy as jsp

@jax.jit
def logpdf_sum_over_events_with_mega_eval(X, wts, mus, precisions, logdets):
    """
    X:          (N, D)           samples theta ~ pop(theta|lambda)  (same X for all events)
    wts:        (E, K)
    mus:        (E, K, D)
    precisions: (E, K, D, D)
    logdets:    (E, K)

    Returns:
      logpdf_per_event: (E, N)  with log p_e(X_n)
      total_logpdf:     ()      equals sum_e log p_e(X_n) if you later sum over e (or whatever reduction you use)
    """
    N, D = X.shape
    E, K = wts.shape

    # ---- Mega evaluation over all EK components (no cross-event averaging) ----
    EK = E * K
    M  = mus.reshape(EK, D)                 # (EK, D)
    P  = precisions.reshape(EK, D, D)       # (EK, D, D)
    LD = logdets.reshape(EK)                # (EK,)

    # Component log-densities at all X: (N,EK)
    diffs = X[:, None, :] - M[None, :, :]                       # (N,EK,D)
    quad  = jnp.einsum('nkd,kdj,nkj->nk', diffs, P, diffs)      # (N,EK)
    log_comp = -0.5 * (D*jnp.log(2*jnp.pi) + LD + quad)         # (N,EK)

    # ---- Reshape back to (N,E,K), then do per-event log-sum-exp over K ----
    log_comp_NEK = log_comp.reshape(N, E, K)                    # (N,E,K)
    logw_NEK     = jnp.log(jnp.clip(wts, 1e-12, 1.0))[None,:,:] # (1,E,K)

    logp_N_E = jsp.special.logsumexp(logw_NEK + log_comp_NEK, axis=-1)  # (N,E)
    logp_E_N = jnp.swapaxes(logp_N_E, 0, 1)                              # (E,N)

    # Example reduction that matches your loop “Logpdf += gmm_e.logpdf(thetas)”:
    # (you may keep logp_E_N to apply Jacobians/weights per event/sample first)
    total_logpdf = jnp.sum(logp_E_N)  # if your loop literally sums event logpdfs over thetas

    return logp_E_N, total_logpdf

zsels = z_of_dL(dL_sam, H0Planck,Om0Planck)
m1sels = m1_sam/(1+zsels)
m2sels = m2_sam/(1+zsels)

x_sorted  = cache["x_sorted"]        # (N,3)

# x, log_qx, _ = sample_and_logq_from_q(master_key, nsamp_pop, q_params, approx=True, top_T=16)
m1dets_pop, m2dets_pop, dLs_pop = x_sorted[:, 0], x_sorted[:, 1], x_sorted[:, 2]

zs_pop = z_of_dL(dLs_pop, H0Planck)
m1s_pop = m1dets_pop/(1+zs_pop)
m2s_pop = m2dets_pop/(1+zs_pop)

X = jnp.column_stack([m1dets_pop, m2dets_pop, dLs_pop])
logp_E_N, total = logpdf_sum_over_events_with_mega_eval(X, wts, mus, precisions, logdets)

logJ = - jnp.log(ddL_of_z(zs_pop, dLs_pop, H0Planck, Om0Planck)) - 2*jnp.log1p(zs_pop) - 2*jnp.log(dLs_pop)  # (N,)

log_q = cache["log_q_ord"]       # (N,)

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

# #sampling

# from dynesty.utils import resample_equal
# from dynesty import NestedSampler, DynamicNestedSampler
# import multiprocessing as multi

# bound = 'multi'
# sample = 'rwalk'
# nprocesses = 1

# d3sampler = NestedSampler(loglike_method_3, prior_transform, ndims, bound=bound, sample=sample, nlive=nlive)
# d3sampler.run_nested(dlogz=0.1)

# import corner

# dres = d3sampler.results

# dlogZdynesty = dres.logz[-1]        # value of logZ
# dlogZerrdynesty = dres.logzerr[-1]  # estimate of the statistcal uncertainty on logZ

# # output marginal likelihood
# print('Marginalised evidence (using dynamic sampler) is {} ± {}'.format(dlogZdynesty, dlogZerrdynesty))

# # get the posterior samples
# dweights = np.exp(dres['logwt'] - dres['logz'][-1])
# dpostsamples = resample_equal(dres.samples, dweights)

# print('Number of posterior samples (using dynamic sampler) is {}'.format(dpostsamples.shape[0]))

# fig3 = corner.corner(dpostsamples,  hist_kwargs={'density': True},labels=labels, fig=fig1, color='orange')
# # plt.savefig('/content/drive/MyDrive/popnflow_goog/GWTC1_meth2.png')
# # plt.show()
# # plt.savefig('/content/drive/MyDrive/popnflow_goog/overplot.png')

fig3

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

#sampling

from dynesty.utils import resample_equal
from dynesty import NestedSampler, DynamicNestedSampler
import multiprocessing as multi

bound = 'multi'
sample = 'rwalk'
nprocesses = 1

d2sampler = NestedSampler(loglike_method_1_sel, prior_transform, ndims, bound=bound, sample=sample, nlive=nlive)
d2sampler.run_nested(dlogz=0.1)

import corner

dres = d2sampler.results

dlogZdynesty = dres.logz[-1]        # value of logZ
dlogZerrdynesty = dres.logzerr[-1]  # estimate of the statistcal uncertainty on logZ

# output marginal likelihood
print('Marginalised evidence (using dynamic sampler) is {} ± {}'.format(dlogZdynesty, dlogZerrdynesty))

# get the posterior samples
dweights = np.exp(dres['logwt'] - dres['logz'][-1])
dpostsamples = resample_equal(dres.samples, dweights)

print('Number of posterior samples (using dynamic sampler) is {}'.format(dpostsamples.shape[0]))

fig2 = corner.corner(dpostsamples,  hist_kwargs={'density': True},labels=labels, fig=fig1, color='green')


dsampler.save(f'{args.outdir}/met1.h5')
d3sampler.save(f'{args.outdir}/met3.h5')




# =============================
# End of production wrapper
# =============================
