#!/usr/bin/env python
# coding: utf-8

# In[4]:
import time
import pandas as pd

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import jax
#jax.config.update('jax_default_device', jax.devices('cpu')[0])

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
from tqdm import tqdm

from line_profiler import LineProfiler, profile

from jax.scipy.stats import norm, multivariate_normal
from functools import partial


import matplotlib
# get_ipython().run_line_magic('matplotlib', 'inline')

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
# jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
jax.config.update('jax_default_matmul_precision', 'highest')

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.4f} seconds")
        return result
    return wrapper

from jaxinterp2d import interp2d, CartesianGrid

H0Planck = Planck15.H0.value
Om0Planck = Planck15.Om0

zMax = 5
zgrid = jnp.expm1(jnp.linspace(jnp.log(1), jnp.log(zMax+1), 500))
Om0grid = jnp.linspace(Om0Planck-0.1,Om0Planck+0.1,50)

cosmo = FlatLambdaCDM(H0=H0Planck,Om0=Planck15.Om0)
speed_of_light = constants.c.to('km/s').value

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


GWTC1=True

with h5py.File('./GWTC-3_posterior_samples_m1detm2detdLradec_4096_1peryear.h5', 'r') as inp:
    if GWTC1:
        nGWTC1 = 10
        nsamps = inp.attrs['nsamp']
        # print(nsamps)
        nEvents = inp.attrs['nobs']
        ra = jnp.array(inp['ra'])[:int(nGWTC1*nsamps)]
        dec = jnp.array(inp['dec'])[:int(nGWTC1*nsamps)]
        # m1det = jnp.array(inp['m1det'])[:int(nGWTC1*nsamps)]
        # m2det = jnp.array(inp['m2det'])[:int(nGWTC1*nsamps)]
        # dL = jnp.array((jnp.array(inp['dL'])*u.Mpc).value)[:int(nGWTC1*nsamps)]
        # ra = jnp.array(inp['ra'])
        # dec = jnp.array(inp['dec'])
        # m1det = jnp.array(inp['m1det'])
        # m2det = jnp.array(inp['m2det'])
        # dL = jnp.array((jnp.array(inp['dL'])*u.Mpc).value)   
    else:
        nGWTC1 = 10
        nsamps = inp.attrs['nsamp']
        nEvents = inp.attrs['nobs'] - nGWTC1
        ra = jnp.array(inp['ra'])[int(nGWTC1*nsamps):]
        dec = jnp.array(inp['dec'])[int(nGWTC1*nsamps):]
        m1det = jnp.array(inp['m1det'])[int(nGWTC1*nsamps):]
        m2det = jnp.array(inp['m2det'])[int(nGWTC1*nsamps):]
        dL = jnp.array((jnp.array(inp['dL'])*u.Mpc).value)[int(nGWTC1*nsamps):]

print(ra.shape)

nEvents=1

ra = ra[:4096*nEvents]
dec = dec[:4096*nEvents]

parameter_translator = dict(
    mass_1_det="m1_detector_frame_Msun",
    mass_2_det="m2_detector_frame_Msun",
    luminosity_distance="luminosity_distance_Mpc",
    a_1="spin1",
    a_2="spin2",
    cos_tilt_1="costilt1",
    cos_tilt_2="costilt2",
)

posteriors = list()
priors = list()

file_str = "../GWTC-1_sample_release/GW{}_GWTC-1.hdf5"

events = [
    "150914",
    "151012",
    "151226",
    "170104",
    "170608",
    "170729",
    "170809",
    "170814",
    "170818",
    "170823",
]

import pandas as pd

m1det, m2det, dL = jnp.array([]), jnp.array([]), jnp.array([])
# ra, dec = jnp.array([]), jnp.array([])

for event in events:
    _posterior = pd.DataFrame()
    _prior = pd.DataFrame()
    with h5py.File(file_str.format(event)) as ff:
        # for my_key, gwtc_key in parameter_translator.items():
        #     _posterior[my_key] = ff["IMRPhenomPv2_posterior"][gwtc_key][:nsamps]
        #     _prior[my_key] = ff["prior"][gwtc_key][:nsamps]
        m1det = jnp.append(m1det, ff['IMRPhenomPv2_posterior']['m1_detector_frame_Msun'][:nsamps])
        m2det = jnp.append(m2det, ff['IMRPhenomPv2_posterior']['m2_detector_frame_Msun'][:nsamps])
        dL = jnp.append(dL, ff['IMRPhenomPv2_posterior']['luminosity_distance_Mpc'][:nsamps])
    posteriors.append(_posterior)
    priors.append(_prior)

GWTC3_events = {}
posteriors = list()
priors = list()

m1det = m1det[:4096*nEvents]
m2det = m2det[:4096*nEvents]
dL = dL[:4096*nEvents]

# with open('../GWTC-3/events_names.txt', 'r') as f:                                                                                                                                                                                                                                                       
#     for line in f:
#         elements = line.strip('\n').split()
#         GWTC3_events[elements[0]] = elements[1]

# parameter_translator_1 = dict(
#     # mass_1="mass_1_source",
#     # mass_2="mass_2_source",
#     m1det = 'mass_1',
#     m2det = 'mass_2',
#     # mass_ratio="mass_ratio",
#     dL="luminosity_distance",
#     # redshift="redshift",
#     ra = 'ra',
#     dec = 'dec'
# )

# print(ra.shape)

# e=0 # +10 for things


# ## Load samples from events
# for event in list(GWTC3_events.keys()):
#     # if e==60:
#     #     break
#     _posterior = pd.DataFrame()
#     waveform = GWTC3_events[event]
#     # if e>=50:
#     # if((ra.shape[0]/4096)!=(e+10)):
#     #     print(e, ra.shape)
#     with h5py.File("../GWTC-3/{}.h5".format(event)) as ff:
#         # for my_key, gwtc_key in parameter_translator_1.items():
#             # _posterior[my_key] = ff[waveform]['posterior_samples'][gwtc_key][:nsamps]
#         m1det = jnp.append(m1det, ff[waveform]['posterior_samples']['mass_1'][:nsamps]) 
#         m2det = jnp.append(m2det, ff[waveform]['posterior_samples']['mass_2'][:nsamps])
#         dL = jnp.append(dL, ff[waveform]['posterior_samples']['luminosity_distance'][:nsamps])
#         ra = jnp.append(ra, ff[waveform]['posterior_samples']['ra'][:nsamps])
#         dec = jnp.append(dec, ff[waveform]['posterior_samples']['dec'][:nsamps])
#     posteriors.append(_posterior)
#     e+=1
# # print(e)

# print(ra.shape)

# nEvents = 69 
nsamp = 4096
# ra = ra.reshape(nEvents,nsamps)[:,0:nsamp]#.flatten()
# dec = dec.reshape(nEvents,nsamps)[:,0:nsamp]#.flatten()
# m1det = m1det.reshape(nEvents,nsamps)[:,0:nsamp]#.flatten()
# m2det = m2det.reshape(nEvents,nsamps)[:,0:nsamp]#.flatten()
# dL = dL.reshape(nEvents,nsamps)[:,0:nsamp]#.flatten()

# ra = ra[0:nEvents].flatten()
# dec = dec[0:nEvents].flatten()
# m1det = m1det[0:nEvents].flatten()
# m2det = m2det[0:nEvents].flatten()
# dL = dL[0:nEvents].flatten()
q = m2det/m1det
# print(nEvents,nsamp)
# print(len(posteriors))

# jnp.savetxt('ra.txt', ra)
# jnp.savetxt('dec.txt', dec)
# jnp.savetxt('m1det.txt', m1det)
# jnp.savetxt('m2det.txt', m2det)
# jnp.savetxt('dL.txt', dL)
# exit()

# m1det1 = []
# m2det1 = []
# dL1 = []
# nsamp = 4096

# from scipy.stats import gaussian_kde
# import pickle

# # for i in range(nEvents):
# #     print(f'kde_det_pkl/{i}de.pkl')
# #     file1 = open(f'kde_det_pkl/{i}de.pkl', 'rb')
# #     kernel = pickle.load(file1)
# #     kde_samples = kernel.resample(size=50000).T
    
# #     m1det1.append(posterior[:,0][:nsamp])
# #     m2det1.append(posterior[:,1][:nsamp])
# #     dL1.append(posterior[:,2][:nsamp])

# m1det1 = np.loadtxt('../models/m1_tkde.txt')
# m2det1 = np.loadtxt('../models/m2_tkde.txt')
# dL1 = np.loadtxt('../models/dL_tkde.txt')

# m1det1 = m1det1.reshape(nEvents,nsamps)[:,0:nsamp]#.flatten()
# m2det1 = m2det1.reshape(nEvents,nsamps)[:,0:nsamp]#.flatten()
# dL1 = dL1.reshape(nEvents,nsamps)[:,0:nsamp]#.flatten()

# m1det1 = m1det1[0:nEvents].flatten()
# m2det1 = m2det1[0:nEvents].flatten()
# dL1 = dL1[0:nEvents].flatten()
# q = m2det1/m1det1

# f = open('../potato_det.pkl', 'rb')
# posteriors = pickle.load(f)


# m1det1 = []
# m2det1 = []
# dL1 = []


# i = 0
# for posterior in posteriors:
#   # print(len(posterior[:,0][:1000]))
#   # m1det1.append(posterior['mass_1_det'][0:1000])
#   # m2det1.append(posterior['mass_2_det'][0:1000])
#   # dL1.append(posterior['luminosity_distance'][0:1000])
#   if i==70:
#     break

#   if i>=60:
#     m1det1.append(posterior[:,0][:nsamp])
#     m2det1.append(posterior[:,1][:nsamp])
#     dL1.append(posterior[:,2][:nsamp])

#   i+=1


# m1det1 = np.concatenate(m1det1)
# m2det1 = np.concatenate(m2det1)
# dL1 = np.concatenate(dL1)

# m1det1[m1det1>100] = 0
# m1det1[m1det1<0] = 0

# m2det1[m2det1>100] = 0
# m2det1[m2det1<0] = 0

# nsamp = 4096
# z1 = z_of_dL(dL1, H0Planck, Om0Planck)

# Read in attributes from injection summary file
injection_file = "./endo3_bbhpop-LIGO-T2100113-v12.hdf5"
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

sels = selection_ifar['cbc']
m1detsels = jnp.array(m1detsels[sels])
m2detsels = jnp.array(m2detsels[sels])
dLsels = jnp.array(dLsels[sels])
rasels = jnp.array(rasels[sels])
decsels = jnp.array(decsels[sels])
p_draw = jnp.array(p_draw[sels])

print(Ndraw)
print(p_draw.shape)
print(z_of_dL(dLsels,115,Om0grid[-1]).max())

# sns.distplot(z_of_dL(dLsels,H0Planck,Om0grid[-1]))
# sns.distplot(z_of_dL(dL,H0Planck,Om0grid[-1]))

m1s = (m1detsels/(1+z_of_dL(dLsels,70)))
print(m1s.max())
sns.distplot(m1s)


@jit
def dV_of_z_normed(z,Om0,gamma):
    dV = dV_of_z(zgrid,H0Planck,Om0)*(1+zgrid)**(gamma-1)
    prob = dV/jnp.trapezoid(dV,zgrid)
    return jnp.interp(z,zgrid,prob)


mass = jnp.linspace(1, 250, 2000)
mass_ratio =  jnp.linspace(1e-5, 1, 2000)

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
    pm1 = pm1/jnp.trapezoid(pm1,mass)
    return jnp.log(jnp.interp(m1,mass,pm1))

@jit
def logpm1_peak(m1,mu,sigma):
    pm1 =  jnp.exp(-(mass - mu)**2 / (2 * sigma ** 2))
    pm1 = pm1/jnp.trapezoid(pm1,mass)
    return jnp.log(jnp.interp(m1,mass,pm1))

@jit
def logpm1_powerlaw_powerlaw(m1,z,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,mu,sigma,f1):
    p1 = jnp.exp(logpm1_powerlaw(m1,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1))
    p2 = jnp.exp(logpm1_peak(m1,mu,sigma))

    pm1 = (1-f1)*p1 + f1*p2
    return jnp.log(pm1)

@jit
def logpm1_powerlaw_GP(m1,z,mu,sigma):
    pass

@jit
def logfq(m1,m2,beta):
    beta=2
    q = m2/m1
    pq = mass_ratio**beta
    pq = pq/jnp.trapezoid(pq,mass_ratio)
    # jax.debug.print("mr: {}", mass_ratio) 
    # jax.debug.print("pq: {}",(mass_ratio**beta))
    # jax.debug.print("trap:{}", jnp.trapezoid(pq, mass_ratio))
    # jax.debug.print("pqd: {}", pq) 

    log_pq = jnp.log(jnp.interp(q,mass_ratio,pq))

    return log_pq

# print(jnp.exp(logfq(2,1,  2)))
# exit()


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
def log_p_pop_pl_pl(m1,m2,z,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,beta,mu,sigma,f1,gamma1):
    # start_time = time.time()
    log_dNdm1 = logpm1_powerlaw_powerlaw(m1,z,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,mu,sigma,f1)
    log_dNdm2 = logpm1_powerlaw_powerlaw(m2,z,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,mu,sigma,f1)
    log_fq = logfq(m1,m2,beta)
    log_dvdz = jnp.log(dV_of_z_normed(z,Om0Planck,gamma1))

    log_p_sz = np.log(0.25) # 1/2 for each spin dimension

    end_time = time.time()
    # print('time0', end_time-start_time)
    return log_p_sz + log_dNdm1 + log_dNdm2 + log_fq + log_dvdz
@jit
def logdiffexp(x, y):
    return x + jnp.log1p(jnp.exp(y-x))

@jit
def pm1_powerlaw_powerlaw(m1,m_min_1=5,m_max_1=80,alpha_1=3.3,dm_min_1=1,dm_max_1=10,mu=50,sigma=3,f1=0.4):
    p1 = jnp.exp(logpm1_powerlaw(m1,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1))
    p2 = jnp.exp(logpm1_peak(m1,mu,sigma))

    pm1 = (1-f1)*p1 + f1*p2
    return pm1

@jit
def powerlaw(xx, high, low, beta=2):
    norm = jnp.where(
        jnp.array(beta) == -1,
        1 / jnp.log(high / low),
        (1 + beta) / jnp.array(high ** (1 + beta) - low ** (1 + beta)),
    )
    prob = jnp.power(xx, beta)
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob

@jit
def log_p_pop_lvk(m1,m2,z,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,beta,mu,sigma,f1,gamma1):
    # start_time = time.time()
    log_dNdm1 = logpm1_powerlaw_powerlaw(m1,z,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,mu,sigma,f1)
    log_dNdm2 = logpm1_powerlaw_powerlaw(m2,z,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,mu,sigma,f1)
    log_fq = logfq(m1,m2,beta)
    log_dvdz = jnp.log(dV_of_z_normed(z,Om0Planck,gamma1))

    log_p_sz = np.log(0.25) # 1/2 for each spin dimension

    # end_time = time.time()
    # print('time0', end_time-start_time)
    return log_p_sz + log_dNdm1 + log_fq + log_dvdz
# @jit
# def log_p_pop_lvk(m1,m2,z,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,beta,mu,sigma,f1,gamma1):
#     # start_time0 = time.time()
    
#     log_dVdz = jnp.log(dV_of_z_normed(z,Om0Planck,gamma1))
#     log_p_sz = jnp.log(0.25)
   
#     # end_time = time.time()
#     # print('time1', end_time-start_time0)
    
#     @jit
#     def log_two_component_primary_mass_ratio(
#         m1, m2, m_min_1, m_max_1, alpha_1, dm_min_1, dm_max_1, mu, sigma, f1
#     ):
#         r"""
#         Power law model for two-dimensional mass distribution, modelling primary
#         mass and conditional mass ratio distribution.
    
#         .. math::
#             p(m_1, q) = p(m1) p(q | m_1)
    
#         """
        
#         # start_time = time.time()
#         log_pm1 = logpm1_powerlaw_powerlaw(m1,z,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,mu,sigma,f1)
#         # p_m1 = pm1_powerlaw_powerlaw(m1, m_min_1, m_max_1, alpha_1, dm_min_1, dm_max_1, mu, sigma, f1)
#         q = m2/m1
#         log_pq = jnp.log(powerlaw(q, beta, 1, m_min_1/m1))
#         # end_time = time.time()
#         # print('time2', end_time-start_time)

#         prob = log_pm1 + log_pq
#         return prob
    
#     # pm1q = two_component_primary_mass_ratio(m1, m2, m_min_1, m_max_1, alpha_1, dm_min_1, dm_max_1, mu, sigma, f1)
#     log_pm1q = log_two_component_primary_mass_ratio(
#         m1, m2, m_min_1, m_max_1, alpha_1, dm_min_1, dm_max_1, mu, sigma, f1
#     )
    
#     # end_time = time.time()
#     # print('time3', end_time-start_time0)
#     return log_pm1q + log_dVdz + log_p_sz

# from scipy.integrate import cumtrapz
from scipy.integrate import cumulative_trapezoid as cumtrapz

## draw samples from p(z) and p(m1, q)
# def z_sampling(n_samples, gamma=3.0):
    
#     @jit
#     def pz(z, gamma=3.0):
#         dV = dV_of_z(z,H0Planck,Om0)*(1+z)**(gamma-1)
#         prob = dV/jnp.trapezoid(dV,z)
#         return prob

#     z_vals = jnp.linspace(0, 5, 2000)
#     pdf_zvalues = pz(z_vals, gamma)
#     cdf_zvalues = cumtrapz(pdf_zvalues, z_vals, initial=0)  # Numerical CDF
#     cdf_zvalues /= cdf_zvalues[-1]  # Normalize to [0, 1]

#     # Interpolate the inverse CDF
#     inverse_cdfz = interp1d(cdf_zvalues, z_vals, bounds_error=False, fill_value=(z_vals[0], z_vals[-1]))
#     key = jax.random.PRNGKey(42)
#     u = jax.random.uniform(key, shape=(n_samples,))    
#     return inverse_cdfz(u)

from jax.scipy.integrate import trapezoid

Z_MAX = 5.0      # Max redshift
N_GRID = 1000    # Reduced grid size (trade-off: speed vs. accuracy)

_z_vals = jnp.linspace(0, Z_MAX, N_GRID)
_dV_cache = dV_of_z(_z_vals, H0Planck, Om0)  # Precompute dV/dz

@partial(jit, static_argnames=['n_samples'])
def z_sampling(n_samples, gamma=3.0, key=None):
    """Handle dynamic gamma while maximizing performance."""
    # Compute (1+z)^(gamma-1) term dynamically
    weight = (1 + _z_vals) ** (gamma - 1)
    
    # Compute normalized PDF (vectorized)
    pdf = _dV_cache * weight
    pdf /= jnp.trapezoid(pdf, _z_vals)  # Normalization
    
    # Compute CDF
    cdf = jnp.cumsum(pdf) * (_z_vals[1] - _z_vals[0])
    cdf /= cdf[-1]
    
    # Sampling
    key = jax.random.PRNGKey(42) if key is None else key
    u = jax.random.uniform(key, (n_samples,))
    indices = jnp.searchsorted(cdf, u)
    return _z_vals[indices]

# # Precompute z_vals and CDF once (if gamma is fixed)
# _z_vals = jnp.linspace(0, Z_MAX, N_GRID)
# _dV_cache = jnp.zeros_like(_z_vals)  # Cache dV_of_z if possible

# @jit
# def pz(z, gamma):
#     dV = dV_of_z(z, H0Planck, Om0) * (1 + z) ** (gamma - 1)
#     norm = trapezoid(dV, z)  # Faster than manual trapezoid
#     return dV / norm

# # Precompute CDF for default gamma=3.0 (if known in advance)
# _pdf_zvalues = pz(_z_vals, 3.0)
# _cdf_zvalues = jnp.cumsum(_pdf_zvalues) * (_z_vals[1] - _z_vals[0])
# _cdf_zvalues /= _cdf_zvalues[-1]

# @jit
# def inverse_cdf(u):
#     indices = jnp.searchsorted(_cdf_zvalues, u)
#     return _z_vals[jnp.clip(indices, 0, len(_z_vals) - 1)]

# @timer
# def z_sampling(n_samples, gamma=3.0, key=None):
#     key = jax.random.PRNGKey(42) if key is None else key
#     u = jax.random.uniform(key, shape=(n_samples,))
#     return inverse_cdf(u)

# @jit
# @timer
# def z_sampling(n_samples, gamma=3.0, key=None):
#     # Assuming dV_of_z is a JAX-compatible function
#     @jit
#     def pz(z, gamma=3.0):
#         dV = dV_of_z(z, H0Planck, Om0) * (1 + z) ** (gamma - 1)
#         prob = dV / trapezoid(dV, z)
#         return prob

#     z_vals = jnp.linspace(0, 5, 2000)
#     pdf_zvalues = pz(z_vals, gamma)
    
#     # Compute cumulative integral (CDF)
#     cdf_zvalues = jnp.cumsum(pdf_zvalues) * (z_vals[1] - z_vals[0])  # Approximate cumulative integral
#     cdf_zvalues = cdf_zvalues / cdf_zvalues[-1]  # Normalize to [0, 1]

#     # Create inverse CDF function
#     @jit
#     def inverse_cdf(u):
#         # JAX-compatible interpolation
#         indices = jnp.searchsorted(cdf_zvalues, u)
#         indices = jnp.clip(indices, 0, len(z_vals) - 1)
#         return z_vals[indices]

#     if key is None:
#         key = jax.random.PRNGKey(42)
#     u = jax.random.uniform(key, shape=(n_samples,))
    
#     return inverse_cdf(u)

# @partial(jit, static_argnums=(0,))
# # @jit
# def sample_from_pdf(n_samples, m1_range, q_range, p_joint):
#     cdf = jnp.cumsum(p_joint.ravel())
#     cdf /= cdf[-1]
#     uniform_samples = jax.random.uniform(jax.random.PRNGKey(42), (n_samples,))
#     indices = jnp.searchsorted(cdf, uniform_samples)
#     q_idx, m1_idx = jnp.unravel_index(indices, p_joint.shape)
#     return m1_range[m1_idx], q_range[q_idx]

# @profile
# def m1_q_samples(n_samples, m_min_1=5,m_max_1=80,alpha_1=3.3,dm_min_1=1,dm_max_1=10,beta=1,mu=50,sigma=3,f1=0.4):
    
#     @timer
#     @jit
#     def two_component_primary_mass_ratio(
#         dataset, m_min_1, m_max_1, alpha_1, dm_min_1, dm_max_1, mu, sigma, f1
#     ):
#         r"""
#         Power law model for two-dimensional mass distribution, modelling primary
#         mass and conditional mass ratio distribution.
    
#         .. math::
#             p(m_1, q) = p(m1) p(q | m_1)
    
#         """

#         p_m1 = pm1_powerlaw_powerlaw(dataset["mass_1"], m_min_1, m_max_1, alpha_1, dm_min_1, dm_max_1, mu, sigma, f1)
#         # p_q = powerlaw(dataset["mass_ratio"], beta, 1, m_min_1/dataset["mass_1"])
        
#         p_q = fq(dataset['mass_ratio'], beta)
#         prob = p_m1 * p_q
#         return prob

#     m1_range = jnp.linspace(m_min_1+0.01, m_max_1, 2000)  # Example range for primary mass
#     q_range = jnp.linspace(0.01, 1, 2000)  # Example range for mass ratio
    
#     m1_grid, q_grid = jnp.meshgrid(m1_range, q_range)
#     dataset = {
#         "mass_1": m1_grid.ravel(),
#         "mass_ratio": q_grid.ravel(),
#     }
    
#     p_joint = two_component_primary_mass_ratio(dataset, m_min_1, m_max_1, alpha_1, dm_min_1, dm_max_1, mu, sigma, f1).reshape(len(q_range), len(m1_range))
    
#     # Step 2: Normalize and compute the CDF
#     p_joint /= jnp.sum(p_joint)  # Normalize the joint probability
    
#     # return sample_from_pdf(n_samples, m1_range, q_range, p_joint)
#     cdf = jnp.cumsum(p_joint.ravel())  # Flatten and compute cumulative sum
#     cdf /= cdf[-1]  # Normalize the CDF to [0, 1]
    
#     # Step 3: Sample from the CDF
    
#     key = jax.random.PRNGKey(42)
#     uniform_samples = jax.random.uniform(key, shape=(n_samples,))
#     sample_indices = jnp.searchsorted(cdf, uniform_samples)
#     sample_q_indices, sample_m1_indices = jnp.unravel_index(sample_indices, p_joint.shape)
    
#     sample_m1 = m1_range[sample_m1_indices]
#     sample_q = q_range[sample_q_indices]

#     return sample_m1, sample_q


GRID_SIZE = 500  # Balance between speed and accuracy

@partial(jit, static_argnums=(0,))  # Make n_samples static
def m1_q_samples(n_samples, m_min_1=5, m_max_1=80, alpha_1=3.3, dm_min_1=1,
                dm_max_1=10, beta=1, mu=50, sigma=3, f1=0.4, key=None):
    """Optimized version that properly handles dynamic n_samples."""
    
    # Create grid - these operations are static
    m1_range = jnp.linspace(m_min_1 + 0.01, m_max_1, GRID_SIZE)
    q_range = jnp.linspace(0.01, 1, GRID_SIZE)
    m1_grid, q_grid = jnp.meshgrid(m1_range, q_range, indexing='ij')

    # Compute joint PDF (jitted internally)
    @jit
    def compute_pdf(m1, q):
        p_m1 = pm1_powerlaw_powerlaw(m1, m_min_1, m_max_1, alpha_1, 
                                    dm_min_1, dm_max_1, mu, sigma, f1)
        p_q = fq(q, beta)
        return p_m1 * p_q

    joint_pdf = compute_pdf(m1_grid.ravel(), q_grid.ravel()).reshape(GRID_SIZE, GRID_SIZE)
    joint_pdf /= jnp.sum(joint_pdf)  # Normalize

    # Compute CDF
    cdf = jnp.cumsum(joint_pdf.ravel())
    cdf /= cdf[-1]

    # Sampling - this part handles dynamic n_samples
    key = jax.random.PRNGKey(42) if key is None else key
    u = jax.random.uniform(key, shape=(n_samples,))
    indices = jnp.searchsorted(cdf, u)
    idx_m1, idx_q = jnp.unravel_index(indices, (GRID_SIZE, GRID_SIZE))

    return m1_range[idx_m1], q_range[idx_q]


def metro_mc(nsamp=10000, m_min_1=5, m_max_1=80, alpha_1=3.3, dm_min_1=1,
                dm_max_1=10, beta=1, mu=50, sigma=3, f1=0.4, key=None):
    """
    Simple Metropolis-Hastings sampling of (m1, q) within [m_min, m_max] × [q_min, q_max].
    - compute_pdf(m1, q) should return non-negative density (unnormalized).
    """
    samples = np.zeros((nsamp, 2))
    p_vals = np.zeros(nsamp)
    
    m_min = m_min_1
    m_max = m_max_1
    
    q_min=0.01
    q_max=1.0
    
    proposal_std_m=5.0 
    proposal_std_q=0.1
    # Example compute_pdf using some user-defined pm1_peak and fq functions.
    # These must be defined elsewhere and accept numpy floats (or arrays).
    def compute_pdf(m1, q):
        # Check bounds; return zero density outside
        if (m1 < m_min_1) or (m1 > m_max_1) or (q < q_min) or (q > q_max):
            return 0.0
        # User’s functions; ensure they accept/return numpy floats
        pm1 = pm1_powerlaw_powerlaw(m1, m_min_1, m_max_1, alpha_1, 
                                    dm_min_1, dm_max_1, mu, sigma, f1)
        pq = fq(q, beta)               # e.g. power-law in q
        # If these return arrays, ensure you index appropriately.
        # Here assume they return scalar floats when inputs are floats.
        return pm1 * pq

    # Initialize first sample somewhere inside the domain.
    # Could choose (mu, midpoint of q-range) or draw random uniform:
    samples[0, 0] = np.clip(mu, m_min, m_max)  # or np.random.uniform(m_min, m_max)
    samples[0, 1] = np.clip((q_min + q_max) / 2, q_min, q_max)
    p_vals[0] = compute_pdf(samples[0,0], samples[0,1])

    for i in range(1, nsamp):
        current_m, current_q = samples[i-1]
        # Propose new point via Gaussian steps:
        prop_m = current_m + np.random.normal(scale=proposal_std_m)
        prop_q = current_q + np.random.normal(scale=proposal_std_q)

        # Boundary check: if out of allowed range, reject immediately
        if (prop_m < m_min) or (prop_m > m_max) or (prop_q < q_min) or (prop_q > q_max):
            # reject: keep previous
            samples[i] = samples[i-1]
            p_vals[i] = p_vals[i-1]
            continue

        # Otherwise compute density at proposal
        p_prop = compute_pdf(prop_m, prop_q)
        p_curr = p_vals[i-1]

        # If current density is zero (should not happen if initialization in support), you might treat carefully:
        if p_curr <= 0:
            # To avoid division by zero, you could automatically accept if p_prop>0,
            # or simply set accept=False; depends on context. Here, if current has zero density but proposal>0,
            # you might accept to move into support:
            accept = (p_prop > 0)
        else:
            # Standard Metropolis acceptance:
            if p_prop >= p_curr:
                accept = True
            else:
                accept = np.random.rand() < (p_prop / p_curr)

        if accept:
            samples[i, 0] = prop_m
            samples[i, 1] = prop_q
            p_vals[i] = p_prop
        else:
            samples[i] = samples[i-1]
            p_vals[i] = p_curr

    return samples, p_vals
# lp = LineProfiler()
# lp_wrapper = lp(m1_q_samples)
# lp_wrapper(10000)
# lp.print_stats()

## Import KDEs


from scipy.stats import gaussian_kde as kde

kdes = []
import pickle
for i in range(nEvents):
    with open(f'./kde_det_pkl/{i}de_jax_scipy.pkl', 'rb') as file:
        kde = pickle.load(file)
    kdes.append(kde)
    # kdes.append((jnp.array(kde.dataset.T), kde.covariance_factor()))

print(len(kdes), 'len_kde')
seed = np.random.randint(1000)
key = jax.random.PRNGKey(1000)

def spectral_siren_log_likelihood_nosky(gamma1=3, m_min_1=5,m_max_1=80,alpha_1=3.3,dm_min_1=1,dm_max_1=10,beta=1,mu=50,sigma=3,f1=0.4):
    zsels = z_of_dL(dLsels, H0Planck,Om0Planck)
    m1sels = m1detsels/(1+zsels)
    m2sels = m2detsels/(1+zsels)

    log_det_weights = log_p_pop_pl_pl(m1sels,m2sels,zsels,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,beta,mu,sigma,f1,gamma1)

    log_det_weights += - jnp.log(p_draw) - 2*jnp.log1p(zsels) - jnp.log(ddL_of_z(zsels,dLsels,H0Planck, Om0Planck))

    log_mu = logsumexp(log_det_weights) - jnp.log(Ndraw)
    log_s2 = logsumexp(2*log_det_weights) - 2.0*jnp.log(Ndraw)
    log_sigma2 = logdiffexp(log_s2, 2.0*log_mu - jnp.log(Ndraw))
    Neff = jnp.exp(2.0*log_mu - log_sigma2)

    ll = -jnp.inf
    ll = jnp.where((Neff <= 4 * nEvents), ll, 0)
    ll += -nEvents*log_mu + nEvents*(3 + nEvents)/(2*Neff)

    z = z_of_dL(dL, H0Planck, Om0Planck)
    m1 = m1det/(1+z)
    m2 = m2det/(1+z)

    # weights = dL**2 / np.sum(dL**2)
    # m1 /= weights
    # m2 /= weights
    
    log_weights = log_p_pop_pl_pl(m1,m2,z,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,beta,mu,sigma,f1,gamma1)
    # log_weights = log_p_pop_lvk(m1,m2,z,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,beta,mu,sigma,f1,gamma1)
    # print('mean', jnp.mean(log_weights))
    log_weights += - jnp.log(ddL_of_z(z,dL,H0Planck,Om0Planck)) - 2*jnp.log(dL) - 2*jnp.log1p(z)

    nsamp1 = 4096
    log_weights = log_weights.reshape((nEvents,nsamp1))
    ll += jnp.sum(-jnp.log(nsamp1) + jnp.nan_to_num(logsumexp(log_weights,axis=-1)))


    # end_time = time.time()
    # etime = end_time - start_time
    # print('etime', etime)
    return ll, Neff


def spectral_siren_log_likelihood_nosky1(gamma1=3, m_min_1=5,m_max_1=80,alpha_1=3.3,dm_min_1=1,dm_max_1=10,beta=1,mu=50,sigma=3,f1=0.4):
    # gamma1=3
    
    zsels = z_of_dL(dLsels, H0Planck,Om0Planck)
    m1sels = m1detsels/(1+zsels)
    m2sels = m2detsels/(1+zsels)

    log_det_weights = log_p_pop_pl_pl(m1sels,m2sels,zsels,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,beta,mu,sigma,f1,gamma1)

    log_det_weights += - jnp.log(p_draw) - 2*jnp.log1p(zsels) - jnp.log(ddL_of_z(zsels,dLsels,H0Planck, Om0Planck))

    log_mu = logsumexp(log_det_weights) - jnp.log(Ndraw)
    log_s2 = logsumexp(2*log_det_weights) - 2.0*jnp.log(Ndraw)
    log_sigma2 = logdiffexp(log_s2, 2.0*log_mu - jnp.log(Ndraw))
    Neff = jnp.exp(2.0*log_mu - log_sigma2)

    ll = -jnp.inf
    ll = jnp.where((Neff <= 4 * nEvents), ll, 0)
    ll += -nEvents*log_mu + nEvents*(3 + nEvents)/(2*Neff)

    z = z_of_dL(dL1, H0Planck, Om0Planck)
    m1 = m1det1/(1+z)
    m2 = m2det1/(1+z)


    log_weights = log_p_pop_pl_pl(m1,m2,z,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,beta,mu,sigma,f1,gamma1)

    log_weights += - jnp.log(ddL_of_z(z,dL1,H0Planck,Om0Planck)) - 2*jnp.log(dL1) - 2*jnp.log1p(z)
    # log_weights += - jnp.log(ddL_of_z(z,dL1,H0Planck,Om0Planck))- 2*jnp.log1p(z)
    log_weights = log_weights.reshape((nEvents,nsamp))
    ll += jnp.sum(-jnp.log(nsamp) + jnp.nan_to_num(logsumexp(log_weights,axis=-1)))

    return ll, Neff

from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count
from functools import partial
import pandas as pd

def get_kde_info(kdes):
    # Extract KDE parameters
    datasets = [kde.dataset for kde in kdes]        # List of (n_dims, n_points)
    weights = [kde.weights for kde in kdes]         # List of (n_points,)
    covariances = [kde.covariance for kde in kdes]  # List of scalars or matrices

    # Find the maximum number of points across all KDEs
    max_n_points = max(dataset.shape[-1] for dataset in datasets)
    n_dims = datasets[0].shape[0]  # Dimensionality of the data
    n_kdes = len(kdes)

    # Pad datasets and weights along the number of points axis
    padded_datasets = []
    padded_weights = []
    dataset_masks = []

    for dataset, weight in zip(datasets, weights):
        n_points = dataset.shape[-1]
        padding = max_n_points - n_points

        # Pad along the second axis (number of points), keeping dimensions intact
        padded_dataset = jnp.pad(dataset, ((0, 0), (0, padding)), constant_values=0)
        padded_weight = jnp.pad(weight, (0, padding), constant_values=0)
        dataset_mask = jnp.pad(jnp.ones(n_points), (0, padding), constant_values=0)

        padded_datasets.append(padded_dataset)
        padded_weights.append(padded_weight)
        dataset_masks.append(dataset_mask)
        
    # padded_datasets = jnp.stack(padded_datasets)   # Shape (n_kdes, n_dims, max_n_points)
    # padded_weights = jnp.stack(padded_weights)     # Shape (n_kdes, max_n_points)
    # dataset_masks = jnp.stack(dataset_masks)       # Shape (n_kdes, max_n_points)
    # covariances = jnp.stack(covariances)           # Shape (n_kdes,)    
        
    return padded_datasets, padded_weights, dataset_masks, covariances


def kde_eval(x, dataset, weights, covariance, mask):
    # Extract per-dimension bandwidth from the diagonal of the covariance matrix
    bandwidth = jnp.sqrt(jnp.diag(covariance))  # shape (n_dims,)

    # Reshape x to (n_dims, 1) so it broadcasts correctly with dataset (n_dims, n_points)
    diff = (x[:, None] - dataset) / bandwidth[:, None]   # shape (n_dims, n_points)

    # Evaluate the normal PDF on each dimension and take the product over dimensions
    kernel_vals = jnp.prod(norm.pdf(diff), axis=0)  # shape (n_points,)

    # Compute density using the mask to ignore padded values and normalize appropriately
    density = jnp.sum(weights * kernel_vals * mask) / jnp.prod(bandwidth)
    return density


def evaluate_kdes_fast(kdes, points, batch_size=1000):
    """
    Optimized evaluation of multiple KDEs with JAX-compatible operations.
    
    Args:
        kdes: List of KDE objects with JAX-compatible evaluate() methods
        points: (3, N) array of evaluation points
        batch_size: Points to process at once
        
    Returns:
        (len(kdes), N) array of densities
    """
    # Extract KDE parameters (assuming they're JAX arrays)
    kde_params = [(kde.dataset, kde.weights, kde.inv_cov) for kde in kdes]
    
    # Define JAX-compatible evaluation function
    @partial(jax.vmap, in_axes=(0, None, None, None))
    def jax_kde_eval(point, dataset, weights, inv_cov):
        diff = point[:, None] - dataset  # (3, N)
        mahalanobis = jnp.einsum('dn,dc,cn->n', diff, inv_cov, diff)
        return jnp.exp(-0.5 * mahalanobis) @ weights
    
    # Process in batches
    n_kdes = len(kdes)
    n_points = points.shape[1]
    results = jnp.zeros((n_kdes, n_points))
    
    for i in range(0, n_points, batch_size):
        batch = points[:, i:i+batch_size]  # (3, B)
        
        # Evaluate all KDEs on this batch
        batch_results = []
        for dataset, weights, inv_cov in kde_params:
            res = jax_kde_eval(batch.T, dataset, weights, inv_cov)  # (B,)
            batch_results.append(res)
        
        # Store results
        results = results.at[:, i:i+batch_size].set(jnp.stack(batch_results))
    
    return results

def spectral_siren_log_likelihood_nosky_kde(gamma1 = 3, m_min_1=5,m_max_1=80,alpha_1=3.3,dm_min_1=1,dm_max_1=10,beta=1,mu=50,sigma=3,f1=0.4):
    n_samples=nEvents*nsamp
    # gamma1 = 0 
    # start_time = time.time()
    zsels = z_of_dL(dLsels, H0Planck,Om0Planck)
    m1sels = m1detsels/(1+zsels)
    m2sels = m2detsels/(1+zsels)

    log_det_weights = log_p_pop_pl_pl(m1sels,m2sels,zsels,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,beta,mu,sigma,f1,gamma1)

    log_det_weights += - jnp.log(p_draw) - 2*jnp.log1p(zsels) - jnp.log(ddL_of_z(zsels,dLsels,H0Planck, Om0Planck))

    log_mu = logsumexp(log_det_weights) - jnp.log(Ndraw)
    log_s2 = logsumexp(2*log_det_weights) - 2.0*jnp.log(Ndraw)
    log_sigma2 = logdiffexp(log_s2, 2.0*log_mu - jnp.log(Ndraw))
    Neff = jnp.exp(2.0*log_mu - log_sigma2)

    ll = -jnp.inf
    ll = jnp.where((Neff <= 4 * nEvents), ll, 0)
    ll += -nEvents*log_mu + nEvents*(3 + nEvents)/(2*Neff)

    # start_time = time.time()
    m1, q = m1_q_samples(n_samples, m_min_1, m_max_1, alpha_1, dm_min_1, dm_max_1, mu, sigma, f1)
    m2 = q * m1
    
    z = z_sampling(n_samples, gamma1)
    dL = dL_of_z(z, H0=H0Planck)
    
    m1det = m1*(1+z)
    m2det = m2*(1+z)
    points = jnp.vstack([m1det, m2det, dL])
    log_weights = np.zeros(m1det.shape[0])

    kde_data = []

    # def evaluate_kde_in_batches(kde_func, points, batch_size=12288):
    #     n = points.shape[0]
    #     results = []

    #     for i in range(0, n, batch_size):
    #         batch = points[:, i:i + batch_size]
    #         batch_result = kde_func.evaluate(batch)
    #         results.append(batch_result)

    #     # return jnp.concatenate(results, axis=0)
    #     print(results)
    #     return results



    def evaluate_kdes(points):
        # return sum(jnp.log(evaluate_kde_in_batches(kde, points)) for kde in kdes)
        log_probs = jnp.log(jnp.array([kde.evaluate(points) for kde in kdes]))
        total_log_prob = jnp.sum(log_probs)
        return total_log_prob
    
    # def kde_sum(dataset_list, weights_list, covariance_list, mask_list, points):
    #     results = []
    #     for dataset, weights, covariance, mask in zip(dataset_list, weights_list, covariance_list, mask_list):
    #         kde_values = jax.vmap(kde_eval, in_axes=(1, None, None, None, None))(
    #                     points, dataset, weights, covariance, mask
    #             )
    #         results.append(kde_values)
                
    #     return jnp.sum(jnp.log(jnp.stack(results), axis=0))

    # def kde_memory_efficient(dataset, weights, covariance, mask, points, point_chunk_size=256):
    #     """
    #     Evaluate KDE in a memory-efficient way by processing points in chunks.

    #     Args:
    #         dataset: (3, 50000) KDE dataset
    #         weights: (50000,) KDE weights
    #         covariance: Covariance matrix
    #         mask: Mask array
    #         points: (3, 4096) array of evaluation points
    #         point_chunk_size: Number of points to process at once

    #     Returns:
    #         KDE estimate of shape (4096,)
    #     """
    #     n_points = points.shape[1]
    #     total_pdf = jnp.zeros(n_points)

    #     for j in range(0, n_points, point_chunk_size):
    #         point_chunk = points[:, j:j + point_chunk_size]

    #         kde_values = jax.vmap(
    #             kde_eval, in_axes=(1, None, None, None, None)
    #         )(point_chunk, dataset, weights, covariance, mask)

    #         total_pdf = total_pdf.at[j:j + point_chunk_size].set(kde_values)

    #     return total_pdf

    # print(len(kdes)) 
    
    # datasets, weights, dataset_masks, covariances = get_kde_info(kdes)
    # datasets = jnp.stack(datasets)[0]
    # weights = jnp.stack(weights)[0]
    # dataset_masks = jnp.stack(dataset_masks)[0]
    # covariances = jnp.stack(covariances)[0]
    # # print(covariances)
    # # print(datasets.shape, weights.shape, dataset_masks.shape, covariances.shape)
    # eval_result = jnp.log(kde_sum(datasets, weights, dataset_masks, covariances, points))
   

    ## Attempt_1 batching over kde parameters - failed due to memory
    # batched_kde_eval = jax.vmap(
    # lambda dataset, weights, covariance, mask: jax.vmap(
    #     kde_eval, in_axes=(1, None, None, None, None)  # x_points along axis 1
    # )(points, dataset, weights, covariance, mask),
    # in_axes=(0, 0, 0, 0)
    # )

    # densities = batched_kde_eval(datasets, weights, covariances, dataset_masks)
    # results = jnp.sum(densities, axis=0)

    ## Attempt_2 simply compute kde using sum(), all built-in methods - fastest
    
    # start_time = time.time()
    
    results = evaluate_kdes_fast(kdes, points)


    # end_time = time.time()
    # etime = end_time - start_time
    
    # print(f'time2:{etime}')
    ## Attempt_3 slicing data to reduce memory burden - slower than 2
    # results = kde_memory_efficient(datasets, weights, covariances, dataset_masks, points)

    # for kde in kdes:
    #     print('num', i)
    # log_weights += sum(Parallel(n_jobs=1)(delayed(evaluate_kde)(kde, points) for kde in kdes))
    # log_weights += parallel_kde_evaluate(kdes, points)
    log_weights += results
    # print('after')
   
    dL1 = dL
    log_weights += jnp.log(ddL_of_z(z,dL1,H0Planck,Om0Planck)) - 2*jnp.log(dL1) + 2*jnp.log1p(z) - jnp.log(m1)
    # log_weights += jnp.log(ddL_of_z(z,dL1,H0Planck,Om0Planck))+ 2*jnp.log1p(z) - jnp.log(m1)
    log_weights = log_weights.reshape((nEvents,nsamp))
    ll += jnp.sum(-jnp.log(nsamp) + jnp.nan_to_num(logsumexp(log_weights,axis=-1)))

    # end_time = time.time()
    # etime = end_time - start_time
    # print('etime', etime)
    return ll, Neff

# def cdf(samples):
#     sorted_samples = np.sort(samples)
#     # The CDF value for each sample is its rank (number of samples <= that value) divided by the total number of samples
#     cdf_values = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)

#     min_val = samples.min()
#     max_val = samples.max()
    
#     U = np.zeros_like(samples)
    
#     # Apply normalization only where theta is not min or max
#     mask = (samples != min_val) & (samples != max_val)

#     def find_cdf(sample_value):
#         # Ensure sample_value is a numpy array for consistent processing
#         sample_value = np.asarray(sample_value)
        
#         # Initialize an array to hold the CDF results
#         cdf_result = np.zeros_like(sample_value, dtype=float)
        
#         for i, value in enumerate(sample_value):
#             # Find the index where the sample value would fit in the sorted array
#             index = np.searchsorted(sorted_samples, value)
#             if index == 0:
#                 cdf_result[i] = 0.0  # If the sample value is less than the smallest sample
#             elif index >= len(cdf_values):
#                 cdf_result[i] = 1.0  # If the sample value is greater than the largest sample
#             else:
#                 cdf_result[i] = cdf_values[index - 1]  # Return the corresponding CDF value
                
#         return cdf_result

#     U[mask] = find_cdf(samples[mask])
#     transformed_samples = np.zeros_like(samples)
#     transformed_samples = norm.ppf(U[mask])
#     return transformed_samples

# def cdf(samples):
#     sorted_samples = np.sort(samples)
#     cdf_values = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
#     min_val = samples.min()
#     max_val = samples.max()
#     U = np.zeros_like(samples)

#     # Apply normalization for all values
#     def find_cdf(sample_value):
#         sample_value = np.asarray(sample_value)
#         cdf_result = np.zeros_like(sample_value, dtype=float)
#         for i, value in enumerate(sample_value):
#             index = np.searchsorted(sorted_samples, value)
#             if index == 0:
#                 cdf_result[i] = 0.0
#             elif index >= len(cdf_values):
#                 cdf_result[i] = 1.0
#             else:
#                 cdf_result[i] = cdf_values[index - 1]
#         return cdf_result

#     # Calculate U for all values
#     U = find_cdf(samples)

#     # Transform all values using norm.ppf
#     # Add small epsilon to avoid inf values at 0 and 1
#     epsilon = 1e-10
#     U = np.clip(U, epsilon, 1 - epsilon)
#     transformed_samples = norm.ppf(U)
#     return transformed_samples



# @partial(jax.jit, static_argnames=['epsilon'])
# def cdf(samples, epsilon=1e-10):
#     # Sort samples and create CDF values (vectorized)
#     sorted_samples = jnp.sort(samples)
#     n = len(sorted_samples)
#     cdf_values = jnp.arange(1, n + 1) / n
    
#     # Vectorized search and interpolation
#     def compute_U(values):
#         indices = jnp.searchsorted(sorted_samples, values)
        
#         # Handle edge cases and lookup in one go
#         return jnp.where(
#             indices == 0,
#             0.0,
#             jnp.where(
#                 indices >= n,
#                 1.0,
#                 cdf_values[indices - 1]
#             )
#         )
    
#     # Process all samples at once
#     U = compute_U(samples)
    
#     # Clip and transform (vectorized)
#     U = jnp.clip(U, epsilon, 1 - epsilon)
#     return norm.ppf(U)

@jax.jit
def jax_cdf(samples, epsilon=1e-10):
    """JAX-optimized CDF computation and normal transform."""
    # Sort samples and create CDF values
    sorted_samples = jnp.sort(samples)
    n = len(sorted_samples)
    cdf_values = jnp.arange(1, n + 1) / n
    
    # Vectorized search and interpolation
    indices = jnp.searchsorted(sorted_samples, samples)
    U = jnp.where(
        indices == 0, 0.0,
        jnp.where(
            indices >= n, 1.0,
            cdf_values[indices - 1]
        )
    )
    
    # Clip and transform
    U = jnp.clip(U, epsilon, 1 - epsilon)
    return norm.ppf(U)

# precomputed_cdf = jax.jit(cdf, static_argnums=(0,)).lower(
#     jax.ShapeDtypeStruct((10000,), jnp.float32)
# ).compile()

gmms = []
for i in range(nEvents):
    with open(f'./gmm_cdf_pkl/{i}de.pkl', 'rb') as f:
        gmm = pickle.load(f)
        gmms.append(gmm)
        

def in_cdf_transform(samples, trans):
    U = norm.cdf(trans)
    return np.quantile(samples, U)

def get_gmm_info(gmms):
    weights = [gmm.weights_ for gmm in gmms]
    means = [gmm.means_ for gmm in gmms]
    covariances = [gmm.covariances_ for gmm in gmms]
    return weights, means, covariances

mnorm = multivariate_normal

@jit
def gmm_logpdf(x, weights, means, covariances):    
       # Compute log-PDF for each component (K, N)
    def component_logpdf(mean, cov):
        return mnorm.logpdf(x, mean=mean, cov=cov)
    
    log_component_pdfs = jax.vmap(component_logpdf)(means, covariances)  # shape (K, N)
    
    # Weighted sum in log-space (logsumexp trick for numerical stability)
    weighted_log_pdfs = jnp.log(weights)[:, jnp.newaxis] + log_component_pdfs  # shape (K, N)
    log_pdf = jax.scipy.special.logsumexp(weighted_log_pdfs, axis=0)  # shape (N,)
    
    return log_pdf.squeeze()

@jit
def gmm_logpdf_optimized(x_batch, weights, means, precisions, logdets):
    """
    x_batch: (N, 3) - Input points to evaluate
    Returns: (N,) log probabilities
    """
    # Compute quadratic forms: (x - μ)^T Σ^{-1} (x - μ) for all K components
    diffs = x_batch[:, None, :] - means[None, :, :]  # (N, K, 3)
    quad_forms = jnp.einsum('nki,kij,nkj->nk', diffs, precisions, diffs)  # (N, K)
    
    # Compute log probabilities for all components
    log_probs = -0.5 * (3 * jnp.log(2 * jnp.pi) + logdets + quad_forms)  # (N, K)
    
    # Weighted sum (logsumexp for stability)
    weighted_log_probs = jnp.log(weights)[None, :] + log_probs  # (N, K)
    return jax.scipy.special.logsumexp(weighted_log_probs, axis=-1)  # (N,)

# @profile
def spectral_siren_log_likelihood_nosky_gmm(gamma1=3, m_min_1=5,m_max_1=80,alpha_1=3.3,dm_min_1=1,dm_max_1=10,beta=1,mu=50,sigma=3,f1=0.4):
    start_time = time.time()
    n_samples=nEvents*nsamp
    
    zsels = z_of_dL(dLsels, H0Planck,Om0Planck)
    m1sels = m1detsels/(1+zsels)
    m2sels = m2detsels/(1+zsels)

    log_det_weights = log_p_pop_pl_pl(m1sels,m2sels,zsels,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,beta,mu,sigma,f1,gamma1)

    log_det_weights += - jnp.log(p_draw) - 2*jnp.log1p(zsels) - jnp.log(ddL_of_z(zsels,dLsels,H0Planck, Om0Planck))

    log_mu = logsumexp(log_det_weights) - jnp.log(Ndraw)
    log_s2 = logsumexp(2*log_det_weights) - 2.0*jnp.log(Ndraw)
    log_sigma2 = logdiffexp(log_s2, 2.0*log_mu - jnp.log(Ndraw))
    Neff = jnp.exp(2.0*log_mu - log_sigma2)

    ll = -jnp.inf
    ll = jnp.where((Neff <= 4 * nEvents), ll, 0)
    ll += -nEvents*log_mu + nEvents*(3 + nEvents)/(2*Neff)

    # start_time = time.time()

    print(m_min_1, m_max_1, alpha_1, dm_min_1, dm_max_1, mu, sigma, f1)

    m1, q = m1_q_samples(n_samples, m_min_1, m_max_1, alpha_1, dm_min_1, dm_max_1, mu, sigma, f1)
    m2 = q * m1
    
    samples, p_vals = metro_mc(n_samples, m_min_1, m_max_1, alpha_1, dm_min_1, dm_max_1, mu, sigma, f1)
    
    # samples = np.array([m1, q])
    np.savetxt('model_samples.txt', samples)
    exit()

    z = z_sampling(n_samples, gamma1)
    dL = dL_of_z(z, H0=H0Planck)
    
    # end_time = time.time()
    # etime = end_time - start_time
    # print('etime', etime)
    
    # 0.02 for above
    
    m1det = m1*(1+z)
    m2det = m2*(1+z)
    points = jnp.vstack([m1det, m2det, dL])

    trans_m1 = jax_cdf(m1det)
    trans_m2 = jax_cdf(m2det)
    trans_dL = jax_cdf(dL)


    trans_param = jnp.column_stack((trans_m1, trans_m2, trans_dL))
    log_weights = jnp.zeros(trans_m1.shape[0])
    

    
    wts, mus, covs = get_gmm_info(gmms)
    wts = wts[0]
    mus= mus[0]
    covs = covs[0]
    
    wts = jnp.asarray(wts, dtype=jnp.float32)
    mus = jnp.asarray(mus, dtype=jnp.float32)
    covs = jnp.asarray(covs, dtype=jnp.float32)
    
    eps = 1e-6
    covs = covs + eps * jnp.eye(3, dtype=jnp.float32)[None, ...]
    precisions = jnp.linalg.inv(covs)  # (K, 3, 3)
    logdets = jnp.log(jnp.linalg.det(covs))
    
    
                                     
    trans_param = jnp.asarray(trans_param, dtype=jnp.float32)
    
    # start_time = time.time()
    # results = gmm.score_samples(trans_param)
    # results = gmm_logpdf(trans_param, wts, mus, covs)
    results = gmm_logpdf_optimized(trans_param, wts, mus, precisions, logdets)
    # end_time = time.time()
    # etime = end_time - start_time
    # print('etime', etime)

#     exit()
                                  
    log_weights += results
    # print('after')
   
    dL1 = dL
    log_weights += -jnp.log(ddL_of_z(z,dL1,H0Planck,Om0Planck)) - 2*jnp.log(dL1) - 2*jnp.log1p(z) + jnp.log(m1)
    # log_weights += jnp.log(ddL_of_z(z,dL1,H0Planck,Om0Planck))+ 2*jnp.log1p(z) - 2*jnp.log(dL1)
    log_weights = log_weights.reshape((nEvents,nsamp))
    ll += jnp.sum(-jnp.log(nsamp) + jnp.nan_to_num(logsumexp(log_weights,axis=-1)))

    # end_time = time.time()
    # etime = end_time - start_time
    # print('etime', etime)
    return ll, Neff


# lp = LineProfiler()
# lp_wrapper = lp(spectral_siren_log_likelihood_nosky_gmm)
# lp_wrapper()
# lp.print_stats()

# exit()

true_param = [2.9, 2.35, 80, 3.5, 0.39, 10, 1.1, 50, 3, 0.4]


# In[58]:


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


lower_bound = np.array([gamma_low,m_min_1_low,m_max_1_low,alpha_1_low,dm_min_1_low,dm_max_1_low,beta_low,mu_low,sigma_low,f1_low])
upper_bound = np.array([gamma_high,m_min_1_high,m_max_1_high,alpha_1_high,dm_min_1_high,dm_max_1_high,beta_high,mu_high,sigma_high,f1_high])

# lower_bound = np.array([m_min_1_low,m_max_1_low,alpha_1_low,dm_min_1_low,dm_max_1_low,beta_low,mu_low,sigma_low,f1_low])
# upper_bound = np.array([m_min_1_high,m_max_1_high,alpha_1_high,dm_min_1_high,dm_max_1_high,beta_high,mu_high,sigma_high,f1_high])

# In[57]:

parameters = ["gamma1", "m_min_1", "m_max_1", "alpha_1", "dm_min_1", "dm_max_1", "beta", "mu", "sigma", "f1"]

def plot_param(gamma1=3, m_min_1=5,m_max_1=80,alpha_1=3.3,dm_min_1=1,dm_max_1=10,beta=1,mu=50,sigma=3,f1=0.4):
    
    fixed_values = {
        "gamma1": 3, "m_min_1": 5, "m_max_1": 80, "alpha_1": 3.3,
        "dm_min_1": 1, "dm_max_1": 10, "beta": 1, "mu": 50, "sigma": 3, "f1": 0.4
    }


    

    for i, param in enumerate(parameters):
        print(i, param)
    # Generate the range for the current parameter based on bounds
        param_range = np.linspace(lower_bound[i], upper_bound[i], 10)
        
        # Get fixed values for the other parameters
        other_params = {k: v for k, v in fixed_values.items() if k != param}
        
        # Evaluate the likelihood functions
        # ll0, n0 = spectral_siren_log_likelihood_nosky(**{param: param_range, **other_params})
        # print(ll0)
        ll1, n1 = [], []
        # ll2, n2 = [], []
        ll0, n0 = [], []
        for val in param_range:
            ll00, n00= spectral_siren_log_likelihood_nosky(**{param: val, **other_params})
            ll0.append(ll00)
            ll10, n10= spectral_siren_log_likelihood_nosky_gmm(**{param: val, **other_params})
            ll1.append(ll10)
            # ll20, n20= spectral_siren_log_likelihood_nosky_kde(**{param: val, **other_params})
            # ll2.append(ll20)
        print(ll0) 
        # Plot the likelihood functions
        plt.figure(figsize=(8, 6))
        plt.plot(param_range, ll0, label=f"$L({param}, fixed)$", color="blue")
        plt.plot(param_range, ll1, label=f"$L_1({param}, fixed)$", color="red", linestyle="--")
        # plt.plot(param_range, ll2, label=f"$L_2({param}, fixed)$", color="green", linestyle="--")
        plt.xlabel(param)
        plt.ylabel("Likelihood")
        plt.title(f"Comparison of Likelihood Functions (Varying {param})")
        plt.legend()
        ll1, n1 = [], []
        plt.grid()
        plt.savefig(f'{param}4.png')
        plt.close()

plot_param()
exit()


def likelihood(coord):
    for i in range(len(coord)):
        if (coord[i]<lower_bound[i] or coord[i]>upper_bound[i]):
            return -np.inf
    ll, Neff = spectral_siren_log_likelihood_nosky_kde(*coord)
    if np.isnan(ll):
        return -np.inf
    if (Neff < 4*nEvents):
        return -np.inf
    else:
        return ll




# In[52]:


ndims = 9
nlive = 200

# ndims=9

labels = ['gamma1','m_min_1','m_max_1','alpha_1','dm_m_min_1','dm_m_max_1','beta','mu','sigma','f1']
labels = ['m_min_1','m_max_1','alpha_1','dm_m_min_1','dm_m_max_1','beta','mu','sigma','f1']


# def prior_transform(theta):
#     gamma1_,m_min_1_,m_max_1_,alpha_1_,dm_min_1_,dm_max_1_,beta_,mu_,sigma_,f1_ = theta

#     gamma1 = gamma1_*(upper_bound[0]-lower_bound[0]) + lower_bound[0]
#     m_min_1 = m_min_1_*(upper_bound[1]-lower_bound[1]) + lower_bound[1]
#     m_max_1 = m_max_1_*(upper_bound[2]-lower_bound[2]) + lower_bound[2]
#     alpha_1 = alpha_1_*(upper_bound[3]-lower_bound[3]) + lower_bound[3]
#     dm_min_1 = dm_min_1_*(upper_bound[4]-lower_bound[4]) + lower_bound[4]
#     dm_max_1 = dm_max_1_*(upper_bound[5]-lower_bound[5]) + lower_bound[5]
#     beta = beta_*(upper_bound[6]-lower_bound[6]) + lower_bound[6]
#     mu = mu_*(upper_bound[7]-lower_bound[7]) + lower_bound[7]
#     sigma = sigma_*(upper_bound[8]-lower_bound[8]) + lower_bound[8]
#     f1 = f1_*(upper_bound[9]-lower_bound[9]) + lower_bound[9]

#     return (gamma1,m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,beta,mu,sigma,f1)


def prior_transform(theta):
    m_min_1_,m_max_1_,alpha_1_,dm_min_1_,dm_max_1_,beta_,mu_,sigma_,f1_ = theta

    m_min_1 = m_min_1_*(upper_bound[0]-lower_bound[0]) + lower_bound[0]
    m_max_1 = m_max_1_*(upper_bound[1]-lower_bound[1]) + lower_bound[1]
    alpha_1 = alpha_1_*(upper_bound[2]-lower_bound[2]) + lower_bound[2]
    dm_min_1 = dm_min_1_*(upper_bound[3]-lower_bound[3]) + lower_bound[3]
    dm_max_1 = dm_max_1_*(upper_bound[4]-lower_bound[4]) + lower_bound[4]
    beta = beta_*(upper_bound[5]-lower_bound[5]) + lower_bound[5]
    mu = mu_*(upper_bound[6]-lower_bound[6]) + lower_bound[6]
    sigma = sigma_*(upper_bound[7]-lower_bound[7]) + lower_bound[7]
    f1 = f1_*(upper_bound[8]-lower_bound[8]) + lower_bound[8]

    return (m_min_1,m_max_1,alpha_1,dm_min_1,dm_max_1,beta,mu,sigma,f1)


from dynesty.utils import resample_equal
from dynesty import NestedSampler, DynamicNestedSampler
import multiprocessing as multi

# try:
#     pool.close()
# except Exception:
#     pass

bound = 'multi'
sample = 'rwalk'
nprocesses = 1
Dynamic = False

pool = multi.Pool()
pool.size = nprocesses

if Dynamic is True:
    dsampler = DynamicNestedSampler(likelihood, prior_transform, ndims, bound=bound, sample=sample)#, pool=pool)
    dsampler.run_nested()
else:
    dsampler = NestedSampler(likelihood, prior_transform, ndims, bound=bound, sample=sample)#, pool=pool)
    dsampler.run_nested(dlogz=0.1)


# In[64]:


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

fig = corner.corner(dpostsamples, labels=labels, hist_kwargs={'density': True})

plt.show()
plt.savefig('./iggy_gwtc3-gmmopt.png')

import pickle

# open a file, where you ant to store the data
file = open('plbump-GWTC3-norm-ng.pkl', 'wb')

# dump information to that file
pickle.dump(dres, file)

# close the file
file.close()
