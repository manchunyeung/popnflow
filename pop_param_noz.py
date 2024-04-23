#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import gwpopulation as gwpop # using the mass model there


# In[2]:


import h5py
import pandas as pd
from scipy.interpolate import interp1d, interp2d
from astropy import cosmology, units, constants
from astropy.cosmology import Planck15, FlatLambdaCDM
from tqdm import tqdm
import astropy.units as u
import scipy
from scipy.special import logsumexp
import matplotlib.pyplot as plt


# In[3]:


H0Planck = Planck15.H0.value
Om0Planck = Planck15.Om0
speed_of_light = constants.c.to('km/s').value

Nsamples = 10000

def infer_required_args_from_function_except_n_args(func, n=1): # Modified from bilby, to get the arguments of posterior
    fullparameters = inspect.getfullargspec(func)
    parameters = fullparameters.args[:len(fullparameters.args) - len(fullparameters.defaults)]
    del parameters[:n]
    return parameters

def Sfilter(m, m_min, dm=0.01):
    def f(mm, dmm):
        return np.exp(dmm/mm + dmm/(mm-dmm))
    Sfilter = 1./(f(m-m_min, dm)+1.)
    Sfilter = np.where(m<m_min+dm, Sfilter, 1.)
    Sfilter = np.where(m>m_min, Sfilter, 0.)
    return Sfilter

def powerlaw(param, exponent, param_min, param_max):
    return param**(exponent) * (1+exponent) / (param_max**(1+exponent) - param_min**(1+exponent))

def gaussian(param, mu, sigma):
    return np.exp(-(param-mu)**2 / (2 * sigma**2)) / np.sqrt(2*np.pi*sigma**2)

### O3 analysis



# Function including dnesities
def E(z,Om0=Om0Planck):
    return np.sqrt(Om0*(1+z)**3 + (1.0-Om0))

# Comoving volume
def dVdz(z, H0=H0Planck):
    cosmo = FlatLambdaCDM(H0=H0Planck,Om0=Om0Planck)
    r_of_z = cosmo.comoving_distance(z).to(u.Mpc).value
    return (speed_of_light / H0*E(z)) * r_of_z**2 


# In[7]:


def p_m(mass1, alpha, m_min, m_max, lamb, mu, sigma):
    mask = mass1<m_max
    mask1 = mass1>m_min
    return np.where(mask & mask1, ((1-lamb) * powerlaw(mass1, -alpha, m_min, m_max) + lamb * gaussian(mass1, mu, sigma)) * Sfilter(mass1, m_min), 0)

def p_q(mass1, mass_ratio, beta, m_min):    
    mask = mass_ratio>(m_min/mass1)
    mask1 = mass_ratio<1
    return np.where(mask & mask1, powerlaw(mass_ratio, beta, m_min/mass1, 1), 0)

def p_z(redshift, kappa): 
    return (1+redshift)**(kappa-1) * dVdz(redshift)


def total_p(mass1, mass2, mass_ratio, redshift, alpha, beta, m_min, m_max, lamb, mu, sigma):
    # total_pm = p_m(mass1, alpha, m_min, m_max)*p_m(mass2, alpha, m_min, m_max)*(mass1>=mass2)*2
    return p_m(mass1, alpha, m_min, m_max, lamb, mu, sigma) * p_q(mass1, mass_ratio, beta, m_min) 

def log_beta(m1sels, m2sels, qsels, zels, p_draw, Ndraw, Lambda):
    alpha, beta, mmin, mmax, lamb, mu, sigma = Lambda
    ratio = np.divide(total_p(m1sels, m2sels, qsels, zels, *Lambda), p_draw, out=np.zeros_like(p_draw), where=p_draw!=0)
    return logsumexp(np.log(ratio)) - np.log(Ndraw)

def compute_log_likelihood_per_event(data, Lambda, prior, Nsamples):
    m1 = data['mass_1'][:Nsamples]
    m2 = data['mass_2'][:Nsamples]
    mass_ratio = data['mass_ratio'][:Nsamples]
    redshift = data['redshift'][:Nsamples]

    log_pm = np.log(total_p(m1, m2, mass_ratio, redshift, *Lambda))
    log_prior = np.log(prior)
    return np.sum(logsumexp(log_pm - log_prior) - np.log(Nsamples))

def log_likelihood(Lambda, posteriors, m1sels, qsels, zels, p_draw, Ndraw):

    alpha, beta, mmin, mmax, lamb, mu, sigma = Lambda
    Nevents = 69
    Nsamples = 5000
    log_likelihood = -Nevents*log_beta(m1sels, m2sels, qsels, zsels, p_draw, Ndraw, Lambda)
    for posterior in posteriors:
        if "prior" in posterior:
            prior = posterior["prior"][:Nsamples]
        else:
            prior = 1.

        log_likelihood += compute_log_likelihood_per_event(posterior, Lambda, prior, Nsamples)
    return log_likelihood

# Load GWTC-1 events
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

file_str = "./GWTC-1_sample_release/GW{}_GWTC-1.hdf5"

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
for event in events:
    _posterior = pd.DataFrame()
    _prior = pd.DataFrame()
    with h5py.File(file_str.format(event)) as ff:
        for my_key, gwtc_key in parameter_translator.items():
            _posterior[my_key] = ff["IMRPhenomPv2_posterior"][gwtc_key]
            _prior[my_key] = ff["prior"][gwtc_key]
    posteriors.append(_posterior)
    priors.append(_prior)

# redshift functions
luminosity_distances = np.linspace(1, 10000, 1000)
redshifts = np.array(
    [
        cosmology.z_at_value(cosmology.Planck15.luminosity_distance, dl * units.Mpc)
        for dl in luminosity_distances
    ]
)
dl_to_z = interp1d(luminosity_distances, redshifts)

# comoving distance
R_z = cosmology.Planck15.comoving_distance(redshifts).to(u.Mpc).value
comoving_dist = interp1d(redshifts, R_z)

dH = cosmology.Planck15.hubble_distance.to(u.Mpc).value

dz_ddl = np.gradient(redshifts, luminosity_distances)

# ddL_dz = luminosity_distances/(1+redshifts) + (1+redshifts)*dH/cosmology.Planck15.efunc(redshifts) # Anarya
ddL_dz = comoving_dist(redshifts) + (1+redshifts)*dH/cosmology.Planck15.efunc(redshifts) # Anarya

luminosity_prior = luminosity_distances**2
redshift_prior = interp1d(redshifts, luminosity_prior * ddL_dz * (1 + redshifts)**2, fill_value="extrapolate")

for posterior in posteriors:
    posterior["redshift"] = dl_to_z(posterior["luminosity_distance"])
    posterior["mass_1"] = posterior["mass_1_det"] / (1 + posterior["redshift"])
    posterior["mass_2"] = posterior["mass_2_det"] / (1 + posterior["redshift"])
    posterior["mass_ratio"] = posterior["mass_2"] / posterior["mass_1"]

# Load GWTC-3 events
GWTC3_events = {}
with open('./GWTC-3/events_names.txt', 'r') as f:
    for line in f:
        elements = line.strip('\n').split()
        GWTC3_events[elements[0]] = elements[1]

parameter_translator_1 = dict(
    mass_1="mass_1_source",
    mass_2="mass_2_source",
    mass_ratio="mass_ratio",
    luminosity_distance="luminosity_distance",
    redshift="redshift"
)

for event in list(GWTC3_events.keys()):
    _posterior = pd.DataFrame()
    waveform = GWTC3_events[event]
    with h5py.File("./GWTC-3/{}.h5".format(event)) as ff:
        for my_key, gwtc_key in parameter_translator_1.items():
            _posterior[my_key] = ff[waveform]['posterior_samples'][gwtc_key]
    posteriors.append(_posterior)

for posterior in posteriors:
    posterior["prior"] = redshift_prior(posterior["redshift"])
    # posterior["prior"] = posterior["redshift"]/posterior["redshift"]

draw_file = "o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5"

with h5py.File(draw_file, 'r') as ff:
    name_df = pd.DataFrame(ff['injections']['name'])
    snr_df = pd.DataFrame(ff['injections']['optimal_snr_net'])
    ifar_gstlal = pd.DataFrame(ff['injections']['ifar_gstlal'])
    ifar_pycbc_hyper = pd.DataFrame(ff['injections']['ifar_pycbc_hyperbank'])
    ifar_pycbc = pd.DataFrame(ff['injections']['ifar_pycbc_bbh'])
    position = list(name_df[(snr_df.iloc[:,0]>=9) & (ifar_pycbc.iloc[:,0]>=1) & (ifar_gstlal.iloc[:,0]>=1) & (ifar_pycbc_hyper.iloc[:,0]>=1)].index)

    Ndraw = ff.attrs['total_generated']

    p_draw = pd.DataFrame(ff['injections']['sampling_pdf']).iloc[position[0:-1], 0]
    m1sels = pd.DataFrame(ff['injections']['mass1_source']).iloc[position[0:-1], 0]
    m2sels = pd.DataFrame(ff['injections']['mass2_source']).iloc[position[0:-1], 0]
    zsels = pd.DataFrame(ff['injections']['redshift']).iloc[position[0:-1], 0]
    qsels = m2sels/m1sels

# plt.scatter(m1sels, m2sels)
# plt.savefig('./sels.pdf')
# exit()

# In[16]:


import bilby as bb
from bilby.core.prior import LogUniform, PriorDict, Uniform
from bilby.hyper.model import Model


# In[17]:


fast_priors = PriorDict()

# mass
fast_priors["alpha"] = Uniform(minimum=-4, maximum=12, latex_label="$\\alpha$")
fast_priors["beta"] = Uniform(minimum=-2, maximum=7, latex_label="$\\beta$")
# fast_priors["kappa"] = Uniform(minimum=-6, maximum=6, latex_label="$\\kappa$")
fast_priors["mmin"] = Uniform(minimum=2, maximum=10, latex_label="$m_{\\min}$")
fast_priors["mmax"] = Uniform(minimum=30, maximum=100, latex_label="$m_{\\max}$")
fast_priors["lam"] = Uniform(minimum=0, maximum=1, latex_label="$\\lambda_{m}$")
fast_priors["mpp"] = Uniform(minimum=20, maximum=50, latex_label="$\\mu_{m}$")
fast_priors["sigpp"] = Uniform(minimum=1, maximum=10, latex_label="$\\sigma_{m}$")


# In[18]:


# def log_prior(Lambda):
#     alpha, beta, mmin, mmax, lam, mpp, sigpp = Lambda
#     if -2<alpha<4 and -4<beta<12 and 5<mmin<10 and 20<mmax<60 and 0<lam<1 and 10<mpp<50 and 0<sigpp<10:
#         return 0.
#     return -np.inf

def log_prior(Lambda):
    alpha, beta, mmin, mmax, lam, mpp, sigpp = Lambda
    if -4<alpha<12 and -2<beta<7 and 2<mmin<10 and 30<mmax<100 and 0<lam<1 and 20<mpp<50 and 1<sigpp<10:
        return 0.
    return -np.inf

def log_probability(Lambda, posteriors, m1sels, qsels, zsels, p_draw, Ndraw):
    lp = log_prior(Lambda)
    if not np.isfinite(lp):
        return -np.inf
    return log_likelihood(Lambda, posteriors, m1sels, qsels, zsels, p_draw, Ndraw) + lp


# In[19]:


pos = list()
for param in list(fast_priors.keys()):
    pos.append(np.random.uniform(fast_priors[param].minimum, high=fast_priors[param].maximum, size=30))
pos = np.array(pos).transpose()
nwalkers, ndim = pos.shape


# In[20]:


import emcee

# To be run on cluster with backending
filename = "data/gwtc3_5000_trial.txt"
backend = emcee.backends.HDFBackend(filename) 
N_samples = 5000

try:
    samples = backend.get_chain() 
    lastpoint = samples[-1] #Find the last point in the existing backend
    p0 = lastpoint #Initiate the walkers from the last point
    existing_steps = len(samples) #Number of steps already completed
    N_samples = N_samples - existing_steps #Update the number of steps to be run
    
except: #If the backend is empty (i.e. a new run)
    
    p0 = pos #np.random.uniform(-1.0e-10,1.0e-10,size=[nwalkers, ndim])
    
sampler=emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(posteriors, m1sels, qsels, zsels, p_draw, Ndraw), backend = backend)
sampler.run_mcmc(p0, N_samples, progress=True)

