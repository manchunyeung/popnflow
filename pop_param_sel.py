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


# In[4]:


def infer_required_args_from_function_except_n_args(func, n=1): # Modified from bilby, to get the arguments of posterior
    fullparameters = inspect.getfullargspec(func)
    parameters = fullparameters.args[:len(fullparameters.args) - len(fullparameters.defaults)]
    del parameters[:n]
    return parameters


# In[5]:


def powerlaw(param, exponent, param_min, param_max):
    return param**(exponent) * (1+exponent) / (param_max**(1+exponent) - param_min**(1+exponent))


# In[6]:


# Function including dnesities
def E(z,Om0=Om0Planck):
    return np.sqrt(Om0*(1+z)**3 + (1.0-Om0))

# Comoving volume
def dVdz(z, H0=H0Planck):
    cosmo = FlatLambdaCDM(H0=H0Planck,Om0=Om0Planck)
    r_of_z = cosmo.comoving_distance(z).to(u.Mpc).value
    return (speed_of_light / H0*E(z)) * r_of_z**2 


# In[7]:


def p_m1(mass1, alpha, m_min, m_max):
    mask = mass1<m_max
    mask1 = mass1>m_min
    return np.where(mask & mask1, powerlaw(mass1, -alpha, m_min, m_max), 0)

def p_q(mass1, mass_ratio, beta, m_min):    
    mask = mass_ratio>(m_min/mass1)
    return np.where(mask, powerlaw(mass_ratio, beta, m_min/mass1, 1), 0)

def p_z(data, kappa): 
    return (1+data['redshift'])**(kappa-1) * dVdz(data['redshift'])


def total_p(mass1, mass_ratio, alpha, beta, m_min, m_max):
    return p_m1(mass1, alpha, m_min, m_max)*p_q(mass1, mass_ratio, beta, m_min)


# In[8]:


def log_beta(m1sels, qsels, p_draw, Nsamples, Lambda):
    ratio = np.divide(total_p(m1sels, qsels, *Lambda), p_draw, out=np.zeros_like(p_draw), where=p_draw!=0)
    return np.log(logsumexp(np.log(ratio)) - np.log(Nsamples))

def compute_log_likelihood_per_event(data, Lambda, prior):
    m1 = data['mass_1']
    mass_ratio = data['mass_ratio']
    ratio = np.divide(total_p(m1, mass_ratio, *Lambda), prior, out=np.zeros_like(prior), where=prior!=0)
    return np.log(np.mean(ratio))

def log_likelihood(Lambda, posteriors, m1sels, qsels, p_draw):
    log_likelihood = 0.
    for posterior in posteriors:
        if "prior" in posterior:
            prior = posterior["prior"]
        else:
            prior = 1.
        Nsamples = len(posterior['mass_1'])
        log_likelihood += compute_log_likelihood_per_event(posterior, Lambda, prior)-10*log_beta(m1sels, qsels, p_draw, Nsamples, Lambda)
    return log_likelihood


# In[9]:


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


draw_file = "o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5"

with h5py.File(draw_file, 'r') as ff:
    name_df = pd.DataFrame(ff['injections']['name'])
    snr_df = pd.DataFrame(ff['injections']['optimal_snr_net'])
    position = list(name_df[(name_df.iloc[:, 0].isin([b'o1', b'o2'])) & (snr_df.iloc[:,0]>=9)].index)
    p_draw = pd.DataFrame(ff['injections']['sampling_pdf']).iloc[position[0:-1], 0]
    m1sels = pd.DataFrame(ff['injections']['mass1_source']).iloc[position[0:-1], 0]
    m2sels = pd.DataFrame(ff['injections']['mass2_source']).iloc[position[0:-1], 0]
    qsels = m2sels/m1sels

plt.scatter(m1sels, m2sels)
plt.savefig('./sels.pdf')
exit()

luminosity_distances = np.linspace(1, 10000, 1000)
redshifts = np.array(
    [
        cosmology.z_at_value(cosmology.Planck15.luminosity_distance, dl * units.Mpc)
        for dl in luminosity_distances
    ]
)
dl_to_z = interp1d(luminosity_distances, redshifts)

luminosity_prior = luminosity_distances**2

dz_ddl = np.gradient(redshifts, luminosity_distances)

redshift_prior = interp1d(redshifts, luminosity_prior / dz_ddl * (1 + redshifts)**2)


# In[15]:


for posterior in posteriors:
    posterior["redshift"] = dl_to_z(posterior["luminosity_distance"])
    posterior["mass_1"] = posterior["mass_1_det"] / (1 + posterior["redshift"])
    posterior["mass_2"] = posterior["mass_2_det"] / (1 + posterior["redshift"])
    posterior["mass_ratio"] = posterior["mass_2"] / posterior["mass_1"]
    # posterior["prior"] = posterior["mass_1"] / posterior["mass_1"]
    posterior["prior"] = redshift_prior(posterior["redshift"])


# In[16]:


import bilby as bb
from bilby.core.prior import LogUniform, PriorDict, Uniform
from bilby.hyper.model import Model


# In[17]:


fast_priors = PriorDict()

# mass
fast_priors["alpha"] = Uniform(minimum=-4, maximum=12, latex_label="$\\alpha$")
fast_priors["beta"] = Uniform(minimum=-4, maximum=12, latex_label="$\\beta$")
# fast_priors["kappa"] = Uniform(minimum=5, maximum=14, latex_label="$\\kappa$")
fast_priors["mmin"] = Uniform(minimum=5, maximum=10, latex_label="$m_{\\min}$")
fast_priors["mmax"] = Uniform(minimum=30, maximum=70, latex_label="$m_{\\max}$")
# fast_priors["lam"] = Uniform(minimum=0, maximum=1, latex_label="$\\lambda_{m}$")
# fast_priors["mpp"] = Uniform(minimum=10, maximum=50, latex_label="$\\mu_{m}$")
# fast_priors["sigpp"] = Uniform(minimum=0, maximum=10, latex_label="$\\sigma_{m}$")


# In[18]:


# def log_prior(Lambda):
#     alpha, beta, mmin, mmax, lam, mpp, sigpp = Lambda
#     if -2<alpha<4 and -4<beta<12 and 5<mmin<10 and 20<mmax<60 and 0<lam<1 and 10<mpp<50 and 0<sigpp<10:
#         return 0.
#     return -np.inf

def log_prior(Lambda):
    alpha, beta, m_min, m_max = Lambda
    if (alpha>-4)and (alpha<12) and (beta>-4) and (beta<12) and (m_max<70) and (m_max>30) and (m_min>5) and (m_min<10):
        return 0.
    return -np.inf

def log_probability(Lambda, posteriors, m1sels, qsels, p_draw):
    lp = log_prior(Lambda)
    if not np.isfinite(lp):
        return -np.inf
    return log_likelihood(Lambda, posteriors, m1sels, qsels, p_draw) + lp


# In[19]:


pos = list()
for param in list(fast_priors.keys()):
    pos.append(np.random.uniform(fast_priors[param].minimum, high=fast_priors[param].maximum, size=30))
pos = np.array(pos).transpose()
nwalkers, ndim = pos.shape


# In[20]:


import emcee

# To be run on cluster with backending
filename = "data/selection_samples.txt"
backend = emcee.backends.HDFBackend(filename) 
N_samples = 1000

try:
    samples = backend.get_chain() 
    lastpoint = samples[-1] #Find the last point in the existing backend
    p0 = lastpoint #Initiate the walkers from the last point
    existing_steps = len(samples) #Number of steps already completed
    N_samples = N_samples - existing_steps #Update the number of steps to be run
    
except: #If the backend is empty (i.e. a new run)
    
    p0 = pos #np.random.uniform(-1.0e-10,1.0e-10,size=[nwalkers, ndim])
    
sampler=emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(posteriors, m1sels, qsels, p_draw), backend = backend)
sampler.run_mcmc(p0, N_samples, progress=True)


# import emcee

# # pos = np.random.randint(0, high=50, size=(30, 5))

# sampler = emcee.EnsembleSampler(
#     nwalkers, ndim, log_probability, args=(posteriors, m1sels, m2sels, p_draw)
# )
# sampler.run_mcmc(pos, 1000, progress=True);


# # In[ ]:


# import emcee
# # filename = "stored_walkers.txt"
# # backend = emcee.backends.HDFBackend(filename) 
# # flat_samples = backend.get_chain(discard=100, thin=100, flat=True)
# flat_samples = sampler.get_chain(discard=100, thin=100, flat=True)


# # In[ ]:


# import corner

# # labels = ['alpha', 'beta', 'mmin', 'mmax', 'lam', 'mpp', 'sigpp']
# labels = ['alpha', 'beta', 'kappa', 'mmin', 'mmax']

# fig = corner.corner(
#     flat_samples, labels=labels, smooth=1.2
# );


# # In[ ]:


# np.random.uniform(fast_priors['alpha'].minimum, high=fast_priors['alpha'].maximum, size=30)


# # In[ ]:




