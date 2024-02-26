#!/usr/bin/env python
# coding: utf-8

# # Using denamrf to estimate posteriors of GW150914

# In[1]:


# Installing denamrf 


# In[7]:


# Import libraries

import numpy as np
from scipy.stats import multivariate_normal
from denmarf import DensityEstimate

# Plotting
import matplotlib
from matplotlib import pyplot as plt
import getdist
from getdist import MCSamples, plots


# In[3]:


# #Import data from ligo
# !wget https://dcc.ligo.org/public/0157/P1800370/002/GWTC-1_sample_release.tar.gz
# !tar -xvzf GWTC-1_sample_release.tar.gz


# In[8]:



import h5py

import pandas as pd
from scipy.interpolate import interp1d
from astropy import cosmology, units

import bilby as bb
from bilby.core.prior import LogUniform, PriorDict, Uniform
from bilby.hyper.model import Model
import gwpopulation as gwpop


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

# In[11]:


# Samples from the first three parameters
first_event = posteriors[0]

parameters = list()
training_index = int(len(first_event['mass_1_det'])*0.7)

for i in range(len(first_event['mass_1_det'])):
    parameters.append([first_event['mass_1_det'][i], first_event['mass_2_det'][i], first_event['luminosity_distance'][i]])


# In[ ]:


lower_bounds = np.full(np.asarray(parameters).shape, [5,5,100])
upper_bounds = np.full(np.asarray(parameters).shape, [60,60,700])


# In[13]:

blocks = [32, 64, 128, 256]
hiddens = [5, 10, 15]
epochs = [100,200]

# True data for evaluation, using the remaining 30% of data
parameters_eval = list()

for i in range(training_index, len(first_event['mass_1_det'])):
    parameters_eval.append([first_event['mass_1_det'][i], first_event['mass_2_det'][i], first_event['luminosity_distance'][i]])

parameters_eval = np.array(parameters_eval)

samples_exact = MCSamples(samples=parameters_eval, label="from samples")


for num_blocks in blocks:
    for num_hidden in hiddens:
        for num_epochs in epochs:
            
            # Denmarf density estimation
            de = DensityEstimate(device="cuda", use_cuda=True).fit(
                parameters,
                num_blocks=num_blocks,
                num_hidden=num_hidden,
                num_epochs=num_epochs
            )
            de.save("denmarf_{}_{}_{}.pkl".format(num_blocks, num_hidden, num_epochs))

            xgen_maf = de.sample(12000)
            samples_maf = MCSamples(samples=xgen_maf, label="from denmarf")
            g1 = plots.get_subplot_plotter()
            g1.triangle_plot([samples_exact, samples_maf], filled=False)
            g1.export("plots/maf_{}_{}_{}.pdf".format(num_blocks, num_hidden, num_epochs))


            # In[ ]:




