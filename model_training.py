import numpy as np
import matplotlib.pyplot as plt
import gwpopulation as gwpop # using the mass model there
import h5py
import pandas as pd
from scipy.interpolate import interp1d, interp2d
from astropy import cosmology, units, constants
from astropy.cosmology import Planck15, FlatLambdaCDM
from tqdm import tqdm
import astropy.units as u
from getdist import MCSamples, plots

from jax import jit

from jaxinterp2d import interp2d, CartesianGrid

from scipy.stats import norm
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

def cdf(samples):
    sorted_samples = np.sort(samples)
    cdf_values = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    min_val = samples.min()
    max_val = samples.max()

    U = np.zeros_like(samples)

    # Apply normalization for all values
    def find_cdf(sample_value):
        sample_value = np.asarray(sample_value)
        cdf_result = np.zeros_like(sample_value, dtype=float)

        for i, value in enumerate(sample_value):
            index = np.searchsorted(sorted_samples, value)
            if index == 0:
                cdf_result[i] = 0.0
            elif index >= len(cdf_values):
                cdf_result[i] = 1.0
            else:
                cdf_result[i] = cdf_values[index - 1]

        return cdf_result

    # Calculate U for all values
    U = find_cdf(samples)

    # Transform all values using norm.ppf
    # Add small epsilon to avoid inf values at 0 and 1
    epsilon = 1e-10
    U = np.clip(U, epsilon, 1 - epsilon)
    transformed_samples = norm.ppf(U)

    return transformed_samples

from sklearn.mixture import GaussianMixture
def GMM(m1, m2, dL, size=4096):
    trans_m1 = cdf(m1)
    trans_m2 = cdf(m2)
    trans_dL = cdf(dL)
    print(trans_m1.shape, trans_m2.shape, trans_dL.shape, 'trans_shape')
    return GaussianMixture(n_components=3).fit(np.column_stack((trans_m1, trans_m2, trans_dL))).sample(size)[0]

def in_cdf_transform(samples, trans):
    U = norm.cdf(trans)
    return np.quantile(samples, U)

jnp=np

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

# @jit
def E(z,Om0=Om0Planck):
    return jnp.sqrt(Om0*(1+z)**3 + (1.0-Om0))

# @jit
def r_of_z(z,H0,Om0=Om0Planck):
    return interp2d(Om0,z,Om0grid,zgrid,rs)*(H0Planck/H0)

# @jit
def dL_of_z(z,H0,Om0=Om0Planck):
    return (1+z)*r_of_z(z,H0,Om0)

# @jit
def z_of_dL(dL,H0=H0Planck,Om0=Om0Planck):
    return jnp.interp(dL,dL_of_z(zgrid,H0,Om0),zgrid)

# @jit
def dV_of_z(z,H0,Om0=Om0Planck):
    return speed_of_light*r_of_z(z,H0,Om0)**2/(H0*E(z,Om0))


def ddL_of_z(z,dL,H0,Om0=Om0Planck):
    return dL/(1+z) + speed_of_light*(1+z)/(H0*E(z,Om0))

def reweight_farr(posteriors):
    m1s, m2s, dls = [],[], []

    for posterior in posteriors:
        m1o = posterior['mass_1'].values
        m2o = posterior['mass_2'].values
        dlo = posterior['luminosity_distance'].values
        
        wt = 1./(dlo*dlo)
        wmax = np.max(wt)
        rs = np.random.uniform(low=0, high=wmax, size = len(m1o))
        sel = rs<wt
        
        m1s.append(m1o[sel])
        m2s.append(m2o[sel])
        dls.append(dlo[sel])
  
    lmin = np.min([len(x) for x in m1s])
    
    m1s_out = []
    m2s_out = []
    dls_out = []
    for m1, m2, dl in zip(m1s, m2s, dls):
        p = np.random.choice(len(m1), size=lmin, replace=False)
        m1s_out.append(m1[p])
        m2s_out.append(m2[p])
        dls_out.append(dl[p])
    m1s_out = np.array(m1s_out)
    m2s_out = np.array(m2s_out)
    dls_out = np.array(dls_out)
    print('nsamples', lmin) 
    return m1s_out, m2s_out, dls_out

def maf(param):
    parameters = param.transpose()

    

    from denmarf import DensityEstimate
    de = DensityEstimate().fit(
        X=parameters,
        num_blocks = 32,
        num_hidden = 5,
        num_epochs = 100
    )
    # sample = de.sample(Nsamples)
    # mass1 = sample[:,0]
    # mass2 = sample[:,0]
    # dL = sample[:,0]
    return de
# def maf(posterior):
#     parameters = np.array([posterior['mass_1'].values, posterior['mass_2'].values, posterior['luminosity_distance'].values])
#     parameters = parameters.transpose()

    

#     from denmarf import DensityEstimate
#     de = DensityEstimate().fit(
#         X=parameters,
#         num_blocks = 32,
#         num_hidden = 5,
#         num_epochs = 100
#     )
#     # sample = de.sample(Nsamples)
#     # mass1 = sample[:,0]
#     # mass2 = sample[:,0]
#     # dL = sample[:,0]
#     return de

# GWTC1
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

ddL_dz = luminosity_distances/(1+redshifts) + (1+redshifts)*cosmology.Planck15.hubble_distance.to(u.Mpc).value/cosmology.Planck15.efunc(redshifts) # Anarya

redshift_prior = interp1d(redshifts, luminosity_prior * ddL_dz * (1 + redshifts)**2, fill_value="extrapolate")

for posterior in posteriors:
    if not "redshift" in posterior:
        # posterior["redshift"] = dl_to_z(posterior["luminosity_distance"])
        posterior["redshift"] = z_of_dL(posterior["luminosity_distance"])
    posterior["mass_1"] = posterior["mass_1_det"] 
    posterior["mass_2"] = posterior["mass_2_det"]
    posterior["mass_ratio"] = posterior["mass_2"] / posterior["mass_1"]
    posterior["prior"] = redshift_prior(posterior["redshift"])


GWTC3_events = {}
with open('./GWTC-3/events_names.txt', 'r') as f:
    for line in f:
        elements = line.strip('\n').split()
        GWTC3_events[elements[0]] = elements[1]

parameter_translator_1 = dict(
    mass_1="mass_1",
    mass_2="mass_2",
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
    if not "redshift" in posterior:
        posterior["redshift"] = z_of_dL(posterior["luminosity_distance"])

m1s, m2s, dls = reweight_farr(posteriors)
        
i=0

labels = ['m1', 'm2', 'dl']
import corner
import pickle
from denmarf import DensityEstimate

m1_gmm = []
m2_gmm, dL_gmm = [], []

for posterior in posteriors:
# for (m1, m2, dl) in zip(m1s, m2s, dls):
    # parameters = np.hstack((posterior['mass_1'].values.reshape(-1,1), posterior['mass_2'].values.reshape(-1,1), posterior['luminosity_distance'].values.reshape(-1,1)))
    # samples_exact = MCSamples(samples = parameters, label = 'from samples')


#     with open(f'./ig_nb/kde_det_pkl/{i}de_farr_full.pkl', 'rb') as file:
#         kde = pickle.load(file)
#     kernel = kde.resample(size = 50000).T


#     de = DensityEstimate.from_file(filename=f"de_pkl/{i}de_farr_full.pkl")
   
#     samples = de.sample(50000)


    m1 = posterior['mass_1']
    m2 = posterior['mass_2']
    dL = posterior['luminosity_distance']

    parameters = np.array([m1[:4096], m2[:4096], dL[:4096]]).transpose()
    print(m1.shape, m2.shape, dL.shape, 'shapes')
    gmm_cdf = GMM(m1, m2, dL)

    m1_fgmm = in_cdf_transform(m1, gmm_cdf[:, 0])
    m2_fgmm = in_cdf_transform(m2, gmm_cdf[:, 1])
    dL_fgmm = in_cdf_transform(dL, gmm_cdf[:, 2])

    m1_gmm.append(m1_fgmm)
    m2_gmm.append(m2_fgmm)
    dL_gmm.append(dL_fgmm)
    
    
    param = np.array([m1_fgmm, m2_fgmm, dL_fgmm]).transpose()
    # param1 = np.array([kernel[:,0][:4096], kernel[:,1][:4096], kernel[:,2][:4096]]).transpose()

    figure = corner.corner(parameters, labels=labels)
    figure1 = corner.corner(param, labels=labels, fig=figure, color='orange')
#     figure2 = corner.corner(param1, labels=labels, fig=figure1, color='red')
    plt.savefig(f'/home/manchun.yeung/population/simon/popnflow/ig_nb/plots/{i}gmm.png')
    plt.close()
    # Samples = de.sample(5000)

    # samples_maf = MCSamples(samples=Samples, label = 'from_denmaf')

    # g1 = plots.get_subplot_plotter()
    # g1.triangle_plot([samples_exact, samples_maf], filled=False)
    # g1.export(f'plots/de_det/{i}de.pdf')
    # print('hi3')
    # posterior = pd.DataFrame()
    # posterior['mass_1_det'] = Samples[:,0]
    # posterior['mass_2_det'] = Samples[:,1]
    # posterior['luminosity_distance'] = Samples[:,2]

#     param = np.array([m1, m2, dl])
#     model = maf(param)
#     model.save(f'de_pkl/{i}de_farr_full.pkl')

    i+=1

m1 = np.concatenate(m1_gmm)
m2 = np.concatenate(m2_gmm)
dL = np.concatenate(dL_gmm)

np.savetxt('m1_gmm.txt', m1)
np.savetxt('m2_gmm.txt', m2)
np.savetxt('dL_gmm.txt', dL)
