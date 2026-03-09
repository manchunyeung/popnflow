import os
# Suppress the CUDA/TF C++ factory registration warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import ot  
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import corner
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

# ==========================================
# 1. Load and Reshape the Data
# ==========================================
indir = "input_data"  

print(f"Loading data from {indir}...")
m1det = np.loadtxt(f'{indir}/m1det.txt')
m2det = np.loadtxt(f'{indir}/m2det.txt')
dL = np.loadtxt(f'{indir}/dL.txt')

Nobs = 69
N_samples = 4096

# Reshape into (Events, Samples)
m1det_ = m1det.reshape(Nobs, N_samples)
m2det_ = m2det.reshape(Nobs, N_samples)
dL_ = dL.reshape(Nobs, N_samples)

# ==========================================
# 2. Train GMMs with K=7 for each Event
# ==========================================
print("Training GMMs (K=7) for all events...")
K = 7
gmms = []
seed = 1234  

for e in tqdm(range(Nobs), desc="Fitting GMMs"):
    X_e = np.column_stack([m1det_[e], m2det_[e], dL_[e]])
    gmm = GaussianMixture(
        n_components=K, 
        covariance_type='full',
        reg_covar=1e-6, 
        n_init=10, 
        random_state=seed
    ).fit(X_e)
    gmms.append(gmm)

# ==========================================
# 3. Evaluate Normalized Sliced Wasserstein Distance & Plot
# ==========================================
output_dir = "results"
os.makedirs(f"{output_dir}/figures/gmm", exist_ok=True)

wasserstein_distances = []

print("Evaluating Normalized Sliced 3D Wasserstein distances...")
for e in tqdm(range(Nobs), desc="Plotting & Evaluating"):
    
    # Original training data for this event
    X_train = np.column_stack([m1det_[e], m2det_[e], dL_[e]])
    
    # Draw resampled data from the trained GMM
    X_resampled, _ = gmms[e].sample(N_samples)
    
    # ---------------------------------------------------------
    # NEW: Z-Score Normalization for Distance Calculation
    # ---------------------------------------------------------
    # Calculate the mean and std of the TRUE posterior
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    
    # Prevent division by zero just in case a parameter is a delta function
    train_std = np.where(train_std == 0, 1e-8, train_std)
    
    # Scale both sets using the TRUE posterior's statistics
    X_train_norm = (X_train - train_mean) / train_std
    X_resampled_norm = (X_resampled - train_mean) / train_std
    
    # Calculate Sliced Wasserstein on the NORMALIZED data
    w_dist = ot.sliced_wasserstein_distance(X_train_norm, X_resampled_norm, n_projections=50)
    wasserstein_distances.append(w_dist)
    
    # ---------------------------------------------------------
    # Generate the Corner Plot (Using Original Unnormalized Data)
    # ---------------------------------------------------------
    labels = [r"$m_1^{\mathrm{det}}$", r"$m_2^{\mathrm{det}}$", r"$d_L$"]
    
    fig = corner.corner(
        X_train,
        color='#1f77b4',
        labels=labels,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        hist_kwargs={'density': True}
    )
    
    corner.corner(
        X_resampled,
        color='#ff7f0e',
        fig=fig,
        hist_kwargs={'density': True}
    )
    
    # Formatting and Legend
    # The title now explicitly states the distance is in units of standard deviations
    fig.suptitle(f"Event {e} | Normalized Wasserstein Distance: {w_dist:.3f} $\sigma$", fontsize=16)
    
    blue_patch = mpatches.Patch(color='#1f77b4', label='Original Samples')
    orange_patch = mpatches.Patch(color='#ff7f0e', label='GMM Resampled')
    fig.axes[1].legend(handles=[blue_patch, orange_patch], loc='center', fontsize=12)
    
    plt.savefig(f"{output_dir}/figures/gmm/events_{e:02d}_corner.pdf", bbox_inches='tight', dpi=150)
    plt.close(fig)

# ==========================================
# 4. Numerical Summary
# ==========================================
wasserstein_distances = np.array(wasserstein_distances)
avg_w_dist = np.mean(wasserstein_distances)

print("\n" + "="*50)
print(" GMM 3D RECOVERY DIAGNOSTICS SUMMARY")
print("="*50)
print(f"Total Events Evaluated: {Nobs}")
print(f"Average Shift from Truth: {avg_w_dist:.4f} standard deviations")
print(f"Worst Fit (Max Distance): {np.max(wasserstein_distances):.4f} std devs (Event {np.argmax(wasserstein_distances)})")
print(f"Best Fit (Min Distance):  {np.min(wasserstein_distances):.4f} std devs (Event {np.argmin(wasserstein_distances)})")
print(f"All {Nobs} corner plots saved to: {output_dir}/figures/gmm/")
