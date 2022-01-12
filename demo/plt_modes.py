#%%
import os, sys
import matplotlib.pyplot as plt
import pltaux
import pickle
import numpy as np
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import utils
import itertools
#%% Load time series
IMG_DIR = "/home/cmendoza/MEGA/Research/Third_Paper/proto"
root = "/home/cmendoza/Research/QSMP/data/Study019/"
folder = "preictal"

fname = 'first56_segments_CSP1.npz'
fpath = os.path.join(root, folder, fname)
with np.load(fpath) as data:
    T = data['time_series']
    splice = data['splice']

#%% Load QSMP
fname = 'qsmp_m350_snr-4.0_-2.0_0.0_2.0_4.0.npz'
fpath = os.path.join(root, folder, fname)
with np.load(fpath) as data:
    density = data['density']
    profile = data['profile']
    neighbor = data['indices']

#%% Hyperparameters
nonan_profile = profile[~np.isnan(profile)]
quantiles = np.quantile(nonan_profile, [0.75, 0.99])
quantiles = np.log2(quantiles)
maxdists = 2 ** np.linspace(*quantiles, 5)

snr = np.r_[-4, -2, 0, 2, 4]
var_noise = 10 ** (-snr/10)
th = 0.1
bandwidths = (9 * var_noise) / np.log(1/th)
sigmas = np.sqrt(bandwidths/2)
n_sigmas = sigmas.size

path_agg = ['add', 'max', 'mean']
path_agg = [path_agg[1]]
#%%
profile = profile.T[:n_sigmas]
neighbor = neighbor.T[:n_sigmas]
density = density.T[:n_sigmas]

#%%
idx = 2
m = 350
max_modes = 9
dpath = os.path.join(root, folder)
grid, indices, energy, fwhm = pltaux.built_grid_fixed_sigma(
    idx, dpath, maxdists, path_agg, T, m, max_modes)
#%%
i_dist = 4
i_agg = 0
plt.plot(grid[i_dist][i_agg].T, color='#1f77b4')
# print(f'Time indices:\n{indices[i_dist][i_agg]}')
ind = indices[i_dist][i_agg]
with np.printoptions(precision=1):
    print(f'Densities:\n{density[idx, ind.flatten()].reshape(ind.shape)}')
    print(f'Energies:\n{energy[i_dist][i_agg]}')
    print(f'FWHM of autocorrelation:\n{fwhm[i_dist][i_agg]}')
plt.title(f'sigma={sigmas[idx]:.3g}, maxdist={maxdists[i_dist]:.3g}, aggregate: {path_agg[i_agg]}')
plt.axis('off')
plt.tight_layout()
plt.show()
# %%
i_dist = 4
dpath = os.path.join(root, folder)
fname = f'tree_maxdist{maxdists[i_dist]:.3g}_max.pickle'
fpath = os.path.join(dpath, fname)
with open(fpath, 'rb') as f:
    modes = pickle.load(f)
# %%
idx, m, max_modes = 2, 350, 128
waves, ind = pltaux.get_waves(modes[idx], T, m, max_modes)
grid, n_rows, n_cols = pltaux.wave_matrix(waves)
ind = ind.reshape((n_rows, n_cols))
plt.plot(grid.T, color='#1f77b4')
plt.title(
    f'sigma={sigmas[idx]:.3g}, maxdist={maxdists[i_dist]:.3g}, aggregate: {path_agg[i_agg]}')
plt.axis('off')
plt.tight_layout()
# plt.show()
fpath = os.path.join(IMG_DIR, f'top128_modes_sigma{sigmas[idx]:.3g}_maxdist{maxdists[i_dist]:.3g}.png')
plt.savefig(fpath)
# %%
