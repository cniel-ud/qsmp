#%%
import os, sys
import numpy as np
import pltaux
from copy import deepcopy
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import tree
import utils
from spectrum import MultiTapering
import matplotlib.pyplot as plt
from numpy.random import default_rng

#%%
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
m = 350
max_modes = 128
n_densities, n_pnts = density.shape
dpath = os.path.join(root, folder)
modes = utils.load_modes(dpath, maxdists[1], path_agg[0])
modes_i = deepcopy(modes[0])
modes_i = tree.reduce_close_modes(modes_i, m)
waves, ind = utils.get_waves(modes_i, T, m)

# %%
rng = default_rng(13)
idx = rng.choice(ind.size, size=100, replace=False)
idx = np.r_[0, idx[:-1]]
sample = waves[idx]
waves_plt, n_rows, n_cols = pltaux.wave_matrix(sample)
plt.figure(figsize=(11, 8.5))
plt.plot(waves_plt.T, color='#1f77b4')
plt.show()
# %%
NFFT = 1024
Px = np.zeros((100, NFFT//2 + 1))
for i in range(100):
    psd = MultiTapering(sample[i], NFFT=NFFT, NW=3, sampling=512)
    Px[i] = psd.psd
# %%
f = psd.frequencies()
Px_plt, n_rows, n_cols = pltaux.wave_matrix(10*np.log(Px))
plt.figure(figsize=(11, 8.5))
plt.plot(Px_plt.T, color='#1f77b4')
plt.show()
# plt.plot(f, 10*np.log10(Px))
# %%
Px_mean = np.mean(Px, axis=0)
plt.plot(f, 10*np.log10(Px_mean))
plt.ylabel('Power [dB]')
plt.xlabel('Frequency [Hz]')
plt.grid(True)
plt.xlim(0,250)

# %%
psd.plot()
# %%
