#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
# %%
DPATH = "/home/cmendoza/Research/QSMP/data/Study019/preictal"
fname = 'qsmp_m350_snr0.0_2.0_4.0_8.0_10.0.npz'
fpath = os.path.join(DPATH, fname)

with np.load(fpath) as data:
    full_density = data['density']
    full_profile = data['profile']
    indices = data['indices']
# %%
snr = np.r_[0, 2, 4, 8, 10]
var_noise = 10 ** (-snr/10)
m = 350
ibw = 2
n_subseq = indices.shape[0]
#%%
inan = np.asarray(np.isnan(full_profile[:, ibw])).nonzero()[0]
is_mode = np.isinf(full_profile[:, ibw])
iinf = np.asarray(is_mode).nonzero()[0]

#%% Ignore subsequences at splice
full_profile[inan, ibw] = np.inf
# %%
# If QSMP == inf, we hit a mode: the point doesn't move, its nearest neighbor
# is itself.
imax = np.argmax(full_density[:, ibw])
assert imax in iinf
indices[iinf, ibw] = iinf
full_profile[iinf, ibw] = 0

#%% Show density and QMSP
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].plot(full_density[:, ibw])
ax[0].set_title('Density')
ax[1].plot(full_profile[:, ibw])
ax[1].set_title('QSMP')
plt.suptitle('Preictal, Study019, SNR = 0 dB')

#%% An example of redundancy in index profile
if ibw == 0:
    x = range(108, 128)
    imax = np.argmax(full_density[indices[x, ibw], ibw])
    aux_ind = copy.deepcopy(indices[x, ibw])
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax[0].stem(x, aux_ind-aux_ind[0])
    mk, _, _ = ax[0].stem(x[imax], aux_ind[imax]-aux_ind[0], 'r')
    mk.set_color('red')
    ax[0].set_title('(Offset) Index profile')
    ax[1].plot(aux_ind, full_density[aux_ind, ibw])
    ax[1].scatter(aux_ind[imax], full_density[aux_ind[imax], ibw], color='red')
    ax[1].set_title('Density')
    ax[2].plot(x, full_profile[x, ibw])
    ax[2].scatter(x[imax], full_profile[x[imax], ibw], color='red')
    ax[2].set_title('QSMP')

#%% Ignore subsequences whose distance to the nearest neighbor is bigger than
# 3*(standard dev. of noise).
dist_th = 3 * np.sqrt(var_noise[ibw])
keep = full_profile[:, ibw] <= dist_th
subseq = np.asarray(keep).nonzero()[0]
neighbor = copy.copy(indices[subseq, ibw])
profile = copy.copy(full_profile[subseq, ibw])
density = copy.copy(full_density[neighbor, ibw])

#%% Find consecutive neighbor indices that are increasing by one
diff_ind = np.abs(np.diff(neighbor))
bins = np.asarray(diff_ind > 1).nonzero()[0] + 1
bins = np.r_[0, bins, n_subseq]
nbins = bins.size - 1
#%% Pick subsequences with max-density neighbor
idx = np.zeros(nbins, dtype=np.int64)
for i in range(nbins):
    aux_neigh = neighbor[bins[i]:bins[i+1]]
    imax = np.argmax(density[aux_neigh])
    idx[i] = bins[i] + imax

subseq = subseq[idx]
neighbor = neighbor[idx]
profile = profile[idx]
density = density[idx]

#%% Sort by descending order of density
isort = np.argsort(-density)
subseq = subseq[isort]
neighbor = neighbor[isort]
profile = profile[isort]
density = density[isort]

#%% Put the results in a data frame for easy visualization
data = {
    'Subsequence': subseq,
    'Neighbor': neighbor,
    'profile': profile,
    'density': density,
}
df = pd.DataFrame(data=data)
fname = 'preictal_Study019_pairings.csv'
dpath = '/home/cmendoza/MEGA/Research/Third_Paper/proto/'
fpath = os.path.join(dpath, fname)
df.to_csv(fpath)

#%% Second pass: for subsequences with nearest-neighbors withing ±m/2 of local
# maxima, move them to that local maxima.
n_picked = subseq.size
cnt = 0
gap = int(np.ceil(m/2))
aux_density = copy.deepcopy(density)
while cnt < n_picked:
    imax = np.argmax(aux_density)
    is_local = np.abs(neighbor - neighbor[imax]) <= gap
    neighbor[is_local] = neighbor[imax]
    density[is_local] = density[imax]
    aux_density[is_local] = -np.inf
    cnt += np.count_nonzero(is_local)

#%% First three maxima
uniq_neigh, idx, inv_idx, counts = np.unique(
    neighbor, return_index=True, return_inverse=True, return_counts=True)

uniq_density = density[idx]
isort = np.argsort(-uniq_density)
imax = uniq_neigh[isort]

fig, ax = plt.subplots(1, 3, figsize=(20, 4))
str_max = ['first', 'second', 'third']
for i in range(3):
    x = range(imax[i]-gap, imax[i]+gap)
    ax[i].plot(x, full_density[x, ibw])
    ax[i].set_title(f'{str_max[i]} maxima')
plt.suptitle(f'Density at SNR = 0 dB, {m}-points around local maxima. '
             'Preictal, Study019.')

#%% Put the results in a data frame for easy visualization
data = {
    'Subsequence': subseq,
    'Neighbor': neighbor,
    'profile': profile,
    'density': density,
}
df = pd.DataFrame(data=data)
fname = 'preictal_Study019_pairings_second_pass.csv'
dpath = '/home/cmendoza/MEGA/Research/Third_Paper/proto/'
fpath = os.path.join(dpath, fname)
df.to_csv(fpath)
