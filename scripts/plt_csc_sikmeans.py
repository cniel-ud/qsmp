#%%
import os
import shift_kmeans.shift_kmeans as sikmeans
from numpy.random import default_rng
import numpy as np
import utils
import matplotlib.pyplot as plt
from demo import pltaux
import pickle
from pathlib import Path
from sparse_coding import plot_X_hat_and_PVE
from learn_z import learn_z
#%%
IMG_DIR = "/home/cmendoza/MEGA/Research/Third_Paper/proto/CSC"
root = '/home/cmendoza/Research/QSMP/data/Study019/preictal'
max_modes, sublen, winlen = 8, 350, 512
init = "random-energy"
metric = "cosine"
n_runs = 3
rng = default_rng(13)

fpath = os.path.join(root, 'qsmp_T_splice.npz')
with np.load(fpath) as data:
    T = data['T']
    splice = data['splice']
#%%
seg_len = np.diff(np.r_[0, splice, T.size-1])
n_seg = seg_len.size
n_win = seg_len//winlen
seg_start_arr = np.r_[0, splice]
cum_win = np.cumsum(n_win)
tot_win = cum_win[-1]
win_idx = np.r_[0, cum_win]
X = np.zeros((tot_win, winlen))
for i_seg in range(n_seg):
    win_start = win_idx[i_seg]
    win_end = win_idx[i_seg+1]
    seg_start = seg_start_arr[i_seg]
    seg_end = seg_start_arr[i_seg] + n_win[i_seg]*winlen
    X[win_start:win_end] = utils.splitdata(
        T[seg_start:seg_end], winlen, keep_dims=True)
#%%
fname = f'sikmeans_{max_modes}atoms.npy'
fpath = os.path.join(root, fname)
fpath = Path(fpath)
if fpath.is_file():
    with fpath.open('rb') as f:
        D = np.load(f)
else:
    with fpath.open('wb') as f:
        # Training begins
        D, _, _, _, _, _ = sikmeans.shift_invariant_k_means(
            X, max_modes, sublen, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)
        np.save(f, D)
# %%
waves_plt, n_rows, n_cols = pltaux.wave_matrix(D, ncols=4)
plt.figure(figsize=(11, 8.5))
plt.plot(waves_plt.T, color='#1f77b4')
plt.title('Dictionary with shift-invariant k-means')
plt.axis('off')
plt.tight_layout()
# %%
rng = default_rng(13)
Fs, NFFT = 512, 256
reg = 5  # regularization factor for l1 regularization
params = dict(
    reg=reg,
    n_iter=1,
    solver_z='l-bfgs',
    solver_z_kwargs=dict(factr=1e9),
    random_state=42,
    n_jobs=1,
    verbose=1)

seg_idx = np.r_[0, splice, T.size-1]
seg_len = np.diff(seg_idx)
n_seg = seg_idx.size - 1
for i_seg in range(1):
    X = T[seg_idx[i_seg]:seg_idx[i_seg+1]]
    X = X[None, :]  # 1 trial
    fname = f'csc_sikmeans_{max_modes}atoms_segment{i_seg+1}_reg{reg}.pickle'
    fpath = os.path.join(root, fname)
    fpath = Path(fpath)
    if fpath.is_file():
       with fpath.open('rb') as f:
           pobj, times, z_hat = pickle.load(f)
    else:
        pobj, times, z_hat = learn_z(X, D, **params)
        with fpath.open('wb') as f:
            pickle.dump((pobj, times, z_hat), f, pickle.HIGHEST_PROTOCOL)

    X = X.squeeze()
    title = f'Shift-invariant k-means. Segment #{i_seg+1}.'
    fig = plot_X_hat_and_PVE(
        X, D, z_hat, Fs=Fs, NFFT=NFFT, rng=i_seg, title=title)
    fname = f'sikmeans_pve_{max_modes}atoms_segment{i_seg+1}_reg{reg}.pdf'
    fpath = os.path.join(IMG_DIR, fname)
    fig.savefig(fpath, bbox_inches='tight')
# %%
