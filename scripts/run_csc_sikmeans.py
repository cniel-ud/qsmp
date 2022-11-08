#%%
import os
import shift_kmeans.shift_kmeans as sikmeans
from numpy.random import default_rng
import numpy as np
import utils
from pathlib import Path
from sparse_coding import compute_PVE
from learn_z import learn_z
from csc_utils import construct_X
from argparse import ArgumentParser
from time import perf_counter
#%%
parser = ArgumentParser()
parser.add_argument('--root', help='Path to data folder')
parser.add_argument('--max-modes', type=int,
                    help='Dictionary size. Pick the `max_modes` most dense to build the the dictionary', default=8)
parser.add_argument('--sublen', type=int, help='Subsequence length')
parser.add_argument('--winlen', type=int, help='Lenght of analysis window')

args = parser.parse_args()
init = "random-energy"
metric = "cosine"
n_runs = 3
rng = default_rng(13)

fpath = os.path.join(args.root, 'qsmp_T_splice.npz')
with np.load(fpath) as data:
    T = data['T']
    splice = data['splice']
#%%
seg_len = np.diff(np.r_[0, splice, T.size-1])
n_seg = seg_len.size
n_win = seg_len//args.winlen
seg_start_arr = np.r_[0, splice]
cum_win = np.cumsum(n_win)
tot_win = cum_win[-1]
win_idx = np.r_[0, cum_win]
X = np.zeros((tot_win, args.winlen))
for i_seg in range(n_seg):
    win_start = win_idx[i_seg]
    win_end = win_idx[i_seg+1]
    seg_start = seg_start_arr[i_seg]
    seg_end = seg_start_arr[i_seg] + n_win[i_seg]*args.winlen
    X[win_start:win_end] = utils.splitdata(
        T[seg_start:seg_end], args.winlen, keep_dims=True)
#%%
fname = f'sikmeans_m{args.sublen}_{args.max_modes}atoms.npy'
fpath = os.path.join(args.root, fname)
fpath = Path(fpath)
if fpath.is_file():
    with fpath.open('rb') as f:
        D = np.load(f)
else:
    with fpath.open('wb') as f:
        # Training begins
        D, _, _, _, _, _ = sikmeans.shift_invariant_k_means(
            X, args.max_modes, args.sublen, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)
        np.save(f, D)
# %%
reg = np.logspace(0, np.log10(1e6), num=10, base=10)
n_reg, n_win = reg.size, 50
pve = np.zeros((n_reg, n_win))
l0norm = np.zeros((n_reg, n_win))
Fs, NFFT = 512, 256
start_idx = np.r_[0, splice]
seg_len = np.diff(np.r_[0, splice, T.size-1])
n_seg = start_idx.size
for i_reg in range(reg.size):
    t_start = perf_counter()
    rng = default_rng(13)
    params = dict(
        reg=reg[i_reg],
        n_iter=1,
        solver_z='l-bfgs',
        solver_z_kwargs=dict(factr=1e9),
        random_state=42,
        n_jobs=1,
        verbose=1)

    for i_win in range(n_win):
        i_seg = rng.choice(n_seg, 1)[0]

        if seg_len[i_seg] < args.winlen:
            i_start = start_idx[i_seg]
            i_stop = i_start + seg_len[i_seg]
        else:
            i_start = rng.choice(seg_len[i_seg]-args.winlen, 1)[0]
            i_start = i_start + start_idx[i_seg]
            i_stop = i_start + args.winlen

        X = T[i_start:i_stop]
        X = X[None, :]  # 1 trial
        pobj, times, z_hat = learn_z(X, D, **params)
        l0norm[i_reg, i_win] = np.count_nonzero(z_hat)
        X_hat = construct_X(z_hat, D)
        _, _, pve[i_reg, i_win] = compute_PVE(X, X_hat, Fs, NFFT=NFFT)
    t_stop = perf_counter()
    print(
        f'Regularization: {reg[i_reg]}. Time elapsed: {t_stop-t_start:.2f} seconds.')
    print(
        f'Average PVE: {np.mean(pve[i_reg])}. Average sparsity: {np.mean(l0norm[i_reg])/z_hat.size}')

fname = f'pve_sparsity_QS_{args.max_modes}atoms.npz'
fpath = os.path.join(args.root, fname)
with open(fpath, 'wb') as f:
    np.savez(f, pve=pve, l0norm=l0norm)
