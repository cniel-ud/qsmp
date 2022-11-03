#%%
import os
from time import perf_counter
import numpy as np
from csc_utils import construct_X
import tree
import utils
from learn_z import learn_z
from numpy.random import default_rng
from sparse_coding import compute_PVE
from argparse import ArgumentParser
#%%
parser = ArgumentParser()
parser.add_argument('--root', help='Path to data folder')
parser.add_argument('--max-modes', type=int,
                    help='Dictionary size. Pick the `max_modes` most dense to build the the dictionary', default=8)
parser.add_argument('--sublen', type=int, help='Subsequence length')
parser.add_argument('--winlen', type=int, help='Lenght of analysis window')
parser.add_argument('--maxdist', type=int,
                    help='Distance threshold in Quick Shift')
parser.add_argument(
    '--fwhm', help='If true, use FWHM scaling.', action='store_true')
#%% Load time series and splice
args = parser.parse_args()
snr = np.r_[-4., -2., 0., 2., 4.]
var_noise = 10 ** (-snr/10)
th = 0.1
bandwidths = (9 * var_noise) / np.log(1/th)
sigmas = np.sqrt(bandwidths/2)
sigmas = sigmas[0]

snr_str = [str(i) for i in snr]
snr_str = '_'.join(snr_str)
fts = 'qsmp_T_splice.npz'
if args.fwhm:
    fqs = f'qsmp_m{args.sublen}_snr{snr_str}_fwhm.npz'
else:
    fqs = f'qsmp_m{args.sublen}_snr{snr_str}.npz'


fpath = os.path.join(args.root, fts)
with np.load(fpath) as data:
    T = data['T']
    splice = data['splice']

fpath = os.path.join(args.root, fqs)
with np.load(fpath) as data:
    density = data['density'][:,0]
    NNdist = data['profile'][:,0]
    NNindex = data['indices'][:,0]

NNi, NNd = tree.cut_tree(NNindex, NNdist, args.maxdist)
#%% Assign each subsequence to its root: clustering
NNi = tree.mark_with_root(NNi)
# %% Merge roots that are less than m/4 apart
# The densest root wins. Ignore orphan roots.
NNi, winning_modes, tree_size = tree.merge_roots(NNi, density, args.sublen/4)
winning_modes = winning_modes[:args.max_modes]
D = utils.get_waves(winning_modes, T, args.sublen) #the dictionary
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
    print(f'Average PVE: {np.mean(pve[i_reg])}. Average sparsity: {np.mean(l0norm[i_reg])/z_hat.size}')

fname = f'pve_sparsity_QS_{args.max_modes}atoms.npz'
fpath = os.path.join(args.root, fname)
with open(fpath, 'wb') as f:
    np.savez(f, pve=pve, l0norm=l0norm)