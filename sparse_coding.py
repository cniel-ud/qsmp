#%%
import os
import matplotlib
import numpy as np
import tree
import matplotlib.pyplot as plt
import utils
from demo import pltaux
from learn_z import learn_z
from csc_utils import construct_X
from scipy import signal
import pickle
from numpy.random import default_rng
#%%
def annotate_axes(ax, text, fontsize=18):
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="darkgrey")
#%% Load time series and splice
IMG_DIR = "/home/cmendoza/MEGA/Research/Third_Paper/proto"
root = '/home/cmendoza/Research/QSMP/data/Study019/preictal'
maxdist = 12
max_modes, k = 8, 9
sublen = 350
snr = np.r_[-4., -2., 0., 2., 4.]
var_noise = 10 ** (-snr/10)
th = 0.1
bandwidths = (9 * var_noise) / np.log(1/th)
sigmas = np.sqrt(bandwidths/2)
sigmas = sigmas[0]

snr_str = [str(i) for i in snr]
snr_str = '_'.join(snr_str)
fnames = {
    'time series': 'qsmp_T_splice.npz',
    'Qtuple fwhm': f'qsmp_m{sublen}_snr{snr_str}_fwhm_max_density_only.npz',
    'Qtuple': f'qsmp_m{sublen}_snr{snr_str}.npz',
}

fpath = os.path.join(root, fnames['time series'])
with np.load(fpath) as data:
    T = data['T']
    splice = data['splice']

fpath = os.path.join(root, fnames['Qtuple fwhm'])
with np.load(fpath) as data:
    density = data['density'][:,0]
    NNdist = data['profile'][:,0]
    NNindex = data['indices'][:,0]

NNi, NNd = tree.cut_tree(NNindex, NNdist, maxdist)
#%% Assign each subsequence to its root: clustering
NNi = tree.mark_with_root(NNi)
# %% Merge roots that are less than m/4 apart
# The densest root wins. Ignore orphan roots.
NNi, winning_modes, tree_size = tree.merge_roots(NNi, density, sublen/4)
winning_modes = winning_modes[:max_modes]
D = utils.get_waves(winning_modes, T, sublen) #the dictionary
waves_plt, n_rows, n_cols = pltaux.wave_matrix(D, ncols=4)
#%%
plt.figure(figsize=(11, 8.5))
plt.plot(waves_plt.T, color='#1f77b4')
plt.title(f'maxdist={maxdist:.3g}, sigma={sigmas:.3g}')
plt.axis('off')
plt.tight_layout()
# %%
# %matplotlib inline
rng = default_rng(13)
Fs, NFFT = 512, 256
reg = 5 # regularization factor for l1 regularization
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
for i_seg in range(n_seg):
    X = T[seg_idx[i_seg]:seg_idx[i_seg+1]]  # first segment
    X = X[None, :]  # 1 trial
    pobj, times, z_hat = learn_z(X, D, **params)
    X = X.squeeze()
    fname = f'csc_fwhm_{max_modes}atoms_segment{i_seg+1}_reg{reg}.pickle'
    fpath = os.path.join(root, fname)
    with open(fpath, 'wb') as f:
        pickle.dump((pobj, times, z_hat), f, pickle.HIGHEST_PROTOCOL)
    
    f, Px = signal.welch(X, fs=Fs, window='hamming', detrend=False,
                         nperseg=Fs//4, nfft=NFFT, noverlap=Fs//8)

    fig = plt.figure(figsize=(10, 15), constrained_layout=False)
    fig.suptitle(f'Segment #{i_seg}.')
    gs = fig.add_gridspec(2*(max_modes+1), 5, hspace=0, wspace=0.5)
    t_atom = np.arange(sublen)/Fs
    rand_start = rng.choice(seg_len[i_seg]-Fs, 1)
    idx_ts = np.arange(rand_start, rand_start+Fs) # one second long
    t = idx_ts/Fs
    xy = (0.03, 0.05)
    for i in range(9):
        if i == 8:
            X_hat = construct_X(z_hat, D)
        else:
            z_i = np.expand_dims(z_hat[i], 0)
            d_i = np.expand_dims(D[i], 0)
            X_hat = construct_X(z_i, d_i)

        X_hat = X_hat.squeeze()
        res = X - X_hat        
        _, Pres = signal.welch(res, fs=512, window='hamming', detrend=False,
                                  nperseg=128, nfft=NFFT, noverlap=64)

        atom = fig.add_subplot(gs[2*i,0])    
        if i == 8:
            annotate_axes(atom, 'All atoms')
            atom.axis('off')
        else:
            atom.plot(t_atom, D[i], color='#1f77b4')
            atom.set_title(f'Atom #{i+1}')
            atom.set_xlim(t_atom[0], t_atom[-1])        
        ts = fig.add_subplot(gs[2*i:2*(i+1), 1:3])
        ts.plot(t, X[idx_ts])
        ts.plot(t, X_hat[idx_ts])
        ts.set_xlim(t[0], t[-1])
        PVE = 1 - Pres/Px
        pve = fig.add_subplot(gs[2*i:2*(i+1), 3:])
        pve.plot(f, PVE)
        pve.set_xscale('log', base=2)
        pve.set_xlim(None, 125)    
        upto = f <= 125
        yticks = [PVE[upto].min(), PVE[upto].mean(), PVE[upto].max()]
        if np.all(np.diff(yticks) < 0.01):
            yticks = [PVE[upto].mean()]
        yticks_labels = [f'{t:.2f}' for t in yticks]
        if yticks[0] < 1e-10:        
            yticks[0] = 0
            yticks_labels[0] = '0'    
        pve.set_yticks(yticks)
        pve.set_yticklabels(yticks_labels)    
        PVE_val = 1 - np.sum(res**2)/np.sum(X**2)
        pve.annotate(f'PVE={PVE_val:.2f}', xy=xy,
                     xytext=xy, xycoords='axes fraction')

        if i == 0:        
            ts.set_title('Original signal vs reconstruction')
            ts.legend(['True', 'Reconstructed'])
            pve.set_title('PVE')
        if i == 7:
            atom.set_xlabel('Time [sec]')
        if i == 8:        
            ts.set_xlabel('Time [sec]')
            pve.set_xlabel('Frequency [Hz]')
            pve.set_xticks([3, 12, 30, 70, 125])
            pve.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        if i < 8:            
            atom.set_xticks([])
            ts.set_xticks([])
            pve.set_xticks([])    

    plt.savefig(
        f'/home/cmendoza/MEGA/Research/Third_Paper/proto/CSC/pve_{max_modes}atoms_fwhm_segment{i_seg}_reg{reg}.pdf', bbox_inches='tight')
    plt.show()
    # %%
