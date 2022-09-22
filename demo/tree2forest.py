#%%
import os
import numpy as np
import tree
import matplotlib.pyplot as plt
from demo import pltaux
from matplotlib.backends.backend_pdf import PdfPages
#%% Load time series and splice
IMG_DIR = "/home/cmendoza/MEGA/Research/Third_Paper/proto"
root = '/home/cmendoza/Research/QSMP/data/Study019/preictal'
max_modes, k = 10, 9
sublen = 350
sigmas = np.r_[4, 3.6, 3.2, 2.8, 2.25]
sigma_str = [str(i) for i in sigmas]
sigma_str = '_'.join(sigma_str)

fts = 'qsmp_T_splice.npz'
fqs = f'qsmp_m{sublen}_sigma{sigma_str}'
fout = f'{max_modes}modes_{k}neighbors'

fwhm = True  # max(,), only on density
if fwhm:
    fqs = fqs + '_fwhm'
    fout = fout + '_fwhm'

fqs = fqs + '.npz'
fout = fout + '.pdf'
    
fpath = os.path.join(root, fts)
with np.load(fpath) as data:
    T = data['T']
    splice = data['splice']

#%%
fpath = os.path.join(root, fqs)
with np.load(fpath) as data:
    density = data['density'].T
    NNdist = data['profile'].T
    NNindex = data['indices'].T

n_bw, n_subseq = NNindex.shape
# XXX: this can be removed in feature experiments, as is now in 
# gpu_qsmp()
NNdist[np.isnan(NNdist)] = 0
#%%
impath = os.path.join(IMG_DIR, fout)
with PdfPages(impath) as pdf:
    for i_sigma in range(sigmas.size):
        quantiles = np.quantile(NNdist[i_sigma], [0.5, 0.99])
        quantiles = np.log2(quantiles)
        maxdists = 2 ** np.linspace(*quantiles, 5)        

        for maxdist in maxdists:
            NNi, NNd = tree.cut_tree(
                NNindex[i_sigma], NNdist[i_sigma], maxdist)

            #%% Assign each subsequence to its root: clustering
            NNi = tree.mark_with_root(NNi)
            # %% Merge roots that are less than m/4 apart
            # The densest root wins. Ignore orphan roots.
            NNi, winning_modes, tree_size = tree.merge_roots(
                NNi, density[i_sigma], sublen/4)

            NNd = tree.recompute_distances(NNi, T, sublen)

            #%%            
            idx = tree.k_neighborhood(winning_modes[:max_modes],
                                      k, NNd, NNi, density[i_sigma], sublen/4)

            #%%
            sample = np.full((idx.size, sublen), fill_value=np.nan)
            is_not_nan = ~np.isnan(idx)
            idx = idx[is_not_nan].astype(np.int64)
            t = idx[:, None] + np.arange(sublen)[None, :]
            sample[is_not_nan] = T[t]
            waves_plt, n_rows, n_cols = pltaux.wave_matrix(sample, ncols=k+1)
    
            plt.figure(figsize=(11, 8.5))
            plt.plot(waves_plt.T, color='#1f77b4')
            plt.title(f'maxdist={maxdist:.3g}, sigma={sigmas[i_sigma]:.3g}')
            plt.axis('off')
            plt.tight_layout()
            pdf.savefig()
            plt.close()
# %%
