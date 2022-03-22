#%%
import os
import numpy as np
import tree
import matplotlib.pyplot as plt
import utils
import core
from demo import pltaux
from matplotlib.backends.backend_pdf import PdfPages
#%% Load time series and splice
IMG_DIR = "/home/cmendoza/MEGA/Research/Third_Paper/proto"
root = '/home/cmendoza/Research/QSMP/data/Study019/preictal'
fname = 'first56_segments_CSP1.npz'
fpath = os.path.join(root, fname)
with np.load(fpath) as data:
    T = data['time_series']
    splice = data['splice']

#%% Whiten the time series and update the splice
# TODO: should we save this data once if doesn't exist when running a new
# experiment?
fs, n_taps = 512, 1001
f, Px_mean = core.mean_PSD(T, splice)
_, coeffs = core.whitening_filter(
    f, Px_mean, n_taps=n_taps, fs=fs)
grp_delay = core.get_group_delay(coeffs, f, fs=fs)
filt_T, filt_splice = core.whiten(T, splice, coeffs, grp_delay)
end_seg = np.r_[filt_splice, filt_T.size]  # end index of each segment

#%%
in_file = 'qsmp_m350_snr-4.0_-2.0_0.0_2.0_4.0_whiten.npz'
outdir = 'qsmp_whiten'
# outdir = os.path.join(root, outdir)
# Path(outdir).mkdir(parents=True, exist_ok=True)
fpath = os.path.join(root, in_file)
with np.load(fpath) as data:
    density = data['density'].T
    NNdist = data['profile'].T
    NNindex = data['indices'].T

snr = np.r_[-4, -2, 0, 2, 4]
var_noise = 10 ** (-snr/10)
th = 0.1
bandwidths = (9 * var_noise) / np.log(1/th)
sigmas = np.sqrt(bandwidths/2)
m = 350

nonan_profile = NNdist[~np.isnan(NNdist)]
quantiles = np.quantile(nonan_profile, [0.75, 0.99])
quantiles = np.log2(quantiles)
maxdists = 2 ** np.linspace(*quantiles, 5)

n_bw, n_subseq = NNindex.shape
#%% Remove parents with distance > max_dist
impath = os.path.join(IMG_DIR, '5modes_4neighbors_whiten.pdf')
with PdfPages(impath) as pdf:
    for maxdist in [maxdists[0]]:
        for i_sigma in range(sigmas.size):

            NNi, NNd = tree.cut_tree(
                NNindex[i_sigma], NNdist[i_sigma], maxdist)

            # this can be removed in next experiments, as it is now included in gpu_qsmp.py:
            in_splice = np.isnan(NNd)
            NNi[in_splice] = np.asarray(in_splice).nonzero()[0]

            #%% Assign each subsequence to its root: clustering
            NNi = tree.mark_with_root(NNi)
            # %% Merge roots that are less than m/4 apart
            # The densest root wins. Ignore orphan roots.
            NNi, winning_modes, tree_size = tree.merge_roots(
                NNi, density[i_sigma], m/4)

            #%% on each cluster, recompute distances from nodes to root
            NNd = tree.recompute_distances(NNi, filt_T, m)

            #%%
            max_modes, k = 5, 4
            idx = tree.k_neighborhood(winning_modes[:max_modes],
                                      k, NNd, NNi, density[i_sigma], m/4)

            #%%
            ind = utils.phase_correction(
                idx, end_seg, grp_delay, direction='backward')
            t = ind[:, None] + np.arange(m)[None, :]
            sample = T[t]
            waves_plt, n_rows, n_cols = pltaux.wave_matrix(sample, ncols=5)
    
            plt.figure(figsize=(11, 8.5))
            plt.plot(waves_plt.T, color='#1f77b4')
            plt.title(f'maxdist={maxdist:.3g}, sigma={sigmas[i_sigma]:.3g}')
            plt.axis('off')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

#%%
#TODO: Create functions to
# - Find k nodes in a cluster with smallest distance that are not temporally 
# close. In the future, we migth recompute the distances...
#    > are the original distances a good proxy for the recomputed distances?? 
# - Build matrix with roots and k closest subsequences. Keep track of distances 
# and time indices!