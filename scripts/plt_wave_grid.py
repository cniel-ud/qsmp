#%%
from pathlib import Path
import numpy as np
import qsmp.tree as tree
import matplotlib.pyplot as plt
from argparse import ArgumentParser
#%%
# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument('experiment', help='Experiment name')
parser.add_argument("--root", help="Path to root folder", default='..')
parser.add_argument("--subseq-len", type=int, default=350,
                    help="Subsequence (query) length")
parser.add_argument("--sigma", type=float, dest="sigma", default=[5],
                    nargs='*', help="Kernel width")
parser.add_argument('--fwhm', action="store_true", default=False,
                    help="Scale distances by the FWHM of the autocorrelation of the subsequences")
parser.add_argument('--max-modes', type=int, default=10,
                    help='Maximum number of modes to plot/save')
parser.add_argument('--n-neighbors', type=int, default=9,
                    help='Number of neighbors for each mode to plot/save')

args = parser.parse_args([
    "morlet",
    "--root",
    "..",
    "--subseq-len",
    "350",
    "--sigma",
    "1", "2", "4", "8", "16",
    "--fwhm"
])
root = Path(args.root)
img_dir = root.joinpath('results', args.experiment, 'img')
img_dir.mkdir(exist_ok=True)
data_dir = root.joinpath('data', args.experiment)
results_dir = root.joinpath('results', args.experiment)
#%%
max_modes, k = args.max_modes, args.n_neighbors
sublen = args.subseq_len
sigma_str = [str(i) for i in args.sigma]
sigma_str = '_'.join(sigma_str)

# max(,), only on density
if args.fwhm:
    in_fname = f'qsmp_m{sublen}_sigma{sigma_str}_fwhm.npz'
    out_fname = f'{max_modes}modes_{k}neighbors_fwhm.pdf'
else:
    in_fname = f'qsmp_m{sublen}_sigma{sigma_str}.npz'
    out_fname = f'{max_modes}modes_{k}neighbors.pdf'

# TODO:
#   * Extend to multiple data files
#   * Use unique data format
fpath = next(iter(data_dir.iterdir()))
if fpath.suffix == '.npz':
    with np.load(fpath) as data:
        T = data['T']
        splice = data['splice']
elif fpath.suffix == '.txt':
    T = np.loadtxt(fpath)

#%%
fpath = results_dir.joinpath(in_fname)
with np.load(fpath) as data:
    density = data['density'].T
    NNdist = data['profile'].T
    NNindex = data['indices'].T

n_bw, n_subseq = NNindex.shape
# XXX: this can be removed in feature experiments, as is now in
# gpu_qsmp()
NNdist[np.isnan(NNdist)] = 0
#%%
impath = img_dir.joinpath(out_fname)
i_sigma = 0
quantiles = np.quantile(NNdist[i_sigma], [0.5, 0.99])
quantiles = np.log2(quantiles)
maxdists = 2 ** np.linspace(*quantiles, 5)

maxdist = maxdists[1]
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
idx_not_nan = idx[is_not_nan].astype(np.int64)
t = idx_not_nan[:, None] + np.arange(sublen)[None, :]
sample[is_not_nan] = T[t]
#%%
fig, ax = plt.subplots(max_modes, k+1, figsize=(k+1, max_modes))
for i in range(max_modes):
    for j in range(k+1):
        ax[i,j].plot(sample[i*max_modes+j])
        ax[i,j].axis('off')
        if ~np.isnan(idx[i*max_modes+j]):
            ax[i, j].set_title(f'{idx[i*max_modes+j]:.0f}', fontsize=8)
plt.tight_layout()
#%%


# plt.plot(waves_plt.T, color='#1f77b4')
plt.title(
    f'maxdist={maxdist:.3g}, sigma={args.sigma[i_sigma]:.3g}')
plt.axis('off')
plt.tight_layout()
plt.close()
# %%
