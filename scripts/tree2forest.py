#%%
from pathlib import Path
import numpy as np
import qsmp.tree as tree
import matplotlib.pyplot as plt
from qsmp.utils import pltaux
from matplotlib.backends.backend_pdf import PdfPages
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
parser.add_argument('--max-modes', type=int, default=10,
                    help='Maximum number of modes to plot/save')
parser.add_argument('--n-neighbors', type=int, default=9,
                    help='Number of neighbors for each mode to plot/save')
parser.add_argument('--window-support', type=float, default=0.5,
                    help='Fraction of Gaussian window with ±3*sigma')
parser.add_argument("--minfilt-size", type=int, help="Length of min-filter")
parser.add_argument('--window-type', default=None)
parser.add_argument('--transform', default=None)

args = parser.parse_args()
root = Path(args.root)
transform = args.transform
img_dir = root.joinpath('results', args.experiment, 'img')
img_dir.mkdir(exist_ok=True)
data_dir = root.joinpath('data', args.experiment)
results_dir = root.joinpath('results', args.experiment)

max_modes, k = args.max_modes, args.n_neighbors
sublen = args.subseq_len
minfilt_size = args.minfilt_size
sigma_str = [str(i) for i in args.sigma]
sigma_str = '_'.join(sigma_str)

if args.window_type is not None:
    win_str = f'_{args.window_type}-{int(100*args.window_support)}'
else:
    win_str = ''

# max(,), only on density
if transform is not None:
    tr_str = f'_{transform}'
else:
    tr_str = ''

# fnames['Qtuple'] = f'qsmp_m{sublen}_sigma{sigma_str}{win_str}{tr_str}.npz'
in_fname = (
    f'qsmp_m{sublen}_sigma{sigma_str}{win_str}'
    f'{tr_str}minfilt-{minfilt_size}.npz'
)
out_fname = (
    f'{max_modes}modes_{k}neighbors{win_str}'
    f'{tr_str}minfilt-{minfilt_size}.pdf'
)

fpath = results_dir.joinpath(in_fname)
with np.load(fpath) as data:
    density = data['density'].T
    NNdist = data['profile'].T
    NNindex = data['indices'].T
    T = data['T']
    splice = data['splice']

if transform == 'whiten':
    # TODO:
    #   * Extend to multiple data files
    #   * Use unique data format
    fpath = next(iter(data_dir.iterdir()))
    if fpath.suffix == '.npz':
        with np.load(fpath) as data:
            T_orig = data['T']      #original T, whitout whitening
            splice_orig = data['splice']
    elif fpath.suffix == '.txt':
        T_orig = np.loadtxt(fpath)
else:
    T_orig = T

n_bw, n_subseq = NNindex.shape
# XXX: this can be removed in feature experiments, as is now in
# gpu_qsmp()
NNdist[np.isnan(NNdist)] = 0
#%%
impath = img_dir.joinpath(out_fname)
with PdfPages(impath) as pdf:
    for i_sigma in range(len(args.sigma)):
        if np.all(NNdist[i_sigma]==0.0):
            continue
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

            #%%
            idx = tree.k_neighborhood(winning_modes[:max_modes],
                                      k, NNd, NNi, density[i_sigma], sublen/4)

            #%%
            sample = np.full((idx.size, sublen), fill_value=np.nan)
            is_not_nan = ~np.isnan(idx)
            idx_not_nan = idx[is_not_nan].astype(np.int64)
            t = idx_not_nan[:, None] + np.arange(sublen)[None, :]
            sample[is_not_nan] = T_orig[t]

            pltaux.wave_subplots(sample, idx, k)

            plt.suptitle(
                f'maxdist={maxdist:.3g}, sigma={args.sigma[i_sigma]:.3g}')
            plt.axis('off')
            plt.tight_layout()
            pdf.savefig()
            plt.close()
# %%
