from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import qsmp.tree as tree
from qsmp.utils import pltaux, utils

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
img_dir = root.joinpath('results', args.experiment, 'img')
img_dir.mkdir(exist_ok=True)
data_dir = root.joinpath('data', args.experiment)
results_dir = root.joinpath('results', args.experiment)

args2filename = utils.Args2Filename(args)
in_fname = args2filename('qsmp')
in_file = results_dir.joinpath(in_fname)
with np.load(in_file) as data:
    density = data['density'].T
    NNdist = data['profile'].T
    NNindex = data['indices'].T

# TODO:
# * Extend to multiple data files (?)
# * Use HDF5 to improve portability
fpath = list(data_dir.glob('*.npz'))[0]
with np.load(fpath, allow_pickle=True) as data:
    T = data['T']
    splice = data['splice']

out_fname = args2filename('report')
out_path = img_dir.joinpath(out_fname)
with PdfPages(out_path) as pdf:
    for i_sigma in range(len(args.sigma)):

        if np.all(NNdist[i_sigma]==0.0):
            continue

        quantiles = np.quantile(NNdist[i_sigma], [0.1, 1])
        quantiles = np.log2(quantiles)
        maxdists = 2 ** np.linspace(*quantiles, 5)

        for max_dist in maxdists:

            NNd, NNi, modes, cluster_size = tree.tree2clusters(
                args.subseq_len, density[i_sigma], NNindex[i_sigma],
                NNdist[i_sigma], max_dist
            )

            if modes.size == 0: # every point points to itself
                continue

            sample, idx = tree.get_neighbors(
                T, args.subseq_len, density[i_sigma], NNi,
                NNd, modes, args.max_modes, args.n_neighbors
            )

            pltaux.wave_subplots(sample, idx, args.n_neighbors)

            plt.suptitle(
                f'maxdist={max_dist:.3g}, sigma={args.sigma[i_sigma]:.3g}')
            plt.axis('off')
            plt.tight_layout()
            pdf.savefig()
            plt.close()
