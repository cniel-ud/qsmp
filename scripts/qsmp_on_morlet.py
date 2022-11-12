import os
from argparse import ArgumentParser
import numba
import numpy as np
from pathlib import Path

from qsmp.gpu_density import gpu_density
from qsmp.gpu_qsmp import gpu_qsmp
import qsmp.utils.utils as utils
from qsmp.utils import windows

from time import perf_counter
t_start = perf_counter()

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument("--root", help="Path to root folder")
parser.add_argument("--subseq-len", type=int,
                    help="Subsequence (query) length")
parser.add_argument("--sigma", type=float, default=[5],
                    nargs='*', help="Kernel width")
parser.add_argument("--minfilt-size", type=int, help="Length of min-filter")
parser.add_argument('--window-support', type=float, default=0.5,
                    help='Fraction of Gaussian window with 3*sigma')
parser.add_argument('--window-type', default=None)
parser.add_argument('--transform', default=None)

args = parser.parse_args()

root = Path(args.root)
sublen = args.subseq_len
minfilt_size = args.minfilt_size
sigma = args.sigma
win_support = args.window_support
transform = args.transform

data_dir = root.joinpath('data/morlet')
results_dir = root.joinpath('results/morlet')
results_dir.mkdir(exist_ok=True)

device_ids = [device.id for device in numba.cuda.list_devices()]
device_ids = [device_ids[0]]  # Use only one GPU




# (Gaussian/Rect) window to penalize patterns that are not centered
if args.window_type is not None:
    win_fn = windows.get_window(args.window_type)
    win = win_fn(sublen, win_support)
else:
    win = None

T_splice_file = 'morlet_signal.npz'
args2filename = utils.Args2Filename(args)
out_file = args2filename('qsmp')

T_splice_file = data_dir.joinpath(T_splice_file)
with np.load(T_splice_file, allow_pickle=True) as data:
    T = data['T']
    splice = data['splice']

compute_density = True
out_file = results_dir.joinpath(out_file)
if out_file.is_file():
    with np.load(out_file) as data:
        if 'density' in data:
            density = data['density']
            T = data['T']
            splice = data['splice']
            compute_density = False
        else:
            print(f'{out_file} is corrupted.\nDeleting it...')
            os.remove(out_file)

# Compute and save density
if compute_density:
    T, splice, density = gpu_density(
        T, sublen, sigma, root, transform=transform,
        splice=None, window=win, device_id=device_ids
    )
    with out_file.open('wb') as f:
        np.savez(f, density=density, T=T, splice=splice)

# Compute QSMP and indices
profile, indices = gpu_qsmp(
    T, sublen, minfilt_size, density, root, splice=splice, device_id=device_ids
)

# Find global maxima (root), and fix neighbor and profile
profile, indices, density = utils.fix_root((profile, indices, density))

# Save density, QSMP, and indices. np.savez doesn't work in append mode.
#XXX: Save splice?
fpath = results_dir.joinpath(out_file)
with fpath.open('wb') as f:
    np.savez(
        f, density=density, profile=profile, indices=indices,
        T=T, splice=splice
    )

t_stop = perf_counter()
print(f'Finished after {t_stop-t_start} seconds!')
