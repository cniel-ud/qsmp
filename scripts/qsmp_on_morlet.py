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
from ipdb import set_trace


t_start = perf_counter()

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument("--root", help="Path to root folder")
parser.add_argument("--subseq-len", type=int,
                    help="Subsequence (query) length")
parser.add_argument("--sigma", type=float, dest="sigma", default=[5],
                    nargs='*', help="Kernel width")
parser.add_argument('--window-support', type=float, default=0.5,
                    help='Fraction of Gaussian window with ±3*sigma')
parser.add_argument('--window-type', default=None)
parser.add_argument('--transform', default=None)

args = parser.parse_args()

root = Path(args.root)
sublen = args.subseq_len
sigma = args.sigma
win_support = args.window_support
transform = args.transform

data_dir = root.joinpath('data/morlet')
results_dir = root.joinpath('results/morlet')
results_dir.mkdir(exist_ok=True)

device_ids = [device.id for device in numba.cuda.list_devices()]

# The Gaussian kernel is of the form
#   f(x) = exp(-x^2/(2*sigma^2))
#   with `sigma` being the bandwidth parameter
sigma = np.array(sigma)
sigma_str = [str(i) for i in sigma]
sigma_str = '_'.join(sigma_str)


# (Gaussian/Rect) window to penalize patterns that are not centered
if args.window_type is not None:
    win_fn = windows.get_window(args.window_type)
    win = win_fn(sublen, win_support)
    win_str = f'_{args.window_type}-{int(100*win_support)}'
else:
    win = None
    win_str = ''

#XXX: we are currently taking the first segments whose cumulative length
# is >= args.train. This parameter is NOT currently reflected in the naming
# of the output files.
fnames = {
    'time series': 'morlet_signal_fs-512.txt'
}

if transform is not None:
    tr_str = f'_{transform}'
else:
    tr_str = ''
fnames['Qtuple'] = f'qsmp_m{sublen}_sigma{sigma_str}{win_str}{tr_str}.npz'


fpath = data_dir.joinpath(fnames['time series'])
T = np.loadtxt(fpath)

compute_density = True
fpath = results_dir.joinpath(fnames['Qtuple'])
if fpath.is_file():
    with np.load(fpath) as data:        
        if 'density' in data:
            density = data['density']
            T = data['T']
            splice = data['splice']
            compute_density = False
        else:
            print(f'{fpath} is corrupted.\nDeleting it...')
            os.remove(fpath)

# Compute and save density
if compute_density:
    T, splice, density = gpu_density(
        T, sublen, sigma, root, transform=transform,
        splice=None, window=win, device_id=device_ids)
    fpath = results_dir.joinpath(fnames['Qtuple'])
    with fpath.open('wb') as f:
        np.savez(f, density=density, T=T, splice=splice)

# Compute QSMP and indices
profile, indices = gpu_qsmp(T, sublen, density, root, transform=transform,
                            splice=splice, device_id=device_ids)

# Find global maxima (root), and fix neighbor and profile
profile, indices, density = utils.fix_root((profile, indices, density))

# Save density, QSMP, and indices. np.savez doesn't work in append mode.
#XXX: Save splice?
fpath = results_dir.joinpath(fnames['Qtuple'])
with fpath.open('wb') as f:
    np.savez(
        f, density=density, profile=profile, indices=indices, 
        T=T, splice=splice
    )

t_stop = perf_counter()
print(f'Finished after {t_stop-t_start} seconds!')