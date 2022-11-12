import os
from argparse import ArgumentParser
from pathlib import Path
import numba
import numpy as np

from qsmp.gpu_density import gpu_density
from qsmp.gpu_qsmp import gpu_qsmp
from qsmp.utils import windows
from qsmp.utils import utils
from qsmp import core

from time import perf_counter

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

if __name__ == '__main__':

    t_start = perf_counter()

    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("-d", "--data-path", dest="dpath",
                        help="Path to folder with time series")
    parser.add_argument("-w", "--W-path", dest="wpath",
                        help="Path to matrix W with spatial filters")
    parser.add_argument("--subseq-len", type=int,
                        help="Subsequence (query) length")
    parser.add_argument("--sigma", type=float, dest="sigma", default=[1],
                        nargs='*', help="SNR in dB")
    parser.add_argument("-t", "--train", type=int, dest="train_len", default=0,
                        help="Number of time points for training")
    parser.add_argument("--minfilt-size", type=int,
                        help="Length of min-filter")
    parser.add_argument('--window-support', type=float, default=0.5,
                        help='Fraction of Gaussian window with 3*sigma')
    parser.add_argument('--window-type', default=None)
    parser.add_argument('--transform', default=None)

    args = parser.parse_args()

    dpath = Path(args.dpath)
    wpath = Path(args.wpath)
    sublen = args.subseq_len
    minfilt_size = args.minfilt_size
    win_support = args.window_support
    sigma = args.sigma
    train_len = args.train_len
    transform = args.transform

    device_ids = [device.id for device in numba.cuda.list_devices()]

    # The Gaussian kernel is of the form
    #   f(x) = exp(-x^2/(2*sigma^2))
    #   with `sigma` being the bandwidth parameter
    sigma = np.array(sigma)

    #XXX: we are currently taking the first segments whose cumulative length
    # is >= args.train. This parameter is NOT currently reflected in the naming
    # of the output files.
    T_splice_file = 'qsmp_T_splice.npz'
    # (Gaussian/Rect) window to penalize patterns that are not centered
    if args.window_type is not None:
        win_fn = windows.get_window(args.window_type)
        win = win_fn(sublen, win_support)
    else:
        win = None

    # No need to include 'preictal ('interictal'), as they are in separate
    # folders (dpath)
    args2filename = utils.Args2Filename(args)
    out_file = args2filename('qsmp')
    params_str = args2filename.base_name

    #XXX: make a more generic (parametrized) name
    if transform == 'whiten':
        whitened_T_splice_file = 'qsmp_T_splice_whitened.npz'
        whitened_T_splice_file = dpath.joinpath(whitened_T_splice_file)

    get_data = True
    T_splice_file = dpath.joinpath(T_splice_file)
    if T_splice_file.is_file():
        with np.load(T_splice_file) as data:
            if 'T' in data:
                T = data['T']
                splice = data['splice']
                get_data = False
            else:
                print(f'{str(T_splice_file)} is corrupted.\nDeleting it...')
                T_splice_file.unlink()

    if get_data:
        #%% Get the CSP filters
        W = utils.loadmat73(wpath, 'W')
        # Pick first(last) CSP filter for preictal(interictal)
        n_csp = W.shape[1]
        if 'preictal' in dpath.parts:
            i_csp = 0
        elif 'interictal' in dpath.parts:
            i_csp = n_csp - 1
        else:
            raise ValueError(
                f"The path '{str(dpath)}' doesn't contain neither 'preictal' nor 'interictal'")
        W = W[:, i_csp]
        T, splice, t_start, t_end, seiz_id = utils.cat_segments(
            dpath, W, train_len=train_len)
        with T_splice_file.open('wb') as f:
            np.savez(f, T=T, splice=splice, t_start=t_start,
                        t_end=t_end, seiz_id=seiz_id)

    compute_density = True
    out_file = dpath.joinpath(out_file)
    if out_file.is_file():
        with np.load(out_file) as data:
            if 'density' in data:
                density = data['density']
                compute_density = False
            else:
                print(f'{str(out_file)} is corrupted.\nDeleting it...')
                out_file.unlink()

    # Compute and save density
    if compute_density:
        if transform == 'whiten':
            if whitened_T_splice_file.is_file():
                with np.load(whitened_T_splice_file) as data:
                    T_w = data['T']
                    splice_w = data['splice']
                    grp_delay = data['grp_delay']

                T_w, splice_w, density = gpu_density(
                    T_w, sublen, sigma, dpath, params_str, transform=None,
                    splice=splice_w, window=win, device_id=device_ids
                )
            else:
                T_w, splice_w, density, grp_delay = gpu_density(
                    T, sublen, sigma, dpath, params_str, transform=transform,
                    splice=splice, window=win, device_id=device_ids
                )
                with whitened_T_splice_file.open('wb') as f:
                    np.savez(f, T=T_w, splice=splice_w, grp_delay=grp_delay)
        else:
            _, _, density = gpu_density(
                T, sublen, sigma, dpath, params_str, transform=transform,
                splice=splice, window=win, device_id=device_ids
            )

        with out_file.open('wb') as f:
            np.savez(f, density=density)

    if transform == 'whiten':
        T, splice = core.whiten_alignment(T, splice, grp_delay)
    # Compute QSMP and indices
    profile, indices = gpu_qsmp(
        T, sublen, minfilt_size, density, dpath, params_str, splice=splice, device_id=device_ids
    )

    # Find global maxima (root), and fix neighbor and profile
    profile, indices, density = utils.fix_root((profile, indices, density))

    # Save density, QSMP, and indices. np.savez doesn't work in append mode.
    with out_file.open('wb') as f:
        np.savez(f, density=density, profile=profile, indices=indices)

    t_stop = perf_counter()
    print(f'Finished after {t_stop-t_start} seconds!')
