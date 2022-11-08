import os
from argparse import ArgumentParser
from pathlib import Path
import numba
import numpy as np

from qsmp.gpu_density import gpu_density
from qsmp.gpu_qsmp import gpu_qsmp
from qsmp.utils import windows
import utils

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
    parser.add_argument("-m", "--subseq-len", dest="sublen", type=int,
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
    wpath = args.wpath
    sublen = args.sublen
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
    sigma_str = [str(i) for i in sigma]
    sigma_str = '_'.join(sigma_str)

    #XXX: we are currently taking the first segments whose cumulative length
    # is >= args.train. This parameter is NOT currently reflected in the naming
    # of the output files.
    fnames = {
        'time series': 'qsmp_T_splice.npz'
    }
    # (Gaussian/Rect) window to penalize patterns that are not centered
    if args.window_type is not None:
        win_fn = windows.get_window(args.window_type)
        win = win_fn(sublen, win_support)
        win_str = f'_{args.window_type}-{int(100*win_support)}'
    else:
        win = None
        win_str = ''

    if transform is not None:
        tr_str = f'_{transform}'
    else:
        tr_str = ''
    fnames['Qtuple'] = (
        f'qsmp_m{sublen}_sigma{sigma_str}{win_str}'
        f'{tr_str}minfilt-{minfilt_size}.npz'
    )
    if transform == 'whiten':
        fnames['whitened time series'] = 'qsmp_T_splice_whitened.npz'

    get_data = True
    fpath = os.path.join(dpath, fnames['time series'])
    if os.path.isfile(fpath):
        with np.load(fpath) as data:
            if 'T' in data:
                T = data['T']
                splice = data['splice']
                get_data = False
            else:
                print(f'{fpath} is corrupted.\nDeleting it...')
                os.remove(fpath)

    if get_data:
        #%% Get the CSP filters
        W = utils.loadmat73(wpath, 'W')
        # Pick first(last) CSP filter for preictal(interictal)
        n_csp = W.shape[1]
        if 'preictal' in dpath:
            i_csp = 0
        elif 'interictal' in dpath:
            i_csp = n_csp - 1
        else:
            raise ValueError(
                f"The path '{dpath}' doesn't contain neither 'preictal' nor 'interictal'")
        W = W[:, i_csp]
        T, splice, t_start, t_end, seiz_id = utils.cat_segments(
            dpath, W, train_len=train_len)
        fpath = os.path.join(dpath, fnames['time series'])
        with open(fpath, 'wb') as f:
            np.savez(f, T=T, splice=splice, t_start=t_start,
                        t_end=t_end, seiz_id=seiz_id)

    compute_density = True
    fpath = os.path.join(dpath, fnames['Qtuple'])
    if os.path.isfile(fpath):
        with np.load(fpath) as data:
            if 'density' in data:
                density = data['density']
                compute_density = False
            else:
                print(f'{fpath} is corrupted.\nDeleting it...')
                os.remove(fpath)

    # Compute and save density
    if compute_density:
        T, splice, density = gpu_density(
            T, sublen, sigma, dpath, transform=transform,
            splice=splice, window=win, device_id=device_ids
        )
        fpath = os.path.join(dpath, fnames['Qtuple'])
        with open(fpath, 'wb') as f:
            np.savez(f, density=density)
        if transform == 'whiten':
            fpath = os.path.join(dpath, fnames['whitened time series'])
            if not os.path.isfile(fpath):
                with open(fpath, 'wb') as f:
                    np.savez(f, T=T, splice=splice)


    # Compute QSMP and indices
    profile, indices = gpu_qsmp(
        T, sublen, minfilt_size, density, dpath, splice=splice, device_id=device_ids
    )

    # Find global maxima (root), and fix neighbor and profile
    profile, indices, density = utils.fix_root((profile, indices, density))

    # Save density, QSMP, and indices. np.savez doesn't work in append mode.
    fpath = os.path.join(dpath, fnames['Qtuple'])
    with open(fpath, 'wb') as f:
        np.savez(f, density=density, profile=profile, indices=indices)

    t_stop = perf_counter()
    print(f'Finished after {t_stop-t_start} seconds!')
