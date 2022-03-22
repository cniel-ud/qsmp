import os
from argparse import ArgumentParser
import numba
import numpy as np

from gpu_density import gpu_density
from gpu_qsmp import gpu_qsmp
import utils

from time import perf_counter

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

if __name__ == "__main__":

    t_start = perf_counter()

    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("-d", "--data-path", dest="dpath",
                        help="Path to folder with time series")
    parser.add_argument("-w", "--W-path", dest="wpath",
                        help="Path to matrix W with spatial filters")
    parser.add_argument("-m", "--subseq-len", dest="sublen", type=int,
                        help="Subsequence (query) length")
    parser.add_argument("--snr", type=float, dest="snr", default=5,
                        nargs='*', help="SNR in dB")
    parser.add_argument("-t", "--train", type=int, dest="train_len", default=0,
                        help="Number of time points for training")
    parser.add_argument('--fwhm', action="store_true", default=False,
                        help="Scale distances by the FWHM of the autocorrelation of the subsequences")
    parser.add_argument('--whiten', action="store_true", default=False,
                        help="Filter the time series with a whitening filter to de-emphasize low frequencies and emphasize high-frequencies")    

    args = parser.parse_args()

    dpath = args.dpath
    wpath = args.wpath
    sublen = args.sublen
    snr = args.snr
    train_len = args.train_len

    device_ids = [device.id for device in numba.cuda.list_devices()]

    # The Gaussian kernel is of the form
    #   f(x) = exp(-x^2/bw)
    #   with `bw` being the bandwidth parameter
    if not isinstance(snr, list): snr = [snr]
    snr = np.array(sorted(snr))
    var_noise = 10 ** (-snr/10) # signal has unit variance (z-normalization)
    # At max noise level (3*sqrt(var_noise)), the contribution to the density of a 
    # given pair is th% (fraction of max. value of density (1)).
    th = 0.1
    bw = (9 * var_noise) / np.log(1/th)

    snr_str = [str(i) for i in snr]
    snr_str = '_'.join(snr_str)

    #XXX: we are currently taking the first segments whose cumulative length 
    # is >= args.train. This parameter is NOT currently reflected in the naming 
    # of the output files.
    fnames = {
        'time series': 'qsmp_T_splice.npz'
    }
    transform = None
    if args.fwhm:
        transform = 'fwhm'
        fnames['Qtuple'] = f'qsmp_m{sublen}_snr{snr_str}_fwhm.npz'
    elif args.whiten:
        transform = 'whiten'
        fnames['whitened time series'] = 'qsmp_T_splice_whitened.npz'
        fnames['Qtuple'] = f'qsmp_m{sublen}_snr{snr_str}_whiten.npz'

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
        T, splice = utils.cat_segments(dpath, W, train_len=train_len)        
        fpath = os.path.join(dpath, fnames['time series'])
        with open(fpath, 'wb') as f:
            np.savez(f, T=T, splice=splice)
    
    
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
            T, sublen, bw, dpath, transform=transform, 
            splice=splice, device_id=device_ids)        
        fpath = os.path.join(dpath, fnames['Qtuple'])
        with open(fpath, 'wb') as f:
            np.savez(f, density=density)
        if transform == 'whiten':
            fpath = os.path.join(dpath, fnames['whitened time series'])
            if not os.path.isfile(fpath):
                with open(fpath, 'wb') as f:
                    np.savez(f, T=T, splice=splice)
        

    # Compute QSMP and indices
    profile, indices = gpu_qsmp(T, sublen, density, dpath, transform=transform,
                        splice=splice, device_id=device_ids)

    
    # Find global maxima (root), and fix neighbor and profile
    profile, indices, density = utils.fix_root((profile, indices, density))

    # Save density, QSMP, and indices. np.savez doesn't work in append mode.    
    fpath = os.path.join(dpath, fnames['Qtuple'])
    with open(fpath, 'wb') as f:
        np.savez(f, density=density, profile=profile, indices=indices)

    t_stop = perf_counter()
    print(f'Finished after {t_stop-t_start} seconds!')