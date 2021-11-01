import os
from argparse import ArgumentParser
import numba
import numpy as np
import math

from gpu_density import gpu_density
from gpu_qsmp import gpu_qsmp
import utils

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

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

args = parser.parse_args()

dpath = args.dpath
wpath = args.wpath
sublen = args.sublen
snr = args.snr
train_len = args.train_len

#%% Get the CSP filters
W = utils.loadmat73(wpath, 'W')

# Pick first(last) CSP filter for preictal(interictal)
n_csp = W.shape[1]
if 'preictal' in dpath:
    i_csp = 0   
elif 'interictal' in dpath:
    i_csp = n_csp - 1
else:
    raise ValueError(f"The path '{dpath}' doesn't contain neither 'preictal' nor 'interictal'")

W = W[:, i_csp]

T, splice = utils.cat_segments(dpath, W, train_len=train_len)

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

density = gpu_density(T, sublen, bw, splice=splice,
            device_id=device_ids, normalize=True)

profile, indices = gpu_qsmp(T, sublen, density,
                    splice=splice, device_id=device_ids)

fname = 'test_gpu_density.npz'
fpath = os.path.join(dpath, fname)
with open(fpath, 'wb') as f:
    np.savez(f, density=density, profile=profile, indices=indices)
