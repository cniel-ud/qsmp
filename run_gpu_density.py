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
parser.add_argument("-t", "--train", type=int, dest="train_len", default=0,
                    help="Number of time points for training")
#parser.add_argument("-n", "--n-gpus", dest="n_gpus", type=int,
                    #default=1, help="Number of GPUS granted by SLURM")

args = parser.parse_args()

dpath = args.dpath
wpath = args.wpath
sublen = args.sublen
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


GAMMA = 1
def make_gauss(gamma=1):
    @numba.cuda.jit("f8(f8)", device=True)
    def gauss(x):
        # Assumes x is distance squared
        return math.exp(-x/gamma)
    return gauss

gauss = make_gauss(gamma=GAMMA)
density = gpu_density(T, sublen, gauss, splice=splice,
            device_id=device_ids, normalize=True)

profile, indices = gpu_qsmp(T, sublen, density, 
                    splice=splice, device_id=device_ids)


fname = 'test_gpu_density.npz'
fpath = os.path.join(dpath, fname)
with open(fpath, 'wb') as f:
    np.savez(f, density=density, profile=profile, indices=indices)
