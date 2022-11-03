#%%
import numpy as np
from .csc_utils import construct_X
from scipy import signal

#%% Load time series and splice
def compute_PVE(X, X_hat, Fs, NFFT=1024):

    f, Px = signal.welch(X, fs=Fs, window='hamming', detrend=False,
                         nperseg=Fs//4, nfft=NFFT, noverlap=Fs//8)
    res = X - X_hat
    _, Pres = signal.welch(res, fs=512, window='hamming', detrend=False,
                           nperseg=128, nfft=NFFT, noverlap=64)

    PVE = 1 - Pres/Px
    PVE_val = 1 - np.sum(res**2)/np.sum(X**2)

    return f, PVE, PVE_val

def get_all_components(X, D, z_hat):
    """
    The i-th component is the convolution of z_hat[i] and D[i]. z_hat[i] is 
    the sparse code vector and D[i] is the atom. We also return the full reconstruction, the result of using all the atoms.
    """

    if X.ndim == 1:
        X = X[None, :]  # 1 trial

    n_atoms = D.shape[0]
    sig_len = X.shape[1]
    X_hat = np.zeros((n_atoms+1, sig_len))
    for i in range(n_atoms+1):
        if i == n_atoms:
            X_hat[i] = construct_X(z_hat, D)
        else:
            z_i = np.expand_dims(z_hat[i], 0)
            d_i = np.expand_dims(D[i], 0)
            X_hat[i] = construct_X(z_i, d_i)

    return X_hat