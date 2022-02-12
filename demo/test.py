#%%
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from numba import njit, prange
#%%
@njit
def rolling_window(a, window):
    """
    Use strides to generate rolling/sliding windows for a numpy array.

    Parameters
    ----------
    a : numpy.ndarray
        numpy array

    window : int
        Size of the rolling window

    Returns
    -------
    output : numpy.ndarray
        This will be a new view of the original input array.
    """
    a = np.asarray(a)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
# %%
@njit(parallel=True, fastmath=True)
def ndxcorr(T, m):
    """
    np.correlate() in Numba only allows for the first two arguments, using the 
    'valid' mode by default. So, we need to do the zero padding of each window 
    in T to get its autocorrelation function. floor(m/2) zeros are padded to left and the remaining zeros (to sum m) are padded to the right. This is done to have the max peak of the autocorrelation at the center of the sequence.
    """
    X = rolling_window(T, m)
    X  = X[0]  #np.squeeze() not available on Numba
    n, m = X.shape
    xcorr = np.zeros((n, m))
    xzpadded = np.zeros(2*m - 1, X.dtype)
    l = int(m/2)
    u = l + m
    for i in range(n):
        xzpadded[l:u] = X[i].copy()
        xcorr[i] = np.correlate(X[i], xzpadded)
    return xcorr

# %%
@njit(parallel=True, fastmath=True)
def fwhm(x):
    """
    Finds the FWHM of the global maximum of x[i], for all i.
    """
    n = x.shape[0]
    fwhm = np.zeros(n, dtype=np.uint64)
    for i in prange(n):
        imax = np.argmax(x[i])
        ind = np.asarray(x[i] < x[i][imax]/2).nonzero()[0]
        isort = np.argsort(np.abs(ind - imax))
        ind = ind[isort[:2]]
        fwhm[i] = ind[1] - ind[0]
    return fwhm
#%%
n = np.arange(0,600) # in samples
F = 1 #Hz
Fs = 200  # samples/s
f = F/Fs
x = np.sin(2*np.pi*f*n)
plt.plot(x)
#%%
N = x.shape[0]
ps = 2*N - 1
xzpadded = np.zeros(ps, x.dtype)
l = int(N/2)
u = l + N
xzpadded[l:u] = x.copy()
# %%
b = np.correlate(xzpadded, x)
plt.plot(b)
b = b[None,:]
fwhm(b)
# %%
c = np.correlate(x, x, mode='same')
plt.plot(c)
c = c[None, :]
fwhm(c)
#%%
y = x[None,:]
ycorr = ndxcorr(y, 200)
#%%
plt.plot(ycorr[13])
#%%
ww = fwhm(ycorr)
ww
#%%
rng = default_rng(13)
a = rng.integers(0,13,1300)
# %%
y = ndxcorr(a, 5)
# %%
y.shape
# %%
