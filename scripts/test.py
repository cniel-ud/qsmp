#%%
import os, sys
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from numba import njit, prange
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from time import perf_counter
#%%
@njit(parallel=True)
def rolling_window(a, window, splice=None):
    """
    Use strides to generate rolling/sliding windows for a numpy array.

    Parameters
    ----------
    a: numpy.ndarray
        numpy array
    window : int
        Size of the rolling window
    splice: numpy.ndarray
        Is `splice` is an array, `a` is the result of `n` concatenated
        segments. `splice` has the start index of second to last segments. In
        other words, the indices where the concatenation of segments takes
        place. The rolling windows are extracted for each segment separately
        and then vertically stacked.  XXX: `a` is assumed to be a 1D array.

    Returns
    -------
    output : numpy.ndarray
        This will be a new view of the original input array.

    Note:
    We are JIT compiling this because it is used in ndxcorr, which is jitted.
    """

    if splice is not None:
        start = np.concatenate((np.array([0]), splice))
        end = np.hstack((splice, np.array([a.size])))
        n_seg = start.size
        n_win = end-start-(window-1)
        cum_win = np.cumsum(n_win)
        idx = np.hstack((np.array([0]), cum_win))
        tot_win = np.sum(n_win)
        out = np.zeros((tot_win, window))
        for i in prange(n_seg):
            shape = a[start[i]:end[i]].shape[:-1] + \
                (a[start[i]:end[i]].shape[-1] - window + 1, window)
            strides = a[start[i]:end[i]].strides +\
                      (a[start[i]:end[i]].strides[-1],)
            out[idx[i]:idx[i+1]] = np.lib.stride_tricks.as_strided(
                a[start[i]:end[i]], shape=shape, strides=strides)

    else:
        a = np.asarray(a)
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        out = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    return out
# %%
@njit(parallel=True, fastmath=True)
def ndxcorr(T, m, splice=None):
    """
    np.correlate() in Numba only allows for the first two arguments, using the
    'valid' mode by default. So, we need to do the zero padding of each window
    in T to get its autocorrelation function. floor(m/2) zeros are padded to left and the remaining zeros (to sum m) are padded to the right. This is done to have the max peak of the autocorrelation at the center of the sequence. Assumes that T has only one dimension. If `splice` is an array, `T` is the result of concatenating `n_seg` segments, and `splice` contains the indices where the concatenation occurs: the start indices of the second to the last segment; in this case, it computes the autocorrelation only for windows that don't fall at the splice.
    """
    X = rolling_window(T, m, splice)
    n, m = X.shape
    xcorr = np.zeros((n, m))
    l = int(m/2)
    u = l + m
    for i in prange(n):
        xzpadded = np.zeros(2*m - 1, X.dtype)
        xzpadded[l:u] = X[i]
        xcorr[i] = np.correlate(X[i], xzpadded)
    return xcorr
# %%
# @njit(parallel=True, fastmath=True)
def fwhm(x):
    """
    Finds the FWHM of the global maximum of x[i], for all i.

    This function is intended for a curve `x` that is symmetric around its
    global maxima (like the autocorrelation function). To compute the FWHM, we
    consider two cases:
    1) The curve increases monotonically from 0 to its global maxima (covered
    in the 'else' clause). There is only one broad peak in `x`.
    2) The curve has one main peak centered around its global maxima, and some side-band ripples (other lower-amplitude local maxima on both sides). This is covered in the 'if' clause.
    """
    n, m = x.shape
    fwhm = np.zeros(n, dtype=np.uint64)
    for i in prange(n):
        imax = np.argmax(x[i])
        diff1 = np.diff(x[i])
        left = diff1[:imax] < 0
        if left.any():
            left = np.asarray(left).nonzero()[0][-1]
            right = imax + np.asarray(diff1[imax:] > 0).nonzero()[0][0]
        else:
            left = -1
            right = m - 1
        idx = np.arange(left+1, right+1)
        half_range = (x[i][idx].max()-x[i][idx].min())/2 + x[i][idx].min()
        idx = idx[x[i][idx] < half_range]
        isort = np.argsort(np.abs(idx - imax))
        idx = np.sort(idx[isort[:2]])
        fwhm[i] = idx[1] - idx[0]
    return fwhm
#%%
def fill_fwhm(fwhm, splice, m):
    """ Extend fwhm to fill the gaps created by the splice

    `fwhm` is the output from `ndxcorr()`. Its length is smaller than what is
    expected in `gpu_density` and `gpu_qsmp` because the subsequences (windows)
    that contain the splice are ignored. Here we extend `fwhm` to get the rigth length, adding an arbitrary value for those subsequences. For each segment
    in the time series, there are m-1 subsequences that are ignored, with `m`  being the length of a subsequence.
    """
    I = np.arange(fwhm.size)
    splice_ext = np.r_[0, splice, I[-1]+(m-1)*splice.size]
    n_seg = splice_ext.size - 1
    new_len = splice_ext[-1] + 1
    ext_fwhm = np.ones(new_len, dtype=np.int64)
    for i in range(n_seg):
        idx = (I >= splice_ext[i]) & (I < splice_ext[i+1])
        ext_idx = I[idx] + (m-1)*i
        ext_fwhm[ext_idx] = fwhm[I[idx]]

    return ext_fwhm
#%%
root = '/home/cmendoza/Research/QSMP/data/Study019/preictal'
fname = 'qsmp_T_splice.npz'
fpath = os.path.join(root, fname)
with np.load(fpath) as data:
    T = data['T']
    splice = data['splice']
#%%
m = 350
tstart = perf_counter()
X = rolling_window(T, m, splice)
tstop = perf_counter()
print(f'Time elapsed: {tstop-tstart:.3g} seconds')
#%%
xcorr = ndxcorr(T, splice, 350)
#%%
fwhm_var = fwhm(xcorr)
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
