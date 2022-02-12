import os
import numpy as np
import pickle
import scipy.signal as signal

def wave(sig, shift, centroid_length):
    sample_length = sig.shape[0]
    wave = sig[shift:shift + centroid_length]
    lnan, rnan = shift, sample_length - shift - centroid_length
    wave = np.r_[np.full(lnan, np.nan), wave, np.full(rnan, np.nan)]

    return wave

def align_waves(X, shift, centroid_length):
    
    nwav, wavlen = X.shape
    zeroshift = wavlen - centroid_length
    shift = zeroshift - shift
    newlen = 2*wavlen - centroid_length
    newX = np.full((nwav, newlen), np.nan)
    
    rowind = np.arange(nwav)[:, None]
    colind = np.arange(wavlen)[None,:] + shift[:, None]

    newX[rowind, colind] = X

    return newX
    
def wave_matrix(X, hgap=4, vgap=None, ncols=None):
    """
    Create a 2D grid of waves

    Parameters
    ----------
    X (2D array)
        A matrix of k waveforms (rows) of length P (columns)
    hgap (int):
        Horizontal gap = wave length / hgap
    vgap (int)
        A fixed gap between the minimum of one row and the maximum of the row below in a 2D grid of waves
    
    Returns
    -------
    X (array)
        A matrix representing a 2D grid of k = m * n waves, with n the greatest 
        power of two that is less than log2(k)/2. X.shape=(m,l), with l=n*P+(n-1)*hgap. hgap = ceil(P/4) is the horizontal gap between waves in a row.
    """
    nwav, wavlen = X.shape

    if ncols is None: # ncols > 1
        ncols, _ = _factor(nwav)    
        hgap = np.ceil(wavlen/hgap).astype('int')
        nangap = np.full((nwav, hgap), np.nan)
        X = np.hstack((X, nangap))        
        ind = np.arange(ncols, nwav-ncols+1, ncols)
        X = np.split(X, ind)
        X = np.vstack(list(map(np.ravel, X)))
        X = X[:, :-hgap]

    nrows = np.int(nwav/ncols)

    maxv = np.nanmax(X, axis=1)
    minv = np.nanmin(X, axis=1)
    rangev = maxv - minv
    if vgap is None:
        vgap = np.median(rangev)//8

    vshift = np.arange(nrows) * vgap
    shiftrange = np.r_[0, minv[:-1]-maxv[1:]]
    curange = np.cumsum(shiftrange)
    vshift = curange - vshift
    X = X + vshift[:, None]

    return X, nrows, ncols


def _factor(x):

    fact1 = np.sqrt(x).astype('int')
    while x % fact1 > 0:
        fact1 += 1

    fact2 = x / fact1
    fact2 = fact2.astype('int')

    return fact1, fact2

def built_grid(waves):
    """ For fixed `maxdist` and `distfunc`, modes[i] has the modes of the i-th
    density.
    """
    pass