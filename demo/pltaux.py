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


def get_waves(modes, ts, m, max_modes):
    idx = np.asarray([mode.index for mode in modes])
    
    n_modes = idx.size
    n_modes = min(n_modes, max_modes)
    idx = idx[:n_modes]
    t = idx[:, None] + np.arange(m)[None, :]
    waves = ts[t]

    return waves, idx

def _fwhm(x):
    n = x.shape[0]
    fwhm = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        imax = np.argmax(x[i])
        ind = np.asarray(x[i]<x[i][imax]/2).nonzero()[0]
        isort = np.argsort(np.abs(ind - imax))
        ind = ind[isort[:2]]
        fwhm[i] = ind[1] - ind[0]
    return fwhm

def _ndcorr(x):
    n, m = x.shape
    xcorr = np.zeros((n, 2*m-1))
    for i in range(n):
        xcorr[i] = signal.correlate(x[i], x[i])
    return xcorr

def built_grid_fixed_sigma(idx, folder, maxdist, distfunc, ts, m, max_modes):
    
    filelist = os.listdir(folder)
    filelist = [file for file in filelist if '.pickle' in file]
    n_maxdist = len(maxdist)
    n_distfunc = len(distfunc)
    grid = [None] * n_maxdist
    indices = [None] * n_maxdist
    energy = [None] * n_maxdist
    fwhm = [None] * n_maxdist
    for i in range(n_maxdist):
        grid[i] = [None] * n_distfunc
        indices[i] = [None] * n_distfunc
        energy[i] = [None] * n_distfunc
        fwhm[i] = [None] * n_distfunc
        for j in range(n_distfunc):
            fname = f'tree_maxdist{maxdist[i]:.3g}_{distfunc[j]}.pickle'
            fpath = os.path.join(folder, fname)
            with open(fpath, 'rb') as f:
                modes = pickle.load(f)
            
            waves, ind = get_waves(modes[idx], ts, m, max_modes)
            energy[i][j] = np.linalg.norm(waves, axis=1)**2            
            fwhm[i][j] = _fwhm(_ndcorr(waves))
            waves, n_rows, n_cols = wave_matrix(waves)
            ind = ind.reshape((n_rows, n_cols))

            energy[i][j] = energy[i][j].reshape((n_rows, n_cols))
            fwhm[i][j] = fwhm[i][j].reshape((n_rows, n_cols))
            grid[i][j] = waves
            indices[i][j] = ind

    return grid, indices, energy, fwhm