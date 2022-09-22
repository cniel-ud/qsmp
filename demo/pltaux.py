import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sparse_coding import compute_PVE, get_all_components
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import utils

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

    if ncols is None:
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


def annotate_axes(ax, text, fontsize=18):
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="darkgrey")


def plot_X_hat_and_PVE(X, D, z_hat, Fs=512, NFFT=1024, rng=None, title=''):
    """
    Assumes X is univariate: shape = (N,), with N number of time points.
    """

    X_hat = get_all_components(X, D, z_hat)
    n_atoms, sublen = D.shape
    rng = utils.check_rng(rng)
    seg_len = X.size
    fig = plt.figure(figsize=(10, 15), constrained_layout=False)
    fig.suptitle(title)
    gs = fig.add_gridspec(2*(n_atoms+1), 5, hspace=0, wspace=0.5)
    t_atom = np.arange(sublen)/Fs
    rand_start = rng.choice(seg_len-2*sublen, 1)
    idx_ts = np.arange(rand_start, rand_start+2*sublen)  # one second long
    t = idx_ts/Fs
    xy = (0.03, 0.05)

    for i in range(n_atoms+1):
        atom = fig.add_subplot(gs[2*i, 0])
        if i == n_atoms:
            annotate_axes(atom, 'All atoms')
            atom.axis('off')
        else:
            atom.plot(t_atom, D[i], color='#1f77b4')
            atom.set_title(f'Atom #{i+1}')
            atom.set_xlim(t_atom[0], t_atom[-1])
        ts = fig.add_subplot(gs[2*i:2*(i+1), 1:3])
        ts.plot(t, X[idx_ts])
        ts.plot(t, X_hat[i][idx_ts])
        ts.set_xlim(t[0], t[-1])

        f, PVE, PVE_val = compute_PVE(X, X_hat[i], Fs, NFFT)
        pve = fig.add_subplot(gs[2*i:2*(i+1), 3:])
        pve.plot(f, PVE)
        pve.set_xscale('log', base=2)
        pve.set_xlim(None, 125)
        upto = f <= 125
        yticks = [PVE[upto].min(), PVE[upto].mean(), PVE[upto].max()]
        if np.all(np.diff(yticks) < 0.01):
            yticks = [PVE[upto].mean()]
        yticks_labels = [f'{t:.2f}' for t in yticks]
        if yticks[0] < 1e-10:
            yticks[0] = 0
            yticks_labels[0] = '0'
        pve.set_yticks(yticks)
        pve.set_yticklabels(yticks_labels)
        pve.annotate(f'PVE={PVE_val:.2f}', xy=xy,
                     xytext=xy, xycoords='axes fraction')

        if i == 0:
            ts.set_title('Original signal vs reconstruction')
            ts.legend(['True', 'Reconstructed'])
            pve.set_title('PVE')
        if i == n_atoms-1:
            atom.set_xlabel('Time [sec]')
        if i == n_atoms:
            ts.set_xlabel('Time [sec]')
            pve.set_xlabel('Frequency [Hz]')
            pve.set_xticks([3, 12, 30, 70, 125])
            pve.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        if i < n_atoms:
            ts.set_xticks([])
            pve.set_xticks([])
        if i < n_atoms-1:
            atom.set_xticks([])

    plt.show()
    return fig
