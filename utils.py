import os
import re
import glob
import h5py
import numpy as np
import scipy.signal as signal
import pickle

# Pattern for path to file rx < id > .mat
FILE_ID_PAT = '.*/rx(?P<id>\d+).mat$'
RX_GLOB = 'rx*.mat'

def loadmat73(fpath, varname):
    """ Load data from a -v7.3 Matlab file."""
    with h5py.File(fpath, 'r') as hdf5:
        dataset = hdf5[varname]
        return _h5py_unpack(dataset, hdf5)


def _h5py_unpack(obj, hdf5):
    """
    Unpack an HDF5 object saved on Matlab v7.3 format.

    It can unpack:
        - (float, int, char) cell arrays. Returned as nested lists with the original Matlab 'shape' and with the same data type.
        - (float, int) arrays. Returned as numpy arrays of the same data type and with the original shape.

    Parameters
    ----------
    obj (array of object references, object reference, dataset):
        The first call should have obj as a dataset type. That dataset might contain a reference or array of references to other datasets.
    hdf5 (File object):
        An instance of h5py.File()

    Returns
    -------
    Numpy arrays for Matlab arrays and nested lists for cell arrays.

    Inspired by https://github.com/skjerns/mat7.3
    """
    if isinstance(obj, np.ndarray): # array of references
        if obj.size == 1:
            obj = obj[0] # an object reference
            obj = hdf5[obj] # a dataset
            return _h5py_unpack(obj, hdf5)
        elif obj.size > 1:
            cell = []
            for ref in obj:
                entry = _h5py_unpack(ref, hdf5)
                cell.append(entry)
            return cell
    elif isinstance(obj, h5py.h5r.Reference): # an object reference
        obj = hdf5[obj]
        return _h5py_unpack(obj, hdf5)
    elif isinstance(obj, h5py._hl.dataset.Dataset):  # a dataset
        vartype = obj.attrs['MATLAB_class']
        if vartype == b'cell':
            cell = []
            for ref in obj:
                entry = _h5py_unpack(ref, hdf5)
                cell.append(entry)
            if len(cell) == 1:
                cell = cell[0]
            if obj.parent.name == '/': # first call
                 if isinstance(cell[0], list): # cell is a nested list
                    cell = list(map(list, zip(*cell)))  # transpose cell
            return cell
        elif vartype == b'char':
            stra = np.array(obj).ravel()
            stra = ''.join([chr(x) for x in stra])
            return stra
        else: #(float or int, not struct)
            array = np.array(obj)
            array = array.T # from C order to Fortran (MATLAB) order
            return array

def apply2list(obj, fun):
    if not isinstance(obj, list):
        return fun(obj)
    else:
        return [apply2list(x, fun) for x in obj]

def make_get_id(p):
    def _get_id(file):
        m = p.search(file)
        return m.group('id')
    return _get_id

def cat_segments(dpath, W, train_len=None):

    if train_len:
        print(f'Requested the first {train_len} time points')

    p = re.compile(FILE_ID_PAT)

    globpath = os.path.join(dpath, RX_GLOB)
    files = glob.glob(globpath)

    get_id = make_get_id(p)
    ids = np.array(list(map(get_id, files)), dtype='u2')
    isort = np.argsort(ids) # sort files in ascending (time) order of ids
    ids = ids[isort]
    files = list(np.array(files)[isort])

    n_files = len(files)

    seglen = []
    ts = []
    pnts = 0
    for i_file in np.arange(n_files):
        epoch = loadmat73(files[i_file], 'epoch')
        x = np.matmul(W.T, epoch)  # spatial filtering
        ts.append(x)
        xlen = x.size
        seglen.append(xlen)
        pnts += xlen

        if train_len and pnts >= train_len:
            break

    print(f'Time series with {pnts} time points after '
          f'concatenating {i_file+1} segments')

    ts = np.hstack(ts)
    seglen = np.array(seglen)
    cumlen = np.cumsum(seglen)
    splice = cumlen[:-1]  # start index for second to last segment

    return ts, splice.astype(np.uint64)

def fix_root(qsmp):
    profile, neighbor, density = qsmp
    n_bw = profile.shape[1]
    for i_bw in np.arange(n_bw):
        is_mode = np.isinf(profile[:, i_bw])
        iinf = np.asarray(is_mode).nonzero()[0]
        # QSMP=inf -> hit a mode: its nearest neighbor is itself.
        imax = np.argmax(density[:, i_bw])
        assert imax in iinf
        neighbor[iinf, i_bw] = iinf
        profile[iinf, i_bw] = 0

    return profile, neighbor, density


def get_waves(modes, ts, m, max_modes=None):
    idx = np.asarray([mode.index for mode in modes])

    n_modes = idx.size
    if max_modes is not None:
        n_modes = min(n_modes, max_modes)

    idx = idx[:n_modes]
    t = idx[:, None] + np.arange(m)[None, :]
    waves = ts[t]

    return waves, idx

def fwhm(x):
    n = x.shape[0]
    fwhm = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        imax = np.argmax(x[i])
        ind = np.asarray(x[i] < x[i][imax]/2).nonzero()[0]
        isort = np.argsort(np.abs(ind - imax))
        ind = ind[isort[:2]]
        fwhm[i] = ind[1] - ind[0]
    return fwhm

def ndxcorr(x):
    n, m = x.shape
    xcorr = np.zeros((n, 2*m-1))
    for i in range(n):
        xcorr[i] = signal.correlate(x[i], x[i])
    return xcorr


def load_modes(folder, maxdist, distfunc):
    fname = f'tree_maxdist{maxdist:.3g}_{distfunc}.pickle'
    fpath = os.path.join(folder, fname)
    with open(fpath, 'rb') as f:
        modes = pickle.load(f)
    return modes
