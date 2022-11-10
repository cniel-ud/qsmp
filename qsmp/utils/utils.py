import os
import re
import h5py
import numpy as np
import pickle
import numbers
from pathlib import Path

# Pattern for path to file rx < id > .mat
FILE_ID_PAT = '.*/rx(?P<id>\d+).mat$'
RX_GLOB = 'rx*.mat'


def check_rng(seed):
    """Turn seed into a np.random.Generator instance

    Parameters
    ----------
    seed : None, int or instance of Generator
        If seed is None, return the Generator using the OS entropy.
        If seed is an int, return a new Generator instance seeded with seed.
        If seed is already a Generator instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, numbers.Integral):
        seed = np.random.SeedSequence(seed)
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.SeedSequence):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.Generator):
        return seed

def loadmat73(fpath:Path, varname):
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

def cat_segments(dpath:Path, W, train_len=None):

    if train_len:
        print(f'Requested the first {train_len} time points')

    p = re.compile(FILE_ID_PAT)

    files = dpath.glob(RX_GLOB)
    #XXX: a better way to do this using Path methods?
    files = [str(f) for f in files]

    get_id = make_get_id(p)
    ids = np.array(list(map(get_id, files)), dtype='u2')
    isort = np.argsort(ids) # sort files in ascending (time) order of ids
    ids = ids[isort]
    files = list(np.array(files)[isort])

    n_files = len(files)

    seglen = []
    ts = []
    seiz_id = []
    t_start, t_end = [], []
    pnts = 0
    for i_file in np.arange(n_files):
        epoch = loadmat73(files[i_file], 'epoch')
        try:
            szid = loadmat73(files[i_file], 'seiz_id')
            seiz_id.append(szid)
        except KeyError:
            pass
        t_start.append(loadmat73(files[i_file], 't_start'))
        t_end.append(loadmat73(files[i_file], 't_end'))
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
    t_start, t_end = np.array(t_start).squeeze(), np.array(t_end).squeeze()

    return ts, splice, t_start, t_end, seiz_id


def splitdata(X, chunk_size, keep_dims=True):
    """
    Split a data into smaller non-overlapping chunks

    Parameters
    ----------
    X (array):
        A 2D array. The rows are observations (data points). X.shape = (m,n)
    chunk_size (int):
        Each observation is split into chunks of size chunk_size. It is assumed that n = k*chunk_size, with k an integer.
    keep_dims (bool):
        It controls the number of dimensions of the returned matrix. See below.

    Returns
    -------
    X (array):
        If keep_dims == True, X.shape = (k*m, chunk_size): the chunks of each observation are stacked vertically as rows of the output matrix. If keep_dims == False, X.shape = (m, k, chunk_size): the chunks are stacked along the second dimension and extend along the third dimension of the output matrix.
    """

    if X.ndim == 1:
        X.shape = (1, -1)

    ind1 = np.arange(X.shape[0]).reshape(-1, 1, 1)
    offset = np.arange(0, X.shape[1], chunk_size).reshape(1, -1, 1)
    chunk_ind = np.arange(chunk_size).reshape(1, 1, -1)
    ind2 = offset + chunk_ind
    X = X[ind1, ind2]

    if keep_dims:
        return X.reshape(-1, chunk_size)
    else:
        return X


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


def get_waves(start_index, time_series, wave_lenght):
    """ Extract waves using start index and wave length

    start_index.shape=(k,)
    time_series.shape=(N,)
    wave_lenght is an int
    """
    if isinstance(start_index, np.ndarray):
        if issubclass(start_index.dtype.type, np.integer):
            t = start_index[:, None] + np.arange(wave_lenght)[None, :]
        else:
            raise TypeError(
                f"Only arrays of integer dtypes are supported, but {(start_index).dtype.type} was passed.")

    elif isinstance(start_index, (int, np.integer)):
        t = np.arange(start_index, start_index + wave_lenght)
    else:
        raise TypeError(
            f"Only int or numpy.ndarray are supported, but {type(start_index).__name__} was passed.")


    waves = time_series[t]

    return waves


def load_modes(folder, maxdist, distfunc):
    fname = f'tree_maxdist{maxdist:.3g}_{distfunc}.pickle'
    fpath = os.path.join(folder, fname)
    with open(fpath, 'rb') as f:
        modes = pickle.load(f)
    return modes


def where_equal(x, y):
    """ Find indices in `x` where `x`==`y`

    Parameters
    ----------
    x: numpy.array
        Reference 1D array. x.shape=(N,). `x` is assumed to be sorted in ascending order.
    y: numpy.array
        Query 1D array. y.shape=(m,). N<m. `y` is assumed to have unique (non-repeating) values.

    Returns
    -------
    idx: numpy.array
        Indices of `x` where values of `x` are in `y`

    Notes
    -----
    A naive for-loop implementation will be O(N*m). Since searchsorted uses binary search, I believe the cost here is O(log(N)*m)?
    """

    start = np.searchsorted(x, y, side='left')
    end = np.searchsorted(x, y, side='right')
    idx = np.r_[tuple(slice(s, e) for s, e in zip(start, end))]

    return idx

def phase_correction(ind, end_seg, grp_delay, direction='backward'):
    """ Shift indices according to group delay caused by linear-phase filter.

    `ind` is a vector of NN-indices of time series subsequences
    `end_seg[i]` has the end index of the i-th segment in the time series.
    `grp_delay` is the grp_delay caused by a linear-phase filter when applied
    to  the time series.
    `direction` indicates the direction of the correction (shift). 'backward'
    is to convert an index in the filtered time series to its equivalent in the
    unfiltered time series.
    ind might contain NaN values, and that is OK, np.searchsorted() returns the end_seg.size in those cases, and NaN + number is NaN.
    """

    if direction == 'forward':
        grp_delay = -np.abs(grp_delay)
    elif direction == 'backward':
        grp_delay = np.abs(grp_delay)

    i_seg = np.searchsorted(end_seg, ind) + 1
    ind = ind + i_seg*grp_delay

    return ind


def args2str(args):
    sigma_str = [str(i) for i in args.sigma]
    sigma_str = '_'.join(sigma_str)

    if args.window_type is not None:
        win_str = f'_{args.window_type}-{int(100*args.window_support)}'
    else:
        win_str = ''

    if args.transform is not None:
        tr_str = f'_{args.transform}'
    else:
        tr_str = ''

    name_pat = (
        f'm-{args.subseq_len}_sigma-{sigma_str}{win_str}'
        f'{tr_str}_minfilt-{args.minfilt_size}'
    )
    return name_pat
