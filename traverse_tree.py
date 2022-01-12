import os
import numpy as np
import tree
import utils
from time import perf_counter
import itertools
import multiprocessing as mp
from multiprocessing import shared_memory, current_process
import pickle

def shared_array(name, arr):
    try:
        shm = shared_memory.SharedMemory(name, size=arr.nbytes, create=True)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name, size=arr.nbytes, create=False)
    arr_sh = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    arr_sh[:] = arr[:]  # Copy the original data into shared memory
    return shm, arr_sh


def _find_modes_worker(qsmp, iter, dirpath):

    n_densities = qsmp.shape[0]
    print(f'Number of densities: {n_densities}', flush=True)

    if isinstance(iter, tuple):
        iter = [iter]

    for maxdist, distfunc in iter:
        modes = [None] * n_densities
        for j in range(n_densities):
            print(
                f'j={j}, maxdist:{maxdist:.3g}, aggregate:{distfunc}, PID={current_process().pid}', flush=True)
            t1 = perf_counter()
            modes[j] = tree.find_modes_no_exclusion_zone(qsmp[j], maxdist, distfunc)
            t2 = perf_counter()
            print(
                f'j={j}, maxdist={maxdist:.3g}, aggregate={distfunc}, time={t2-t1:.3g} seconds, PID={current_process().pid}', flush=True)

        fname = f'tree_maxdist{maxdist:.3g}_{distfunc}.pickle'
        fpath = os.path.join(dirpath, fname)
        with open(fpath, 'wb') as f:
            pickle.dump(modes, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    mp.set_start_method('spawn')

    dirpath = "/home/cmendoza/Research/QSMP/data/Study019/preictal"
    fname = 'qsmp_m350_snr-4.0_-2.0_0.0_2.0_4.0.npz'
    fpath = os.path.join(dirpath, fname)

    with np.load(fpath) as data:
        density = data['density']
        profile = data['profile']
        neighbor = data['indices']

    # snr = np.r_[0, 2, 4, 8, 10]
    snr = np.r_[-4, -2, 0, 2, 4]
    var_noise = 10 ** (-snr/10)
    th = 0.1
    bandwidths = (9 * var_noise) / np.log(1/th)
    sigmas = np.sqrt(bandwidths/2)

    m = 350
    nonan_profile = profile[~np.isnan(profile)]
    quantiles = np.quantile(nonan_profile, [0.75, 0.99])
    quantiles = np.log2(quantiles)
    maxdists = 2 ** np.linspace(*quantiles, 5)
    # maxdists = maxdists[:2]

    n_subseq, n_bw = neighbor.shape
    path_agg = ['add', 'max', 'mean']
    path_agg = [path_agg[1]]

    n_cpus = 5
    p = mp.Pool(processes=n_cpus)
    params = list(itertools.product(maxdists, path_agg))
    n_params = len(params)
    step = int(np.ceil(n_params/n_cpus))

    modes = [None] * n_cpus
    results = [None] * n_cpus

    profile = profile.T
    neighbor = neighbor.T
    density = density.T

    # profile = profile[3:]
    # neighbor = neighbor[3:]
    # density = density[3:]

    n_densities = density.shape[0]
    qsmp = np.zeros((n_densities, 3, density.shape[1]))
    for i in np.arange(n_densities):
        qsmp[i][0] = profile[i]
        qsmp[i][1] = neighbor[i]
        qsmp[i][2] = density[i]

    del profile
    del neighbor
    del density

    shm, qsmp = shared_array('qsmp', qsmp)

    for idx, start in enumerate(range(0, n_params, step)):
        stop = min(start+step, n_params)
        if n_cpus > 1:
            p.apply_async(
                _find_modes_worker,
                (
                    qsmp,
                    params[start:stop],
                    dirpath
                )
            )
        else:
            _find_modes_worker(
                qsmp,
                params[start:stop],
                dirpath
            )

    # Clean up process pool
    if n_cpus > 1:  # pragma: no cover
        p.close()
        p.join()

    # shm.close()
    # shm.unlink()
