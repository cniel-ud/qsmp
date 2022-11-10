# STUMPY Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause
# BSD license. STUMPY is a trademark of TD Ameritrade IP Company, Inc. All
# rights reserved.
# QSMPY Copyright (c) 2021, Carlos H. Mendoza-Cardenas. Released under the
# terms of the 3-Clause BSD license.

# This module adapts and expands the module gpu_stump in
# https://github.com/TDAmeritrade/stumpy

import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
import nvtx

import numpy as np
from numba import cuda

from qsmp import core, config
from time import perf_counter
import cupy


logger = logging.getLogger(__name__)

@cuda.jit
def _compute_and_update_dist_kernel(
    i,
    T,
    m,
    QT_even,
    QT_odd,
    QT_first,
    M_T,
    Σ_T,
    k,
    excl_zone,
    dist_vec,
    compute_QT,
    ):
    """
    A Numba CUDA kernel to update the matrix profile and matrix profile indices

    Parameters
    ----------
    i : int
        sliding window `i`
    T : ndarray
        The time series or sequence for which to compute the dot product
    m : int
        Window size
    QT_even : ndarray
        The input QT array (dot product between the query sequence,`Q`, and
        time series, `T`) to use when `i` is even
    QT_odd : ndarray
        The input QT array (dot product between the query sequence,`Q`, and
        time series, `T`) to use when `i` is odd
    QT_first : ndarray
        Dot product between the first query sequence,`Q`, and time series, `T`
    M_T : ndarray
        Sliding mean of time series, `T`
    Σ_T : ndarray
        Sliding standard deviation of time series, `T`
    k : int
        The total number of sliding windows to iterate over
    excl_zone : int
        The half width for the exclusion zone relative to the current
        sliding window
    dist_vec : ndarray
        D[i,:], where D[i,j] is the distance between subsequences x_i and x_j
    compute_QT : bool
        A boolean flag for whether or not to compute QT

    Returns
    -------
    None
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    if i % 2 == 0:
        QT_out = QT_even
        QT_in = QT_odd
    else:
        QT_out = QT_odd
        QT_in = QT_even

    for j in range(start, QT_out.shape[0], stride):
        zone_start = max(0, j - excl_zone)
        zone_stop = min(k, j + excl_zone)

        # if i==2 and j==2:
        #     set_trace()

        if compute_QT:
            QT_out[j] = (
                QT_in[j - 1] - T[i - 1] * T[j - 1]
                             + T[i + m - 1] * T[j + m - 1]
            )
            QT_out[0] = QT_first[i]

        if math.isinf(M_T[j]) or math.isinf(M_T[i]):
            D = np.inf
        else:
            if (
                Σ_T[i] < config.STUMPY_STDDEV_THRESHOLD
                or Σ_T[j] < config.STUMPY_STDDEV_THRESHOLD
            ):
                D = m
            else:
                denom = m * Σ_T[i] * Σ_T[j]
                if math.fabs(denom) < config.STUMPY_DENOM_THRESHOLD:  # pragma nocover
                    denom = config.STUMPY_DENOM_THRESHOLD
                D = abs(2 * m * (1.0 -
                        (QT_out[j] - m * M_T[i] * M_T[j]) / denom)
                    )

            if (
                Σ_T[i] < config.STUMPY_STDDEV_THRESHOLD
                and Σ_T[j] < config.STUMPY_STDDEV_THRESHOLD
            ) or D < config.STUMPY_D_SQUARED_THRESHOLD:
                D = 0

        # Ignore subsequences in the exclusion zone
        if (i <= zone_stop and i >= zone_start):
            D = np.inf

        dist_vec[j] = D


def chkpt_write(dpath: Path, params_str, device_id, i, profile, indices,
                device_QT_even, device_QT_odd, range_start, t_elapsed_hr):
    """ Save distance and index profile to checkpointing file

    The density is saved periodically (see QSMP_CHECKPOINT_PERIOD in config.py) in case the job gets killed (SIGTERM or preemption in SLURM).
    """

    fname = f'GPU-{device_id}_{params_str}_chkpt.npz'
    fpath = dpath.joinpath(fname)

    profile = cupy.asnumpy(profile)
    indices = cupy.asnumpy(indices)
    QT_even = device_QT_even.copy_to_host()
    QT_odd = device_QT_odd.copy_to_host()

    with fpath.open('wb') as f:
        np.savez(f, range_start=i+1, profile=profile,
                 indices=indices, QT_even=QT_even, QT_odd=QT_odd)

    print(f'GPU #{device_id}\n'
          f'{t_elapsed_hr:.3g} hours elapsed. Checkpointing...\n'
          f'i = {i}, range_start = {range_start}', flush=True)

def chkpt_read(dpath: Path, params_str, device_id, range_start, k, n_bw):
    """ Read distance and index profile from checkpointing file """

    fname = f'GPU-{device_id}_{params_str}_chkpt.npz'
    fpath = dpath.joinpath(fname)
    if fpath.is_file():
        old_start = range_start
        with np.load(fpath) as data:
            range_start = data['range_start']
            profile = data['profile']
            indices = data['indices']
            QT_even = data['QT_even']
            QT_odd = data['QT_odd']

        print(f'Checkpoint found for GPU #{device_id}\n'
              f'Previous start: {old_start}\n'
              f'New start: {range_start}', flush=True)
    else:
        profile = np.full((k, n_bw), np.inf)  # float64
        indices = np.full((k, n_bw), -1, dtype=np.int64)  # int64
        QT_even = np.full(0, 0)
        QT_odd = np.full(0, 0)

    return range_start, profile, indices, QT_even, QT_odd


def chkpt_clean(dpath: Path, params_str, device_id):
    """ Remove checkpointing file """

    fname = f'GPU-{device_id}_{params_str}_chkpt.npz'
    fpath = dpath.joinpath(fname)
    if fpath.is_file():
        fpath.unlink()

@cuda.jit
def _min_argmin_kernel(dist_vec, mindist, mindist_idx, minfilt_size):
    i = cuda.grid(1)
    # stride = cuda.gridsize(1) XXX: useful with multiple GPUS??

    prev_min = math.inf
    sz = dist_vec.size
    if i < sz:
        for j in range(max(0,i-minfilt_size//2), min(sz, i+minfilt_size//2)):
            #XXX: multiple threads reading from the same memory. Is this a
            # problem?
            if dist_vec[j] < prev_min:
                mindist_idx[i] = j
                mindist[i] = dist_vec[j]
                prev_min = dist_vec[j]

def _update_NNdist_and_NNdindex(
    i, mindist, density, profile, indices):
    # if i == 0:
    #     set_trace()
    n_sigmas = density.shape[1]
    mindist_cp = cupy.asarray(mindist)
    for sigma_idx in range(n_sigmas): #XXX: vectorize???
        higher_density = cupy.nonzero(
            density[:, sigma_idx] > density[i, sigma_idx])[0]
        if higher_density.size > 0:
            j_min = cupy.argmin(mindist_cp[higher_density])
            j_min = higher_density[j_min]
            profile[i, sigma_idx] = mindist_cp[j_min]
            indices[i, sigma_idx] = j_min

def _gpu_qsmp(
    T_fname,
    m,
    minfilt_size,
    density,
    splice,
    range_stop,
    excl_zone,
    M_T_fname,
    Σ_T_fname,
    QT_fname,
    QT_first_fname,
    k,
    dpath,
    params_str,
    range_start=1,
    device_id=0,
):
    """
    A Numba CUDA version of STOMP for parallel computation of the
    Quick Shift Matrix Profile (QSMP), and QSMP indices.

    Parameters
    ----------
    T_fname : str
        The file name for the time series or sequence for which to compute
        the matrix profile
    m : int
        Window size
    minfilt_size: int
        Lenght of min filter. This filter is used to compute a shift-invariant
        distance vector from the distance profile of each subsequence.
    density : ndarray
        Density of subsequences of length m in T. density.shape(n, k), with `k`
        being the number of density estimates. A different tuple of distances
        and indices is computed for each density estimate. `n` is the number of
        subsequences.
    splice : numpy.ndarray
        If not None, T is the concatenation of multiple smaller time
        series (segments), and `splice` has the start indices of the
        second to the last segment, in the order of concatenation (from
        left to right).
    range_stop : int
        The index value along T_B for which to stop the matrix profile
        calculation. This parameter is here for consistency with the
        distributed `stumped` algorithm.
    excl_zone : int
        The half width for the exclusion zone relative to the current
        sliding window
    M_T_fname : str
        The file name for the sliding mean of time series, `T`
    Σ_T_fname : str
        The file name for the sliding standard deviation of time series, `T`
    QT_fname : str
        The file name for the dot product between some query sequence,`Q`, and time series, `T`
    QT_first_fname : str
        The file name for the QT for the first window relative to the
        current sliding window
    k : int
        The total number of sliding windows to iterate over
    dpath: Path
        Absolute path to folder where checkpointing files are to be saved
    params_str: str
        A string to be used in naming the checkpointing files
    range_start : int
        The starting index value along T for which to start the distance
        and index calculation. Default is 1.
    device_id : int
        The (GPU) device number to use. The default value is `0`.

    Returns
    -------
    profile_fname : str
        The file name for the quick shift matrix profile

    indices_fname : str
        The file name for the QSMP indices.
    """
    threads_per_block = config.STUMPY_THREADS_PER_BLOCK
    blocks_per_grid = math.ceil(k / threads_per_block)

    T = np.load(T_fname, allow_pickle=False)
    QT = np.load(QT_fname, allow_pickle=False)
    QT_first = np.load(QT_first_fname, allow_pickle=False)
    M_T = np.load(M_T_fname, allow_pickle=False)
    Σ_T = np.load(Σ_T_fname, allow_pickle=False)
    mindist = np.full(k, np.inf)
    mindist_idx = np.zeros(k, dtype=int)

    # XXX: Parent function adds a 1 to `start`, and then substracts that 1
    # here. This is probably unnecesary and might cause confusion. The 1 added
    # in the else clause follows the same behaviour.
    n_sigmas = density.shape[1]
    new_range_start, profile, indices, QT_even, QT_odd = chkpt_read(
        dpath, params_str, device_id, range_start, k, n_sigmas)
    if new_range_start == range_start:
        QT_odd = QT
        QT_even = QT
        compute_QT = False
    else:
        compute_QT = True
        range_start = new_range_start + 1

    #XXX: Use https://github.com/rapidsai/rmm ???
    with cuda.gpus[device_id]:

        dist_vec = np.full(k, np.inf)  # float64

        for i in splice:
            profile[i-m+1:i, :] = 0
            splice_ind = np.arange(i-m+1, i)[:, None]
            indices[i-m+1:i, :] = np.repeat(splice_ind, n_sigmas, axis=1)

        device_T = cuda.to_device(T)
        device_QT_odd = cuda.to_device(QT_odd)
        device_QT_even = cuda.to_device(QT_even)
        device_QT_first = cuda.to_device(QT_first)
        device_M_T = cuda.to_device(M_T)
        device_Σ_T = cuda.to_device(Σ_T)
        device_dist_vec = cuda.to_device(dist_vec)
        device_mindist = cuda.to_device(mindist)
        device_mindist_idx = cuda.to_device(mindist_idx)
        density_cp = cupy.asarray(density)
        profile_cp = cupy.asarray(profile)
        indices_cp = cupy.asarray(indices)

        _compute_and_update_dist_kernel[blocks_per_grid, threads_per_block](
            range_start - 1,
            device_T,
            m,
            device_QT_even,
            device_QT_odd,
            device_QT_first,
            device_M_T,
            device_Σ_T,
            k,
            excl_zone,
            device_dist_vec,
            compute_QT,
        )

        _min_argmin_kernel[blocks_per_grid, threads_per_block](
            device_dist_vec, device_mindist, device_mindist_idx, minfilt_size)
        _update_NNdist_and_NNdindex(
            range_start-1, device_mindist, density_cp, profile_cp, indices_cp)

        t_elapsed_hr = 0
        tot_elapsed_hr = 0
        for i in range(range_start, range_stop):
            if i == 13:
                st_rng = nvtx.start_range("profile and indices", color="green")
            elif i == 14:
                nvtx.end_range(st_rng)
            t_start = perf_counter()
            _compute_and_update_dist_kernel[blocks_per_grid, threads_per_block](
                i,
                device_T,
                m,
                device_QT_even,
                device_QT_odd,
                device_QT_first,
                device_M_T,
                device_Σ_T,
                k,
                excl_zone,
                device_dist_vec,
                True,
            )
            if i % 10000 == 0:
                print(f'=== {i}/{range_stop} ===')

            _min_argmin_kernel[blocks_per_grid, threads_per_block](
                device_dist_vec, device_mindist,
                device_mindist_idx, minfilt_size
            )
            _update_NNdist_and_NNdindex(
                i, device_mindist, density_cp, profile_cp, indices_cp
            )

            t_stop = perf_counter()
            t_elapsed_hr += (t_stop - t_start)/3600
            if t_elapsed_hr > config.QSMP_CHECKPOINT_PERIOD:
                tot_elapsed_hr += t_elapsed_hr
                t_elapsed_hr = 0
                chkpt_write(dpath, params_str, device_id, i, profile_cp,
                            indices_cp, device_QT_even, device_QT_odd, range_start, tot_elapsed_hr)

        chkpt_clean(dpath, params_str, device_id)

        profile = cupy.asnumpy(profile_cp)
        indices = cupy.asnumpy(indices_cp)
        profile = np.sqrt(profile)

        profile_fname = core.array_to_temp_file(profile)
        indices_fname = core.array_to_temp_file(indices)

    return profile_fname, indices_fname


def gpu_qsmp(
    T, m, minfilt_size, density, dpath, params_str, splice=None, device_id=0):
    """
    Compute the z-normalized Quick Shift Matrix Profile with one or more
    GPU devices.

    This is a convenience wrapper around the Numba `cuda.jit`
    `_gpu_qsmp` function which computes the QSMP according to GPU-STOMP.

    Parameters
    ----------
    T : ndarray
        The time series or sequence for which to compute the QSMP
    m : int
        Window size
    minfilt_size: int
        Lenght of min filter. This filter is used to compute a shift-invariant
        distance vector from the distance profile of each subsequence.
    density : ndarray
        Density of subsequences of length m in T
    dpath: Path
        Absolute path to folder where checkpointing files are to be saved
    params_str: str
        A string to be used in naming the checkpointing files
    transform: None or string
        Transform to be applied to either the distances or the time series. If 'fwhm', scale the distances by the FWHM of the autocorrelation of the subsequences. If 'whiten', build a whitening filter from the average PSD of the data and apply the filter to de-emphasize low frequencies and emphasize high frequencies.
    splice : numpy.ndarray
        If not None, T is the concatenation of multiple smaller time
        series (segments), and `splice` has the start indices of the
        second to the last segment, in the order of concatenation (from
        left to right).
    device_id : int or list, default 0
        The (GPU) device number to use. The default value is `0`. A list of
        valid device ids (int) may also be provided for parallel GPU-STUMP
        computation. A list of all valid device ids can be obtained by
        executing `[device.id for device in numba.cuda.list_devices()]`.

    Returns
    -------
    profile : ndarray
        The nearest-neighbor distances
    indices : ndarray
        The nearest-neighbor indices
    """

    # Create a 0-dimensional array if splice is None. This is needed to avoid
    # the error: "CudaAPIError: [1] Call to cuLaunchKernel results in
    # CUDA_ERROR_INVALID_VALUE".
    if splice is None:
        splice = np.full(0, 0)

    T, M_T, Σ_T = core.preprocess(T, m)

    if T.ndim != 1:  # pragma: no cover
        raise ValueError(
            f"T is {T.ndim}-dimensional and must be 1-dimensional. "
        )

    core.check_window_size(m, max_size=T.shape[0])

    k = T.shape[0] - m + 1
    excl_zone = int(
        np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM)
    )  # See Definition 3 and Figure 3

    T_fname = core.array_to_temp_file(T)
    M_T_fname = core.array_to_temp_file(M_T)
    Σ_T_fname = core.array_to_temp_file(Σ_T)

    if isinstance(device_id, int):
        device_ids = [device_id]
    else:
        device_ids = device_id

    profile = [None] * len(device_ids)
    indices = [None] * len(device_ids)

    for _id in device_ids:
        with cuda.gpus[_id]:
            if (
                cuda.current_context().__class__.__name__ != "FakeCUDAContext"
            ):  # pragma: no cover
                cuda.current_context().deallocations.clear()

    step = 1 + k // len(device_ids)

    # Start process pool for multi-GPU request
    if len(device_ids) > 1:  # pragma: no cover
        mp.set_start_method("spawn", force=True)
        p = mp.Pool(processes=len(device_ids))
        results = [None] * len(device_ids)

    QT_fnames = []
    QT_first_fnames = []

    for idx, start in enumerate(range(0, k, step)):
        stop = min(k, start + step)

        QT, QT_first = core._get_QT(start, T, T, m)
        QT_fname = core.array_to_temp_file(QT)
        QT_first_fname = core.array_to_temp_file(QT_first)
        QT_fnames.append(QT_fname)
        QT_first_fnames.append(QT_first_fname)

        if len(device_ids) > 1 and idx < len(device_ids) - 1:  # pragma: no cover
            # Spawn and execute in child process for multi-GPU request
            results[idx] = p.apply_async(
                _gpu_qsmp,
                (
                    T_fname,
                    m,
                    minfilt_size,
                    density,
                    splice,
                    stop,
                    excl_zone,
                    M_T_fname,
                    Σ_T_fname,
                    QT_fname,
                    QT_first_fname,
                    k,
                    dpath,
                    params_str,
                    start + 1,
                    device_ids[idx],
                ),
            )
        else:
            # Execute last chunk in parent process
            # Only parent process is executed when a single GPU is requested
            profile[idx], indices[idx] = _gpu_qsmp(
                T_fname,
                m,
                minfilt_size,
                density,
                splice,
                stop,
                excl_zone,
                M_T_fname,
                Σ_T_fname,
                QT_fname,
                QT_first_fname,
                k,
                dpath,
                params_str,
                start + 1,
                device_ids[idx],
            )

    # Clean up process pool for multi-GPU request
    if len(device_ids) > 1:  # pragma: no cover
        p.close()
        p.join()

        # Collect results from spawned child processes if they exist
        for idx, result in enumerate(results):
            if result is not None:
                profile[idx], indices[idx] = result.get()

    os.remove(T_fname)
    os.remove(M_T_fname)
    os.remove(Σ_T_fname)
    for QT_fname in QT_fnames:
        os.remove(QT_fname)
    for QT_first_fname in QT_first_fnames:
        os.remove(QT_first_fname)

    for idx in range(len(device_ids)):
        profile_fname = profile[idx]
        indices_fname = indices[idx]
        profile[idx] = np.load(profile_fname, allow_pickle=False)
        indices[idx] = np.load(indices_fname, allow_pickle=False)
        os.remove(profile_fname)
        os.remove(indices_fname)

    n_density = density.shape[1]
    for i in range(1, len(device_ids)):
        # Update all matrix profiles and matrix profile indices
        # (global, left, right) and store in profile[0] and indices[0]
        for ic in range(n_density):
            cond = profile[0][:, ic] < profile[i][:, ic]
            profile[0][:, ic] =\
                np.where(cond, profile[0][:, ic], profile[i][:, ic])
            indices[0][:, ic] =\
                np.where(cond, indices[0][:, ic], indices[i][:, ic])

    threshold = 10e-6
    if core.are_distances_too_small(profile[0], threshold=threshold):  # pragma: no cover
        logger.warning(f"A large number of values are smaller than {threshold}.")

    return profile[0], indices[0]
