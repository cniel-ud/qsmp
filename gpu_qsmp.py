# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.
import logging
import math
import multiprocessing as mp
import os

import numpy as np
from numba import cuda

import core, config
from time import perf_counter

logger = logging.getLogger(__name__)


#(
#    "(i8, f8[:], i8, f8[:], f8[:], f8[:], f8[:], f8[:], f8[:],"
#    "i8, i8, f8[:, :], i8[:, :], b1)"
#)
@cuda.jit
def _compute_and_update_QI_kernel(
    i,
    T,
    m,
    density,
    QT_even,
    QT_odd,
    QT_first,
    M_T,
    Σ_T,
    k,
    excl_zone,
    profile,
    indices,
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

    density : ndarray
        Density of subsequences of length m in T

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

    profile : ndarray
        Matrix profile. The first column consists of the global matrix profile,
        the second column consists of the left matrix profile, and the third
        column consists of the right matrix profile.

    indices : ndarray
        The first column consists of the matrix profile indices, the second
        column consists of the left matrix profile indices, and the third
        column consists of the right matrix profile indices.

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

        # Ignore neighbors that are trivial (on excl. zone) or if at splice
        if (i <= zone_stop and i >= zone_start) or math.isnan(profile[j, 0]):
            D = np.inf

        n_density = density.shape[1]

        for ic in range(n_density):
            # Ignore neighbors that don't increase density
            if density[i, ic] > density[j, ic]:
                if D < profile[j, ic]:
                    profile[j, ic] = D
                    indices[j, ic] = i


def chkpt_read(dpath, device_id, range_start, k, n_bw):
    # Checkpoint file that saves last `i` processed before SIGTERM
    fname = f'device{device_id}_checkpoint.npz'
    fpath = os.path.join(dpath, fname)
    if os.path.isfile(fpath):
        old_start = range_start
        with np.load(fpath) as data:
            range_start = data['range_start']
            profile = data['profile']
            indices = data['indices']
        print(f'Checkpoint found for GPU #{device_id}\n'
              f'Previous start: {old_start}\n'
              f'New start: {range_start}', flush=True)
    else:
        profile = np.full((k, n_bw), np.inf)  # float64
        indices = np.full((k, n_bw), -1, dtype=np.int64)  # int64

    return range_start, profile, indices


def chkpt_write(dpath, device_id, i, device_profile, device_indices, \
    range_start, t_elapsed_hr):
    # Checkpoint file that saves last `i` processed before SIGTERM
    fname = f'device{device_id}_checkpoint.npz'
    fpath = os.path.join(dpath, fname)

    profile = device_profile.copy_to_host()
    indices = device_indices.copy_to_host()

    with open(fpath, 'wb') as f:
        np.savez(f, range_start=i, profile=profile, indices=indices)

    print(f'GPU #{device_id}\n'
          f'{t_elapsed_hr:.3g} hours elapsed. Checkpointing...\n'
          f'i = {i}, range_start = {range_start}', flush=True)


def chkpt_clean(dpath, device_id):
    """ Remove checkpointing file """
    fname = f'device{device_id}_checkpoint.npz'
    fpath = os.path.join(dpath, fname)

    os.remove(fpath)


def _gpu_qsmp(
    T_fname,
    m,
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

    density : ndarray
        Density of subsequences of length m in T

    splice : numpy.ndarray
        If not None, T is the concatenation of multiple smaller time series
        (segments), and `splice` has the start indices of the second to the
        last segment, in the order of concatenation (from left to right).

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
        The file name for the dot product between some query sequence,`Q`,
        and time series, `T`

    QT_first_fname : str
        The file name for the QT for the first window relative to the current
        sliding window

    k : int
        The total number of sliding windows to iterate over

    dpath: string
        Absolute path to folder where checkpointing files are to be saved

    range_start : int
        The starting index value along T_B for which to start the matrix
        profile calculation. Default is 1.

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

    with cuda.gpus[device_id]:

        range_start, profile, indices = chkpt_read(\
            dpath, device_id, range_start, k, density.shape[1])

        for i in splice:
            profile[i-m+1:i, :] = np.nan
            indices[i-m+1:i, :] = -1


        device_T = cuda.to_device(T)
        device_QT_odd = cuda.to_device(QT)
        device_QT_even = cuda.to_device(QT)
        device_QT_first = cuda.to_device(QT_first)
        device_M_T = cuda.to_device(M_T)
        device_Σ_T = cuda.to_device(Σ_T)
        device_density = cuda.to_device(density)

        device_profile = cuda.to_device(profile)
        device_indices = cuda.to_device(indices)
        _compute_and_update_QI_kernel[blocks_per_grid, threads_per_block](
            range_start - 1,
            device_T,
            m,
            device_density,
            device_QT_even,
            device_QT_odd,
            device_QT_first,
            device_M_T,
            device_Σ_T,
            k,
            excl_zone,
            device_profile,
            device_indices,
            False,
        )

        t_elapsed_hr = 0
        tot_elapsed_hr = 0
        for i in range(range_start, range_stop):
            t_start = perf_counter()
            _compute_and_update_QI_kernel[blocks_per_grid, threads_per_block](
                i,
                device_T,
                m,
                device_density,
                device_QT_even,
                device_QT_odd,
                device_QT_first,
                device_M_T,
                device_Σ_T,
                k,
                excl_zone,
                device_profile,
                device_indices,
                True,
            )
            t_stop = perf_counter()
            t_elapsed_hr += (t_stop - t_start)/3600
            if t_elapsed_hr > config.QSMP_CHECKPOINT_PERIOD:
                tot_elapsed_hr += t_elapsed_hr
                t_elapsed_hr = 0
                chkpt_write(dpath, device_id, i, device_profile, device_indices,
                            range_start, tot_elapsed_hr)

        chkpt_clean(dpath, device_id)

        profile = device_profile.copy_to_host()
        indices = device_indices.copy_to_host()
        profile = np.sqrt(profile)

        profile_fname = core.array_to_temp_file(profile)
        indices_fname = core.array_to_temp_file(indices)

    return profile_fname, indices_fname


def gpu_qsmp(T, m, density, dpath, splice=None, device_id=0):
    """
    Compute the z-normalized Quick Shift Matrix Profile with one or more GPU
    devices

    This is a convenience wrapper around the Numba `cuda.jit` `_gpu_qsmp` function which computes the QSMP according to GPU-STOMP.

    Parameters
    ----------
    T : ndarray
        The time series or sequence for which to compute the QSMP

    m : int
        Window size

    density : ndarray
        Density of subsequences of length m in T

    splice : numpy.ndarray
         If not None, T is the concatenation of multiple smaller time series
         (segments), and `splice` has the start indices of the second to the
         last segment, in the order of concatenation (from left to right).

    device_id : int or list, default 0
        The (GPU) device number to use. The default value is `0`. A list of
        valid device ids (int) may also be provided for parallel GPU-STUMP
        computation. A list of all valid device ids can be obtained by
        executing `[device.id for device in numba.cuda.list_devices()]`.

    Returns
    -------
    out : ndarray
        The first column consists of the QSMP, the second column
        consists of the QSMP indices.
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
