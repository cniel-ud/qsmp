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
from time import perf_counter

import numpy as np
from numba import cuda


from qsmp import core, config


logger = logging.getLogger(__name__)

@cuda.jit
def _compute_and_update_density_kernel(
    i,
    T,
    m,
    sigma,
    splice,
    QT_even,
    QT_odd,
    QT_first,
    M_T,
    Σ_T,
    centeredness,
    fwhm,
    k,
    excl_zone,
    density,
    compute_QT,
    ):
    """
    A Numba CUDA kernel to update the density
    Parameters
    ----------
    i : int
        sliding window `i`
    T : numpy.ndarray
        The time series or sequence for which to compute the dot product
    m : int
        Window size
    sigma : numpy.ndarray
        Standard deviation of the Gaussian kernel used to estimate the density.
    splice : numpy.ndarray
         If not None, T is the concatenation of multiple smaller time series
         (segments), and `splice` has the start indices of the second to the
         last segment, in the order of concatenation (from left to right).
    window: numpy.ndarray
        A window to give more weight to patterns centered in the sliding window
        during computation of the density.
    QT_even : numpy.ndarray
        The input QT array (dot product between the query sequence,`Q`, and
        time series, `T`) to use when `i` is even
    QT_odd : numpy.ndarray
        The input QT array (dot product between the query sequence,`Q`, and
        time series, `T`) to use when `i` is odd
    QT_first : numpy.ndarray
        Dot product between the first query sequence,`Q`, and time series, T`
    M_T : numpy.ndarray
        Sliding mean of time series, `T`
    Σ_T : numpy.ndarray
        Sliding standard deviation of time series, `T`
    centeredness:numpy.ndarray
        Centeredness of each subsequence. This is meant to penalize patterns that are not centered.
    fwhm : numpy.ndarray
        FWHM of the autocorrelation function of each subsequence
    k : int
        The total number of sliding windows to iterate over
    excl_zone : int
        The half width for the exclusion zone relative to the current
        sliding window
    density : numpy.ndarray
        Density of subsequences of length m in time series T
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

        # Ignore subsequences in the exclusion zone or that are at one splice
        is_in_splice = core.is_in_splice(splice, m, j)
        if (i <= zone_stop and i >= zone_start) or is_in_splice:
            D = np.inf

        # if start == 1263: #XXX: Debugging..
        #     set_trace()

        if centeredness.size > 0:
            D = D / centeredness[j]
            if fwhm.size > 0 and centeredness[j] < max(fwhm[j], fwhm[i]):
                    D = D * max(fwhm[j], fwhm[i])
        elif fwhm.size > 0:
            D = D * max(fwhm[j], fwhm[i])

        n_sigma = sigma.size
        for ic in range(n_sigma):
            P = math.exp(-0.5*D/sigma[ic]**2)
            density[j, ic] = density[j, ic] + P



def chkpt_write(dpath:Path, device_id, i, device_density, range_start, t_elapsed_hr):
    """ Save density to checkpointing file

    The density is saved periodically (see QSMP_CHECKPOINT_PERIOD in config.py) in case the job gets killed (SIGTERM or preemption in SLURM).
    """
    # Checkpoint file that saves last `i` processed before SIGTERM
    fname = f'device{device_id}_checkpoint.npz'
    fpath = dpath.joinpath(fname)

    density = device_density.copy_to_host()

    with fpath.open('wb') as f:
        np.savez(f, range_start=i, density=density)

    print(f'GPU #{device_id}\n'
          f'{t_elapsed_hr:.3g} hours elapsed. Checkpointing...\n'
          f'i = {i}, range_start = {range_start}', flush=True)

def chkpt_read(dpath:Path, device_id, range_start, k, n_bw):
    """ Read density from checkpointing file """

    fname = f'device{device_id}_checkpoint.npz'
    fpath = dpath.joinpath(fname)
    if fpath.is_file():
        old_start = range_start
        with np.load(fpath) as data:
            range_start = data['range_start']
            density = data['density']
        print(f'Checkpoint found for GPU #{device_id}\n'
              f'Previous start: {old_start}\n'
              f'New start: {range_start}', flush=True)
    else:
        density = np.zeros((k, n_bw))  # float64

    return range_start, density


def chkpt_clean(dpath:Path, device_id):
    """ Remove checkpointing file """
    fname = f'device{device_id}_checkpoint.npz'
    fpath = dpath.joinpath(fname)
    if fpath.is_file():
        os.remove(fpath)

def _gpu_density(
    T_fname,
    m,
    sigma,
    splice,
    range_stop,
    excl_zone,
    M_T_fname,
    Σ_T_fname,
    centeredness_fname,
    fwhm_fname,
    QT_fname,
    QT_first_fname,
    k,
    dpath,
    range_start=1,
    device_id=0,
):
    """
    A Numba CUDA version of STOMP for parallel computation of the
    density of subsequences of the time series T.

    Parameters
    ----------
    T_fname : str
        The file name for the time series or sequence for which to compute
        the density
    m : int
        Window size
    sigma : numpy.ndarray
        Standard deviation of the Gaussian kernel used to estimate the density.
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
    centeredness_fname: str
        The file name for the centeredness of each patttern. This is meant to
        be used to penalize patterns are not centered (i.e., reduce their
        density).
    fwhm_fname : str
        The file name for the FWHM of the autocorrelation of the subsequences
    QT_fname : str
        The file name for the dot product between some query sequence,`Q`, and time series, `T`
    QT_first_fname : str
        The file name for the QT for the first window relative to the
        current sliding window
    k : int
        The total number of sliding windows to iterate over
    dpath: string
        Absolute path to folder where checkpointing files are to be saved
    range_start : int
        The starting index value along T for which to start the density
        calculation. Default is 1.
    device_id : int
        The (GPU) device number to use. The default value is `0`.

    Returns
    -------
    density_fname : str
        The file name for the density
    """
    threads_per_block = config.STUMPY_THREADS_PER_BLOCK
    blocks_per_grid = math.ceil(k / threads_per_block)

    T = np.load(T_fname, allow_pickle=False)
    QT = np.load(QT_fname, allow_pickle=False)
    QT_first = np.load(QT_first_fname, allow_pickle=False)
    M_T = np.load(M_T_fname, allow_pickle=False)
    Σ_T = np.load(Σ_T_fname, allow_pickle=False)
    centeredness = np.load(centeredness_fname, allow_pickle=False)
    fwhm = np.load(fwhm_fname, allow_pickle=False)

    n_sigma = sigma.size

    with cuda.gpus[device_id]:

        range_start, density = chkpt_read(
            dpath, device_id, range_start, k, n_sigma)
        device_T = cuda.to_device(T)
        device_QT_odd = cuda.to_device(QT)
        device_QT_even = cuda.to_device(QT)
        device_QT_first = cuda.to_device(QT_first)
        device_M_T = cuda.to_device(M_T)
        device_Σ_T = cuda.to_device(Σ_T)
        device_centeredness = cuda.to_device(centeredness)
        device_fwhm = cuda.to_device(fwhm)
        device_splice = cuda.to_device(splice)
        device_sigma = cuda.to_device(sigma)

        device_density = cuda.to_device(density)
        _compute_and_update_density_kernel[blocks_per_grid, threads_per_block](
            range_start - 1,
            device_T,
            m,
            device_sigma,
            device_splice,
            device_QT_even,
            device_QT_odd,
            device_QT_first,
            device_M_T,
            device_Σ_T,
            device_centeredness,
            device_fwhm,
            k,
            excl_zone,
            device_density,
            False,
        )
        # set_trace()

        t_elapsed_hr = 0
        tot_elapsed_hr = 0
        for i in range(range_start, range_stop):
            if i == 13:
                st_rng = nvtx.start_range("density", color="blue")
            elif i == 14:
                nvtx.end_range(st_rng)

            t_start = perf_counter()
            _compute_and_update_density_kernel[blocks_per_grid, threads_per_block](
                i,
                device_T,
                m,
                device_sigma,
                device_splice,
                device_QT_even,
                device_QT_odd,
                device_QT_first,
                device_M_T,
                device_Σ_T,
                device_centeredness,
                device_fwhm,
                k,
                excl_zone,
                device_density,
                True,
            )
            if i % 10000 == 0:
                print(f'=== {i}/{range_stop} ===')

            # set_trace()
            t_stop = perf_counter()
            t_elapsed_hr += (t_stop - t_start)/3600
            if t_elapsed_hr > config.QSMP_CHECKPOINT_PERIOD:
                tot_elapsed_hr += t_elapsed_hr
                t_elapsed_hr = 0
                chkpt_write(dpath, device_id, i, device_density,
                            range_start, tot_elapsed_hr)

        chkpt_clean(dpath, device_id)
        density = device_density.copy_to_host()

        density_fname = core.array_to_temp_file(density)

    return density_fname


def gpu_density(T, m, sigma, dpath, transform=None,
                splice=None, window=None, device_id=0):
    """
    Estimate the density of subsequences of the z-normalized matrix
    profile with one or more GPU devices.

    This is a convenience wrapper around the Numba `cuda.jit`
    `_gpu_density` function which computes the density at each
    subsequence according to GPU-STOMP.

    Parameters
    ----------
    T : numpy.ndarray
        The time series or sequence for which to compute the matrix profile
    m : int
        Window size
    sigma : numpy.ndarray
        Standard deviation of the Gaussian kernel used to estimate the density.
    dpath: string
        Absolute path to folder where checkpointing files are to be saved
    transform: None or string
        Transform to be applied to either the distances or the time series. If 'fwhm', scale the distances by the FWHM of the autocorrelation of the subsequences. If 'whiten', build a whitening filter from the average PSD of the data and apply the filter to de-emphasize low frequencies and emphasize high frequencies.
    splice : numpy.ndarray(dtype=uint64)
        If not None, T is the concatenation of multiple smaller time
        series (segments), and `splice` has the start indices of the
        second to the last segment, in the order of concatenation (from
        left to right).
    window: numpy.ndarray
        A window to compute standard deviation in the center of the subsequence.
    device_id : int or list, default 0
        The (GPU) device number to use. The default value is `0`. A list of
        valid device ids (int) may also be provided for parallel GPU-STUMP
        computation. A list of all valid device ids can be obtained by
        executing `[device.id for device in numba.cuda.list_devices()]`.

    Returns
    -------
    density : numpy.ndarray
        The density.
    """

    # Create a 0-dimensional array if splice is None. This is needed to avoid
    # the error: "CudaAPIError: [1] Call to cuLaunchKernel results in
    # CUDA_ERROR_INVALID_VALUE".
    if splice is None:
        splice = np.full(0, 0)

    if transform == 'whiten':
        fs, n_taps = 512, 1001
        f, Px_mean = core.mean_PSD(T, splice)
        _, coeffs = core.whitening_filter(
            f, Px_mean, n_taps=n_taps, fs=fs)
        grp_delay = core.get_group_delay(coeffs, f, fs=fs)
        T, splice = core.whiten(T, splice, coeffs, grp_delay)

    T, M_T, Σ_T = core.preprocess(T, m)
    if window is not None:
        Σ_centered_T = core.compute_centered_std(T, window)
        centeredness = Σ_centered_T/Σ_T
        centeredness = centeredness/centeredness.max()
    else:
        centeredness = np.full(0, 0)

    if transform == 'fwhm':
        #XXX: this can take quite some time. Should we compute it in the GPU?
        fwhm = core.fwhm(core.ndxcorr(T, m, splice))
        if splice.size > 0:
            fwhm = core.fill_fwhm(fwhm, splice, m)
    else:
        fwhm = np.full(0, 0)

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
    centeredness_fname = core.array_to_temp_file(centeredness)
    fwhm_fname = core.array_to_temp_file(fwhm)

    if isinstance(device_id, int):
        device_ids = [device_id]
    else:
        device_ids = device_id

    density = [None] * len(device_ids)

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

    # Asynchronously call _gpu_density() to compute the density over a
    # range of values for i (the index of the query). Each range is
    # processed by a different GPU.
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
                _gpu_density,
                (
                    T_fname,
                    m,
                    sigma,
                    splice,
                    stop,
                    excl_zone,
                    M_T_fname,
                    Σ_T_fname,
                    centeredness_fname,
                    fwhm_fname,
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
            density[idx] = _gpu_density(
                T_fname,
                m,
                sigma,
                splice,
                stop,
                excl_zone,
                M_T_fname,
                Σ_T_fname,
                centeredness_fname,
                fwhm_fname,
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
                density[idx] = result.get()

    os.remove(T_fname)
    os.remove(M_T_fname)
    os.remove(Σ_T_fname)
    for QT_fname in QT_fnames:
        os.remove(QT_fname)
    for QT_first_fname in QT_first_fnames:
        os.remove(QT_first_fname)

    for idx in range(len(device_ids)):
        density_fname = density[idx]
        density[idx] = np.load(density_fname, allow_pickle=False)
        os.remove(density_fname)

    # If multiple GPUs requested, aggregate density results from all the devices
    if len(device_ids) > 1:
        # asumme densities are columns in density[i]
        density[0] = np.sum(density, axis=0)

    density[0][density[0] < config.QSMP_DENSITY_THRESHOLD] = 0

    threshold = 10e-6
    if core.are_distances_too_small(density[0], threshold=threshold):  # pragma: no cover
        logger.warning(f"A large number of values are smaller than {threshold}.")

    return T, splice, density[0]
