# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

"""Numerical thresholds and GPU tunables.

Most ``STUMPY_*`` values are inherited from the upstream STUMPY project
unchanged and cover edge cases of the z-normalised Euclidean distance
formula. The two ``QSMP_*`` values are specific to this project.
"""

import numpy as np

# Threads per CUDA block in `_compute_and_update_dist_kernel` and
# `_compute_and_update_density_kernel`. The grid size is derived as
# `ceil(N / STUMPY_THREADS_PER_BLOCK)`. Tune to match your GPU's warp size
# and shared-memory budget.
STUMPY_THREADS_PER_BLOCK = 512

# Number of chunks used by `compute_mean_std` / `compute_centered_std` to
# stream the rolling mean/std computation when the time series is too large
# to fit a single materialised rolling-window matrix in memory. Doubled
# automatically on `MemoryError`, up to `STUMPY_MEAN_STD_MAX_ITER` retries.
STUMPY_MEAN_STD_NUM_CHUNKS = 1
STUMPY_MEAN_STD_MAX_ITER = 10

# Floor applied to the denominator of the z-normalised Euclidean distance
# (Eq. 4 in the paper), to prevent division by zero on near-constant
# subsequences.
STUMPY_DENOM_THRESHOLD = 1e-14

# A subsequence with rolling standard deviation below this threshold is
# treated as a "flat" subsequence and assigned a fixed distance of `m`
# instead of being z-normalised.
STUMPY_STDDEV_THRESHOLD = 1e-7

# Squared distances below this threshold are clamped to zero. This avoids
# negative-but-near-zero squared distances caused by floating-point
# cancellation in the subtraction step of Eq. 4.
STUMPY_D_SQUARED_THRESHOLD = 1e-14

# Number of decimal places used by the test suite (legacy, from STUMPY).
STUMPY_TEST_PRECISION = 5

# Convenience constants for "infinite" distance values; rarely used directly.
STUMPY_MAX_SQUARED_DISTANCE = np.finfo(np.float64).max
STUMPY_MAX_DISTANCE = np.sqrt(STUMPY_MAX_SQUARED_DISTANCE)

# Half-width of the *exclusion zone* expressed as a fraction of the
# subsequence length: subsequences within `m / STUMPY_EXCL_ZONE_DENOM` of
# each other are considered trivial matches and ignored. The paper uses
# `m/4` (Definition 5 and §3.4) which corresponds to the default value of 4.
STUMPY_EXCL_ZONE_DENOM = 4

# Period (in hours) between writes of the partial GPU state to a checkpoint
# `.npz`. If a SLURM/preemption signal kills the job, restarting it with the
# same `params_str` resumes from the last checkpoint. See
# `gpu_density.chkpt_write` and `gpu_qsmp.chkpt_write`.
QSMP_CHECKPOINT_PERIOD = 1  # hours

# Density values smaller than this are zeroed out after the GPU pass. This
# avoids spurious tiny density bumps influencing the Quick-Shift NN
# selection (Eq. 7).
QSMP_DENSITY_THRESHOLD = 1e-14
