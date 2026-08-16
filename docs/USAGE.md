# Running QSMP on new data

The minimal API for arbitrary 1D time series:

```python
from pathlib import Path
import numpy as np
from numba import cuda

from qsmp.gpu_density import gpu_density
from qsmp.gpu_qsmp import gpu_qsmp
from qsmp.utils import windows
from qsmp.utils.utils import fix_root
from qsmp import tree

# Your data
T = np.load("my_signal.npy").astype(np.float64)   # 1D
splice = None                                     # or an array of segment-boundary indices

# Parameters
m = 512                       # subsequence length
sigma = np.array([1.0, 2.0])  # kernel bandwidth(s) — can pass several at once
minfilt_size = m // 4         # min-pooling filter length B (≈ exclusion zone)
device_ids = [d.id for d in cuda.list_devices()]

# Optional: rectangular window to penalise off-centred patterns
win = windows.get_window("rect")(m, support_frac=0.5)

dpath = Path(".")             # where checkpoints are written
params_str = "myrun"

# 1. Density estimate (Algorithm 1)
T, splice, density = gpu_density(
    T, m, sigma, dpath, params_str,
    splice=splice, window=win, device_id=device_ids,
)

# 2. Shift-invariant NN-distance + NN-index (Algorithm 2)
profile, indices = gpu_qsmp(
    T, m, minfilt_size, density, dpath, params_str,
    splice=splice, device_id=device_ids,
)

# 3. Make global modes self-rooting
profile, indices, density = fix_root((profile, indices, density))

# 4. Cut the tree post-hoc at any tau to extract modes
i_sigma = 0
tau = np.quantile(profile[:, i_sigma], 0.999)   # heuristic
NNd, NNi, modes, cluster_size = tree.tree2clusters(
    m, density[:, i_sigma], indices[:, i_sigma], profile[:, i_sigma], tau,
)

# 5. (Optional) Find tau that yields exactly k modes
tau_k = tree.find_tau(k=6, subseq_len=m,
                      density=density[:, i_sigma],
                      NNindex=indices[:, i_sigma],
                      NNdist=profile[:, i_sigma])
```

## Output format

The `.npz` files written by the scripts contain:

| Key       | Shape       | Meaning                                                     |
|-----------|-------------|-------------------------------------------------------------|
| `T`       | `(n,)`      | The 1D time series (z-normalisation is done internally)     |
| `splice`  | `(M,)`      | Segment-boundary indices (empty if not segmented)           |
| `density` | `(N, n_σ)`  | Kernel density estimate per subsequence, per kernel width   |
| `profile` | `(N, n_σ)`  | Shift-invariant NN-distance (the QSMP)                      |
| `indices` | `(N, n_σ)`  | NN-index (which higher-density subsequence each one points to) |

with `N = n - m + 1` and `n_σ = len(sigma)`. Together, `(density, profile, indices)`
form the **QS-tuple** defined in §3.2 of the paper.

## Interactive exploration

Once you have a QSMP `.npz` saved, you can explore modes across different
distance thresholds and kernel widths interactively from a Jupyter notebook
without re-running the GPU pipeline:

```python
import numpy as np
from qsmp.viz import apps

with np.load("results/morlet/qsmp_m-512_sigma-0.9_1.0_2.0_rect-50_minfilt-256.npz") as d:
    T, density, NNdist, NNindex = d["T"], d["density"].T, d["profile"].T, d["indices"].T

# Slider over distance thresholds, fixed kernel width
apps.modes_across_maxdist(T, wave_len=512, density=density,
                          NNindex=NNindex, NNdist=NNdist,
                          sigmas=[0, 1, 2])

# Slider over kernel widths, fixed distance threshold
apps.modes_across_sigma(T, wave_len=512, density=density,
                        NNindex=NNindex, NNdist=NNdist,
                        sigmas=[0.9, 1.0, 2.0])
```

The `qsmp.viz.viz` module also exposes the underlying plotting functions
(`show_modes_across_maxdist`, `show_modes_across_sigma`, `show_density`,
`show_segment_start`) for use in non-interactive scripts.

## Checkpointing

`gpu_density` and `gpu_qsmp` periodically write
`GPU-<id>_<params_str>_chkpt.npz` to `dpath` (period controlled by
`config.QSMP_CHECKPOINT_PERIOD`, default 1 hour). If a job is preempted (e.g.
SLURM SIGTERM), restarting it with the same `params_str` resumes from the
last checkpoint.
