# Known issues

A running list of code-quality and compatibility issues that are not yet
fixed. None of these block the documented reproducibility flow, but they
will affect anyone who upgrades the dependency pins or wants to run on a
recent Python/NumPy/SciPy.

## Compatibility — will break on modern dependencies

These will throw at runtime on the modern (verified-importable) dep set
listed in [`README.md`](../README.md#installation): `numpy>=1.26`,
`scipy>=1.12`, `numba>=0.61`.

### `np.int` is removed in NumPy 1.20+

`np.int` was deprecated in NumPy 1.20 and removed in 1.24. Replace with the
Python builtin `int`.

| File | Line | Snippet |
|---|---|---|
| `qsmp/tree.py` | 210 | `max_num_of_rows = np.int(max_chunk_size/nbytes_per_wave)` |
| `qsmp/tree.py` | 221 | `n_chunks = np.int(n_children/max_num_of_rows) + 1` |
| `qsmp/utils/pltaux.py` | 79 | `nrows = np.int(nwav/ncols)` |
| `qsmp/shift_kmeans/wrappers.py` | 72 | `dtype=np.int` (inside `np.empty(...)`) — use `dtype=int` or `np.int64` |
| `qsmp/shift_kmeans/datasets/generator.py` | 19, 20, 21 | three uses around `data_size`, `num_training_samples`, `num_validation_samples` |

### `scipy.ndimage.filters` is removed in SciPy 1.12

`from scipy.ndimage.filters import maximum_filter1d, minimum_filter1d` in
`qsmp/core.py:13` should be `from scipy.ndimage import maximum_filter1d,
minimum_filter1d`. The submodule was deprecated in 1.10 and removed in 1.12.

### `requirements.txt` pins are stale for modern Python

The pins (`numpy==1.17.4`, `cupy==11.2.0`, `numba==0.56.4`, ...) work on the
original Python 3.9.13 environment but won't all install cleanly on Python
3.10+. Verified-working modern alternatives that import the library are:

- `numpy>=1.26`
- `numba>=0.61`
- `scipy>=1.11` (note: drops `scipy.ndimage.filters`, see below)
- `h5py>=3.16`
- `cupy-cuda11x` matching your CUDA toolkit

A pin refresh — possibly using `uv pip compile` — would let new contributors
install on a current toolchain without trial and error.

Also: `webunit==1.3.10` appears unused (no imports found in the codebase).

## Cleanup — non-functional, low priority

### Hardcoded NVTX profiling markers

Both `gpu_density.py` and `gpu_qsmp.py` start an NVTX range on iteration 13
and end it on iteration 14:

```python
if i == 13:
    st_rng = nvtx.start_range("density", color="blue")
elif i == 14:
    nvtx.end_range(st_rng)
```

Locations: `qsmp/gpu_density.py:351-353` and `qsmp/gpu_qsmp.py:370-372`.

This is leftover scaffolding from a profiling session. A nicer pattern is to
gate it behind an env var (e.g. `QSMP_PROFILE_NVTX=1`) so the capability is
preserved but doesn't fire on hardcoded iterations during regular runs.
Iteration 13 is also unreachable when `range_start > 13` (e.g. resuming
from a checkpoint), which is another reason to remove the magic number.

### Commented-out `set_trace()` calls

Debugger leftovers that should be removed:

- `qsmp/gpu_density.py:345` — `# set_trace()`
- `qsmp/gpu_density.py:377` — `# set_trace()`
- `qsmp/gpu_qsmp.py:94` — `# if i==2 and j==2:` / `#     set_trace()`
- `qsmp/gpu_qsmp.py:208` — `# if i == 0:` / `#     set_trace()`

### `transform='whiten'` and `transform='fwhm'` in `gpu_density`

The `gpu_density()` driver supports two `transform` modes — `'whiten'`
(builds a whitening filter from the average PSD and applies it before the
density estimate) and `'fwhm'` (scales distances by the FWHM of each
subsequence's autocorrelation). Neither appears in the paper. They're
experimental and have not been validated end-to-end against a published
result.

If you keep them, the docstring should call them out as experimental. If
you remove them, you can also drop `core.mean_PSD`, `core.whitening_filter`,
`core.get_group_delay`, `core.whiten`, `core.whiten_alignment`,
`core.fwhm`, `core.ndxcorr`, and `core.fill_fwhm` from `core.py`.

### `tree.merge_roots` has a flagged in-place bug

Line 117 in `qsmp/tree.py`:

```python
roots[isort[idx]] = winning_roots[i]  #XXX: in-place bug?
```

The `XXX` was left by the original author. The function takes `roots` and
mutates it in place even though callers may not expect that. Either drop the
in-place mutation (return a copy) or rename the parameter to make the
contract explicit.
