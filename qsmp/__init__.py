"""Quick Shift + Matrix Profile (QSMP) for time series subsequence clustering.

QSMP is a hierarchical, density-based clustering algorithm for the set of all
subsequences of length ``m`` in a long 1D time series. It connects each
subsequence to its nearest higher-density neighbor (forming an
*anti-arborescence*), and that tree is then cut at a distance threshold
``tau`` to produce clusters whose roots are the *representative subsequences*
(modes).

Submodules
----------
``core``
    Distance, mean/std and dot-product primitives derived from STOMP.
``gpu_density``
    GPU kernel + Python driver implementing Algorithm 1 (kernel density
    estimate of all subsequences). Requires CUDA.
``gpu_qsmp``
    GPU kernel + Python driver implementing Algorithm 2 (shift-invariant
    NN-distance and NN-index, i.e., the QSMP). Requires CUDA.
``tree``
    Tree-cutting, root merging, neighbor extraction, and binary-search of
    ``tau`` for a target number of modes. Pure NumPy, CPU-only.
``datasets``
    Synthetic data generators used in the paper experiments.
``config``
    Numerical thresholds and GPU-block tunables.
``viz``
    Matplotlib helpers and an ``ipywidgets``-based interactive app for
    exploring modes across kernel widths and distance thresholds.
``utils``
    Window functions, MATLAB v7.3 loader, deterministic file-naming helper.

Reference
---------
Mendoza-Cardenas, C. H. & Brockmeier, A. J. *QSMP: finding representative
time series subsequences through Quick Shift + Matrix Profile* (under
resubmission).
"""

__version__ = "0.0.1"
