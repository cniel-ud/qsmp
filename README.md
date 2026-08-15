# QSMP

Python implementation of **Quick Shift + Matrix Profile** for finding
representative subsequences (waveforms) in long time series through
density-guided clustering.

QSMP is a hierarchical, density-based clustering algorithm for time series
subsequences. It connects each subsequence to its nearest neighbor with higher
density to form a tree (an *anti-arborescence*), which is then cut at a
distance threshold `tau` to produce clusters whose roots are the
*representative subsequences* (modes). It is locally shift-invariant via a
min-pooling filter and amenable to multi-GPU acceleration.

The reference algorithm and experiments are described in:

> Mendoza-Cardenas, C. H., Silva, R. F., Brockmeier, A. J. *QSMP: finding
> representative time series subsequences through Quick Shift + Matrix
> Profile*. 2026 IEEE International Workshop on Machine Learning for Signal
> Processing (MLSP), 2026.

## Repository layout

```
qsmp/
├── qsmp/                      # Library code
│   ├── core.py                # STOMP-derived primitives (z-norm distance, dot-product update, FFT conv, etc.)
│   ├── gpu_density.py         # GPU kernel + driver for the kernel-density estimate (Algorithm 1)
│   ├── gpu_qsmp.py            # GPU kernel + driver for shift-invariant NN-distance/NN-index (Algorithm 2)
│   ├── tree.py                # Tree-cutting, root merging, mode/neighbor extraction, binary-search of tau
│   ├── datasets.py            # Synthetic Morlet-wavelet generator (power-law dataset, §5.1)
│   ├── config.py              # Numerical thresholds and GPU-block tunables
│   ├── utils/                 # Windows, MATLAB v7.3 loader, file-naming helpers
│   ├── viz/                   # Matplotlib plots + interactive ipywidgets app for exploring modes
│   ├── shift_kmeans/          # Shift-invariant k-means baseline (sikmeans, ref [12])
│   └── alphacsc/              # Convolutional sparse-coding helpers (used by some plots)
├── scripts/                   # Reproducibility entry points
│   ├── gen_morlet_signal.py   # Generate the synthetic power-law dataset (§5.2)
│   ├── qsmp_on_morlet.py      # Run QSMP on the synthetic dataset (§5.2-5.4)
│   ├── qsmp_on_ecog.py        # Run QSMP on Study019 preictal/interictal (§5.4-5.5)
│   ├── sikmeans.py            # Run sikmeans baseline (§5.5, §5.6)
│   └── build_report.py        # Render top-mode + neighbor figures into a multi-page PDF
├── notebooks/                 # Exploratory notebooks + the MixedBag pipeline
│   └── QSMP_on_MixedBag.py    # Reproduces Table 1 (§5.6)
├── data/
│   ├── MixedBag/              # 100 time series from Snippet-Finder [9]
│   └── study019-preictal/
│       └── qsmp_T_splice.npz  # Preprocessed preictal time series (CSP-filtered, spliced)
├── docs/
│   ├── REPRODUCING.md         # Step-by-step reproduction of the paper experiments
│   ├── USAGE.md               # Running QSMP on new data + output format reference
│   └── KNOWN_ISSUES.md        # Compatibility breakages and cleanup TODOs
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Hardware and software requirements

QSMP relies on **GPU acceleration via Numba CUDA + CuPy** for the
density-estimate and NN-distance/NN-index computations. The following are
required to run anything beyond the synthetic-data generator and the tree
post-processing:

- An NVIDIA GPU (the original experiments were run on UD's HPC cluster under SLURM)
- **CUDA Toolkit 11.x** (matching `cupy==11.2.0`)
- Python **3.9.13** (the version used for the original submission)

> **Note on CPU-only use.** The library code is importable on a CPU-only
> machine (macOS, Linux, Windows) and the synthetic-data generator
> (`qsmp.datasets.morlet_signal`), the tree-cutting utilities (`qsmp.tree`),
> and the post-processing scripts work without a GPU. The
> `gpu_density`/`gpu_qsmp` entry points cannot be exercised without CUDA.


## Installation

We recommended using [uv](https://docs.astral.sh/uv/) for fast environment + package
management.

```sh
uv venv --python 3.9.13
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

> **Heads-up on `requirements.txt`.** The pins
> (`numpy==1.17.4`, `cupy==11.2.0`, `numba==0.56.4`, etc.) reflect the
> environment used for the original submission. If you want to use a more
> recent Python (3.10+), see [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md)
> for the working modern alternatives that have been verified to import the
> library.

## Reproducing the paper

See [`docs/REPRODUCING.md`](docs/REPRODUCING.md) for step-by-step instructions
covering the three experiments:

- §5.2 — Power-law (Morlet) synthetic dataset
- §5.4–5.5 — Study019 ECoG (preictal/interictal)
- §5.6 — MixedBag binary-segment task

## Running QSMP on new data

See [`docs/USAGE.md`](docs/USAGE.md) for the minimal Python API, the format of
the saved `.npz` outputs, the interactive ipywidgets app for post-hoc
exploration of the QS tree, and the checkpointing behaviour.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{mendozacardenas2026qsmp,
  title     = {QSMP: finding representative time series subsequences through Quick Shift + Matrix Profile},
  author    = {Mendoza-Cardenas, Carlos H. and Silva, Rogers F. and Brockmeier, Austin J.},
  booktitle = {2026 IEEE International Workshop on Machine Learning for Signal Processing (MLSP)},
  year      = {2026},
  publisher = {IEEE},
  % TODO: add pages and DOI once the proceedings are published; arXiv preprint link when available.
}
```

## License

This project is released under the BSD 3-Clause License (see [`LICENSE`](LICENSE)).
The GPU-density and GPU-QSMP modules adapt routines from
[STUMPY](https://github.com/TDAmeritrade/stumpy)
(© 2019 TD Ameritrade, BSD 3-Clause).
