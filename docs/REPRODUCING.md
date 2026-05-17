# Reproducing the paper

This guide walks through reproducing the three experiments in the QSMP paper.
All scripts accept a `--root <PATH>` (or `-d` / `-w`) so you can place data
and results anywhere. Output `.npz` files are named deterministically from
the parameter combination via `qsmp.utils.utils.Args2Filename`, which makes
parameter sweeps easy to enumerate.

> **Prerequisite.** Install the package as described in the
> [main README](../README.md#installation). The Morlet experiment can be run
> partially without a GPU (the dataset generator), but the QSMP/density
> computations require CUDA.

## §5.2 — Power-law (Morlet) dataset

Generates a 1,000-second synthetic time series at 512 Hz from six 1-second
Morlet wavelets (1, 5, 12, 30, 100, 150 Hz) drawn from a power-law
distribution, plus Gaussian noise (σ=0.07).

```sh
# Generate the synthetic dataset
python scripts/gen_morlet_signal.py --root .

# Run QSMP on it (uses subsequence length m=512, kernel widths σ ∈ {0.9, 1.0, 2.0})
python scripts/qsmp_on_morlet.py \
    --root . \
    --subseq-len 512 \
    --sigma 0.9 1.0 2.0 \
    --minfilt-size 256 \
    --window-type rect \
    --window-support 0.5

# Build the figure of top modes + neighbors
python scripts/build_report.py morlet \
    --root . \
    --subseq-len 512 \
    --sigma 0.9 1.0 2.0 \
    --minfilt-size 256 \
    --window-type rect \
    --window-support 0.5 \
    --max-modes 6 \
    --n-neighbors 9
```

The QSMP output is saved to
`./results/morlet/qsmp_m-512_sigma-0.9_1.0_2.0_rect-50_minfilt-256.npz` and
contains `density`, `profile` (NN-distance), `indices` (NN-index), `T`, and
`splice`.

To reproduce the panels in **Figure 5** (QSMP vs Snippet-Finder vs sikmeans),
also run:

```sh
python scripts/sikmeans.py morlet --root . --centroid-len 512 --window-len 768 --num-clusters 6
```

The Snippet-Finder baseline is not reimplemented in this repo; use the
authors' implementation from Imani et al., DAMI 2020 (ref [9]).

## §5.4–5.5 — Study019 ECoG dataset

The preprocessed preictal segment is checked in at
`data/study019-preictal/qsmp_T_splice.npz`. It contains:

- `T` — float64 array of length 5,794,755 (the linearly-combined
  single-channel ECoG, matching §5.1 — "first n=5,794,755 samples of
  preictal")
- `splice` — int64 array of 55 splice indices (boundaries between
  concatenated raw segments, used to mask trivial cross-segment matches)

Run QSMP on it:

```sh
python scripts/qsmp_on_ecog.py \
    -d data/study019-preictal \
    -w <PATH-TO-W.mat>          # only needed if you want to regenerate the .npz; see below
    --subseq-len 512 \
    --sigma 0.9 1.0 2.0 \
    --minfilt-size 128 \
    --window-type rect \
    --window-support 0.5
```

Because `data/study019-preictal/qsmp_T_splice.npz` already exists, the script
skips the .mat → .npz preprocessing step and the `-w` argument is not
strictly required (the script will still parse it, but you can pass
`-w /dev/null` or any path).

The interictal segment used in §5.5 is **not** included in the repo. To
reproduce that part, or to regenerate the preictal `.npz` from scratch, you
need:

1. **Raw multi-channel ECoG segments** from
   [ieeg.org](https://www.ieeg.org/) (study `Study019`). Each segment is
   saved as a v7.3 MATLAB file `rxNNN.mat` in a directory whose path
   contains either `preictal` or `interictal`. The expected variables
   are: `epoch` (n_channels × n_samples), `t_start`, `t_end`, and
   optionally `seiz_id`.
2. **CSP spatial filters** `W` saved as a v7.3 MATLAB file (referenced as
   `-w` above). The first column maximizes preictal energy; the last
   column maximizes interictal energy. The CSP pipeline is the one
   described in Mendoza-Cardenas & Brockmeier, EMBC 2021 (ref [12]).
3. The first run of `qsmp_on_ecog.py` writes `qsmp_T_splice.npz` into
   `-d`. Subsequent runs reuse it.

> **TODO (data provenance).** Add a short script and pointer to the
> upstream IEEG.org dataset and the CSP-filter computation.

## §5.6 — MixedBag dataset

The 100 time series from Imani et al. (DAMI 2020) are checked in under
`data/MixedBag/`. The reproducibility entry point is the script-style
notebook:

```sh
python notebooks/QSMP_on_MixedBag.py
```

This sweeps `m_scale ∈ {1, 0.75, 0.5, 0.25}` and
`σ ∈ {0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5}`, runs QSMP on each time
series, performs a binary search for the `tau` that yields exactly 2 modes,
and tags as a *success* the runs where the two modes fall on opposite sides
of the known split point (within an `m`-sized tolerance). The success rate is
saved to `results/MixedBag/sucess_rate.pickle`. This reproduces **Table 1**
(QSMP column) and **Tables 2 & 3**.
