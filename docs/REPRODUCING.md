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

## §4.1 — Power-law (Morlet) synthetic recovery

The synthetic dataset is a 1,000-second time series at 512 Hz built from six
1-second Morlet wavelets (1, 5, 12, 30, 100, 150 Hz) drawn from a power-law
prevalence, plus Gaussian noise (σ=0.07).

The main-paper quantitative result (**Table 1**) is the **ground-truth recovery
experiment**: it scores how well QSMP, Snippet-Finder, and sikmeans recover the
known prototypes, with each method's hyperparameter chosen by its own
**unsupervised** criterion (no tuning to the ground truth). It is documented
end-to-end — metrics (FreqRec, CosSim, PeakErr), matching, spacing, and
per-method selection — in
[`RECOVERY_EXPERIMENT.md`](RECOVERY_EXPERIMENT.md). In brief:

```sh
# QSMP + sikmeans (GPU) and Snippet-Finder (CPU, via stumpy.snippets),
# 20 seeds, uniform spacing (the main-paper signal).
for s in $(seq 0 19); do
    python scripts/eval_recovery.py --root . --seed $s --spacing uniform \
        --subseq-len 512 --sigma 0.5 0.9 1 2 3 --minfilt-size 256 --k 6 \
        --window-len 640 768 1024
    python scripts/snippetfinder_recovery.py --root . --seed $s --spacing uniform \
        --subseq-len 512 --k 6 --percentage 0.15 0.2 0.3 0.4 0.5
done

# Aggregate Table 1 and render the qualitative comparison figure from the
# same saved, unsupervised-selected prototypes (so figure and table agree).
python scripts/aggregate_recovery.py --root . --spacing uniform \
    --out results/recovery/uniform/table.tex
python scripts/fig_synthetic_patterns.py --root . --spacing uniform \
    --out img/QSMP_vs_Snippet-Finder_vs_sikmeans_morlet.pdf
```

The Snippet-Finder baseline runs via STUMPY's `stumpy.snippets`, so no external
MATLAB implementation is needed.

An older exploratory pipeline (`gen_morlet_signal.py` → `qsmp_on_morlet.py` →
`build_report.py`, with a hand-picked σ grid) remains available for browsing
top modes and their nearest neighbors, but it is *not* what produces the
paper's Table 1 or the comparison figure.

## §4.2 — Study019 ECoG dataset

The preprocessed preictal segment is checked in at
`data/study019-preictal/qsmp_T_splice.npz`. It contains:

- `T` — float64 array of length 5,794,755 (the linearly-combined
  single-channel ECoG — "first n=5,794,755 samples of preictal")
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

The interictal segment (§4.2) is **not** included in the repo. To
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

## §4.3 — MixedBag dataset

The 100 time series from Imani et al. (DAMI 2020) are checked in under
`data/MixedBag/`. Each series is split into two regimes at a known change
point. The reproducibility entry points are the two script-style drivers:

```sh
python notebooks/QSMP_on_MixedBag.py       # QSMP column of Table 2
python notebooks/sikmeans_on_MixedBag.py   # sikmeans column of Table 2
```

For each series the subsequence length `m` is **fixed** to the per-series
default provided with the dataset (encoded in the file name); the QS tree is
cut (binary search on `tau`) to exactly 2 modes, and a run is a *success* when
the two modes fall on opposite sides of the known split point (within an
`m`-sized tolerance). The remaining hyperparameter is chosen **unsupervised**
(never tuned to the success label): QSMP sweeps
`σ ∈ {0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5}` and keeps the width that
maximizes the separation between the two modes; sikmeans keeps the window
length that minimizes clustering distortion. Snippet-Finder is **not** re-run
here — the paper cites the authors' reported result at their fixed setting
(`S = 50%`, Imani et al. 2020). Per-series successes are saved under
`results/MixedBag/`. Together these reproduce the QSMP and sikmeans columns of
**Table 2** (the success-rate comparison); random sampling is the analytical
reference from the same paper.
