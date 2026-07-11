# Ground-truth recovery experiment (synthetic `power-law` dataset)

This experiment quantifies how well QSMP, Snippet-Finder, and shift-invariant
$k$-means (sikmeans) recover the *known* waveforms of the synthetic `power-law`
dataset, aggregated over several random seeds with 95% confidence intervals.

The `power-law` dataset (`qsmp.datasets.powerlaw_dataset`) places Morlet
wavelets drawn from a known frequency alphabet `[1, 5, 12, 30, 100, 150]` Hz
under a power-law prevalence. Activations are scattered with **Poisson
(non-uniform) spacing**, so nearby wavelets *superimpose additively* — a
length-`m` window is generally a partial superposition of overlapping wavelets,
a harder and more realistic test than the edge-to-edge `uniform` tiling (where
every window is a single clean wavelet). The methods can still recover the
underlying shapes because the interference is sparse and, pooled over the ~1000
instances, incoherent (it averages toward zero, like the additive noise), while
the shared prototype survives.

Crucially, the generator produces the clean prototypes (`morlet_waveforms`)
*before* placing them (`powerlaw_signal`), so the ground truth is exactly those
prototypes — known even where the signal superimposes them. This enables three
complementary, method-agnostic **prototype-recovery** metrics (defined in
`qsmp/eval_metrics.py`):

- **`n_freqs_recovered`** — how many of the ground-truth frequencies the method
  recovers (each returned prototype is assigned to the nearest alphabet
  frequency by its spectral peak; count the distinct frequencies hit).
- **`recovery_error`** — morphology error of the recovered prototypes: the mean
  shift-invariant, z-normalised Euclidean distance of the optimally
  (Hungarian-)matched prototype→ground-truth pairs.
- **`peak_freq_error`** — mean absolute error (Hz) between each matched
  prototype's spectral peak and the true frequency.

A method is represented only by the `k` prototype waveforms it returns (QSMP
modes, sikmeans centroids, Snippet-Finder snippets), so all methods are scored
identically.

## Files

The design separates **producing** prototypes from **scoring** them. Each runner
saves only the raw `k × m` prototype waveforms (plus the dataset parameters
needed to re-derive the ground truth) in one self-describing file per
`(method, seed)`. `aggregate_recovery.py` is the single place that scores — it
loads every prototype file, re-derives the ground truth, and applies the same
metrics to all methods. So metrics can be changed and everything re-scored with
no re-run, and the methods can be produced in any order.

| File | Role | Needs |
|---|---|---|
| `qsmp/eval_metrics.py` | Ground-truth prototypes, the three recovery metrics, and the `save_prototypes`/`load_prototypes` file format. Has a `__main__` self-test. | CPU |
| `scripts/eval_recovery.py` | Per-seed runner for QSMP + sikmeans; writes `results/recovery/{qsmp,sikmeans}_seed-<seed>.npz` (prototypes only). | GPU (QSMP) |
| `scripts/snippetfinder_recovery.py` | Per-seed Snippet-Finder runner (via `stumpy.snippets`); writes `results/recovery/snippetfinder_seed-<seed>.npz`. | CPU |
| `scripts/aggregate_recovery.py` | Scores every method's prototypes, pools per-seed → mean ± 95% CI → LaTeX table. | CPU |

## Reproduce

The QSMP step requires a CUDA GPU (see the environment notes in
[`README.md`](../README.md) and [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md)); the
Snippet-Finder and aggregation steps are CPU-only.

```sh
# 0. Sanity-check the metric logic (CPU, seconds).
python -m qsmp.eval_metrics

# 1. Per-seed QSMP + sikmeans (GPU). One invocation per seed; resumable.
#    Each method sweeps its own grid and self-selects (see "Parameter
#    selection" below); the defaults shown are the swept grids.
for s in $(seq 0 19); do
    python scripts/eval_recovery.py --root . --seed $s \
        --subseq-len 512 --sigma 0.5 0.9 1 2 3 --minfilt-size 256 --k 6 \
        --window-len 640 768 1024
done

# 2. Per-seed Snippet-Finder (CPU). Also resumable. Order vs. step 1 is
#    irrelevant -- scoring happens only in step 3.
for s in $(seq 0 19); do
    python scripts/snippetfinder_recovery.py --root . --seed $s \
        --subseq-len 512 --k 6 --percentage 0.15 0.2 0.3 0.4 0.5
done

# 3. Score every method's prototypes and aggregate into the LaTeX table.
python scripts/aggregate_recovery.py --root . --out results/recovery/table.tex
```

Steps 1 and 2 are embarrassingly parallel over seeds and are convenient to run
as job-array tasks on a cluster (one seed per task); `snippetfinder_recovery.qs`
is an example SLURM array script. `stumpy` is required for the Snippet-Finder
step (`pip install stumpy`).

## Notes

- **Reproducibility across machines.** `powerlaw_dataset` threads one seeded
  generator through the prototype phases, the activation draw, the Poisson
  placement, and the noise, so it is deterministic in the seed: the
  QSMP/sikmeans runs and the Snippet-Finder runs see the *same* signal per seed
  even when run on different machines, and the aggregator re-derives the exact
  matching ground truth.
- **Parameter selection is unsupervised.** The harder Poisson signal shifts
  each method's optimal settings, so parameters are *not* hand-picked and are
  *never* tuned to the recovery metric (that would leak the ground truth and
  make the comparison circular). Instead every method sweeps a grid and selects
  its configuration by its **own internal, ground-truth-free objective**, then
  that single selected configuration is scored:
  - **QSMP `sigma`** — the tree is cut (by binary search on the distance
    threshold `tau`) to exactly `k` modes for each width; the width kept is the
    one that **maximises the minimum pairwise shift-invariant distance between
    its `k` modes** (max-min diversity). This encodes QSMP's goal of `k`
    *distinct* representatives. (It replaces an earlier min-total-NN-distance
    tie-break, which rewarded tight — hence redundant — clusters and so
    preferred a width whose modes collapse onto the most-prevalent frequencies.)
  - **sikmeans `window-len`** — swept; the length with the smallest clustering
    **distortion** (mean inertia, comparable across window counts under cosine +
    z-normalisation) is kept.
  - **Snippet-Finder `percentage`** — the MPdist sub-subsequence length as a
    fraction of `m` (`stumpy.snippets(..., percentage=...)`, equivalent to
    `per/100` in the Matlab `snippetfinder(data, N, sub, per)`); swept, keeping
    the run with the highest snippet **coverage**.
- **`k = 6` and `m = 512` stay fixed** for all methods: the alphabet size and
  the pattern length are known domain facts, given equally to every method, and
  match the assumptions behind the paper's qualitative figure.
