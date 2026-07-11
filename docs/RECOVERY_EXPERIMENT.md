# Ground-truth recovery experiment (synthetic `power-law` dataset)

This experiment quantifies how well QSMP, Snippet-Finder, and shift-invariant
$k$-means (sikmeans) recover the *known* waveforms of the synthetic `power-law`
dataset, aggregated over several random seeds with 95% confidence intervals.

The `power-law` dataset (`qsmp.datasets.morlet_signal`) concatenates 1-second
Morlet wavelets drawn from a known frequency alphabet
`[1, 5, 12, 30, 100, 150]` Hz under a power-law prevalence. Because the
generating frequencies and phases are known, each dataset instance comes with
noise-free ground-truth prototype waveforms, enabling three complementary,
method-agnostic **prototype-recovery** metrics (defined in
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
for s in $(seq 0 19); do
    python scripts/eval_recovery.py --root . --seed $s \
        --subseq-len 512 --sigma 0.9 1 2 --minfilt-size 256 --k 6
done

# 2. Per-seed Snippet-Finder (CPU). Also resumable. Order vs. step 1 is
#    irrelevant -- scoring happens only in step 3.
for s in $(seq 0 19); do
    python scripts/snippetfinder_recovery.py --root . --seed $s \
        --subseq-len 512 --k 6 --percentage 0.30
done

# 3. Score every method's prototypes and aggregate into the LaTeX table.
python scripts/aggregate_recovery.py --root . --out results/recovery/table.tex
```

Steps 1 and 2 are embarrassingly parallel over seeds and are convenient to run
as job-array tasks on a cluster (one seed per task); `snippetfinder_recovery.qs`
is an example SLURM array script. `stumpy` is required for the Snippet-Finder
step (`pip install stumpy`).

## Notes

- **Reproducibility across machines.** `morlet_signal` is deterministic in the
  seed, so the QSMP/sikmeans runs and the Snippet-Finder runs see the *same*
  signal per seed even when run on different machines; the ground truth aligns.
- **`k = 6`.** The tree is cut (by binary search on the distance threshold
  `tau`) to `k = 6` modes; among the kernel widths `sigma` that yield exactly
  `k` modes, the one with the smallest total nearest-neighbour distance is kept.
  Choosing `k` uses knowledge of the alphabet size, matching the qualitative
  figure in the paper.
- **Snippet-Finder `percentage`.** This is the MPdist sub-subsequence length as
  a fraction of `m` (`stumpy.snippets(..., percentage=...)`), equivalent to
  `per/100` in the original Matlab `snippetfinder(data, N, sub, per)`.
