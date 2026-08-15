# Uniform-spacing recovery run (main-paper table)

The recovery experiment is spacing-aware. The **uniform** set backs the
main-paper table; the **poisson** set backs the supplement. Uniform tiles
wavelets edge-to-edge, so each analysis window is one clean wavelet -- the
easier, easier-to-explain signal for the main text.

Results land in `results/recovery/<spacing>/`, so uniform and poisson never
collide. Every runner takes `--spacing uniform`; the aggregator defaults to
uniform.

Prereqs: the GPU runs (QSMP, sikmeans) go on a machine with a CUDA GPU;
Snippet-Finder is CPU and can run locally or on a CPU node. If you split the
work across machines, first sync the code to the GPU machine (the runs need the
`--spacing` flag). Adjust the host names and paths below to your setup.

## 0. Sync code to the GPU machine (only if running remotely)

```bash
# from your local repo root; mirrors code only (results/ is gitignored anyway)
rsync -av --exclude '.git' --exclude '.venv' --exclude 'results' \
  ~/qsmp/ <gpu-host>:~/qsmp/
```

## 1. QSMP + sikmeans, 20 seeds, uniform (GPU machine)

```bash
cd ~/qsmp
for s in $(seq 0 19); do
  python scripts/eval_recovery.py --root . --seed "$s" --spacing uniform \
    --subseq-len 512 --sigma 0.5 0.9 1.0 2.0 3.0 --minfilt-size 256 --k 6 \
    --window-len 640 768 1024
done
# writes results/recovery/uniform/{qsmp,sikmeans}_seed-*.npz
```

## 2. Snippet-Finder, 20 seeds, uniform (CPU: local machine or a CPU node)

```bash
cd ~/qsmp
for s in $(seq 0 19); do
  python scripts/snippetfinder_recovery.py --root . --seed "$s" --spacing uniform \
    --subseq-len 512 --k 6 --percentage 0.15 0.20 0.30 0.40 0.50
done
# writes results/recovery/uniform/snippetfinder_seed-*.npz
```

## 3. Collect all uniform files onto one machine

```bash
# pull GPU-produced files back to the local machine (into the uniform subdir)
rsync -av <gpu-host>:~/qsmp/results/recovery/uniform/ \
  ~/qsmp/results/recovery/uniform/
# (and <cpu-host>:... if Snippet-Finder ran on a separate node)
```

## 4. Aggregate the main-paper table

```bash
python scripts/aggregate_recovery.py --root . --spacing uniform \
  --out results/recovery/uniform/table.tex
# label is tab:recovery ; poisson would be tab:recovery_poisson
```

## Poisson supplement -- regenerate any time

```bash
python scripts/aggregate_recovery.py --root . --spacing poisson
python scripts/viz_recovery.py        --root . --spacing poisson
python scripts/paired_tests.py        --root . --spacing poisson
python scripts/sf_cossim_analysis.py  --root . --spacing poisson
```
