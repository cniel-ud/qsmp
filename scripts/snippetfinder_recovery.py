"""Per-seed Snippet-Finder run for the ground-truth recovery experiment.

Companion to ``eval_recovery.py`` (which runs QSMP + sikmeans). This script
runs *only* Snippet-Finder on one instance of the synthetic ``power-law``
dataset (one seed) and saves its ``k`` snippet waveforms in the same
self-describing format the other runners use::

    <root>/results/recovery/snippetfinder_seed-<seed>.npz

``aggregate_recovery.py`` picks these up alongside the QSMP/sikmeans files and
scores every method the same way, so Snippet-Finder becomes a third row in the
recovery table. This script does not score anything itself.

Snippet-Finder is CPU-based and comparatively slow, so it is convenient to run
on a cluster, one seed per array task. It uses STUMPY's native
``stumpy.snippets`` -- equivalent to the authors' Matlab
``snippetfinder(data, N, sub, per)``:

    Matlab ``sub``  -> stumpy ``m``            (snippet length)
    Matlab ``N``    -> stumpy ``k``            (number of snippets)
    Matlab ``per``  -> stumpy ``percentage``   (MPdist sub-subsequence fraction; S = round(m*percentage))
    Matlab MPdist threshold 0.05 -> stumpy ``mpdist_percentage`` default 0.05.

The synthetic signal is deterministic in the seed (same
``qsmp.datasets.morlet_signal``), so the ground truth the aggregator re-derives
matches this signal even if this ran on a different machine than the QSMP runs.

Example (single seed, e.g. inside a SLURM array task)::

    python scripts/snippetfinder_recovery.py --root . --seed $SLURM_ARRAY_TASK_ID \
        --subseq-len 512 --k 6 --percentage 0.30

The paper reports that only S~=154 (30% of m=512) let Snippet-Finder recover all
six patterns; pass ``--percentage`` accordingly, or a list to sweep and keep the
best-covering run (see ``--percentage`` help).
"""

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import numpy as np
import stumpy

from qsmp import eval_metrics as em
from qsmp.datasets import powerlaw_dataset

FREQS = np.array([1, 5, 12, 30, 100, 150])


def run_snippetfinder(T, m, k, percentage):
    """Run STUMPY Snippet-Finder, return ``(snippets (k, m), info)``.

    If ``percentage`` is a list, run once per value and keep the run whose
    snippet coverage fraction sums highest (a ground-truth-free selection that
    mirrors the paper searching over the sub-subsequence length S).
    """
    percentages = np.atleast_1d(percentage).astype(float)
    best = None
    t0 = perf_counter()
    for per in percentages:
        snippets, indices, profiles, fractions, areas, regimes = stumpy.snippets(
            T, m=m, k=k, percentage=float(per))
        cover = float(np.sum(fractions))
        if best is None or cover > best["cover"]:
            best = dict(snippets=np.asarray(snippets), indices=np.asarray(indices),
                        fractions=np.asarray(fractions), percentage=float(per),
                        cover=cover)
    info = dict(percentage=best["percentage"], cover=best["cover"],
                indices=best["indices"].ravel().tolist(),
                fractions=best["fractions"].ravel().tolist(),
                t_snippetfinder=perf_counter() - t0)
    return best["snippets"], info


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".", help="Repository root")
    p.add_argument("--seed", type=int, required=True,
                   help="Random seed for this dataset instance")
    p.add_argument("--subseq-len", type=int, default=512, help="Snippet length m")
    p.add_argument("--k", type=int, default=6, help="Number of snippets")
    p.add_argument("--percentage", type=float, nargs="+",
                   default=[0.15, 0.20, 0.30, 0.40, 0.50],
                   help="MPdist sub-subsequence fraction(s) of m to sweep. The "
                        "best-covering run is kept -- an unsupervised selection "
                        "mirroring QSMP's sigma / sikmeans' window-len sweeps. "
                        "Paper used ~0.30; pass a single value to fix it.")
    p.add_argument("--n-waves", type=int, default=1000)
    p.add_argument("--noise-std", type=float, default=0.07)
    p.add_argument("--spacing", choices=["poisson", "uniform"], default="poisson",
                   help="Wavelet arrival spacing; must match the QSMP/sikmeans "
                        "runs being compared. Results are written under "
                        "results/recovery/<spacing>/.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    root = Path(args.root)
    out_dir = root.joinpath("results", "recovery", args.spacing)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir.joinpath(f"snippetfinder_seed-{args.seed}.npz")
    if out_file.is_file() and not args.overwrite:
        print(f"{out_file} exists; skipping (use --overwrite to redo).")
        return

    m = args.subseq_len
    # Same generator + seed + spacing as the QSMP/sikmeans runs -> identical
    # signal, so the ground truth the aggregator re-derives matches.
    T, _, _, _ = powerlaw_dataset(
        FREQS, seed=args.seed, fs=512, wave_len=m, n_waves=args.n_waves,
        noise_std=args.noise_std, spacing=args.spacing)
    ds = dict(seed=args.seed, m=m, n_waves=args.n_waves,
              noise_std=args.noise_std, freqs=FREQS, spacing=args.spacing)

    protos, info = run_snippetfinder(T, m, args.k, args.percentage)
    print(f"[seed {args.seed}] Snippet-Finder: k={protos.shape[0]}, "
          f"percentage={info['percentage']}, cover={info['cover']:.3f}, "
          f"t={info['t_snippetfinder']:.1f}s, indices={info['indices']}")

    # Same self-describing format as qsmp/sikmeans; scoring is done by the
    # aggregator, uniformly across methods.
    em.save_prototypes(out_file, protos, method="snippetfinder", k=args.k,
                       ds=ds, info=info)
    print(f"[seed {args.seed}] wrote {out_file}")


if __name__ == "__main__":
    main()
