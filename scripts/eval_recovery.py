"""Per-seed QSMP + sikmeans runs on the synthetic power-law dataset.

Runs QSMP and sikmeans on one instance of the synthetic Morlet dataset (one
random seed) and saves each method's ``k`` prototype waveforms -- *without*
scoring them. Scoring against the known ground truth is done separately, for all
methods at once, by ``aggregate_recovery.py``. This keeps the (expensive,
GPU-bound) production of prototypes decoupled from the (cheap, CPU-bound)
metric computation, so metrics can be changed and results re-scored without any
re-run, and methods can be run in any order.

Each method writes one file per seed via :func:`qsmp.eval_metrics.save_prototypes`::

    <root>/results/recovery/qsmp_seed-<seed>.npz
    <root>/results/recovery/sikmeans_seed-<seed>.npz

with the ``k x m`` prototypes plus the dataset parameters needed to
re-derive the ground truth (seed, m, n_waves, noise_std, freqs) and method
metadata (chosen sigma/tau, timings). Snippet-Finder is produced by
``snippetfinder_recovery.py`` in the same format.

The script is **resumable and SLURM-array friendly**: one seed per invocation,
skips a method whose output already exists (unless ``--overwrite``).

Example (single seed, e.g. inside a SLURM array task)::

    python scripts/eval_recovery.py --root . --seed $SLURM_ARRAY_TASK_ID \
        --subseq-len 512 --sigma 0.9 1 2 --minfilt-size 256 --k 6

GPU is required for the QSMP portion (``gpu_density`` / ``gpu_qsmp`` need CUDA).
sikmeans runs on CPU.
"""

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import numpy as np

from qsmp import eval_metrics as em
from qsmp import tree
from qsmp.datasets import morlet_signal
import qsmp.utils.utils as utils
from qsmp.utils import windows

FREQS = np.array([1, 5, 12, 30, 100, 150])


def run_qsmp(T, splice, m, sigma, minfilt, k, root, params_str, window,
             device_ids):
    """Run QSMP, cut to ``k`` modes, return the ``k`` mode waveforms.

    Returns ``(protos, info)`` where ``protos`` is ``(k, m)`` (or fewer rows if
    no threshold yields exactly ``k`` modes for some sigma) and ``info`` is a
    dict with the chosen ``sigma``/``tau`` and timing.
    """
    from qsmp.gpu_density import gpu_density
    from qsmp.gpu_qsmp import gpu_qsmp

    t0 = perf_counter()
    T, splice, density = gpu_density(
        T, m, sigma, root, params_str, splice=splice, window=window,
        device_id=device_ids)
    profile, indices = gpu_qsmp(
        T, m, minfilt, density, root, params_str, splice=splice,
        device_id=device_ids)
    profile, indices, density = utils.fix_root((profile, indices, density))
    t_qsmp = perf_counter() - t0

    # `density`, `indices` and `profile` have shape (N, n_sigma): one column per
    # kernel width. Per-sigma vectors are accessed as ``density.T[i_sigma]``,
    # matching the canonical MixedBag pipeline (notebooks/QSMP_on_MixedBag.py).
    density_T = np.atleast_2d(density.T)
    indices_T = np.atleast_2d(indices.T)
    profile_T = np.atleast_2d(profile.T)

    # For each sigma, binary-search tau for exactly k modes; among the sigmas
    # that succeed, keep the one whose k modes have the smallest total NN-distance
    # (the data-driven, ground-truth-free tie-break from the KDD rebuttal).
    best = None
    for si in range(sigma.size):
        dens_i, ind_i, prof_i = density_T[si], indices_T[si], profile_T[si]
        if np.all(prof_i == 0):        # this sigma produced no usable tree
            continue
        tau = tree.find_tau(k, m, dens_i, ind_i, prof_i)
        if tau is None:
            continue
        NNd, NNi, modes, csize = tree.tree2clusters(m, dens_i, ind_i, prof_i,
                                                     tau)
        if modes.size != k:
            continue
        score = float(np.sum(NNd[NNi != np.arange(NNi.size)]))
        if best is None or score < best["score"]:
            best = dict(sigma=float(sigma[si]), tau=float(tau), modes=modes,
                        score=score)

    if best is None:
        return np.empty((0, m)), dict(sigma=None, tau=None, t_qsmp=t_qsmp,
                                      n_modes=0)

    protos = utils.get_waves(np.sort(best["modes"]).astype(np.int64), T, m)
    return protos, dict(sigma=best["sigma"], tau=best["tau"], t_qsmp=t_qsmp,
                        n_modes=int(best["modes"].size))


def run_sikmeans(T, splice, m, k, win_len, n_runs=30, seed=13):
    """Run sikmeans, return its ``k`` centroids as prototypes ``(k, m)``."""
    from qsmp.shift_kmeans.shift_kmeans import shift_invariant_k_means

    t0 = perf_counter()
    start_arr = np.r_[0, splice] if splice.size else np.r_[0]
    end_arr = np.r_[splice, T.size] if splice.size else np.r_[T.size]
    tot_win = int(np.sum((end_arr - start_arr) // win_len))
    X = np.zeros((tot_win, win_len))
    sx = 0
    for start, end in zip(start_arr, end_arr):
        seg = T[start:end]
        n_win = seg.size // win_len
        idx = np.arange(0, n_win * win_len, win_len)[:, None] + \
            np.arange(win_len)[None, :]
        X[sx:sx + n_win] = seg[idx]
        sx += n_win
    centroids, labels, shifts, distances, _, _ = shift_invariant_k_means(
        X, k, m, metric="cosine", init="random", n_init=n_runs, rng=seed,
        verbose=False)
    return np.asarray(centroids), dict(t_sikmeans=perf_counter() - t0)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".", help="Repository root")
    p.add_argument("--seed", type=int, required=True,
                   help="Random seed for this dataset instance")
    p.add_argument("--subseq-len", type=int, default=512)
    p.add_argument("--sigma", type=float, nargs="*", default=[0.9, 1.0, 2.0])
    p.add_argument("--minfilt-size", type=int, default=256)
    p.add_argument("--k", type=int, default=6, help="Number of prototypes")
    p.add_argument("--window-len", type=int, default=768,
                   help="sikmeans non-overlapping window length")
    p.add_argument("--n-waves", type=int, default=1000)
    p.add_argument("--noise-std", type=float, default=0.07)
    p.add_argument("--window-support", type=float, default=0.5)
    p.add_argument("--window-type", default="rect",
                   help="Centeredness window type (rect/gauss/None)")
    p.add_argument("--methods", nargs="+", default=["qsmp", "sikmeans"],
                   choices=["qsmp", "sikmeans"],
                   help="Which method(s) to run")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    root = Path(args.root)
    out_dir = root.joinpath("results", "recovery")
    out_dir.mkdir(parents=True, exist_ok=True)

    m = args.subseq_len
    sigma = np.array(args.sigma)
    # Dataset parameters saved with every prototype file so the aggregator can
    # re-derive the exact ground truth for this seed.
    ds = dict(seed=args.seed, m=m, n_waves=args.n_waves,
              noise_std=args.noise_std, freqs=FREQS)

    # --- dataset (no ground truth needed here; scoring is in the aggregator) --
    T, freq, _ = morlet_signal(
        FREQS, fs=512, wave_len=m, n_waves=args.n_waves,
        noise_std=args.noise_std, rng=args.seed)
    splice = np.full(0, 0)

    if "qsmp" in args.methods:
        out_file = out_dir.joinpath(f"qsmp_seed-{args.seed}.npz")
        if out_file.is_file() and not args.overwrite:
            print(f"{out_file} exists; skipping (use --overwrite to redo).")
        else:
            if args.window_type and args.window_type.lower() != "none":
                win = windows.get_window(args.window_type)(m, args.window_support)
            else:
                win = None
            import numba
            device_ids = [d.id for d in numba.cuda.list_devices()][:1]
            params_str = f"recovery_seed-{args.seed}"
            protos, info = run_qsmp(T, splice, m, sigma, args.minfilt_size,
                                    args.k, root, params_str, win, device_ids)
            em.save_prototypes(out_file, protos, method="qsmp", k=args.k,
                               ds=ds, info=info)
            print(f"[seed {args.seed}] QSMP: n_modes={info['n_modes']} "
                  f"sigma={info['sigma']} tau={info['tau']} -> {out_file}")

    if "sikmeans" in args.methods:
        out_file = out_dir.joinpath(f"sikmeans_seed-{args.seed}.npz")
        if out_file.is_file() and not args.overwrite:
            print(f"{out_file} exists; skipping (use --overwrite to redo).")
        else:
            protos, info = run_sikmeans(T, splice, m, args.k, args.window_len)
            em.save_prototypes(out_file, protos, method="sikmeans", k=args.k,
                               ds=ds, info=info)
            print(f"[seed {args.seed}] sikmeans: {protos.shape[0]} centroids "
                  f"-> {out_file}")


if __name__ == "__main__":
    main()
