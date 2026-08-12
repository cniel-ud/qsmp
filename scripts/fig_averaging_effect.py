"""Supplement figure: effect of shift-aligned averaging on returned prototypes.

Addresses the reviewer's intuition that raw returned waveforms preserve noise
while *averaging* their occurrences denoises them. For each prototype a method
returns (a QSMP mode or a Snippet-Finder snippet), we locate its nearest
occurrences in the signal with a sliding-window z-normalised distance profile
(``stumpy.match``; the sliding search handles the shift, a trivial-match
exclusion zone prevents self-overlap), take a FIXED top-10, and average the
z-normalised windows. The same procedure is applied to both methods.

The story is deliberately two-sided (reviewer decision "2+3"):
  * Averaging denoises the *common* frequencies (their occurrences are plentiful,
    so all 10 windows are genuine): cosine-to-ground-truth goes up.
  * A fixed top-10 *cannot rescue the rarest* frequencies: e.g. 150 Hz has only a
    handful of activations, so most of the 10 windows are unrelated/superimposed
    segments and the average is worse than the raw waveform. This is a limitation
    of averaging, not of either method, and it is stated as such in the caption
    and supplement text.

sikmeans is intentionally omitted: its centroids are already averages.

CPU only. Run after the poisson prototype files exist::

    python scripts/fig_averaging_effect.py --root . --seed 18 \
        --out img/supp_averaging_seed-18.pdf
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stumpy

from qsmp import eval_metrics as em
from qsmp.datasets import powerlaw_dataset
from viz_recovery import (ALPHABET, METHOD_COLOR, METHOD_LABEL, best_lag_align,
                          load_all, pick_representative_seed, snapped_columns,
                          _znorm)

TOP_K = 10                       # fixed number of occurrences to average
AVG_METHODS = ["qsmp", "snippetfinder"]   # sikmeans already averages -> excluded


def top_k_average(query, signal, k=TOP_K):
    """Average the ``k`` nearest z-normalised occurrences of ``query`` in ``signal``.

    Uses ``stumpy.match`` (forced to return the top ``k`` regardless of an
    absolute distance threshold) so the sliding search supplies the shift and
    the built-in exclusion zone (``m/4``) forbids overlapping self-matches. Each
    matched length-``m`` window is z-normalised, sign-aligned to the query (a
    z-normalised match can be anti-correlated), and averaged. Returns the
    averaged waveform and the number of windows actually used (``<= k`` near the
    signal edges).
    """
    query = np.asarray(query, dtype=float)
    m = query.size
    matches = stumpy.match(query, np.asarray(signal, dtype=float),
                           max_distance=np.inf, max_matches=int(k))
    zq = _znorm(query)
    windows = []
    for _, start in matches:
        s = int(start)
        w = signal[s:s + m]
        if w.size < m:
            continue
        zw = _znorm(w)
        if float(zw @ zq) < 0:        # resolve z-norm sign ambiguity
            zw = -zw
        windows.append(zw)
    if not windows:
        return _znorm(query), 0
    return np.mean(windows, axis=0), len(windows)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".")
    p.add_argument("--spacing", choices=["poisson", "uniform"], default="poisson")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rec_dir = em.resolve_rec_dir(args.root, args.spacing)
    protos, gt_cache = load_all(rec_dir)
    if not protos:
        raise SystemExit(f"No prototype files found in {rec_dir}")
    seed = args.seed if args.seed is not None else \
        pick_representative_seed(protos, gt_cache)

    # Rebuild the exact signal for this seed (deterministic from the metadata).
    _, meta0 = em.load_prototypes(next(rec_dir.glob(f"*_seed-{seed}.npz")))
    signal, _, wave_counts, _ = powerlaw_dataset(
        np.asarray(meta0["freqs"]), seed=seed, wave_len=int(meta0["m"]),
        n_waves=int(meta0["n_waves"]), noise_std=float(meta0["noise_std"]),
        spacing=str(meta0["spacing"]))
    counts_by_freq = {int(f): int(c) for f, c in zip(meta0["freqs"], wave_counts)}

    gt, gt_freqs = gt_cache[seed]
    gt_by_col = {int(np.argmin(np.abs(ALPHABET - f))): g
                 for g, f in zip(gt, gt_freqs)}

    methods = [m for m in AVG_METHODS if seed in protos.get(m, {})]
    ncol = ALPHABET.size
    fig, axes = plt.subplots(len(methods), ncol, squeeze=False, sharex=True,
                             figsize=(2.3 * ncol, 2.2 * len(methods)))

    summary = []   # (method, freq, cos_raw, cos_avg, n_used) for the printed note
    for r, m in enumerate(methods):
        P = protos[m][seed]
        cols = snapped_columns(P)
        for ci in range(ncol):
            ax = axes[r][ci]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_alpha(0.25)
            gref = gt_by_col.get(ci)
            idx = np.where(cols == ci)[0]
            if gref is None or idx.size == 0:
                if gref is None:
                    ax.set_facecolor("0.97")
                else:
                    ax.text(0.5, 0.5, "missed", transform=ax.transAxes,
                            fontsize=8, ha="center", va="center",
                            color="0.5", style="italic")
                if r == 0:
                    ax.set_title(f"{ALPHABET[ci]} Hz", fontsize=10)
                continue
            # Best-matching raw prototype for this frequency (the one the
            # with-replacement metric credits), and its top-10 average.
            best_j, best_cos = None, -np.inf
            for j in idx:
                _, d = best_lag_align(P[j], gref)
                cos = 1.0 - d ** 2 / 2.0
                if cos > best_cos:
                    best_cos, best_j = cos, j
            raw_aligned, d_raw = best_lag_align(P[best_j], gref)
            avg, n_used = top_k_average(P[best_j], signal)
            avg_aligned, d_avg = best_lag_align(avg, gref)
            cos_raw = 1.0 - d_raw ** 2 / 2.0
            cos_avg = 1.0 - d_avg ** 2 / 2.0
            summary.append((METHOD_LABEL[m], int(ALPHABET[ci]), cos_raw, cos_avg,
                            n_used, counts_by_freq.get(int(ALPHABET[ci]), 0)))

            ax.plot(_znorm(gref), color="0.6", lw=2.4, alpha=0.8, zorder=1)
            ax.plot(raw_aligned, color=METHOD_COLOR[m], lw=1.1, alpha=0.55,
                    ls="--", zorder=2)
            ax.plot(avg_aligned, color=METHOD_COLOR[m], lw=1.8, zorder=3)
            better = cos_avg >= cos_raw
            ax.text(0.03, 0.03,
                    f"raw {cos_raw:.2f}\navg {cos_avg:.2f} (n={n_used})",
                    transform=ax.transAxes, fontsize=7, va="bottom",
                    color=("#1a7d3c" if better else "#b22222"),
                    linespacing=1.1)
            if r == 0:
                ax.set_title(f"{ALPHABET[ci]} Hz", fontsize=10)
        axes[r][0].set_ylabel(METHOD_LABEL[m], fontsize=11,
                              color=METHOD_COLOR[m], fontweight="bold")

    fig.suptitle(
        f"Effect of shift-aligned averaging on returned prototypes "
        f"(seed {seed}, Poisson spacing)\n"
        f"grey = clean ground truth; dashed = raw returned prototype; "
        f"solid = average of its top-{TOP_K} occurrences; "
        f"cosine-to-truth shown (green = averaging helps, red = hurts)",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out = args.out or str(rec_dir.joinpath(f"figs/averaging_seed-{seed}.pdf"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out} (seed {seed})")
    print("\nmethod        freq  n_act  n_used  cos_raw  cos_avg  delta")
    for lbl, f, cr, ca, nu, nact in summary:
        print(f"{lbl:13s} {f:4d} {nact:6d} {nu:6d}   {cr:6.3f}  {ca:6.3f}  "
              f"{ca - cr:+.3f}")


if __name__ == "__main__":
    main()
