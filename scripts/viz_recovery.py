"""Qualitative visualisations for the ground-truth recovery experiment.

Complements the three quantitative metrics (``aggregate_recovery.py``) with a
qualitative story, using the *same* peak-frequency snapping and shift-invariant
matching the metrics use, so the figures and the table stay consistent.

It discovers the per-(method, seed) prototype files written by the runners
(``results/recovery/<method>_seed-<seed>.npz``), re-derives the deterministic
ground truth for each seed, and produces four figures under
``results/recovery/figs/``:

1. ``gallery_seed-<s>.png`` -- for a representative seed (chosen automatically as
   the one whose per-method ``#freq`` is closest to each method's mean, so it is
   not cherry-picked), a grid with one row per method and one column per
   alphabet frequency. Each recovered prototype is drawn in the column of its
   snapped spectral peak, shift-aligned and overlaid on the clean ground-truth
   prototype (grey). Empty columns are *misses*; several curves stacked in one
   column are *collapse*. Shows morphology fidelity, misses and collapse at once.
2. ``coverage_heatmap.png`` -- fraction of the 20 seeds in which each method
   recovers each alphabet frequency (method x frequency). The failure story:
   which frequencies are systematically missed.
3. ``collapse_multiplicity.png`` -- mean number of recovered prototypes that
   snap to each alphabet frequency, per method. Reveals prototypes piling onto
   the prevalent low frequencies while the rare high ones are starved.
4. ``metric_distributions.png`` -- per-seed spread (strip + box) of the three
   metrics per method, so the reader sees variability, not just mean +/- CI.

CPU only. Run after the prototype files exist::

    python scripts/viz_recovery.py --root .
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qsmp import eval_metrics as em

ALPHABET = np.array([1, 5, 12, 30, 100, 150])
METHOD_ORDER = ["qsmp", "snippetfinder", "sikmeans"]
METHOD_LABEL = {"qsmp": "QSMP", "snippetfinder": "Snippet-Finder",
                "sikmeans": "sikmeans"}
METHOD_COLOR = {"qsmp": "#1b9e77", "snippetfinder": "#7570b3",
                "sikmeans": "#d95f02"}


# --------------------------------------------------------------------------- #
# Loading + ground truth
# --------------------------------------------------------------------------- #
def load_all(rec_dir):
    """Return ``{method: {seed: prototypes}}`` and ``{seed: (gt, gt_freqs)}``."""
    protos = {}
    gt_cache = {}
    for f in sorted(rec_dir.glob("*_seed-*.npz")):
        p, meta = em.load_prototypes(f)
        mth, seed = meta["method"], int(meta["seed"])
        protos.setdefault(mth, {})[seed] = np.atleast_2d(p)
        if seed not in gt_cache:
            g, gfreqs, _ = em.gt_clean_prototypes(
                np.asarray(meta["freqs"]), seed, wave_len=int(meta["m"]),
                n_waves=int(meta["n_waves"]), noise_std=float(meta["noise_std"]))
            gt_cache[seed] = (np.atleast_2d(g), np.asarray(gfreqs, dtype=float))
    return protos, gt_cache


# --------------------------------------------------------------------------- #
# Alignment for display (consistent with em._pairwise_si_dist)
# --------------------------------------------------------------------------- #
def _znorm(a, eps=1e-8):
    a = np.asarray(a, dtype=float)
    sd = a.std()
    return (a - a.mean()) / (sd if sd > eps else 1.0)


def best_lag_align(rec, gt, max_lag=None, min_overlap=None):
    """Shift-align ``rec`` to ``gt`` at the lag minimising the z-norm distance.

    Mirrors :func:`em._pairwise_si_dist` for a single pair -- same full-range
    lag search and ``m/2`` minimum-overlap guard -- so the alignment drawn in
    the gallery matches the distance/cosine the metrics report. Returns the
    z-normalised, lag-shifted ``rec`` (rolled; the Morlet envelope decays to ~0
    at the edges so the wrap is negligible for display) and the achieved
    distance.
    """
    zr, zg = _znorm(rec), _znorm(gt)
    m = zr.size
    if max_lag is None:
        max_lag = m - 1
    if min_overlap is None:
        min_overlap = m // 2
    best_d, best_lag = np.inf, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = zr[:m + lag], zg[-lag:]
        elif lag > 0:
            a, b = zr[lag:], zg[:m - lag]
        else:
            a, b = zr, zg
        if a.size < min_overlap:
            continue
        az, bz = _znorm(a), _znorm(b)
        corr = float(az @ bz) / az.size
        d = np.sqrt(max(2.0 - 2.0 * corr, 0.0))
        if d < best_d:
            best_d, best_lag = d, lag
    # In _pairwise_si_dist's convention a positive ``lag`` compares ``rec[lag:]``
    # against ``gt[:m-lag]`` -- i.e. ``rec`` is shifted LEFT by ``lag`` to meet
    # ``gt``. To overlay ``rec`` on ``gt`` for display we therefore roll by
    # ``-best_lag`` (rolling by ``+best_lag`` would double the offset).
    return _znorm(np.roll(zr, -best_lag)), best_d


def snapped_columns(prototypes):
    """Alphabet-column index each prototype snaps to (by spectral peak)."""
    cols = []
    for p in prototypes:
        f = em.snap_to_alphabet(em.peak_frequency(p), ALPHABET)
        cols.append(int(np.argmin(np.abs(ALPHABET - f))))
    return np.asarray(cols)


# --------------------------------------------------------------------------- #
# Figure 1: prototype gallery for one representative seed
# --------------------------------------------------------------------------- #
def pick_representative_seed(protos, gt_cache):
    """Seed whose per-method ``#freq`` is jointly closest to each method's mean."""
    seeds = sorted(gt_cache)
    nfreq = {m: {} for m in METHOD_ORDER}
    for m in METHOD_ORDER:
        for s in seeds:
            if s in protos.get(m, {}):
                gt, gf = gt_cache[s]
                rec = em.prototype_recovery(protos[m][s], gt, gf)
                nfreq[m][s] = rec["n_freqs_recovered"]
    means = {m: np.mean(list(v.values())) for m, v in nfreq.items() if v}
    score = {}
    for s in seeds:
        if all(s in nfreq[m] for m in means):
            score[s] = sum(abs(nfreq[m][s] - means[m]) for m in means)
    return min(score, key=score.get) if score else seeds[0]


def fig_gallery(protos, gt_cache, seed, out_path):
    gt, gt_freqs = gt_cache[seed]
    # map alphabet index -> clean GT prototype (only for present freqs)
    gt_by_col = {}
    for g, f in zip(gt, gt_freqs):
        gt_by_col[int(np.argmin(np.abs(ALPHABET - f)))] = g

    methods = [m for m in METHOD_ORDER if seed in protos.get(m, {})]
    ncol = ALPHABET.size
    fig, axes = plt.subplots(len(methods), ncol, figsize=(2.3 * ncol, 2.1 * len(methods)),
                             squeeze=False, sharex=True)

    for r, m in enumerate(methods):
        P = protos[m][seed]
        cols = snapped_columns(P)
        hit_cols = set()
        for ci in range(ncol):
            ax = axes[r][ci]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_alpha(0.25)
            # ground truth reference (grey)
            gref = gt_by_col.get(ci)
            if gref is not None:
                ax.plot(_znorm(gref), color="0.6", lw=2.4, alpha=0.8, zorder=1)
            # recovered prototypes that snapped to this column
            idx = np.where(cols == ci)[0]
            # Draw every prototype in the column, but track the BEST match --
            # the closest prediction to this ground-truth frequency, which is
            # exactly what the (with-replacement) recovery metric credits -- so
            # the label reflects the metric, not an arbitrary first prototype.
            best_cos, best_df = -np.inf, None
            for j in idx:
                if gref is not None:
                    aligned, d = best_lag_align(P[j], gref)
                    # cosine similarity (shift-invariant): cos = 1 - d^2/2, the
                    # same measure the RecErr distance is derived from.
                    cos = 1.0 - d ** 2 / 2.0
                    if cos > best_cos:
                        best_cos = cos
                        best_df = abs(em.peak_frequency(P[j]) - ALPHABET[ci])
                else:
                    aligned = _znorm(P[j])
                ax.plot(aligned, color=METHOD_COLOR[m], lw=1.4, alpha=0.9, zorder=2)
            if idx.size and gref is not None:
                # best cos/Delta-f in this column (see caption: for collapsed
                # columns, marked x n, this is the best-matching prototype).
                lbl = f"cos={best_cos:.2f}\n$\\Delta$f={best_df:.0f}Hz"
                ax.text(0.03, 0.03, lbl, transform=ax.transAxes, fontsize=7,
                        color=METHOD_COLOR[m], va="bottom", linespacing=1.1)
                hit_cols.add(ci)
            if len(idx) > 1:                      # collapse marker
                # White bbox so it stays legible even where the trace passes
                # through the top-right corner (e.g. QSMP's 1 Hz plateau).
                ax.text(0.97, 0.92, rf"$\times${len(idx)}", transform=ax.transAxes,
                        fontsize=9, ha="right", va="top", color=METHOD_COLOR[m],
                        fontweight="bold", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec=METHOD_COLOR[m], alpha=0.85, lw=0.8))
            if ci not in hit_cols and gref is not None and len(idx) == 0:
                ax.text(0.5, 0.5, "missed", transform=ax.transAxes, fontsize=8,
                        ha="center", va="center", color="0.5", style="italic")
            if r == 0:
                ax.set_title(f"{ALPHABET[ci]} Hz", fontsize=10)
        axes[r][0].set_ylabel(METHOD_LABEL[m], fontsize=11,
                              color=METHOD_COLOR[m], fontweight="bold")

    n_present = int(np.isin(np.arange(ncol),
                            [int(np.argmin(np.abs(ALPHABET - f))) for f in gt_freqs]).sum())
    fig.suptitle(f"Recovered prototypes vs. ground truth (seed {seed}; "
                 f"{n_present}/6 frequencies present)\n"
                 f"grey = clean ground truth, colour = recovered (shift-aligned); "
                 f"cos = shift-inv. cosine similarity, "
                 r"$\Delta$f = peak-freq. error, "
                 r"'$\times n$' = $n$ prototypes on one frequency "
                 r"(cos/$\Delta$f are for the best-matching one)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return seed


# --------------------------------------------------------------------------- #
# Figure 2 + 3: coverage heatmap and collapse multiplicity (over all seeds)
# --------------------------------------------------------------------------- #
def coverage_and_multiplicity(protos, gt_cache):
    """Return per-method (coverage_frac, mean_multiplicity) over the alphabet."""
    ncol = ALPHABET.size
    cov = {m: np.zeros(ncol) for m in METHOD_ORDER}
    mult = {m: np.zeros(ncol) for m in METHOD_ORDER}
    nseed = {m: 0 for m in METHOD_ORDER}
    for m in METHOD_ORDER:
        for s, P in protos.get(m, {}).items():
            cols = snapped_columns(P)
            counts = np.bincount(cols, minlength=ncol)
            cov[m] += (counts > 0)
            mult[m] += counts
            nseed[m] += 1
    for m in METHOD_ORDER:
        if nseed[m]:
            cov[m] /= nseed[m]
            mult[m] /= nseed[m]
    return cov, mult, nseed


def fig_coverage(cov, nseed, out_path):
    methods = [m for m in METHOD_ORDER if nseed[m]]
    M = np.vstack([cov[m] for m in methods])
    fig, ax = plt.subplots(figsize=(1.1 * ALPHABET.size + 1.5, 0.9 * len(methods) + 1.5))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(ALPHABET.size))
    ax.set_xticklabels([f"{f} Hz" for f in ALPHABET])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_LABEL[m] for m in methods])
    for i in range(len(methods)):
        for j in range(ALPHABET.size):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.6 else "black", fontsize=9)
    ax.set_title("Fraction of seeds in which each frequency is recovered")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="recovery rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_multiplicity(mult, nseed, out_path):
    methods = [m for m in METHOD_ORDER if nseed[m]]
    x = np.arange(ALPHABET.size)
    w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(1.1 * ALPHABET.size + 1.5, 4))
    for i, m in enumerate(methods):
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, mult[m], w,
               label=METHOD_LABEL[m], color=METHOD_COLOR[m], alpha=0.9)
    ax.axhline(1.0, color="0.4", ls="--", lw=1,
               label="1 prototype (ideal)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f} Hz" for f in ALPHABET])
    ax.set_ylabel("mean # recovered prototypes")
    ax.set_title("Where each method spends its k=6 prototypes\n"
                 "(>1 = collapse/redundancy; 0 = frequency starved)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 4: per-seed metric distributions
# --------------------------------------------------------------------------- #
def fig_metric_distributions(protos, gt_cache, out_path):
    metrics = [("n_freqs_recovered", "#freq recovered (of 6)", True),
               ("recovery_cosine", "CosSim (shift-inv. cosine)", True),
               ("peak_freq_error", "PeakErr (Hz)", False)]
    methods = [m for m in METHOD_ORDER if protos.get(m)]
    vals = {m: {k: [] for k, *_ in metrics} for m in methods}
    for m in methods:
        for s, P in protos[m].items():
            gt, gf = gt_cache[s]
            rec = em.prototype_recovery(P, gt, gf)
            for k, *_ in metrics:
                vals[m][k].append(rec[k])

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4))
    rng = np.random.default_rng(0)
    for ax, (k, title, higher) in zip(axes, metrics):
        data = [vals[m][k] for m in methods]
        bp = ax.boxplot(data, positions=range(len(methods)), widths=0.5,
                        showfliers=False, patch_artist=True, zorder=1)
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(METHOD_COLOR[m]); patch.set_alpha(0.25)
        for med in bp["medians"]:
            med.set_color("0.2")
        for i, m in enumerate(methods):
            y = np.asarray(vals[m][k], dtype=float)
            xj = i + (rng.random(y.size) - 0.5) * 0.28
            ax.scatter(xj, y, s=18, color=METHOD_COLOR[m], alpha=0.8,
                       edgecolor="white", linewidth=0.4, zorder=2)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([METHOD_LABEL[m] for m in methods], rotation=15)
        arrow = "higher better" if higher else "lower better"
        ax.set_title(f"{title}\n({arrow})", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Per-seed metric distributions (20 seeds)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".")
    p.add_argument("--seed", type=int, default=None,
                   help="Force the gallery seed (default: representative seed)")
    args = p.parse_args()

    rec_dir = Path(args.root).joinpath("results", "recovery")
    fig_dir = rec_dir.joinpath("figs")
    fig_dir.mkdir(parents=True, exist_ok=True)

    protos, gt_cache = load_all(rec_dir)
    if not protos:
        raise SystemExit(f"No prototype files found in {rec_dir}")
    print("methods:", {m: len(v) for m, v in protos.items()})

    seed = args.seed if args.seed is not None else \
        pick_representative_seed(protos, gt_cache)
    g_path = fig_dir.joinpath(f"gallery_seed-{seed}.png")
    fig_gallery(protos, gt_cache, seed, g_path)
    print(f"wrote {g_path} (representative seed {seed})")

    cov, mult, nseed = coverage_and_multiplicity(protos, gt_cache)
    fig_coverage(cov, nseed, fig_dir.joinpath("coverage_heatmap.png"))
    print(f"wrote {fig_dir.joinpath('coverage_heatmap.png')}")
    fig_multiplicity(mult, nseed, fig_dir.joinpath("collapse_multiplicity.png"))
    print(f"wrote {fig_dir.joinpath('collapse_multiplicity.png')}")
    fig_metric_distributions(protos, gt_cache,
                             fig_dir.joinpath("metric_distributions.png"))
    print(f"wrote {fig_dir.joinpath('metric_distributions.png')}")


if __name__ == "__main__":
    main()
