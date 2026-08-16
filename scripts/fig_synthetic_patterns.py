"""Main-paper synthetic figure: the patterns each method returns.

One row per method, one column per returned pattern (k), each pattern z-normalised, 
drawn, and titled with its peak frequency. Within a row, patterns are ordered by 
peak frequency so misses and duplicates are easy to see. The seed is chosen automatically
as the one whose per-method number of recovered frequencies is jointly closest to each method's
mean (not cherry-picked), or forced with ``--seed``.

CPU only::

    python scripts/fig_synthetic_patterns.py --root . --spacing uniform \
        --out img/QSMP_vs_Snippet-Finder_vs_sikmeans_morlet.pdf
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qsmp import eval_metrics as em

METHOD_ORDER = ["qsmp", "snippetfinder", "sikmeans"]
METHOD_LABEL = {"qsmp": "QSMP", "snippetfinder": "Snippet-Finder",
                "sikmeans": "sikmeans"}
METHOD_COLOR = {"qsmp": "#1b9e77", "snippetfinder": "#7570b3",
                "sikmeans": "#d95f02"}


def load_all(rec_dir):
    """Return ``{method: {seed: prototypes}}`` and ``{seed: (gt, gt_freqs)}``."""
    protos, gt_cache = {}, {}
    for f in sorted(rec_dir.glob("*_seed-*.npz")):
        if f.parent != rec_dir:
            continue
        p, meta = em.load_prototypes(f)
        mth, seed = meta["method"], int(meta["seed"])
        protos.setdefault(mth, {})[seed] = np.atleast_2d(p)
        if seed not in gt_cache:
            g, gf, _ = em.gt_clean_prototypes(
                np.asarray(meta["freqs"]), seed, wave_len=int(meta["m"]),
                n_waves=int(meta["n_waves"]), noise_std=float(meta["noise_std"]),
                spacing=str(meta["spacing"]))
            gt_cache[seed] = (np.atleast_2d(g), np.asarray(gf, dtype=float))
    return protos, gt_cache


def pick_representative_seed(protos, gt_cache):
    """Seed whose per-method #freq is jointly closest to each method's mean."""
    seeds = sorted(gt_cache)
    nfreq = {m: {} for m in METHOD_ORDER}
    for m in METHOD_ORDER:
        for s in seeds:
            if s in protos.get(m, {}):
                gt, gf = gt_cache[s]
                nfreq[m][s] = em.prototype_recovery(
                    protos[m][s], gt, gf)["n_freqs_recovered"]
    means = {m: np.mean(list(v.values())) for m, v in nfreq.items() if v}
    score = {s: sum(abs(nfreq[m][s] - means[m]) for m in means)
             for s in seeds if all(s in nfreq[m] for m in means)}
    return min(score, key=score.get) if score else seeds[0]


def _znorm(a, eps=1e-8):
    a = np.asarray(a, dtype=float)
    sd = a.std()
    return (a - a.mean()) / (sd if sd > eps else 1.0)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".")
    p.add_argument("--spacing", choices=["uniform", "poisson"], default="uniform")
    p.add_argument("--seed", type=int, default=None,
                   help="Force the seed (default: representative seed)")
    p.add_argument("--out", default=None, help="Output path (.pdf or .png)")
    args = p.parse_args()

    rec_dir = em.resolve_rec_dir(args.root, args.spacing)
    protos, gt_cache = load_all(rec_dir)
    if not protos:
        raise SystemExit(f"No prototype files found in {rec_dir}")

    seed = args.seed if args.seed is not None else \
        pick_representative_seed(protos, gt_cache)
    methods = [m for m in METHOD_ORDER if seed in protos.get(m, {})]
    k = max(protos[m][seed].shape[0] for m in methods)

    fig, axes = plt.subplots(len(methods), k, squeeze=False,
                             figsize=(1.7 * k, 1.6 * len(methods)))
    for r, m in enumerate(methods):
        P = protos[m][seed]
        order = np.argsort([em.peak_frequency(w) for w in P])
        for c in range(k):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c < P.shape[0]:
                w = P[order[c]]
                ax.plot(_znorm(w), color=METHOD_COLOR[m], lw=1.0)
                ax.set_title(f"{em.peak_frequency(w):.0f} Hz", fontsize=9)
        axes[r][0].set_ylabel(METHOD_LABEL[m], fontsize=11,
                              color=METHOD_COLOR[m], fontweight="bold")

    fig.tight_layout()
    out = args.out or str(rec_dir.joinpath(f"method_patterns_seed-{seed}.pdf"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out} (seed {seed}; {len(methods)} methods x {k} patterns)")


if __name__ == "__main__":
    main()
