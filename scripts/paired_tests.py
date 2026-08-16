"""Paired two-sample tests between methods on the recovery metrics.

The methods are evaluated on the *same* per-seed signals (``morlet`` power-law
dataset is deterministic in the seed), so a given seed yields matched
observations across methods. This script pairs them by seed and, for each
metric, reports:

- a **paired t-test** (``scipy.stats.ttest_rel``), and
- a **Wilcoxon signed-rank test** (``scipy.stats.wilcoxon``) as a
  distribution-free check -- advisable here because ``n_freqs_recovered`` is a
  small, bounded integer count for which the t-test's normality assumption is
  weak,

plus the mean paired difference, the per-seed win/loss/tie tally, and the
paired effect size Cohen's ``dz`` (mean difference / SD of differences).

It reuses the single scoring path (``qsmp.eval_metrics``) and the saved
prototype files -- no method is re-run. By default it contrasts the two
strongest methods, QSMP vs. Snippet-Finder, on all three metrics.

Usage::

    python scripts/paired_tests.py --root .
    python scripts/paired_tests.py --root . --a qsmp --b sikmeans
    python scripts/paired_tests.py --root . --metrics n_freqs_recovered recovery_cosine
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy import stats

from qsmp import eval_metrics as em

# (key, label, higher-is-better)
METRICS = {
    "n_freqs_recovered": ("#freq recovered", True),
    "recovery_cosine": ("CosSim", True),
    "peak_freq_error": ("PeakErr (Hz)", False),
}
METHOD_LABEL = {"qsmp": "QSMP", "snippetfinder": "Snippet-Finder",
                "sikmeans": "sikmeans"}


def score_method(rec_dir, method, gt_cache):
    """Return ``{seed: {metric: value}}`` for one method (ground truth cached)."""
    out = {}
    for f in sorted(rec_dir.glob(f"{method}_seed-*.npz")):
        protos, meta = em.load_prototypes(f)
        seed = int(meta["seed"])
        key = (seed, int(meta["m"]), int(meta["n_waves"]),
               float(meta["noise_std"]), str(meta["spacing"]))
        if key not in gt_cache:
            gt_cache[key] = em.gt_clean_prototypes(
                np.asarray(meta["freqs"]), seed, wave_len=int(meta["m"]),
                n_waves=int(meta["n_waves"]), noise_std=float(meta["noise_std"]),
                spacing=str(meta["spacing"]))
        gt_protos, gt_freqs, _ = gt_cache[key]
        protos = np.atleast_2d(protos)
        if protos.shape[0] == 0:
            out[seed] = {k: np.nan for k in METRICS}
        else:
            out[seed] = em.prototype_recovery(protos, gt_protos, gt_freqs)
    return out


def paired_report(a_vals, b_vals, a_label, b_label, metric_key):
    """Print the paired comparison of one metric between two methods."""
    label, higher = METRICS[metric_key]
    seeds = sorted(set(a_vals) & set(b_vals))
    a = np.array([a_vals[s][metric_key] for s in seeds], dtype=float)
    b = np.array([b_vals[s][metric_key] for s in seeds], dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = a.size
    d = a - b                                   # a minus b
    # Orient "wins" toward whichever direction is better for this metric.
    a_better = (d > 0) if higher else (d < 0)
    b_better = (d < 0) if higher else (d > 0)

    t, p_t = stats.ttest_rel(a, b)
    try:
        w, p_w = stats.wilcoxon(a, b)
        wil = f"W={w:.1f}, p={p_w:.4f}"
    except ValueError as e:                     # all-zero differences
        wil = f"n/a ({e})"
    sd = d.std(ddof=1)
    dz = d.mean() / sd if sd > 0 else np.nan

    arrow = "higher better" if higher else "lower better"
    print(f"[{label}]  ({arrow};  n={n} paired seeds)")
    print(f"  {a_label:15s} mean={a.mean():.3f}  sd={a.std(ddof=1):.3f}")
    print(f"  {b_label:15s} mean={b.mean():.3f}  sd={b.std(ddof=1):.3f}")
    print(f"  paired diff ({a_label}-{b_label}): mean={d.mean():+.3f}  sd={sd:.3f}")
    print(f"  wins: {a_label} {int(a_better.sum())}, "
          f"{b_label} {int(b_better.sum())}, tie {int((d == 0).sum())}")
    print(f"  paired t-test:        t={t:+.3f}, p={p_t:.4f}")
    print(f"  Wilcoxon signed-rank: {wil}")
    print(f"  Cohen's dz:           {dz:+.3f}")
    print()


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".")
    p.add_argument("--a", default="qsmp", choices=list(METHOD_LABEL),
                   help="First method (default: qsmp)")
    p.add_argument("--b", default="snippetfinder", choices=list(METHOD_LABEL),
                   help="Second method (default: snippetfinder)")
    p.add_argument("--metrics", nargs="+", default=list(METRICS),
                   choices=list(METRICS), help="Metrics to test")
    p.add_argument("--spacing", choices=["poisson", "uniform"], default="poisson",
                   help="Which result set to test (default poisson, the "
                        "supplement's harder signal). Reads "
                        "results/recovery/<spacing>/.")
    args = p.parse_args()

    rec_dir = em.resolve_rec_dir(args.root, args.spacing)
    if not any(f.parent == rec_dir for f in rec_dir.glob("*_seed-*.npz")):
        raise SystemExit(f"No prototype files found in {rec_dir}")

    gt_cache = {}
    a_vals = score_method(rec_dir, args.a, gt_cache)
    b_vals = score_method(rec_dir, args.b, gt_cache)
    a_label, b_label = METHOD_LABEL[args.a], METHOD_LABEL[args.b]

    print(f"Paired tests: {a_label} vs. {b_label}\n" + "=" * 44 + "\n")
    for key in args.metrics:
        paired_report(a_vals, b_vals, a_label, b_label, key)


if __name__ == "__main__":
    main()
