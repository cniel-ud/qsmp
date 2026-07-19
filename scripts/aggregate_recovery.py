"""Score every method's prototypes and aggregate into a mean +/- 95% CI table.

This is the single scoring point of the recovery experiment. It discovers every
``results/recovery/<method>_seed-<seed>.npz`` file (raw prototypes saved by
``eval_recovery.py`` and ``snippetfinder_recovery.py``), re-derives the
ground truth for each file's seed/parameters, scores the prototypes with
:mod:`qsmp.eval_metrics` (identically across methods), pools per-seed metrics,
and prints (a) a human-readable summary and (b) a LaTeX ``tabular``. Confidence
intervals are the 95% t-interval over seeds.

Because scoring lives only here -- not in the runners -- metrics can be changed
and everything re-scored without re-running any (expensive, GPU) method, and the
methods may be produced in any order.

Usage::

    python scripts/aggregate_recovery.py --root .
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy import stats

from qsmp import eval_metrics as em

METHOD_LABEL = {"qsmp": "QSMP", "snippetfinder": "Snippet-Finder",
                "sikmeans": "sikmeans"}
METHOD_ORDER = ["qsmp", "snippetfinder", "sikmeans"]
# (key, LaTeX header, format, higher-is-better)
METRICS = [
    ("n_freqs_recovered", r"\#freq $\uparrow$", "{:.1f}", True),
    ("recovery_cosine", r"CosSim $\uparrow$", "{:.2f}", True),
    ("recovery_error", r"RecErr $\downarrow$", "{:.2f}", False),
    ("peak_freq_error", r"PeakErr (Hz) $\downarrow$", "{:.1f}", False),
]


def ci95(x):
    """Mean and half-width of the 95% t-CI of a 1D sample (NaNs dropped)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.nan, np.nan
    mean = x.mean()
    if n == 1:
        return mean, np.nan
    sem = x.std(ddof=1) / np.sqrt(n)
    h = sem * stats.t.ppf(0.975, n - 1)
    return mean, h


def _gt_cache_key(meta):
    return (int(meta["seed"]), int(meta["m"]), int(meta["n_waves"]),
            float(meta["noise_std"]), tuple(np.asarray(meta["freqs"]).tolist()))


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".")
    p.add_argument("--out", default=None,
                   help="Optional path to write the LaTeX table")
    args = p.parse_args()

    rec_dir = Path(args.root).joinpath("results", "recovery")
    files = sorted(rec_dir.glob("*_seed-*.npz"))
    if not files:
        raise SystemExit(f"No <method>_seed-*.npz found in {rec_dir}")

    # Score every prototype file, caching ground truth per (seed, params) so it
    # is computed once even though several methods share it.
    pooled = {}                       # method -> {metric_key: [values]}
    n_seeds = {}                      # method -> count
    gt_cache = {}
    for f in files:
        protos, meta = em.load_prototypes(f)
        mth = meta["method"]
        key = _gt_cache_key(meta)
        if key not in gt_cache:
            gt_cache[key] = em.gt_clean_prototypes(
                meta["freqs"], meta["seed"], wave_len=meta["m"],
                n_waves=meta["n_waves"], noise_std=meta["noise_std"])
        gt_protos, gt_freqs, _ = gt_cache[key]

        if protos.shape[0] == 0:      # method failed to return prototypes
            rec = {k: np.nan for k, *_ in METRICS}
        else:
            rec = em.prototype_recovery(protos, gt_protos, gt_freqs)

        pooled.setdefault(mth, {k: [] for k, *_ in METRICS})
        n_seeds[mth] = n_seeds.get(mth, 0) + 1
        for k, *_ in METRICS:
            pooled[mth][k].append(rec.get(k, np.nan))

    methods = [m for m in METHOD_ORDER if m in pooled] + \
              [m for m in pooled if m not in METHOD_ORDER]

    print(f"Files scored: {len(files)} "
          f"({', '.join(f'{m}:{n_seeds[m]}' for m in methods)})\n")

    # human-readable
    for mth in methods:
        label = METHOD_LABEL.get(mth, mth)
        print(f"{label} (n={n_seeds[mth]} seeds):")
        for key, hdr, fmt, _ in METRICS:
            mean, h = ci95(pooled[mth][key])
            hstr = "" if np.isnan(h) else " +/- " + fmt.format(h)
            print(f"  {hdr:<22} {fmt.format(mean)}{hstr}")
        print()

    # LaTeX
    header = " & ".join(["Method"] + [hdr for _, hdr, *_ in METRICS])
    lines = [
        r"\begin{table}[htb]",
        r"    \centering",
        r"    \caption{Ground-truth recovery on the \texttt{power-law} dataset "
        r"(mean $\pm$ 95\% CI over seeds). $\uparrow$/$\downarrow$: higher/lower "
        r"is better. \#freq is out of 6.}",
        r"    \label{tab:recovery}",
        r"    \resizebox{\linewidth}{!}{%",
        r"    \begin{tabular}{l" + "c" * len(METRICS) + "}",
        r"    \toprule",
        "    " + header + r" \\",
        r"    \midrule",
    ]
    for mth in methods:
        cells = [METHOD_LABEL.get(mth, mth)]
        for key, _, fmt, _ in METRICS:
            mean, h = ci95(pooled[mth][key])
            if np.isnan(mean):
                cells.append("--")
            elif np.isnan(h):
                cells.append(fmt.format(mean))
            else:
                cells.append(f"{fmt.format(mean)}$\\pm${fmt.format(h)}")
        lines.append("    " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"    \end{tabular}}", r"\end{table}"]
    latex = "\n".join(lines)

    print("=" * 60)
    print("LaTeX table:\n")
    print(latex)
    if args.out:
        Path(args.out).write_text(latex + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
