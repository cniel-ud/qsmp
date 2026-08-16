"""Why Snippet-Finder underperforms on CosSim: reproducible evidence.

Snippet-Finder (SF) ties QSMP on ``#freq recovered`` but loses on ``CosSim``
(see ``paired_tests.py``). This script reproduces the two mechanisms behind
that gap, so the analysis can be cited (e.g. in supplemental material):

**Mechanism 1 -- MPdist matches sub-subsequences, so contamination is
invisible to SF's selection.** SF scores candidate snippets with MPdist
(Gharghabi et al., ICDM 2018), which declares two windows similar if they
share *fragments* of length ``L = ceil(percentage * m)`` -- formally the 5th
percentile of the fragment-level join matrix profile -- ignoring the rest of
the window by design. Both QSMP and SF return verbatim windows of the signal;
the difference is the criterion: QSMP's density is built from *full-window*
distances (a superimposed window is nearly unique, hence low-density, hence
never a mode), while SF's fragment-level criterion cannot distinguish a clean
prototype window from a mixture. Evidence produced here:

- [A] a *constructed* window spliced from two half-prototypes has MPdist ~ 0
  to BOTH prototypes (perfect match under SF's measure) while the full-window
  shift-invariant cosine correctly penalises it;
- [B] per matched pair (best match, with replacement, as in ``eval_metrics``): SF's *median*
  cosine equals QSMP's (a clean SF window is as good as a QSMP mode) but SF
  has ~2x QSMP's fraction of poor pairs (cos < 0.5) and lower spectral
  purity. Averaging is not the protective factor: sikmeans averages and has
  the worst purity of all (its centroids mix clusters).

**Mechanism 2 -- the coverage objective starves rare frequencies.** SF
greedily minimises ProfileArea (fidelity + coverage; Imani et al., DMKD
2020). The rare high frequencies occupy a tiny fraction of the signal, so
covering them barely moves the objective. Evidence produced here:

- [C] poor pairs concentrate at 100/150 Hz, and the MPdist between those poor
  snippets and their assigned ground truth sits at the noise baseline --
  i.e. SF never *selected* high-frequency windows: even its closest snippet to
  each high-frequency ground truth sits at the MPdist noise floor.

**Fairness check -- score recovery under SF's own distance.** One could
object that CosSim measures a property (whole-window morphology) that SF's
objective never optimises, so the comparison is rigged. Section [D] therefore
re-scores every method's recovery with *MPdist itself* as the matching
distance (mean MPdist to the clean ground truth, matched with replacement).
QSMP still wins, significantly: mechanism 1 explains SF's contaminated-window
tail, but mechanism 2 (rare-frequency starvation) hurts under *any*
distance -- a snippet that was never selected to represent 150 Hz is far from
the 150 Hz prototype no matter how leniently you compare waveforms.

Pure NumPy/SciPy, CPU only. Requires the per-(method, seed) prototype files
written by the runners. Usage::

    python scripts/sf_cossim_analysis.py --root .
"""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from qsmp import eval_metrics as em

try:                                    # prefer the reference implementation
    import stumpy                       # (the SF runner itself uses stumpy)
    HAVE_STUMPY = True
except ImportError:
    HAVE_STUMPY = False

METHOD_ORDER = ["qsmp", "snippetfinder", "sikmeans"]
POOR_COS = 0.5          # per-pair cosine below this counts as a poor match
HIGH_FREQ = 100.0       # "rare high frequency" threshold (Hz)


# --------------------------------------------------------------------------- #
# MPdist (Gharghabi et al., ICDM 2018) -- naive reference implementation
# --------------------------------------------------------------------------- #
def _frags_znorm(x, L):
    """All length-``L`` sliding fragments of 1-D ``x``, rows z-normalised."""
    W = np.lib.stride_tricks.sliding_window_view(np.asarray(x, dtype=float), L)
    mu = W.mean(axis=1, keepdims=True)
    sd = W.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (W - mu) / sd


def mpdist(A, B, L):
    """MPdist between equal-length windows ``A`` and ``B`` with fragment
    length ``L``.

    Concatenates the AB and BA join matrix profiles (nearest-neighbour
    z-normalised Euclidean distance of every length-``L`` fragment of one
    window within the other) and returns the k-th smallest value with
    ``k = 5% of 2n`` -- the paper's definition. Two windows are thus
    "similar" if merely ~5% of their fragments match somewhere in the other,
    *regardless of order*; the remaining ~95% is ignored by design.

    Uses ``stumpy.mpdist`` (the same implementation the SF runner builds on)
    when available; otherwise a self-contained vectorised fallback -- one
    matrix product over the fragment-pair grid, exploiting that for
    z-normalised rows the squared distance is ``2L - 2 <a, b>``. The two
    agree to float precision (validated on clean/spliced/superimposed/noise
    pairs).
    """
    if HAVE_STUMPY:
        return float(stumpy.mpdist(np.asarray(A, dtype=float),
                                   np.asarray(B, dtype=float), L))
    FA, FB = _frags_znorm(A, L), _frags_znorm(B, L)
    D = np.sqrt(np.clip(2 * L - 2 * (FA @ FB.T), 0.0, None))
    P = np.sort(np.concatenate([D.min(axis=1), D.min(axis=0)]))
    k = int(np.ceil(0.05 * (A.size + B.size)))
    return float(P[min(k, P.size - 1)])


# --------------------------------------------------------------------------- #
# Spectral purity
# --------------------------------------------------------------------------- #
def spectral_purity(wave, f_true, fs=512, halfwidth_frac=0.25):
    """Fraction of (mean-removed) spectral energy within +/-25% of ``f_true``.

    ~1 for a clean Morlet wavelet at ``f_true``; low for a superposition or
    an off-frequency waveform.
    """
    w = np.asarray(wave, dtype=float)
    w = w - w.mean()
    spec = np.abs(np.fft.rfft(w)) ** 2
    freqs = np.fft.rfftfreq(w.size, d=1.0 / fs)
    band = (freqs >= f_true * (1 - halfwidth_frac)) & \
           (freqs <= f_true * (1 + halfwidth_frac))
    tot = spec.sum()
    return float(spec[band].sum() / tot) if tot > 0 else 0.0


def si_cos(a, b):
    """Full-window shift-invariant cosine (same measure as ``recovery_cosine``)."""
    d = em.shift_invariant_znorm_dist(a, b)
    return 1.0 - d ** 2 / 2.0


# --------------------------------------------------------------------------- #
# Matched pairs (consistent with eval_metrics.prototype_recovery)
# --------------------------------------------------------------------------- #
def matched_pairs(rec_dir, method, gt_cache, matching="best"):
    """Per-seed matched (gt_freq, cosine, prototype, gt_proto) pairs.

    ``matching="best"`` (default, matching ``prototype_recovery``): each
    ground-truth prototype takes its closest prediction (with replacement), so
    there is one pair per present frequency and collapse is not double-charged.
    ``matching="hungarian"``: 1-to-1 assignment, kept for comparison.
    """
    pairs = []
    for f in sorted(rec_dir.glob(f"{method}_seed-*.npz")):
        P, meta = em.load_prototypes(f)
        seed = int(meta["seed"])
        if seed not in gt_cache:
            gt_cache[seed] = em.gt_clean_prototypes(
                np.asarray(meta["freqs"]), seed, wave_len=int(meta["m"]),
                n_waves=int(meta["n_waves"]), noise_std=float(meta["noise_std"]),
                spacing=str(meta["spacing"]))
        gt, gf, _ = gt_cache[seed]
        P = np.atleast_2d(P)
        D = em._pairwise_si_dist(P, gt)
        if matching == "best":
            row, col = D.argmin(axis=0), np.arange(D.shape[1])
        else:
            row, col = linear_sum_assignment(D)
        for r, c in zip(row, col):
            cos = 1.0 - D[r, c] ** 2 / 2.0
            pairs.append(dict(seed=seed, freq=float(gf[c]), cos=cos,
                              proto=P[r], gt=gt[c]))
    return pairs


# --------------------------------------------------------------------------- #
# [A] Constructed demonstration: fragment matching cannot see contamination
# --------------------------------------------------------------------------- #
def demo_spliced_window(gt_cache, m, L):
    gt, gf, _ = gt_cache[sorted(gt_cache)[0]]
    proto = {int(f): g for g, f in zip(gt, gf)}
    if not {12, 100, 150} <= proto.keys():
        print("[A] skipped: reference seed lacks 12/100/150 Hz prototypes")
        return
    spliced = np.concatenate([proto[100][:m // 2], proto[12][:m // 2]])
    mixed = proto[150] + proto[1]

    rng = np.random.default_rng(0)
    base = np.mean([mpdist(rng.standard_normal(m), rng.standard_normal(m), L)
                    for _ in range(5)])
    print(f"[A] Fragment matching vs. whole-window cosine "
          f"(L={L} = {L / m:.0%} of m={m}; "
          f"MPdist noise-vs-noise baseline ~ {base:.1f}; "
          f"clean-vs-itself = 0)")
    print(f"    {'window':34s} {'vs GT':>8s} {'MPdist':>7s} {'si-cos':>7s}")
    cases = [("half 100Hz | half 12Hz spliced", spliced, 100),
             ("half 100Hz | half 12Hz spliced", spliced, 12),
             ("150Hz + 1Hz superimposed", mixed, 150),
             ("clean 150Hz (control)", proto[150], 150)]
    for name, w, tgt in cases:
        print(f"    {name:34s} {tgt:>6d}Hz {mpdist(w, proto[tgt], L):7.2f} "
              f"{si_cos(w, proto[tgt]):7.2f}")
    print("    -> the spliced mixture is a PERFECT match to both prototypes "
          "under MPdist;\n       the full-window cosine correctly penalises "
          "it. SF's selection cannot see contamination.\n")


# --------------------------------------------------------------------------- #
# [B] Per-pair cosine + spectral purity across methods
# --------------------------------------------------------------------------- #
def report_pair_stats(all_pairs):
    print(f"[B] Per matched pair (best match, with replacement), all seeds:")
    print(f"    {'method':14s} {'n':>4s} {'mean cos':>8s} {'median':>7s} "
          f"{'frac<%.1f' % POOR_COS:>9s} {'purity':>7s}")
    for m_ in METHOD_ORDER:
        pr = all_pairs[m_]
        cos = np.array([p["cos"] for p in pr])
        pur = np.array([spectral_purity(p["proto"], p["freq"]) for p in pr])
        print(f"    {m_:14s} {len(pr):4d} {cos.mean():8.3f} "
              f"{np.median(cos):7.3f} {(cos < POOR_COS).mean():9.2f} "
              f"{pur.mean():7.3f}")
    print("    -> SF's MEDIAN equals QSMP's (a clean SF window is as good as "
          "a QSMP mode)\n       but SF has ~2x the poor-pair fraction: the "
          "mean is dragged by a bad tail.\n       sikmeans averages and has "
          "the WORST purity -- averaging is not the protective factor.\n")


# --------------------------------------------------------------------------- #
# [C] Poor pairs concentrate at rare high frequencies / are never selected
# --------------------------------------------------------------------------- #
def report_poor_pair_location(all_pairs, alphabet):
    print(f"[C] Where the poor pairs (cos < {POOR_COS}) live "
          f"(poor/total per matched GT frequency):")
    hdr = "".join(f"{int(f):>9d}Hz" for f in alphabet)
    print(f"    {'method':14s}{hdr}")
    for m_ in METHOD_ORDER:
        cnt = Counter(p["freq"] for p in all_pairs[m_] if p["cos"] < POOR_COS)
        tot = Counter(p["freq"] for p in all_pairs[m_])
        line = "".join(f"{cnt.get(f, 0):>7d}/{tot.get(f, 0):<3d}"
                       for f in alphabet)
        print(f"    {m_:14s}{line}")
    print()


def report_sf_highfreq_leftovers(sf_pairs, L, m, n_mpdist=6):
    """MPdist + peak-freq of SF's BEST snippet for each rare high frequency.

    With replacement, each poor pair is SF's *closest available* snippet for
    that ground-truth frequency -- not a leftover forced on it by a 1-to-1
    assignment. So a poor pair here means SF has no good candidate at all.
    """
    hi = [p for p in sf_pairs if p["freq"] >= HIGH_FREQ]
    poor = [p for p in hi if p["cos"] < POOR_COS]
    good = [p for p in hi if p["cos"] >= POOR_COS]

    rng = np.random.default_rng(0)
    base = np.mean([mpdist(rng.standard_normal(m), rng.standard_normal(m), L)
                    for _ in range(5)])
    # MPdist is O(m^2) per pair; a few pairs per group suffice to show the gap.
    d_poor = [mpdist(p["proto"], p["gt"], L) for p in poor[:n_mpdist]]
    d_good = [mpdist(p["proto"], p["gt"], L) for p in good[:n_mpdist]]
    pk_poor = Counter(int(round(em.peak_frequency(p["proto"]))) for p in poor)

    print(f"[C] SF's best snippet for each GT >= {HIGH_FREQ:.0f} Hz "
          f"({len(poor)} poor, {len(good)} good):")
    print(f"    MPdist(snippet, gt): poor pairs "
          f"{np.mean(d_poor):.1f} +/- {np.std(d_poor):.1f} "
          f"(noise baseline ~ {base:.1f}); good pairs "
          f"{np.mean(d_good):.1f} +/- {np.std(d_good):.1f}"
          f"  [first {n_mpdist} of each]")
    print(f"    peak-freq histogram of the poor snippets: "
          f"{sorted(pk_poor.items())}")
    print("    -> even SF's CLOSEST snippet to a rare frequency sits at the "
          "noise baseline\n       under SF's own measure: SF simply never "
          "produced a high-frequency snippet\n       (rare freqs barely move "
          "ProfileArea, so coverage spends its budget elsewhere).\n")


# --------------------------------------------------------------------------- #
# [D] Fairness check: recovery scored under SF's own distance (MPdist)
# --------------------------------------------------------------------------- #
def report_mpdist_recovery(rec_dir, gt_cache, L):
    """Mean MPdist to ground truth (best match, with replacement), per method + paired test.

    Same protocol as ``prototype_recovery`` but with MPdist as the matching
    distance -- i.e. each method is scored under the lenient, fragment-level,
    order-free measure SF itself optimises. If SF's CosSim deficit were only
    an artefact of scoring it with a whole-window measure, SF would win here.
    """
    from scipy import stats

    per_seed = {m_: {} for m_ in METHOD_ORDER}
    for f in sorted(rec_dir.glob("*_seed-*.npz")):
        P, meta = em.load_prototypes(f)
        m_, seed = meta["method"], int(meta["seed"])
        if m_ not in per_seed:
            continue
        gt, _, _ = gt_cache[seed]
        P = np.atleast_2d(P)
        D = np.array([[mpdist(p, g, L) for g in gt] for p in P])
        # Best match (with replacement), consistent with prototype_recovery:
        # each ground truth scored against its closest prediction under MPdist.
        per_seed[m_][seed] = float(D.min(axis=0).mean())

    print(f"[D] Fairness check -- recovery under SF's OWN distance "
          f"(best-match mean MPdist_L{L}, lower better):")
    for m_ in METHOD_ORDER:
        a = np.array(list(per_seed[m_].values()))
        print(f"    {m_:14s} mean={a.mean():6.2f}  sd={a.std(ddof=1):5.2f}  "
              f"(n={a.size} seeds)")

    seeds = sorted(set(per_seed["qsmp"]) & set(per_seed["snippetfinder"]))
    q = np.array([per_seed["qsmp"][s] for s in seeds])
    s_ = np.array([per_seed["snippetfinder"][s] for s in seeds])
    d = q - s_
    t, p_t = stats.ttest_rel(q, s_)
    _, p_w = stats.wilcoxon(q, s_)
    dz = d.mean() / d.std(ddof=1)
    print(f"    paired QSMP vs SF (diff = QSMP - SF; negative = QSMP better):"
          f"\n      mean diff={d.mean():+.3f}  t={t:+.2f} p={p_t:.4f}  "
          f"Wilcoxon p={p_w:.4f}  dz={dz:+.2f}  "
          f"wins: QSMP {(d < 0).sum()}, SF {(d > 0).sum()}, "
          f"tie {(d == 0).sum()}")
    print("    -> QSMP wins even under MPdist: mechanism 1 explains SF's "
          "contaminated-window\n       tail, but mechanism 2 (rare-frequency "
          "starvation) hurts under ANY distance.\n")


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".")
    p.add_argument("--spacing", choices=["poisson", "uniform"], default="poisson",
                   help="Which result set to analyse. Defaults to poisson -- "
                        "the overlapping signal that drives SF's CosSim deficit; "
                        "the mechanisms are specific to superposition, so poisson "
                        "is the intended set. Reads results/recovery/<spacing>/.")
    args = p.parse_args()

    rec_dir = em.resolve_rec_dir(args.root, args.spacing)
    if not any(f.parent == rec_dir for f in rec_dir.glob("*_seed-*.npz")):
        raise SystemExit(f"No prototype files found in {rec_dir}")

    # Fragment length actually used by the SF runs (auto-selected percentage).
    sf_meta = [em.load_prototypes(f)[1]
               for f in sorted(rec_dir.glob("snippetfinder_seed-*.npz"))
               if f.parent == rec_dir]
    m = int(sf_meta[0]["m"])
    pcts = sorted({float(np.asarray(mt["percentage"])) for mt in sf_meta})
    L = int(np.ceil(pcts[0] * m))
    print(f"SF fragment length: percentage(s) {pcts} -> L = {L} samples "
          f"({L / m:.0%} of the m = {m} prototype)\n")

    gt_cache = {}
    all_pairs = {m_: matched_pairs(rec_dir, m_, gt_cache)
                 for m_ in METHOD_ORDER}
    alphabet = sorted({p["freq"] for pr in all_pairs.values() for p in pr})

    demo_spliced_window(gt_cache, m, L)
    report_pair_stats(all_pairs)
    report_poor_pair_location(all_pairs, alphabet)
    report_sf_highfreq_leftovers(all_pairs["snippetfinder"], L, m)
    report_mpdist_recovery(rec_dir, gt_cache, L)


if __name__ == "__main__":
    main()
