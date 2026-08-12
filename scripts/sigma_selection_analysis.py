"""Diagnose the unsupervised sigma-selection criterion for QSMP (supplement).

The max-min *diversity* criterion picks the sigma whose k modes are most
mutually distinct. This script quantifies its two documented limitations on the
Poisson recovery set, all from the saved winning-sigma prototype files
(``results/recovery/poisson/qsmp_seed-*.npz``, which record the chosen ``sigma``
and ``diversity``) plus the deterministic signal:

  1. Which sigma the criterion selects across seeds (it structurally prefers wide
     kernels, never exploring the small end of the grid), and mean FreqRec by
     chosen sigma.
  2. Frequency-collapse: how often two shape-distinct modes snap to the same
     peak frequency (a wasted slot + a missed frequency elsewhere), and how that
     concentrates at the low frequencies where a wavelet spans ~1 cycle over the
     window.
  3. The representative-seed 5 Hz failure: the raw cosine of QSMP's contaminated
     mode vs. the clean prototype, and the neighbour-count evidence that a clean
     cluster existed but a wide kernel washed out the local density difference.

CPU only::

    python scripts/sigma_selection_analysis.py --root . --seed 18
"""

from argparse import ArgumentParser
from collections import Counter, defaultdict

import numpy as np

from qsmp import eval_metrics as em
from qsmp.datasets import powerlaw_dataset

ALPHABET = np.array([1, 5, 12, 30, 100, 150])
SHAPE_DISTINCT = 0.5   # shift-inv z-norm distance above which two modes differ


def _load_qsmp(rec_dir):
    out = {}
    for f in sorted(rec_dir.glob("qsmp_seed-*.npz")):
        p, meta = em.load_prototypes(f)
        out[int(meta["seed"])] = (np.atleast_2d(p), meta)
    return out


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--seed", type=int, default=18, help="Seed to profile in detail")
    args = ap.parse_args()

    rec_dir = em.resolve_rec_dir(args.root, "poisson")
    data = _load_qsmp(rec_dir)
    if not data:
        raise SystemExit(f"No QSMP files in {rec_dir}")
    seeds = sorted(data)

    # --- 1. chosen sigma across seeds + FreqRec by sigma -------------------- #
    sig_by_seed = {s: float(np.asarray(m["sigma"]).ravel()[0]) for s, (_, m) in data.items()}
    sig_counts = Counter(round(v, 2) for v in sig_by_seed.values())
    freqrec_by_sigma = defaultdict(list)
    for s, (P, meta) in data.items():
        gt, gf, _ = em.gt_clean_prototypes(
            np.asarray(meta["freqs"]), s, wave_len=int(meta["m"]),
            n_waves=int(meta["n_waves"]), noise_std=float(meta["noise_std"]),
            spacing="poisson")
        n = em.prototype_recovery(P, gt, gf)["n_freqs_recovered"]
        freqrec_by_sigma[round(sig_by_seed[s], 2)].append(n)
    print(f"[sigma] chosen across {len(seeds)} seeds: "
          f"{dict(sorted(sig_counts.items()))}")
    for sg in sorted(freqrec_by_sigma):
        v = freqrec_by_sigma[sg]
        print(f"        sigma={sg}: n={len(v)}  mean FreqRec={np.mean(v):.2f}")

    # --- 2. frequency-collapse duplicates ----------------------------------- #
    seeds_with_dup, pairs, shape_distinct, dup_at_low = 0, 0, 0, 0
    for s, (P, _) in data.items():
        peaks = np.array([em.peak_frequency(w) for w in P])
        snapped = np.array([em.snap_to_alphabet(f, ALPHABET) for f in peaks])
        has_dup = False
        for f in np.unique(snapped):
            idx = np.where(snapped == f)[0]
            if idx.size < 2:
                continue
            has_dup = True
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    pairs += 1
                    d = em.shift_invariant_znorm_dist(P[idx[a]], P[idx[b]])
                    if d >= SHAPE_DISTINCT:
                        shape_distinct += 1
                    if f <= 5:
                        dup_at_low += 1
        seeds_with_dup += int(has_dup)
    print(f"[collapse] seeds with >=1 frequency-collapsed duplicate: "
          f"{seeds_with_dup}/{len(seeds)}")
    print(f"           duplicate mode-pairs: {pairs}  "
          f"(shape-distinct, dist>={SHAPE_DISTINCT}: {shape_distinct}; "
          f"at <=5 Hz: {dup_at_low})")

    # --- 3. representative-seed 5 Hz failure -------------------------------- #
    P, meta = data[args.seed]
    signal, protos, counts, _ = powerlaw_dataset(
        np.asarray(meta["freqs"]), seed=args.seed, wave_len=int(meta["m"]),
        n_waves=int(meta["n_waves"]), noise_std=float(meta["noise_std"]),
        spacing="poisson")
    m = int(meta["m"])
    clean5 = protos[list(meta["freqs"]).index(5)]
    # QSMP mode that snapped to 5 Hz
    snapped = np.array([em.snap_to_alphabet(em.peak_frequency(w), ALPHABET) for w in P])
    idx5 = np.where(snapped == 5)[0]
    print(f"[seed {args.seed}] snapped freqs = "
          f"{[int(em.snap_to_alphabet(em.peak_frequency(w), ALPHABET)) for w in P]}")
    if idx5.size:
        mode5 = P[idx5[0]]
        d = em.shift_invariant_znorm_dist(mode5, clean5)
        print(f"           QSMP 5 Hz mode raw cosine to clean prototype = "
              f"{1 - d**2/2:.3f}")

    # neighbour counts: windows with cos>=0.9 to clean-5Hz vs to QSMP's pick
    def neighbour_count(query, cos_thr=0.9, stride=8):
        zq = em._znorm(np.asarray(query, float))
        cnt = 0
        for st in range(0, signal.size - m + 1, stride):
            w = signal[st:st + m]
            zw = em._znorm(w)
            if abs(float(zw @ zq) / m) >= cos_thr:  # |Pearson| over full window
                cnt += 1
        return cnt
    nc_clean = neighbour_count(clean5)
    print(f"           windows with |cos|>=0.9 (stride 8) to clean 5 Hz: {nc_clean}")
    if idx5.size:
        nc_pick = neighbour_count(P[idx5[0]])
        print(f"           windows with |cos|>=0.9 (stride 8) to QSMP's pick: {nc_pick}")
    print(f"           5 Hz activation count this seed: "
          f"{int(counts[list(meta['freqs']).index(5)])}")


if __name__ == "__main__":
    main()
