"""Quantitative ground-truth-recovery metrics for the synthetic power-law dataset.

The synthetic ``power-law`` dataset (:func:`qsmp.datasets.morlet_signal`) is
built by concatenating 1-second Morlet wavelets drawn from a *known* set of
frequencies, so it comes with per-wavelet ground-truth labels *and* noise-free
prototype waveforms. This module turns those into three complementary,
method-agnostic **prototype-recovery** scores so the paper's qualitative
comparison (QSMP vs. sikmeans vs. Snippet-Finder) can be backed by numbers with
confidence intervals:

- ``n_freqs_recovered`` -- how many of the ground-truth frequencies each method
  recovers.
- ``recovery_cosine`` -- the morphology similarity of the matched prototypes (a
  shift-invariant cosine similarity; ``1`` = identical).
- ``peak_freq_error`` -- the peak-frequency error in Hz.

Everything here is pure NumPy / SciPy and runs on CPU, so the measurement logic
can be unit-tested without a GPU (see the ``__main__`` block). A method is
represented only by the set of prototype waveforms it returns (shape
``(k, m)``); QSMP modes, sikmeans centroids and Snippet-Finder snippets all fit
that interface, which keeps the comparison symmetric.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from qsmp.datasets import powerlaw_dataset


def resolve_rec_dir(root, spacing):
    """Locate the recovery prototype directory for a given ``spacing``.

    Prefers ``<root>/results/recovery/<spacing>/``. Falls back to the flat
    legacy ``<root>/results/recovery/`` (whose files predate spacing subdirs
    and are all ``poisson``) so older result sets still load. Returns the
    (possibly empty) ``<spacing>`` subdir when nothing is found, so the caller
    can report a clear missing-directory error.
    """
    base = Path(root).joinpath("results", "recovery")
    sub = base.joinpath(spacing)
    if any(sub.glob("*_seed-*.npz")):
        return sub
    if spacing == "poisson" and any(
            f.parent == base for f in base.glob("*_seed-*.npz")):
        return base
    return sub


# --------------------------------------------------------------------------- #
# Prototype I/O (uniform across methods)
# --------------------------------------------------------------------------- #
# Every method (QSMP, sikmeans, Snippet-Finder) saves only the raw prototype
# waveforms it returns, in one self-describing file per (method, seed). Scoring
# is done later, for all methods at once, by ``aggregate_recovery.py`` -- so the
# expensive production of prototypes is decoupled from the metric computation.
def save_prototypes(path, prototypes, *, method, k, ds, info=None):
    """Save a method's prototypes plus the info needed to score them later.

    Parameters
    ----------
    path : str or pathlib.Path
        Output ``.npz`` path.
    prototypes : numpy.ndarray
        The ``(k, m)`` prototype waveforms (fewer rows allowed if a method
        failed to return ``k``).
    method : str
        Method name, e.g. ``"qsmp"``, ``"sikmeans"``, ``"snippetfinder"``.
    k : int
        Requested number of prototypes.
    ds : dict
        Dataset parameters needed to re-derive the ground truth:
        ``seed``, ``m``, ``n_waves``, ``noise_std``, ``freqs``, and optionally
        ``spacing`` (``"poisson"`` or ``"uniform"``; defaults to ``"poisson"``
        for backward compatibility with files written before spacing was a
        sweep dimension). The exact spacing is saved so the aggregator
        re-derives the matching ground truth regardless of where the file
        lives on disk.
    info : dict, optional
        Free-form method metadata (chosen sigma/tau, timings, ...), stored under
        a JSON-free flat namespace with an ``info_`` prefix.
    """
    payload = dict(prototypes=np.asarray(prototypes), method=method, k=int(k),
                   seed=int(ds["seed"]), m=int(ds["m"]),
                   n_waves=int(ds["n_waves"]),
                   noise_std=float(ds["noise_std"]),
                   freqs=np.asarray(ds["freqs"]),
                   spacing=str(ds.get("spacing", "poisson")))
    for key, val in (info or {}).items():
        payload[f"info_{key}"] = np.asarray(val)
    np.savez(path, **payload)


def load_prototypes(path):
    """Load a prototype file saved by :func:`save_prototypes`.

    Returns ``(prototypes, meta)`` where ``meta`` has ``method, k, seed, m,
    n_waves, noise_std, freqs, spacing`` plus any ``info_*`` fields (prefix
    stripped). ``spacing`` defaults to ``"poisson"`` for files written before
    it was recorded.
    """
    with np.load(path, allow_pickle=True) as d:
        protos = d["prototypes"]
        meta = dict(method=str(d["method"]), k=int(d["k"]), seed=int(d["seed"]),
                    m=int(d["m"]), n_waves=int(d["n_waves"]),
                    noise_std=float(d["noise_std"]), freqs=d["freqs"],
                    spacing=str(d["spacing"]) if "spacing" in d.files
                    else "poisson")
        for key in d.files:
            if key.startswith("info_"):
                meta[key[len("info_"):]] = d[key]
    return protos, meta


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
def gt_clean_prototypes(freqs, seed, *, fs=512, wave_len=512, n_waves=1000,
                        noise_std=0.07, pmf_exp=3.1, spacing="poisson"):
    """Return the noise-free prototype waveform for every frequency present.

    With :func:`qsmp.datasets.powerlaw_dataset` the prototypes are produced
    *before* they are placed into the signal, so the ground truth is simply the
    clean prototype set it returns -- no need to read waveforms back out of the
    signal. This is what makes the harder ``'poisson'`` spacing well-defined:
    even though overlapping wavelets superimpose in the signal, the underlying
    shapes are known exactly. A frequency is reported as *present* only if it
    was actually activated for this seed (the rarest frequency can be absent).

    Parameters
    ----------
    freqs : numpy.ndarray
        The frequency alphabet, e.g. ``[1, 5, 12, 30, 100, 150]``.
    seed : int
        Random seed passed to :func:`qsmp.datasets.powerlaw_dataset`.
    fs, wave_len, n_waves, noise_std, pmf_exp, spacing
        Forwarded to :func:`qsmp.datasets.powerlaw_dataset`; must match the
        values used to build the dataset under evaluation. (``noise_std`` does
        not affect the clean prototypes but is accepted so callers can pass the
        full parameter set.)

    Returns
    -------
    prototypes : numpy.ndarray
        Shape ``(n_present, wave_len)``. Noise-free prototype per present
        frequency, ordered as ``freqs_present``.
    freqs_present : numpy.ndarray
        The subset of ``freqs`` that actually occur for this seed.
    wave_counts : numpy.ndarray
        Shape ``(freqs.size,)``. Number of activations per frequency.
    """
    freqs = np.asarray(freqs)
    _, prototypes, wave_counts, _ = powerlaw_dataset(
        freqs, seed=seed, fs=fs, wave_len=wave_len, n_waves=n_waves,
        noise_std=noise_std, pmf_exp=pmf_exp, spacing=spacing)
    present = wave_counts > 0
    return prototypes[present], freqs[present], wave_counts


# --------------------------------------------------------------------------- #
# Distance
# --------------------------------------------------------------------------- #
def _znorm(a, axis=-1, eps=1e-8):
    mu = a.mean(axis=axis, keepdims=True)
    sd = a.std(axis=axis, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (a - mu) / sd


def _pairwise_si_dist(P, G, max_lag=None, min_overlap=None):
    """Shift-invariant z-normalised distance matrix between rows of ``P`` and ``G``.

    In the spirit of the paper's shift-invariant distance (the min-pooling
    dissimilarity :math:`\\breve{D}_{i,j}`), each waveform is z-normalised, one
    is slid against the other, and the minimum length-normalised z-normalised
    Euclidean distance over the overlap is kept. After re-z-normalising an
    overlap of length ``L`` to zero-mean/unit-std, its sum of squares is exactly
    ``L``, so the length-normalised squared distance reduces to ``2 - 2*corr``
    where ``corr`` is the Pearson correlation over the overlap. Each lag is
    therefore a single matrix multiply.

    Shift range vs. overlap. When comparing method prototypes to ground-truth
    prototypes, a returned waveform can be an arbitrarily positioned *sub-segment*
    of the recurring pattern -- Snippet-Finder snippets in particular are
    extracted at essentially any phase, so their energy can be offset from the
    canonical (centred) ground-truth prototype by more than ``m/4``. Capping the
    shift too tightly would therefore under-credit exactly those methods. We thus
    default to searching the *full* lag range while requiring a minimum overlap
    of ``m/2`` samples (``min_overlap``): this reaches large genuine offsets
    while forbidding the degenerate tiny-overlap alignments (an overlap of a few
    samples is almost always spuriously well-correlated, giving fake ``cos~1``).

    Parameters
    ----------
    P : numpy.ndarray
        ``(n, m)`` waveforms (rows).
    G : numpy.ndarray
        ``(g, m)`` waveforms (rows).
    max_lag : int or None
        Maximum absolute shift, in samples. ``None`` (default) searches the full
        range ``m-1`` subject to ``min_overlap``.
    min_overlap : int or None
        Minimum number of overlapping samples for a lag to be considered.
        ``None`` (default) uses ``m // 2``. Must be ``>= 2``.

    Returns
    -------
    numpy.ndarray
        ``(n, g)`` matrix of minimum-over-lag distances.
    """
    P, G = np.atleast_2d(P).astype(float), np.atleast_2d(G).astype(float)
    m = P.shape[1]
    if max_lag is None:
        max_lag = m - 1
    if min_overlap is None:
        min_overlap = m // 2
    min_overlap = max(int(min_overlap), 2)
    best = np.full((P.shape[0], G.shape[0]), np.inf)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            Pi, Gj = P[:, :m + lag], G[:, -lag:]
        elif lag > 0:
            Pi, Gj = P[:, lag:], G[:, :m - lag]
        else:
            Pi, Gj = P, G
        L = Pi.shape[1]
        if L < min_overlap:
            continue
        Pz, Gz = _znorm(Pi), _znorm(Gj)          # rows: mean 0, sum-sq = L
        corr = (Pz @ Gz.T) / L                    # (n, g) Pearson correlation
        d = np.sqrt(np.clip(2.0 - 2.0 * corr, 0.0, None))
        np.minimum(best, d, out=best)
    return best


def shift_invariant_znorm_dist(a, b, max_lag=None, min_overlap=None):
    """Shift-invariant z-normalised distance between two 1D waveforms.

    Thin scalar wrapper around :func:`_pairwise_si_dist` (kept for readability
    and tests). See that function for the definition.
    """
    return float(_pairwise_si_dist(np.atleast_2d(a), np.atleast_2d(b),
                                   max_lag, min_overlap)[0, 0])


# --------------------------------------------------------------------------- #
# Frequency estimation
# --------------------------------------------------------------------------- #
def peak_frequency(wave, fs=512):
    """Peak frequency (Hz) of a waveform from its magnitude spectrum."""
    wave = np.asarray(wave, dtype=float)
    wave = wave - wave.mean()
    spec = np.abs(np.fft.rfft(wave))
    freqs = np.fft.rfftfreq(wave.size, d=1.0 / fs)
    return float(freqs[np.argmax(spec)])


def snap_to_alphabet(freq_hz, alphabet):
    """Snap an estimated frequency to the nearest ground-truth frequency."""
    alphabet = np.asarray(alphabet, dtype=float)
    return alphabet[np.argmin(np.abs(alphabet - float(freq_hz)))]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def prototype_recovery(pred_protos, gt_protos, gt_freqs, *, fs=512,
                       max_lag=None, min_overlap=None, matching="best"):
    """Prototype-recovery metrics for one method on one dataset instance.

    Parameters
    ----------
    pred_protos : numpy.ndarray
        Prototypes returned by the method, shape ``(k, m)`` (e.g. the ``k``
        QSMP modes, sikmeans centroids, or Snippet-Finder snippets).
    gt_protos : numpy.ndarray
        Ground-truth noise-free prototypes, shape ``(n_present, m)``
        (from :func:`gt_clean_prototypes`).
    gt_freqs : numpy.ndarray
        Frequencies (Hz) aligned with ``gt_protos`` rows.
    fs : int
        Sampling rate, for peak-frequency estimation.
    max_lag : int or None
        Max shift for the shift-invariant distance; ``None`` (default) searches
        the full lag range subject to ``min_overlap`` (see
        :func:`_pairwise_si_dist`), so arbitrarily positioned sub-segments (e.g.
        Snippet-Finder snippets) are matched on morphology rather than penalised
        for their position.
    min_overlap : int or None
        Minimum overlap (samples) for a lag; ``None`` (default) uses ``m/2``.
    matching : {"best", "hungarian"}
        How predictions are paired to ground-truth prototypes for the
        morphology/peak-error metrics.

        ``"best"`` (default) matches *with replacement*: every ground-truth
        prototype is scored against its closest prediction, and one prediction
        may serve several ground truths. Each metric then answers "how well is
        each true prototype represented?" in isolation; redundancy/collapse is
        deliberately NOT penalised here because ``n_freqs_recovered`` already
        measures it, keeping the metrics orthogonal. Without replacement, a
        method that (say) duplicates the prevalent 1 Hz wavelet is *also*
        charged on CosSim for a forced leftover pairing (e.g. a 1 Hz prototype
        assigned to the 150 Hz ground truth), double-counting the same failure.

        ``"hungarian"`` matches without replacement (1-to-1, minimum-cost
        assignment over ``min(k, n_present)`` pairs), so morphology and
        coverage failures are entangled in one number. Kept for comparison.

    Returns
    -------
    dict
        ``n_freqs_recovered`` : distinct ground-truth frequencies whose peak is
        hit by at least one predicted prototype.
        ``n_freqs_total`` : number of ground-truth frequencies present.
        ``recovery_cosine`` : mean shift-invariant cosine similarity of the
        matched pairs (``1`` = identical morphology). Higher is better.
        ``peak_freq_error`` : mean |estimated - true| peak frequency (Hz) over
        matched pairs.

        With ``matching="best"`` there is one pair per present ground-truth
        prototype (``n_present`` pairs); with ``matching="hungarian"`` there
        are ``min(k, n_present)`` pairs.
    """
    pred_protos = np.atleast_2d(pred_protos)
    gt_protos = np.atleast_2d(gt_protos)
    gt_freqs = np.asarray(gt_freqs, dtype=float)

    # --- distinct-frequency recovery via peak spectrum ---------------------- #
    pred_peaks = np.array([peak_frequency(p, fs) for p in pred_protos])
    snapped = np.array([snap_to_alphabet(f, gt_freqs) for f in pred_peaks])
    n_recovered = np.unique(snapped).size

    # --- morphology + peak-frequency error over matched pairs --------------- #
    # D[i, j] = shift-invariant distance from prediction i to ground truth j.
    D = _pairwise_si_dist(pred_protos, gt_protos, max_lag, min_overlap)
    if matching == "best":
        # With replacement: each ground truth (column) takes its closest
        # prediction. One prediction may serve several ground truths, so
        # collapse is not charged here -- n_freqs_recovered already measures it.
        pred_idx = D.argmin(axis=0)                    # (n_present,)
        gt_idx = np.arange(D.shape[1])
    elif matching == "hungarian":
        # Without replacement: 1-to-1 minimum-cost assignment over
        # min(k, n_present) pairs.
        pred_idx, gt_idx = linear_sum_assignment(D)
    else:
        raise ValueError(f"matching must be 'best' or 'hungarian', got {matching!r}")
    d_pairs = D[pred_idx, gt_idx]
    # Shift-invariant cosine similarity of the matched pairs. For z-normalised
    # waveforms cos = 1 - d^2/2, but we average the *per-pair* cosine -- the mean
    # does not commute with the non-linear 1 - d^2/2, so converting an aggregate
    # distance would be wrong. This is the interpretable morphology score
    # (1 = identical).
    recovery_cosine = float(np.mean(1.0 - d_pairs ** 2 / 2.0))
    peak_err = float(np.mean([
        abs(pred_peaks[r] - gt_freqs[c]) for r, c in zip(pred_idx, gt_idx)
    ]))
    return dict(
        n_freqs_recovered=int(n_recovered),
        n_freqs_total=int(gt_freqs.size),
        recovery_cosine=recovery_cosine,
        peak_freq_error=peak_err,
    )


# --------------------------------------------------------------------------- #
# Self-test (CPU only, no GPU / no QSMP run required)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # A minimal, self-contained sanity check of the measurement logic:
    #   * a method that returns the true prototypes should score perfectly;
    #   * a method that returns only duplicates of the most-prevalent waveform
    #     (the sikmeans failure mode the paper describes) should score worse.
    rng_freqs = np.array([1, 5, 12, 30, 100, 150])
    SEED = 13

    gt_protos, gt_freqs, wave_counts = gt_clean_prototypes(rng_freqs, SEED)
    print(f"[self-test] frequencies present: {gt_freqs.tolist()} "
          f"(counts {wave_counts.tolist()})")

    # Oracle method: returns the ground-truth prototypes.
    oracle = prototype_recovery(gt_protos, gt_protos, gt_freqs)
    print(f"[self-test] oracle recovery : {oracle}")
    assert oracle["n_freqs_recovered"] == gt_freqs.size
    assert oracle["recovery_cosine"] > 1.0 - 1e-6
    assert oracle["peak_freq_error"] == 0.0

    # Collapse method: k copies of the most-prevalent (lowest-freq) prototype.
    collapse = np.repeat(gt_protos[:1], gt_freqs.size, axis=0)
    coll = prototype_recovery(collapse, gt_protos, gt_freqs)
    print(f"[self-test] collapse recovery : {coll}")
    assert coll["n_freqs_recovered"] == 1
    assert coll["recovery_cosine"] < oracle["recovery_cosine"]

    # Both matching modes recover the oracle perfectly. They differ on
    # collapse: "best" (with replacement) scores each ground truth against its
    # closest prediction (here all identical, so every GT maps to the same 1 Hz
    # copy); "hungarian" (1-to-1) is forced into leftover pairings. The oracle
    # is invariant to the choice; both must flag collapse as worse.
    for mode in ("best", "hungarian"):
        orc = prototype_recovery(gt_protos, gt_protos, gt_freqs, matching=mode)
        col = prototype_recovery(collapse, gt_protos, gt_freqs, matching=mode)
        assert orc["recovery_cosine"] > 1.0 - 1e-6, mode
        assert col["recovery_cosine"] < orc["recovery_cosine"], mode
    try:
        prototype_recovery(gt_protos, gt_protos, gt_freqs, matching="bogus")
        raise AssertionError("expected ValueError for unknown matching")
    except ValueError:
        pass

    print("[self-test] OK")
