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
- ``recovery_error`` -- the morphology error of the matched prototypes (a
  shift-invariant z-normalised distance).
- ``peak_freq_error`` -- the peak-frequency error in Hz.

Everything here is pure NumPy / SciPy and runs on CPU, so the measurement logic
can be unit-tested without a GPU (see the ``__main__`` block). A method is
represented only by the set of prototype waveforms it returns (shape
``(k, m)``); QSMP modes, sikmeans centroids and Snippet-Finder snippets all fit
that interface, which keeps the comparison symmetric.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from qsmp.datasets import morlet_signal


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
        ``seed``, ``m``, ``n_waves``, ``noise_std``, ``freqs``.
    info : dict, optional
        Free-form method metadata (chosen sigma/tau, timings, ...), stored under
        a JSON-free flat namespace with an ``info_`` prefix.
    """
    payload = dict(prototypes=np.asarray(prototypes), method=method, k=int(k),
                   seed=int(ds["seed"]), m=int(ds["m"]),
                   n_waves=int(ds["n_waves"]),
                   noise_std=float(ds["noise_std"]),
                   freqs=np.asarray(ds["freqs"]))
    for key, val in (info or {}).items():
        payload[f"info_{key}"] = np.asarray(val)
    np.savez(path, **payload)


def load_prototypes(path):
    """Load a prototype file saved by :func:`save_prototypes`.

    Returns ``(prototypes, meta)`` where ``meta`` has ``method, k, seed, m,
    n_waves, noise_std, freqs`` plus any ``info_*`` fields (prefix stripped).
    """
    with np.load(path, allow_pickle=True) as d:
        protos = d["prototypes"]
        meta = dict(method=str(d["method"]), k=int(d["k"]), seed=int(d["seed"]),
                    m=int(d["m"]), n_waves=int(d["n_waves"]),
                    noise_std=float(d["noise_std"]), freqs=d["freqs"])
        for key in d.files:
            if key.startswith("info_"):
                meta[key[len("info_"):]] = d[key]
    return protos, meta


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
def gt_clean_prototypes(freqs, seed, *, fs=512, wave_len=512, n_waves=1000,
                        noise_std=0.07, pmf_exp=3.1):
    """Return the noise-free prototype waveform for every frequency present.

    Re-generates the dataset for ``seed`` twice: once with the configured
    ``noise_std`` (to recover the per-slot frequency assignment) and once with
    ``noise_std=0`` under the *same* seed (so the phases and the frequency draw
    are identical, but the wavelets are noise-free). The clean prototype of a
    frequency is then read directly out of the noiseless signal at the first
    slot carrying that frequency -- so it has the exact phase/morphology of the
    instances that appear in the noisy signal.

    Parameters
    ----------
    freqs : numpy.ndarray
        The frequency alphabet, e.g. ``[1, 5, 12, 30, 100, 150]``.
    seed : int
        Random seed passed to :func:`qsmp.datasets.morlet_signal`.
    fs, wave_len, n_waves, noise_std, pmf_exp
        Forwarded to :func:`qsmp.datasets.morlet_signal`; must match the values
        used to build the dataset under evaluation.

    Returns
    -------
    prototypes : numpy.ndarray
        Shape ``(n_present, wave_len)``. Noise-free prototype per present
        frequency, ordered as ``freqs_present``.
    freqs_present : numpy.ndarray
        The subset of ``freqs`` that actually occur for this seed (usually all
        of them, but the rarest frequency can be absent for some seeds).
    slot_freq : numpy.ndarray
        Shape ``(n_waves,)``. Ground-truth frequency of each 1-second slot.
    """
    freqs = np.asarray(freqs)
    common = dict(fs=fs, wave_len=wave_len, n_waves=n_waves, pmf_exp=pmf_exp)
    # Same seed -> identical phases and per-slot frequency draw.
    _, freq_noisy, _ = morlet_signal(freqs, noise_std=noise_std, rng=seed,
                                     **common)
    sig_clean, freq_clean, _ = morlet_signal(freqs, noise_std=0.0, rng=seed,
                                             **common)
    slot_freq = freq_clean.reshape(-1).astype(freqs.dtype)
    assert np.array_equal(slot_freq, freq_noisy.reshape(-1)), (
        "noise_std must not change the frequency draw for a fixed seed"
    )

    freqs_present = np.array([f for f in freqs if np.any(slot_freq == f)])
    prototypes = np.empty((freqs_present.size, wave_len))
    for i, f in enumerate(freqs_present):
        first_slot = int(np.flatnonzero(slot_freq == f)[0])
        s = first_slot * wave_len
        prototypes[i] = sig_clean[s:s + wave_len]
    return prototypes, freqs_present, slot_freq


# --------------------------------------------------------------------------- #
# Distance
# --------------------------------------------------------------------------- #
def _znorm(a, axis=-1, eps=1e-8):
    mu = a.mean(axis=axis, keepdims=True)
    sd = a.std(axis=axis, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (a - mu) / sd


def _pairwise_si_dist(P, G, max_lag):
    """Shift-invariant z-normalised distance matrix between rows of ``P`` and ``G``.

    Mirrors the paper's shift-invariant distance (Eq. 7): each waveform is
    z-normalised, one is slid against the other over ``[-max_lag, +max_lag]``,
    and the minimum length-normalised z-normalised Euclidean distance over the
    overlap is kept. After re-z-normalising an overlap of length ``L`` to
    zero-mean/unit-std, its sum of squares is exactly ``L``, so the
    length-normalised squared distance reduces to ``2 - 2*corr`` where ``corr``
    is the Pearson correlation over the overlap. Each lag is therefore a single
    matrix multiply, which keeps the (n_slots x n_prototypes) computation fast.

    Parameters
    ----------
    P : numpy.ndarray
        ``(n, m)`` waveforms (rows).
    G : numpy.ndarray
        ``(g, m)`` waveforms (rows).
    max_lag : int
        Maximum absolute shift, in samples (the paper uses ``m/4``).

    Returns
    -------
    numpy.ndarray
        ``(n, g)`` matrix of minimum-over-lag distances.
    """
    P, G = np.atleast_2d(P).astype(float), np.atleast_2d(G).astype(float)
    m = P.shape[1]
    best = np.full((P.shape[0], G.shape[0]), np.inf)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            Pi, Gj = P[:, :m + lag], G[:, -lag:]
        elif lag > 0:
            Pi, Gj = P[:, lag:], G[:, :m - lag]
        else:
            Pi, Gj = P, G
        L = Pi.shape[1]
        if L < 2:
            continue
        Pz, Gz = _znorm(Pi), _znorm(Gj)          # rows: mean 0, sum-sq = L
        corr = (Pz @ Gz.T) / L                    # (n, g) Pearson correlation
        d = np.sqrt(np.clip(2.0 - 2.0 * corr, 0.0, None))
        np.minimum(best, d, out=best)
    return best


def shift_invariant_znorm_dist(a, b, max_lag):
    """Shift-invariant z-normalised distance between two 1D waveforms.

    Thin scalar wrapper around :func:`_pairwise_si_dist` (kept for readability
    and tests). See that function for the definition.
    """
    return float(_pairwise_si_dist(np.atleast_2d(a), np.atleast_2d(b),
                                   max_lag)[0, 0])


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
                       max_lag=None):
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
        Max shift for the shift-invariant distance; defaults to ``m/4``.

    Returns
    -------
    dict
        ``n_freqs_recovered`` : distinct ground-truth frequencies whose peak is
        hit by at least one predicted prototype.
        ``n_freqs_total`` : number of ground-truth frequencies present.
        ``recovery_error`` : mean shift-invariant z-norm distance of the
        Hungarian-matched (prototype -> ground truth) pairs. Lower is better.
        ``peak_freq_error`` : mean |estimated - true| peak frequency (Hz) over
        matched pairs.
    """
    pred_protos = np.atleast_2d(pred_protos)
    gt_protos = np.atleast_2d(gt_protos)
    gt_freqs = np.asarray(gt_freqs, dtype=float)
    m = pred_protos.shape[1]
    if max_lag is None:
        max_lag = m // 4

    # --- distinct-frequency recovery via peak spectrum ---------------------- #
    pred_peaks = np.array([peak_frequency(p, fs) for p in pred_protos])
    snapped = np.array([snap_to_alphabet(f, gt_freqs) for f in pred_peaks])
    n_recovered = np.unique(snapped).size

    # --- morphology + peak-frequency error via Hungarian matching ----------- #
    D = _pairwise_si_dist(pred_protos, gt_protos, max_lag)
    # Match min(k, n_present) pairs; if k > n_present, only n_present matched.
    row, col = linear_sum_assignment(D)
    recovery_error = float(D[row, col].mean())
    peak_err = float(np.mean([
        abs(pred_peaks[r] - gt_freqs[c]) for r, c in zip(row, col)
    ]))
    return dict(
        n_freqs_recovered=int(n_recovered),
        n_freqs_total=int(gt_freqs.size),
        recovery_error=recovery_error,
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

    gt_protos, gt_freqs, slot_freq = gt_clean_prototypes(rng_freqs, SEED)
    print(f"[self-test] frequencies present: {gt_freqs.tolist()}")

    # Oracle method: returns the ground-truth prototypes.
    oracle = prototype_recovery(gt_protos, gt_protos, gt_freqs)
    print(f"[self-test] oracle recovery : {oracle}")
    assert oracle["n_freqs_recovered"] == gt_freqs.size
    assert oracle["recovery_error"] < 1e-6
    assert oracle["peak_freq_error"] == 0.0

    # Collapse method: k copies of the most-prevalent (lowest-freq) prototype.
    collapse = np.repeat(gt_protos[:1], gt_freqs.size, axis=0)
    coll = prototype_recovery(collapse, gt_protos, gt_freqs)
    print(f"[self-test] collapse recovery : {coll}")
    assert coll["n_freqs_recovered"] == 1
    assert coll["recovery_error"] > oracle["recovery_error"]

    print("[self-test] OK")
