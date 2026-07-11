from typing import Any
import numpy as np
from qsmp.utils.utils import check_rng

def morlet_signal(
    freqs: np.ndarray,
    fs: int = 512,
    wave_len: int = 512,
    n_waves: int = 1000,
    noise_std: float = 0.07,
    pmf_exp: float = 3.1,
    rng: Any = 13,
):
    """Generate the synthetic *power-law* dataset used in §5.1 of the paper.

    Builds a time series by concatenating ``n_waves`` Morlet wavelets, each of
    length ``wave_len``, drawn from ``freqs`` according to a power-law
    probability mass function ``p_k ∝ 1 / k**pmf_exp`` (so the lowest
    frequency is most prevalent). I.i.d. zero-mean Gaussian noise with
    standard deviation ``noise_std`` is added on top.

    Each wavelet is independently amplitude-normalised to ``[-1, 1]`` and
    given a random phase shift drawn at construction time.

    Parameters
    ----------
    freqs : numpy.ndarray
        1D array of wavelet centre frequencies, in Hz, sorted from least to
        most prevalent. The paper uses ``[1, 5, 12, 30, 100, 150]``.
    fs : int, default 512
        Sampling frequency in Hz.
    wave_len : int, default 512
        Length of a single wavelet, in samples. With the default ``fs`` and
        ``wave_len``, each wavelet spans 1 second.
    n_waves : int, default 1000
        Number of wavelets to concatenate. The total signal length is
        ``n_waves * wave_len`` samples.
    noise_std : float, default 0.07
        Standard deviation of the additive Gaussian noise.
    pmf_exp : float, default 3.1
        Exponent of the power-law PMF over ``freqs``. Larger values
        concentrate more mass on the lowest frequency (matching the 1/f
        spectrum of EEG that the paper takes inspiration from).
    rng : None, int, or numpy.random.Generator, default 13
        Random state. An ``int`` gives a reproducible run.

    Returns
    -------
    sig : numpy.ndarray
        1D float64 signal of shape ``(wave_len * n_waves,)``.
    freq : numpy.ndarray
        Per-wavelet frequency assignments, shape ``(n_waves, 1)``. The ``i``-th
        entry is the frequency of the wavelet starting at sample
        ``i * wave_len``.
    SNRdB : float
        Signal-to-noise ratio in dB, computed as
        ``10 * log10(mean(sig**2) / mean(noise**2))`` *before* the noise was
        added (so it reflects the noiseless wavelets vs the noise floor).
    """
    rng = check_rng(rng)
    n_freqs = freqs.size
    phis = rng.random((n_freqs,)) * 2*np.pi
    k = np.arange(1, n_freqs+1)
    sum_k = (1/k**pmf_exp).sum()
    pmf = (1/k**pmf_exp)/sum_k
    t = np.linspace(-2 * np.pi, 2 * np.pi, wave_len).reshape(1, -1)
    i_freq = rng.choice(np.arange(n_freqs), size=n_waves, p=pmf)
    freq, phi = freqs[i_freq].reshape(-1, 1), phis[i_freq].reshape(-1, 1)
    i_freq, freq_cnts = np.unique(i_freq, return_counts=True)
    w0 = freq * wave_len / (2*fs)
    sig = np.exp(1j * (w0 * t + phi)) - np.exp(-0.5 * (w0**2))
    sig *= np.exp(-0.5 * (t**2)) * np.pi**(-0.25)
    sig = sig.real
    row_max = np.abs(sig).max(axis=1).reshape(-1, 1)
    sig = sig / row_max
    sig = sig.flatten()
    noise = rng.normal(scale=noise_std, size=wave_len*n_waves)
    sig += noise

    sig_pow = np.mean(sig**2)
    noise_pow = np.mean(noise**2)
    SNRdB = 10 * np.log10(sig_pow / noise_pow)

    return sig, freq, SNRdB


def morlet_waveforms(
    freqs: np.ndarray,
    fs: int = 512,
    wave_len: int = 512,
    rng: Any = 13,
):
    """Return the clean prototype waveform for each frequency (the ground truth).

    Unlike :func:`morlet_signal`, this produces the *prototypes* on their own,
    decoupled from how they are later placed into a signal (see
    :func:`powerlaw_signal`). Each row is one amplitude-normalised Morlet
    wavelet with a DC correction and a Gaussian envelope; a random phase per
    frequency is drawn from ``rng``. Because the prototypes are returned
    directly, they serve as exact, uncontaminated ground truth even when the
    generated signal superimposes overlapping wavelets.

    Parameters
    ----------
    freqs : numpy.ndarray
        1D array of centre frequencies (Hz), e.g. ``[1, 5, 12, 30, 100, 150]``.
    fs : int, default 512
        Sampling frequency (Hz).
    wave_len : int, default 512
        Waveform length in samples (spans ``wave_len / fs`` seconds).
    rng : None, int, or numpy.random.Generator, default 13
        Random state for the per-frequency phase.

    Returns
    -------
    numpy.ndarray
        Clean prototypes, shape ``(freqs.size, wave_len)``, each row scaled so
        its peak magnitude is 1.
    """
    rng = check_rng(rng)
    n_freqs = freqs.size
    freqs = freqs.reshape(-1, 1)
    phis = (rng.random((n_freqs,)) * 2 * np.pi).reshape(-1, 1)
    wave_duration = wave_len / fs
    t = np.linspace(-0.5 * wave_duration, 0.5 * wave_duration,
                    wave_len).reshape(1, -1)
    waves = np.cos(2 * np.pi * freqs * t + phis) \
        - np.exp(-0.5 * ((freqs * wave_duration / 2) ** 2))
    waves *= np.exp(-0.5 * t ** 2 / (wave_duration / 4 / np.pi) ** 2)
    row_max = np.abs(waves).max(axis=1).reshape(-1, 1)
    waves = waves / row_max
    return waves


def powerlaw_signal(
    waves: np.ndarray,
    fs: int = 512,
    signal_duration: float = 1000,
    wave_rate: float = 1,
    noise_std: float = 0.07,
    pmf_exp: float = 3.1,
    spacing: str = 'uniform',
    rng: Any = 13,
):
    """Place the prototype ``waves`` into a noisy signal under a power-law PMF.

    ``N = signal_duration * wave_rate`` activations are drawn from the
    frequency alphabet with power-law prevalence ``p_k ∝ 1 / k**pmf_exp`` and
    convolved with the corresponding prototype. Two spacings are supported:

    - ``'uniform'`` places activations exactly ``wave_len`` apart, so wavelets
      tile edge-to-edge and every length-``wave_len`` window is a single clean
      wavelet (the easy case).
    - ``'poisson'`` scatters activations to random sample positions (via a
      row-permutation of the activation train) before convolving, so nearby
      wavelets *superimpose additively* -- a window is generally a partial
      superposition of overlapping wavelets, a harder and more realistic test.

    Both spacings use the same number of activations ``N``, so the signal
    length and mean activation density match.

    Parameters
    ----------
    waves : numpy.ndarray
        Prototype waveforms, shape ``(n_freqs, wave_len)`` (from
        :func:`morlet_waveforms`).
    fs : int, default 512
        Sampling frequency (Hz).
    signal_duration : float, default 1000
        Signal length in seconds (total samples ``T = signal_duration * fs``).
    wave_rate : float, default 1
        Activations per second; ``N = signal_duration * wave_rate``.
    noise_std : float, default 0.07
        Standard deviation of the additive Gaussian noise.
    pmf_exp : float, default 3.1
        Power-law exponent of the frequency PMF.
    spacing : {'uniform', 'poisson'}, default 'uniform'
        Activation placement (see above).
    rng : None, int, or numpy.random.Generator, default 13
        Random state.

    Returns
    -------
    sig : numpy.ndarray
        1D signal of shape ``(T,)``.
    wave_counts : numpy.ndarray
        Number of activations per frequency, ``np.bincount(i_freq)``.
    SNRdB : float
        Signal-to-noise ratio (dB) of the noiseless signal vs the noise floor.
    act : numpy.ndarray
        Activation train, shape ``(T, n_freqs)``.
    """
    rng = check_rng(rng)
    n_freqs, wave_len = waves.shape
    k = np.arange(1, n_freqs + 1)
    sum_k = (1 / k ** pmf_exp).sum()
    pmf = (1 / k ** pmf_exp) / sum_k
    N = int(signal_duration * wave_rate)
    i_freq = rng.choice(np.arange(n_freqs), size=N, p=pmf)
    T = int(signal_duration * fs)
    act = np.zeros((T, n_freqs))
    amps = np.ones(N)                       # common amplitude
    if spacing == 'poisson':
        act[np.arange(N), i_freq] = amps
        act = rng.permutation(act)          # scatter activations across time
    elif spacing == 'uniform':
        act[np.arange(0, N * wave_len, wave_len), i_freq] = amps
    else:
        raise NotImplementedError(f"unknown spacing {spacing!r}")

    X = np.vstack([np.convolve(waves[i], act[:, i], mode='same')
                   for i in range(n_freqs)])
    noise = rng.normal(scale=noise_std, size=T)
    sig = np.sum(X, axis=0)
    sig_pow = np.mean(sig ** 2)
    noise_pow = np.mean(noise ** 2)
    SNRdB = 10 * np.log10(sig_pow / noise_pow)
    sig += noise
    return sig, np.bincount(i_freq, minlength=n_freqs), SNRdB, act


def powerlaw_dataset(
    freqs: np.ndarray,
    seed: Any = 13,
    *,
    fs: int = 512,
    wave_len: int = 512,
    n_waves: int = 1000,
    noise_std: float = 0.07,
    pmf_exp: float = 3.1,
    spacing: str = 'poisson',
):
    """Deterministic power-law dataset: clean prototypes *and* the noisy signal.

    Threads a single :class:`numpy.random.Generator` (seeded by ``seed``)
    through :func:`morlet_waveforms` (phases) and :func:`powerlaw_signal`
    (activation draw, placement, noise), so the whole dataset -- including the
    ground-truth prototypes -- is reproducible from ``seed`` alone and matches
    across machines. This is the entry point used by the recovery experiment:
    the runners take ``sig``; the aggregator takes ``prototypes`` + ``counts``
    to build the ground truth.

    Parameters
    ----------
    freqs : numpy.ndarray
        Frequency alphabet (Hz).
    seed : None, int, or numpy.random.Generator, default 13
        Random state for the whole dataset.
    fs, wave_len, noise_std, pmf_exp
        Forwarded to the generators.
    n_waves : int, default 1000
        Number of activations ``N`` (kept as ``n_waves`` for continuity with
        :func:`morlet_signal`); with ``wave_rate = 1`` the signal spans
        ``n_waves`` seconds.
    spacing : {'poisson', 'uniform'}, default 'poisson'
        Activation placement; the recovery experiment uses ``'poisson'``.

    Returns
    -------
    sig : numpy.ndarray
        1D signal, shape ``(wave_len * n_waves,)`` (with ``fs == wave_len``).
    prototypes : numpy.ndarray
        Clean ground-truth waveforms, shape ``(freqs.size, wave_len)``.
    wave_counts : numpy.ndarray
        Activations per frequency, shape ``(freqs.size,)``.
    SNRdB : float
        Signal-to-noise ratio in dB.
    """
    freqs = np.asarray(freqs)
    rng = check_rng(seed)
    prototypes = morlet_waveforms(freqs, fs=fs, wave_len=wave_len, rng=rng)
    sig, wave_counts, snr_db, _ = powerlaw_signal(
        prototypes, fs=fs, signal_duration=n_waves, wave_rate=1,
        noise_std=noise_std, pmf_exp=pmf_exp, spacing=spacing, rng=rng)
    return sig, prototypes, wave_counts, snr_db
