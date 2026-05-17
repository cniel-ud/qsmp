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
