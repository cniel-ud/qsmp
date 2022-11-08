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
    return sig, freq, freq_cnts
