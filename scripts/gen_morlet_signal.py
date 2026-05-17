"""Generate the synthetic power-law (Morlet) dataset used in §5.2.

Produces a 1,000-second signal at 512 Hz built from six 1-second Morlet
wavelets at frequencies (1, 5, 12, 30, 100, 150) Hz, drawn from a power-law
distribution and corrupted with i.i.d. Gaussian noise (σ=0.07).

The output is saved as ``<root>/data/morlet/morlet_signal.npz`` with keys:
    - ``T``: 1D float64 time series
    - ``splice``: empty int array (the signal has no missing-data splices)
    - ``fs``, ``wave_len``, ``n_waves``, ``noise_std``, ``pmf_exp``, ``SNRdB``,
      ``freq``: signal-generation parameters
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from spectrum import MultiTapering

from qsmp.datasets import morlet_signal


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root. The dataset is written to <root>/data/morlet/. "
        "Defaults to the current directory.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip the diagnostic time-series and PSD plots.",
    )
    args = parser.parse_args()

    sig_params = dict(
        fs=512,
        wave_len=512,
        n_waves=1000,
        noise_std=0.07,
        pmf_exp=3.1,
    )
    freqs = np.array([1, 5, 12, 30, 100, 150])

    sig, freq, snr_db = morlet_signal(freqs, **sig_params)
    sig_params["SNRdB"] = snr_db
    sig_params["freq"] = freq

    out_dir = args.root.joinpath("data", "morlet")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir.joinpath("morlet_signal.npz")
    with out_path.open("wb") as f:
        np.savez(f, T=sig, splice=np.full(0, 0), **sig_params)
    print(f"Wrote {out_path} (n={sig.size}, SNR={snr_db:.2f} dB)")

    if not args.no_plots:
        plt.figure()
        plt.plot(sig[13 * sig_params["wave_len"]: 20 * sig_params["wave_len"]])
        plt.title("Sample of the synthetic power-law signal")

        psd = MultiTapering(sig, NW=3, sampling=sig_params["fs"])
        psd.plot()

        plt.show()


if __name__ == "__main__":
    main()
