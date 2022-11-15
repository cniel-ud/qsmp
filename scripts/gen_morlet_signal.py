#%%
import numpy as np
import matplotlib.pyplot as plt
from spectrum import MultiTapering
from qsmp.datasets import morlet_signal
from pathlib import Path

#%% Generate signal from Morlet wavelets
sig_params = dict(
    fs=512,
    wave_len=512,
    n_waves=1000,
    noise_std=0.07,
    pmf_exp=3.1
)
freqs = np.array([1, 5, 12, 30, 100, 150])
sig, freq, SNRdB = morlet_signal(freqs, **sig_params)
plt.plot(sig[13*sig_params['wave_len']:20*sig_params['wave_len']])
sig_params['SNRdB'] = SNRdB
sig_params['freq'] = freq

#%% Save data
root = '/home/cmendoza/software/qsmp/data/morlet'
fpath = Path(root).joinpath('morlet_signal.npz')
with fpath.open('wb') as f:
    np.savez(
        f, T=sig, splice=np.full(0,0), **sig_params
    )
#%%
X = MultiTapering(sig, NW=3, sampling=512)
X.plot()

# %%
