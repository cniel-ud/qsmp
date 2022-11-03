#%%
import numpy as np
import matplotlib.pyplot as plt
from spectrum import MultiTapering
from qsmp.datasets import morlet_signal
from pathlib import Path
# %% Generate signal from Morlet wavelets
freqs = np.array([1, 5, 12, 30, 100, 150])
sig, freq, freq_cnts = morlet_signal(freqs)
plt.plot(sig)
#%%
root = '/home/cmendoza/software/qsmp-python'
fname = Path(root).joinpath('data', 'morlet_signal_fs-512.txt')
np.savetxt(fname, sig, delimiter='\n')
#%%
sig = np.loadtxt(fname)
#%%

X = MultiTapering(sig, NW=3, sampling=512)
X.plot()
PSD, f = X.psd, X.frequencies()
imax = np.argmax(PSD)
f[imax]
# %%
wnames = ['db4']
for wname in wnames:
    wavelet = pywt.Wavelet(wname)
    phi, psi, x = wavelet.wavefun(level=9)
    plt.figure()
    plt.plot(x, psi)
    plt.title(wname)
# %%
cfreq = pywt.central_frequency('db4', precision=9)
y = np.cos(2*np.pi*cfreq*x+3*np.pi/4)
plt.plot(x, psi)
plt.plot(x, y)
#%%
Psi = MultiTapering(psi, NW=3, sampling=512)
Psi.plot()
#%%
PSD, f = Psi.psd, Psi.frequencies()
imax = np.argmax(PSD)
f[imax]
# %%
