#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from spectrum import MultiTapering
from scipy.signal import lfilter
# %% Generate Weierstrass function
rng = default_rng()
#%%
Fs = 512
t = np.reshape(np.arange(0, 1, 1/Fs), (1, -1))
phi = 2 * np.pi * rng.random() - np.pi
phi = np.reshape(phi, (-1, 1))
n = np.arange(0, 100)
omega, H = 4.0, 0.5
omega = np.reshape(omega**n, (-1, 1))
x = np.sum(
    omega**(-H) * np.cos(omega * t + phi), axis=0
)
plt.figure()
plt.plot(x)
plt.figure()
X = MultiTapering(x, NFFT=1024, NW=3, sampling=Fs)
X.plot()
# %%
# https://ccrma.stanford.edu/~jos/sasp/Example_Synthesis_1_F_Noise.html
Nx = 2**16;  # number of samples to synthesize
B = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
A = np.array([1, -2.494956002, 2.017265875, -0.522189400])
nT60 = np.round(np.log(1000)/(1-max(abs(np.roots(A)))))
nT60 = int(nT60)
v = rng.normal(scale=100, size=(1, Nx+nT60))
x = lfilter(B, A, v)
x.shape = (-1,)
x = x[nT60+1:-1]
plt.plot(x[100:200])
# %%
X = MultiTapering(x, NFFT=1024, NW=3)
X.plot()

# %%
# Morlet wavelet
w = 10 # Hz
x = np.pi**-0.25 * (np.exp(1j*w*t) - np.exp(-0.5*(w**2))) * np.exp(-0.5*(t**2))
plt.plot(x.imag.squeeze())
# %%
w, M = 10, 512
x = np.linspace(-2 * np.pi, 2 * np.pi, M)
wav = np.exp(1j * w * x) - np.exp(-0.5 * (w**2))
wav *= np.exp(-0.5 * (x**2)) * np.pi**(-0.25)
plt.plot(wav.real.squeeze())
# %%
Wav = MultiTapering(wav, NFFT=1024, NW=3, sampling=512)
Wav.plot()

# %%
PSD, f = Wav.psd, Wav.frequencies()
# %%
imax = np.argmax(PSD)
f[imax]
# %%
