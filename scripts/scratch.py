#%%
import os, sys
import numpy as np
import qsmp.utils.pltaux as pltaux
from copy import deepcopy
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import tree
import utils
from spectrum import MultiTapering, pmtm
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.fft import fft, ifft
import scipy.signal as signal
from scipy.interpolate import UnivariateSpline
import core

#%%
root = "/home/cmendoza/Research/QSMP/data/Study019/"
folder = "preictal"

fname = 'first56_segments_CSP1.npz'
fpath = os.path.join(root, folder, fname)
with np.load(fpath) as data:
    T = data['time_series']
    splice = data['splice']

#%%
fs, n_taps = 512, 1001
f, Px_mean = core.mean_PSD(T, splice)
Px_mean_rev, coeffs = core.whitening_filter(f, Px_mean, n_taps=n_taps, fs=fs)
Px_list = [Px_mean, Px_mean_rev]
#%%
plt.clf()
plt.figure(figsize=(11.5, 8))
titles = ['Average amplitude spectrum',
          'Desired filter response', 'Filter response']
for i, Px_i in enumerate(Px_list):    
    plt.plot(f, 10*np.log10(np.sqrt(Px_i)))

freq, response = signal.freqz(coeffs, fs=fs)
y = 10*np.log10(np.abs(response))
plt.plot(freq, y)
plt.legend(titles)
plt.ylabel('Power [dB]')
plt.xlabel('Frequency [Hz]')
plt.grid(True)
plt.xlim(0, 256)
plt.grid(True)
plt.title(f'Linear-phase filter, {n_taps} taps')
plt.show()
#%% filt_T has the first `group_delay` samples from each filtered segment 
# removed
grp_delay = core.get_group_delay(coeffs, f, fs=fs)
filt_T, new_splice = core.whiten(T, splice, coeffs, grp_delay)
#%% Load QSMP
fname = 'qsmp_m350_snr-4.0_-2.0_0.0_2.0_4.0.npz'
fpath = os.path.join(root, folder, fname)
with np.load(fpath) as data:
    density = data['density']
    profile = data['profile']
    neighbor = data['indices']

#%% Hyperparameters
nonan_profile = profile[~np.isnan(profile)]
quantiles = np.quantile(nonan_profile, [0.75, 0.99])
quantiles = np.log2(quantiles)
maxdists = 2 ** np.linspace(*quantiles, 5)

snr = np.r_[-4, -2, 0, 2, 4]
var_noise = 10 ** (-snr/10)
th = 0.1
bandwidths = (9 * var_noise) / np.log(1/th)
sigmas = np.sqrt(bandwidths/2)
n_sigmas = sigmas.size

path_agg = ['add', 'max', 'mean']
path_agg = [path_agg[1]]
#%%
profile = profile.T[:n_sigmas]
neighbor = neighbor.T[:n_sigmas]
density = density.T[:n_sigmas]

#%%
m = 350
max_modes = 128
n_densities, n_pnts = density.shape
# dpath = os.path.join(root, folder, "qsmp-fwhm")
dpath = os.path.join(root, folder)
modes = utils.load_modes(dpath, maxdists[3], path_agg[0])
#%%
modes_i = deepcopy(modes[4])
modes_i = tree.reduce_close_modes(modes_i, m)
waves, ind = utils.get_waves(modes_i, T, m)
#%%
titles = [
    'Waveforms from original time series', 'Waveforms from filtered time series (linear-phase filter)']
n_modes = 100
rng = default_rng(13)
idx = rng.choice(ind.size, size=n_modes, replace=False)
idx = np.r_[0, idx[:-1]]
idx = idx[:n_modes]
sample = waves[idx]
filt_ind = ind[idx]
end_seg = np.r_[splice, T.size]  # end index of each segment
i_seg = np.searchsorted(end_seg, filt_ind) + 1
filt_ind = filt_ind - i_seg*grp_delay
t = filt_ind[:, None] + np.arange(m)[None, :]
filt_sample = filt_T[t]
samples = [sample, filt_sample]
fig, ax = plt.subplots(1, 2, figsize=(2*11, 8.5))
for i, x in enumerate(samples):
    waves_plt, n_rows, n_cols = pltaux.wave_matrix(x)
    ax[i].plot(waves_plt.T, color='#1f77b4')
    ax[i].set_title(titles[i])
    ax[i].axis('off')

plt.tight_layout()
plt.show()
# fpath = os.path.join(root, folder, 'test_linear_phase_filter_waves_fullseries.pdf')
# plt.savefig(fpath)
#%%
i_wav = 32
t = np.arange(m)/fs
fig, ax = plt.subplots(1,2,figsize=(2*11.5,4))
titles = [f'Original waveform (#{i_wav})', '"Whitened" waveform']
X = [sample[i_wav], filt_sample[i_wav]]
for i, x in enumerate(X):
    ax[i].plot(t, x, color='#1f77b4')
    ax[i].set_title(titles[i])
    ax[i].set_xlabel('Time (seconds)')
    ax[i].set_ylabel('Amplitude')
    ax[i].grid(True)
    ax[i].set_xlim(0, t[-1])
plt.show()
#%%
titles = ['Waveforms from original time series',
          'Waveforms from filtered time series (linear-phase filter)']
fig, ax = plt.subplots(1, 2, figsize=(2*11, 8.5))
Px = np.zeros((n_modes, NFFT//2 + 1))
sample_list = [sample, filt_sample]
for i, sample_i in enumerate(sample_list):
    for j in range(n_modes):
        psd = MultiTapering(sample_i[j], NFFT=NFFT, NW=3, sampling=512)
        Px[j] = psd.psd

    f = psd.frequencies()    
    Px_plt, n_rows, n_cols = pltaux.wave_matrix(10*np.log(Px))
    ax[i].plot(Px_plt.T, color='#1f77b4')
    ax[i].set_title(titles[i])
    ax[i].axis('off')
plt.suptitle('PSD of windows')
plt.tight_layout()
fpath = os.path.join(
    root, folder, 'test_linearphase_filter_waves_PSD_fullseries.pdf')
plt.savefig(fpath)
#%%
rng = default_rng(13)
idx = rng.choice(ind.size, size=100, replace=False)
idx = np.r_[0, idx[:-1]]
sample = waves[idx]
waves_plt, n_rows, n_cols = pltaux.wave_matrix(sample)
plt.figure(figsize=(11, 8.5))
plt.plot(waves_plt.T, color='#1f77b4')
plt.show()
# %%
NFFT = 1024
Px = np.zeros((100, NFFT//2 + 1))
for i in range(100):
    psd = MultiTapering(sample[i], NFFT=NFFT, NW=3, sampling=512)
    Px[i] = psd.psd
# %%
f = psd.frequencies()
Px_plt, n_rows, n_cols = pltaux.wave_matrix(10*np.log(Px))
plt.figure(figsize=(11, 8.5))
plt.plot(Px_plt.T, color='#1f77b4')
plt.show()
# plt.plot(f, 10*np.log10(Px))
# %%
Px_mean = np.mean(Px, axis=0)
plt.plot(f, 10*np.log10(Px_mean))
plt.ylabel('Power [dB]')
plt.xlabel('Frequency [Hz]')
plt.grid(True)
plt.xlim(0,250)

# %%
Px_mean_rev = Px_mean[::-1]
plt.plot(f, 10*np.log10(Px_mean_rev))
plt.ylabel('Power [dB]')
plt.xlabel('Frequency [Hz]')
plt.grid(True)
plt.xlim(0, 250)
# %%
fs = 512
n_taps = 1001
fig, ax = plt.subplots(2, 2, figsize=(2*11, 2*8.5))
desired_gain = np.sqrt(Px_mean_rev)
coeffs = signal.firls(n_taps, f[:-1], desired_gain[:-1], fs=fs)
h_min = signal.minimum_phase(coeffs, n_fft=NFFT)
titles = [f'Linear-phase filter, {n_taps} taps', 'Minimum-phase filter']
ylabels = ['Amplitude[dB]', 'Phase[Radians]']
filts = [coeffs, h_min]
for i, filt in enumerate(filts):
    freq, response = signal.freqz(filt, fs=fs)
    y = [10*np.log10(np.abs(response)), np.angle(response)]
    for j in range(2):
        ax[i,j].plot(freq, y[j])
        if j==0:
            ax[i,j].plot(f, 10*np.log10(desired_gain))
            ax[i, j].legend(['Filter response', 'Desired response'])
        ax[i,j].set_ylabel(ylabels[j])
        ax[i,j].set_xlabel('Frequency [Hz]')
        ax[i,j].set_xlim(0, 256)
        ax[i,j].grid(True)
        ax[i,j].set_title(titles[i])

plt.tight_layout()
fpath = os.path.join(
    root, folder, 'test_linear_vs_minimum_phase_filter_response.pdf')
plt.savefig(fpath)
#%%
f, h = signal.freqz(coeffs, fs=fs)
plt.plot(f, np.angle(h))
#%%
fig, ax = plt.subplots(1, 2, figsize=(2*11, 8.5))
filt_sample = signal.lfilter(coeffs, 1, sample, axis=1)
mu = np.mean(filt_sample, axis=1)
std = np.std(filt_sample, axis=1)
filt_sample = (filt_sample-mu[:,None])/std[:,None]
waves_plt, n_rows, n_cols = pltaux.wave_matrix(filt_sample)
ax[1].plot(waves_plt.T, color='#1f77b4')
ax[1].set_title('After filtering with linear-phase filter')
ax[1].axis('off')
waves_plt, n_rows, n_cols = pltaux.wave_matrix(sample)
ax[0].plot(waves_plt.T, color='#1f77b4')
ax[0].set_title('Original waveforms')
ax[0].axis('off')
plt.tight_layout()
fpath = os.path.join(root, folder, 'test_linear_phase_filter_waves.pdf')
plt.savefig(fpath)

#%%
m = 350
dist_filt = 2*m - 2*filt_sample@filt_sample.T
mu = np.mean(sample, axis=1)
std = np.std(sample, axis=1)
sample_norm = (sample-mu[:, None])/std[:, None]
dist = 2*m - 2*sample_norm@sample_norm.T
vmin = np.min(np.concatenate((dist, dist_filt)))
vmax = np.max(np.concatenate((dist, dist_filt)))
fig, ax = plt.subplots(1, 2)
ax[1].imshow(dist_filt, vmin=vmin, vmax=vmax, cmap='inferno')
ax[0].imshow(dist, vmin=vmin, vmax=vmax, cmap='inferno')
#%%
#%%
fig, ax = plt.subplots(1, 2, figsize=(2*11, 8.5))
filt_sample = signal.lfilter(h_min, 1, sample, axis=1)
mu = np.mean(filt_sample, axis=1)
std = np.std(filt_sample, axis=1)
filt_sample = (filt_sample-mu[:, None])/std[:, None]
waves_plt, n_rows, n_cols = pltaux.wave_matrix(filt_sample)
ax[1].plot(waves_plt.T, color='#1f77b4')
ax[1].set_title('After filtering with minimum-phase filter')
ax[1].axis('off')
waves_plt, n_rows, n_cols = pltaux.wave_matrix(sample)
ax[0].plot(waves_plt.T, color='#1f77b4')
ax[0].set_title('Original waveforms')
ax[0].axis('off')
plt.tight_layout()
fpath = os.path.join(root, folder, 'test_min_phase_filter_waves.pdf')
plt.savefig(fpath)
#%%
fs = 512
freq, response = signal.freqz(h_min)
plt.plot(0.5*fs*freq/np.pi, 10*np.log10(np.abs(response)))
plt.plot(f, 10*np.log10(desired_gain))
plt.ylabel('Power [dB]')
plt.xlabel('Frequency [Hz]')
plt.grid(True)
plt.xlim(0, 256)
plt.legend(['Filter response', 'Desired gain'])
plt.show()
#%%
imp = np.zeros(350)
imp[0] = 1
imp_resp = signal.lfilter(h_min, 1, imp, axis=0)
psd_imp_resp = MultiTapering(imp_resp, NFFT=NFFT, NW=3, sampling=512)
psd_imp_resp.plot()
#%%
fig, ax = plt.subplots(1, 2)
f, h = signal.freqz(h_min, fs=fs)
ax[1].plot(f, np.angle(h))
ax[0].plot(imp_resp)
#%%
Sk_complex, weights,_ = pmtm(Ti, NW=3, NFFT=NFFT)
# %%
Sk = np.abs(Sk_complex)**2
Sk = Sk.transpose()
Sk = np.mean(Sk * weights, axis=1)
Px = Sk[0:int(NFFT/2 + 1)] * 2
plt.plot(f, 10*np.log10(Px))
plt.ylabel('Power [dB]')
plt.xlabel('Frequency [Hz]')
plt.grid(True)
plt.xlim(-10, 250)
# %%
X = np.sqrt(Sk)
plt.plot(f, 10*np.log10(X[:int(NFFT/2)+1]))
plt.ylabel('Power [dB]')
plt.xlabel('Frequency [Hz]')
plt.grid(True)
plt.xlim(-10, f[-1])
# %%
w = np.real(ifft(X))
plt.plot(w)
#%%
X2 = np.r_[X[int(NFFT/2)+1::-1], X[:int(NFFT/2)+1:-1]]
plt.plot(f, 10*np.log10(X2[:int(NFFT/2)+1]))
plt.ylabel('Power [dB]')
plt.xlabel('Frequency [Hz]')
plt.grid(True)
plt.xlim(0, f[-1]+10)
#%%
# Sk = Sk_complex.transpose()
# X = np.mean(Sk * weights, axis=1)
# X = np.r_[X[int(NFFT/2)+1::-1], X[:int(NFFT/2)+1:-1]]
# plt.plot(10*np.log10(np.abs(X)))
# %%
w = np.real(ifft(X2))
plt.plot(w)
# %%
plt.plot(w[:Ti.size])
# %%
