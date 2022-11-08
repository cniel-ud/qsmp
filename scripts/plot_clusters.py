#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import qsmp.utils.pltaux as pltaux

#%%
def mds(X):
    mu, std = np.mean(X, axis=1), np.std(X, axis=1)
    X = (X - mu[:, None])/std[:, None]
    D = 2*m - 2*X @ X.T
    N = D.shape[0]
    H = np.eye(N) - (1/N) * np.ones((N, N))
    G = -(1/2) * H @ D @ H
    L, U = np.linalg.eig(G)
    assert np.max(np.imag(L)) < 1e-10
    L, U = np.real(L), np.real(U)
    isort = np.argsort(np.sort(-L))
    L, U = L[isort[:2]], U[:, isort[:2]]
    L = np.diag(np.sqrt(L))
    return U @ L

#%%
IMG_DIR = "/home/cmendoza/MEGA/Research/Third_Paper/proto"
DPATH = "/home/cmendoza/Research/QSMP/data/Study019/preictal"
fname = 'qsmp_m350_snr0.0_2.0_4.0_8.0_10.0.npz'
fpath = os.path.join(DPATH, fname)

with np.load(fpath) as data:
    density = data['density']
    profile = data['profile']
    neighbor = data['indices']

#%%
fname = 'first56_segments_CSP1.npz'
fpath = os.path.join(DPATH, fname)
with np.load(fpath) as data:
    T = data['time_series']
    splice = data['splice']
#%%
density, profile, neighbor = density[:, 1][:,None], profile[:, 1][:, None], neighbor[:, 1][:, None]
#%%
imax = np.argmax(density)
profile[np.asarray(np.isinf(profile)).nonzero()[0]] = 0
#%%
plt.rcParams['text.usetex'] = True
vfactor = 0.2658
fig, ax = plt.subplots(3, 1, figsize=(10.5, 8))
ax[0].plot(T * vfactor)
ax[0].set_ylabel(r'$\mu$V')
ax[0].set_xticks([])
ax[1].plot(density)
ax[1].set_xticks([])
ax[2].plot(profile)
ax[2].set_xticks([0, T.size-1])
plt.setp(ax, xlim=[0, T.size-1])
plt.rcParams.update({'font.size': 35})
plt.tight_layout(pad=0)
fname = 'preictal_full_QSMP.png'
fpath = os.path.join(IMG_DIR, fname)
plt.savefig(fpath, bbox_inches='tight')
#%%
fig, ax = plt.subplots(3, 1, figsize=(10.5, 8))
ax[0].plot(T[imax-13131:imax+13131])
ax[0].set_xticks([])
ax[1].plot(density[imax-13131:imax+13131-349])
ax[1].set_xticks([])
ax[2].plot(profile[imax-13131:imax+13131-349])
ax[2].set_xticks([0, 13131*2])
plt.setp(ax, xlim=[0, 13131*2])
plt.rcParams.update({'font.size': 35})
plt.tight_layout(pad=0)
fname = 'real_demo.png'
fpath = os.path.join(IMG_DIR, fname)
plt.savefig(fpath, bbox_inches='tight')
#%%

snr = np.r_[0, 2, 4, 8, 10]
var_noise = 10 ** (-snr/10)
th = 0.1
bandwidths = (9 * var_noise) / np.log(1/th)
m = 350
n_modes = 5
maxdists = 2 ** np.linspace(2.6, 4, 5)
n_bw = snr.size
path_agg = ['add', 'max', 'mean']

#%%
m = 350
fname = 'tree_bw2.47_maxdist7.73_max.npz'
fpath = os.path.join(DPATH, fname)
with np.load(fpath) as data:
    modes = data['modes']
    new_neighbor = data['new_neighbor']

t = modes[:16, None] + np.arange(m)[None, :]
waves = T[t]
# X[i][j].append(waves)
# labels[i][j].append(modes[:20])
# f_count[i][j].append(waves.shape[0])
waves, _, _ = pltaux.wave_matrix(waves)
plt.plot(waves.T, color='#1f77b4')
plt.axis('off')
fname = 'real_demo_modes.png'
fpath = os.path.join(IMG_DIR, fname)
plt.savefig(fpath, bbox_inches='tight')

#%%
fig, ax = plt.subplots(5, 9, figsize=(10.5, 8))
ax = ax.flatten()
zero_ref = np.zeros(m)
cnt_ax = 0
X  = [0] * maxdists.size
markers = ['+', '*', 's']
f_count = [0] * maxdists.size
labels = [0] * maxdists.size
for i, maxdist in enumerate(maxdists):
    X[i] = [0] * 3
    labels[i] = [0] * 3
    f_count[i] = [0] * 3
    for j, bw in enumerate(bandwidths[:3]):
        X[i][j] = []
        labels[i][j] = []
        f_count[i][j] = []
        for f in path_agg:
            fname = f'tree_bw{bw:.3g}_maxdist{maxdist:.3g}_{f}.npz'
            fpath = os.path.join(DPATH, fname)
            with np.load(fpath) as data:
                modes = data['modes']
                new_neighbor = data['new_neighbor']

            t = modes[:20, None] + np.arange(m)[None, :]
            waves = T[t]
            X[i][j].append(waves)
            labels[i][j].append(modes[:20])
            f_count[i][j].append(waves.shape[0])
            waves,_,_ = pltaux.wave_matrix(waves[:9])
            ax[cnt_ax].plot(waves.T, color='#1f77b4')
            if cnt_ax % 9 == 0:
                ax[cnt_ax].set(frame_on=False, xticks=[], yticks = [])
                ax[cnt_ax].set_ylabel(f'{maxdist:.3g}')
            else:
                ax[cnt_ax].axis('off')
            ax[cnt_ax].set_title(f'{bw:.3g}, {f}')
            cnt_ax += 1
fig.supylabel('maxdist')
fig.supxlabel('bandwidth, path distance aggregation')
plt.tight_layout()
fname = 'maxdist_bw_path_agg_comparison_first_9_modes.pdf'
fpath = os.path.join(IMG_DIR, fname)
fig.savefig(fpath, bbox_inches='tight')
# %%
Y = [0] * maxdists.size
fig, ax = plt.subplots(5, 3, figsize=(10.5, 8))
ax = ax.flatten()
cnt_ax = 0
for i, maxdist in enumerate(maxdists):
    Y[i] = [0] * 3
    for j, bw in enumerate(bandwidths[:3]):
        X[i][j] = np.vstack(X[i][j])
        Y[i][j] = mds(X[i][j])
        start = 0
        for k, modecnt in enumerate(f_count[i][j]):
            stop = start + modecnt
            ax[cnt_ax].scatter(
                Y[i][j][start:stop,0], Y[i][j][start:stop,1], marker=markers[k], label=path_agg[k])
            start = stop

        if cnt_ax % 3 == 0:
            ax[cnt_ax].set_ylabel(f'{maxdist:.3g}')
        if cnt_ax < 3:
            ax[cnt_ax].set_title(f'{bw:.3g}')
        cnt_ax += 1
handles, leg_labels = ax[-1].get_legend_handles_labels()
fig.legend(handles, leg_labels, loc='upper center')
plt.subplots_adjust()
fname = 'MDS_first_20_modes.pdf'
fpath = os.path.join(IMG_DIR, fname)
fig.savefig(fpath, bbox_inches='tight')

# %%
XX = []
for i, maxdist in enumerate(maxdists):
    for j, bw in enumerate(bandwidths[:3]):
        XX.append(np.vstack(X[i][j]))

XX = np.vstack(XX)
Y = mds(XX)
fig, ax = plt.subplots(5, 3, figsize=(10.5, 8))
ax = ax.flatten()
cnt_ax = 0
start = 0
for i, maxdist in enumerate(maxdists):
    for j, bw in enumerate(bandwidths[:3]):
        for k, modecnt in enumerate(f_count[i][j]):
            stop = start + modecnt
            ax[cnt_ax].scatter(
                Y[start:stop, 0], Y[start:stop, 1], marker=markers[k], label=path_agg[k])
            start = stop

        if cnt_ax % 3 == 0:
            ax[cnt_ax].set_ylabel(f'{maxdist:.3g}')
        if cnt_ax < 3:
            ax[cnt_ax].set_title(f'{bw:.3g}')
        cnt_ax += 1
handles, labels = ax[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
xlim, ylim = [0] * 2, [0] * 2
xlim[0] = np.floor(min([plt.getp(ax[i], 'xlim')[0] for i in range(15)]))
xlim[1] = np.ceil(max([plt.getp(ax[i], 'xlim')[1] for i in range(15)]))
ylim[0] = np.floor(min([plt.getp(ax[i], 'ylim')[0] for i in range(15)]))
ylim[1] = np.ceil(max([plt.getp(ax[i], 'ylim')[1] for i in range(15)]))
plt.setp(ax, xlim=xlim, ylim=ylim)
plt.subplots_adjust()
fname = 'OneMDS_first_20_modes.pdf'
fpath = os.path.join(IMG_DIR, fname)
fig.savefig(fpath, bbox_inches='tight')

# %%
