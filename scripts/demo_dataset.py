#%%
import os
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import utils
import tree
#%%
IMG_DIR = "/home/cmendoza/MEGA/Research/Third_Paper/proto"
snr = np.r_[0, 2, 4, 8, 10]
var_noise = 10 ** (-snr/10)
th = 0.1
bandwidths = (9 * var_noise) / np.log(1/th)

rng = default_rng(42)
N, k = 1000, 4
p = np.r_[0.4, 0.25, 0.2, 0.15]
n_samples = np.zeros(k, dtype=np.int64)
n_samples[:k-1] = np.ceil(p[:k-1] * N)
n_samples[k-1] = N - np.sum(n_samples[:k-1])
max_r = 20000
min_r = 400

X = np.zeros((N, 2))
r = rng.uniform(min_r, max_r, n_samples[0])
r = np.sqrt(r)
θ = rng.uniform(0, 2*np.pi, n_samples[0])
X[:n_samples[0], 0] = r * np.cos(θ)
X[:n_samples[0], 1] = r * np.sin(θ)

mu = np.array([
    [-7, 7],
    [7, 7],
    [0, -7]]
)
cov = np.array([
    [[1, 0.1],
     [0.1, 1]],
    [[1, 0.1],
     [0.1, 1]],
    [[1, 0.1],
     [0.1, 1]]],
)

start = n_samples[0]
for ic in range(1, k):
    stop = start + n_samples[ic]
    X[start:stop] = rng.multivariate_normal(mu[ic-1], cov[ic-1], size=n_samples[ic])
    start = stop

fig = plt.figure(figsize=(10.5, 8))
plt.scatter(X[:,0], X[:,1], s=1)
plt.tight_layout()
fname = 'toy_dataset.pdf'
fpath = os.path.join(IMG_DIR, fname)
fig.savefig(fpath, bbox_inches='tight')

rng.shuffle(X)
# μ_X = np.mean(X, axis=1)[:, None]
sq_row_norms = np.sum(X * X, axis=1)
# X = (X - μ_X) / σ_X
# %%
m = 2
density = np.zeros((5, N))
for i in range(N):
    dist = sq_row_norms + sq_row_norms[i] - 2*X @ X[i]
    dist[i] = np.inf
    for ibw, bw in enumerate(bandwidths):
        density[ibw] += np.exp(-dist/bw)

plt.plot(density[0])
plt.title('Density')
# %%
density = density.T
profile = np.full((N,5), fill_value=np.inf)
neighbor = np.full((N,5), fill_value=-1, dtype=np.int64)
for i in range(N):
    dist = sq_row_norms + sq_row_norms[i] - 2*X @ X[i]
    for ibw, bw in enumerate(bandwidths):
        inc_density = density[:, ibw] > density[i, ibw]
        inc_density = np.asarray(inc_density).nonzero()[0]
        if inc_density.size > 0:
            imin = np.argmin(dist[inc_density])
            imin = inc_density[imin]
            if dist[imin] < profile[i, ibw]:
                neighbor[i, ibw] = imin
                profile[i, ibw] = dist[imin]
#%% Find global maxima (root), and fix neighbor and profile
profile, neighbor, density = utils.fix_root(
    (profile, neighbor, density))
plt.plot(profile)
# %%
DPATH = "/home/cmendoza/Research/QSMP/data/toy"
# maxdists = 2 ** np.linspace(3.1, 4.2, 5)
maxdists = 2 ** np.linspace(7, 9, 5)
n_subseq, n_bw = neighbor.shape
path_agg = ['add', 'max', 'mean']
markers = ['v', '<', '>']
voff = np.r_[0, 4, 8, 12]
fig = plt.figure(constrained_layout=False, figsize=(10.5, 8))
gs = fig.add_gridspec(5,6)
for i, bw in enumerate(bandwidths):
    for j, maxdist in enumerate(maxdists):
        ax = fig.add_subplot(gs[i,j])
        for kk, f in enumerate(path_agg):
            qsmp = (profile, neighbor, density)
            modes = tree.find_modes(
                (profile[:, i], neighbor[:, i], density[:, i]),
                maxdist, distfunc=f)
            x0 = X[modes[:4], 0]
            x0[x0<0] -= 2.5*np.arange(np.size((x0<0).nonzero()))
            x0[x0>0] += 2.5*np.arange(np.size((x0>0).nonzero()))
            x1 = X[modes[:4], 1]
            x1[x1<0] -= voff[kk]
            x1[x1>0] += voff[kk]
            ax.scatter(x0, x1, marker=markers[kk], label=path_agg[kk], s=10, alpha=0.8, edgecolors='none', lw=0.1)
            ax.set(xticks=[], yticks=[])
        if j == 0:
            sigma =  np.sqrt(bw/2)
            ax.set_ylabel(f'{sigma:.3g}')
        if i == 0:
            ax.set_title(f'{maxdist:.3g}')
            #fname = f'tree_bw{bw:.3g}_maxdist{maxdist:.3g}_{f}.npz'
handles, labels = ax.get_legend_handles_labels()
ax = fig.add_subplot(gs[2,-1])
ax.scatter(mu[:, 0], mu[:, 1], marker='d', s=5)
ax.set_title('True modes')
l, b, w, h = ax.get_position().bounds
b = b + h*1.5
l = l + w*.5
fig.legend(handles, labels, loc=(l,b))
xlim, ylim = [0] * 2, [0] * 2
ax = fig.get_axes()
xlim[0] = np.floor(min([plt.getp(ax[i], 'xlim')[0] for i in range(len(ax))]))
xlim[1] = np.ceil(max([plt.getp(ax[i], 'xlim')[1] for i in range(len(ax))]))
ylim[0] = np.floor(min([plt.getp(ax[i], 'ylim')[0] for i in range(len(ax))]))
ylim[1] = np.ceil(max([plt.getp(ax[i], 'ylim')[1] for i in range(len(ax))]))
plt.setp(ax, xlim=xlim, ylim=ylim)
plt.setp(ax, frame_on=True)
l, b, w, h = ax[-6].get_position().bounds
supx = fig.supxlabel('maxdist', y=b*0.7)
supy = fig.supylabel('sigma', x=l*0.7)
plt.show()
plt.tight_layout()
fname = 'clustering_toy_dataset.pdf'
fpath = os.path.join(IMG_DIR, fname)
fig.savefig(fpath, bbox_inches='tight')

# %%
