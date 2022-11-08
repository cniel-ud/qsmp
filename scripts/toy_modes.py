#%%
import os, sys
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import utils
import tree
from demo.core import qsmp
from demo.utils import mds, make_2D_modes
#%%
DPATH = "/home/cmendoza/Research/QSMP/data/toy"
IMG_DIR = "/home/cmendoza/MEGA/Research/Third_Paper/proto"
sigmas  = 2 ** np.linspace(-1, 1, 5)
sigma_noise = 2
normalize = True
norm_str = 'z-normalized' if normalize else 'unnormalized'

rng = default_rng(42)
n_samples, n_dims = 1000, 10

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
p = np.r_[0.4, 0.2, 0.2, 0.2]
k = p.size - 1

X = make_2D_modes(mu, cov, n_samples, n_dims, p, sigma_noise, rng=rng)

X = np.vstack((
    X,
    np.hstack((mu, np.zeros((k, n_dims-2))))
))

X2D = mds(X, normalize=normalize)
true_modes = X2D[-k:]
X = X[:-k]
X2D = X2D[:-k]
isort = rng.permutation(X.shape[0])
X = X[isort]
X2D = X2D[isort]

fig = plt.figure(figsize=(10.5, 8))
plt.scatter(X2D[:, 0], X2D[:, 1], s=1)
plt.tight_layout()
plt.title(f'{norm_str}')
fname = 'toy_dataset.pdf'
fpath = os.path.join(IMG_DIR, fname)
fig.savefig(fpath, bbox_inches='tight')

profile, neighbor, density = qsmp(X, sigmas, normalize=normalize)
profile, neighbor, density = utils.fix_root(
    (profile, neighbor, density))
profile = np.sqrt(profile)


q2_3 = np.quantile(profile.flatten(), [0.5, 0.9])
q2_3 = np.log2(q2_3)
maxdists = 2 ** np.linspace(*q2_3, 5)
n_subseq, n_bw = neighbor.shape
# path_agg = ['add', 'max', 'mean']
path_agg = ['max']
# markers = ['v', '<', '>']
markers = ['o']
voff = np.r_[0, 4, 8, 12]/10.0
fig = plt.figure(constrained_layout=False, figsize=(10.5, 8))
gs = fig.add_gridspec(5, 6)
for i, sigma in enumerate(sigmas):
    for j, maxdist in enumerate(maxdists):
        ax = fig.add_subplot(gs[i, j])
        for kk, f in enumerate(path_agg):
            modes = tree.find_modes(
                (profile[:, i], neighbor[:, i], density[:, i]),
                maxdist, distfunc=f)
            modes_idx = np.array([mode.index for mode in modes])
            x0 = X2D[modes_idx[:4], 0]
            # x0[x0 < 0] -= .5*np.arange(np.size((x0 < 0).nonzero()))
            # x0[x0 > 0] += .5*np.arange(np.size((x0 > 0).nonzero()))
            x1 = X2D[modes_idx[:4], 1]
            # x1[x1 < 0] -= voff[kk]
            # x1[x1 > 0] += voff[kk]
            ax.scatter(x0, x1, marker=markers[kk], label=path_agg[kk],
                       s=10, alpha=1, edgecolors='none', lw=0.1)

            x_tail, y_tail = x0[0], x1[0]
            for x_head, y_head in zip(x0[1:], x1[1:]):
                arrow = mpatches.FancyArrowPatch(
                    (x_tail, y_tail), (x_head, y_head),
                    mutation_scale=7,
                    facecolor='red',
                    edgecolor='none')
                ax.add_patch(arrow)
                x_tail, y_tail = x_head, y_head

            ax.set(xticks=[], yticks=[])
        if j == 0:
            ax.set_ylabel(f'{sigma:.3g}')
        if i == 0:
            ax.set_title(f'{maxdist:.3g}')
            #fname = f'tree_bw{bw:.3g}_maxdist{maxdist:.3g}_{f}.npz'
handles, labels = ax.get_legend_handles_labels()
ax = fig.add_subplot(gs[2, -1])
ax.scatter(true_modes[:, 0], true_modes[:, 1], marker='d', s=5)
ax.set_title('True modes')
l, b, w, h = ax.get_position().bounds
b = b + h*1.5
l = l + w*.5
fig.legend(handles, labels, loc=(l, b))
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
plt.suptitle(f'{norm_str}')
plt.show()
plt.tight_layout()
fname = 'clustering_toy_dataset.pdf'
fpath = os.path.join(IMG_DIR, fname)
fig.savefig(fpath, bbox_inches='tight')

# %%

# %%
