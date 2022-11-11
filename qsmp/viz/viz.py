from matplotlib import pyplot as plt
import numpy as np
from qsmp import tree

def show_segment_start(T, text, wave_len=512, n_seg=4, start=1000):

    idx = np.arange(start, start+n_seg*wave_len)
    plt.plot(idx, T[idx])
    plt.xlabel('Sample index')
    plt.xlim(idx[0], idx[-1])
    plt.title(f'A small segment of {text}')


def show_density(density, sigma, wave_len=512, n_seg=4):

    density = density[sigma]
    center = np.argmax(density)
    idx = np.arange(max(0, center-wave_len*n_seg),
                    min(density.size, center+n_seg*wave_len))
    plt.plot(idx, density[idx])
    plt.xlabel('Sample index')
    plt.xlim(idx[0], idx[-1])
    plt.title(f'Density around global maxima')


def show_modes_across_maxdist(
    T, wave_len, density, NNindex,
    NNdist, q=[0.75, 0.99], n_dist=5, sigma=0, max_modes=10, n_neighbors=9, show_neighbors=False,
    ):

    quantiles = np.quantile(NNdist, q)
    quantiles = np.log2(quantiles)
    max_dist = 2 ** np.linspace(*quantiles, n_dist)
    fig, ax = plt.subplots(max_modes, n_dist, figsize=(n_dist, max_modes))

    for j, dist in enumerate(max_dist):
        NNd, NNi, modes, cluster_size = tree.tree2clusters(
            wave_len, density[sigma], NNindex[sigma],
            NNdist[sigma], dist
        )

        sample, idx = tree.get_neighbors(
            T, wave_len, density[sigma], NNi,
            NNd, modes, max_modes, n_neighbors
        )
        n_modes = len(sample)
        for i in range(n_modes):
            ax[i, j].axis('off')
            ax[i, j].plot(sample[i][0])
            if show_neighbors:
                ax[i, j].plot(sample[i][1:].T)
            if i == 0:
                ax[i, j].set_title(f'{dist:.2f}')

        for i in range(n_modes, max_modes):
            ax[i,j].remove()

    plt.suptitle('Modes across different distance thresholds')
    plt.tight_layout()
