import numpy as np
from demo.utils import zscore

def qsmp(X, sigmas, normalize=True):
    n_sigmas = sigmas.size
    N, m = X.shape
    density = np.zeros((5, N))

    if normalize:
        X = zscore(X)

    row_norms = np.sum(X * X, axis=1)

    for i in range(N):

        dist = row_norms + row_norms[i] - 2*X @ X[i]

        for isig, sigma in enumerate(sigmas):
            density[isig] += np.exp(-dist/(2*sigma**2))

    density = density.T
    profile = np.full((N, 5), fill_value=np.inf)
    neighbor = np.full((N, 5), fill_value=-1, dtype=np.int64)
    for i in range(N):

        dist = row_norms + row_norms[i] - 2*X @ X[i]

        for isig in range(n_sigmas):
            inc_density = density[:, isig] > density[i, isig]
            inc_density = np.asarray(inc_density).nonzero()[0]
            if inc_density.size > 0:
                imin = np.argmin(dist[inc_density])
                imin = inc_density[imin]
                if dist[imin] < profile[i, isig]:
                    neighbor[i, isig] = imin
                    profile[i, isig] = dist[imin]

    below_th = profile < 1e-13
    profile[below_th] = 0

    return profile, neighbor, density
