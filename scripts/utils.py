import numpy as np
from numpy.random import default_rng

def zscore(X):
    mu = np.mean(X, axis=1)[:, None]
    sigma = np.std(X, axis=1)[:, None]
    sigma[sigma<1e-10] = 1
    X = (X - mu) / sigma
    return X

def mds(X, normalize=True):
    if normalize:
        X = zscore(X)

    row_norms = np.sum(X * X, axis=1)[None, :]
    D = row_norms.T + row_norms - 2*X @ X.T

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


def make_2D_modes(mu, cov, n_samples, n_dims, partition, std_noise, rng=None):

    if rng is None:
        rng = default_rng()

    k = partition.size - 1

    n_samples_part = np.ceil(partition[1:] * n_samples).astype(int)
    tot_mode_samples = np.sum(n_samples_part)
    n_noise_samples = int(n_samples - tot_mode_samples)

    b = np.sqrt(3) * std_noise
    noise = b*rng.uniform(-1, 1, size=(n_noise_samples, n_dims))

    modes = np.zeros((tot_mode_samples, n_dims))

    start = 0
    for ic in range(k):
        stop = start + n_samples_part[ic]
        modes[start:stop, :2] = rng.multivariate_normal(
            mu[ic], cov[ic], size=n_samples_part[ic])
        start = stop

    X = np.vstack((noise, modes))

    return X
