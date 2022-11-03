import numpy as np
import qsmp.utils.utils as utils

def get_lambda_max(X, D_hat, sample_weights=None):

    # univariate case, add a dimension (n_channels = 1)
    if X.ndim == 2:
        X = X[:, None, :]
        D_hat = D_hat[:, None, :]
        if sample_weights is not None:
            sample_weights = sample_weights[:, None, :]

    n_trials, n_channels, n_times = X.shape

    if sample_weights is None:
        # no need for the last dimension if we only have ones
        if D_hat.ndim == 2:
            sample_weights = np.ones(n_trials)
        else:
            sample_weights = np.ones((n_trials, n_channels))

    # multivariate rank-1 case
    if D_hat.ndim == 2:
        return np.max([[
            np.convolve(
                np.dot(uv_k[:n_channels], X_i * W_i), uv_k[:n_channels - 1:-1],
                mode='valid') for X_i, W_i in zip(X, sample_weights)
        ] for uv_k in D_hat], axis=(1, 2))[:, None]

    # multivariate general case
    else:
        return np.max([[
            np.sum([
                np.correlate(D_kp, X_ip * W_ip, mode='valid')
                for D_kp, X_ip, W_ip in zip(D_k, X_i, W_i)
            ], axis=0) for X_i, W_i in zip(X, sample_weights)
        ] for D_k in D_hat], axis=(1, 2))[:, None]


def construct_X(z, ds):
    """
    Parameters
    ----------
    z : array, shape (n_atoms, n_trials, n_times_valid)
        The activations
    ds : array, shape (n_atoms, n_times_atom)
        The atoms
    Returns
    -------
    X : array, shape (n_trials, n_times)
    """
    assert z.shape[0] == ds.shape[0]
    n_atoms, n_trials, n_times_valid = z.shape
    n_atoms, n_times_atom = ds.shape
    n_times = n_times_valid + n_times_atom - 1

    X = np.zeros((n_trials, n_times))
    for i in range(n_trials):
        X[i] = _choose_convolve(z[:, i], ds)
    return X


def _choose_convolve(z_i, ds):
    """Choose between _dense_convolve and _sparse_convolve with a heuristic
    on the sparsity of z_i, and perform the convolution.
    z_i : array, shape(n_atoms, n_times_valid)
        Activations
    ds : array, shape(n_atoms, n_times_atom)
        Dictionary
    """
    assert z_i.shape[0] == ds.shape[0]

    if np.sum(z_i != 0) < 0.01 * z_i.size:
        return _sparse_convolve(z_i, ds)
    else:
        return _dense_convolve(z_i, ds)


def _sparse_convolve(z_i, ds):
    """Same as _dense_convolve, but use the sparsity of zi."""
    n_atoms, n_times_atom = ds.shape
    n_atoms, n_times_valid = z_i.shape
    n_times = n_times_valid + n_times_atom - 1
    Xi = np.zeros(n_times)
    for zik, dk in zip(z_i, ds):
        for nnz in np.where(zik != 0)[0]:
            Xi[nnz:nnz + n_times_atom] += zik[nnz] * dk
    return Xi


def _dense_convolve(z_i, ds):
    """Convolve z_i[k] and ds[k] for each atom k, and return the sum."""
    return sum([np.convolve(zik, dk) for zik, dk in zip(z_i, ds)], 0)


def power_iteration(lin_op, n_points=None, b_hat_0=None, max_iter=1000,
                    tol=1e-7, random_state=None):
    """Estimate dominant eigenvalue of linear operator A.
    Parameters
    ----------
    lin_op : callable or array
        Linear operator from which we estimate the largest eigenvalue.
    n_points : tuple
        Input shape of the linear operator `lin_op`.
    b_hat_0 : array, shape (n_points, )
        Init vector. The estimated eigen-vector is stored inplace in `b_hat_0`
        to allow warm start of future call of this function with the same
        variable.
    Returns
    -------
    mu_hat : float
        The largest eigenvalue
    """
    if hasattr(lin_op, 'dot'):
        n_points = lin_op.shape[1]
        lin_op = lin_op.dot
    elif callable(lin_op):
        msg = ("power_iteration require n_points argument when lin_op is "
               "callable")
        assert n_points is not None, msg
    else:
        raise ValueError("lin_op should be a callable or a ndarray")

    rng = utils.check_rng(random_state)
    if b_hat_0 is None:
        b_hat = rng.random(n_points)
    else:
        b_hat = b_hat_0

    mu_hat = np.nan
    for ii in range(max_iter):
        b_hat = lin_op(b_hat)
        norm = np.linalg.norm(b_hat)
        if norm == 0:
            return 0
        b_hat /= norm
        fb_hat = lin_op(b_hat)
        mu_old = mu_hat
        mu_hat = np.dot(b_hat, fb_hat)
        # note, we might exit the loop before b_hat converges
        # since we care only about mu_hat converging
        if (mu_hat - mu_old) / mu_old < tol:
            break

    assert not np.isnan(mu_hat)

    if b_hat_0 is not None:
        # copy inplace into b_hat_0 for next call to power_iteration
        np.copyto(b_hat_0, b_hat)

    return mu_hat
