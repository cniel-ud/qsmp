import numpy as np
import qsmp.utils.utils as utils

def cut_tree(parent, distance, max_dist, in_place=False):
    """ Cut the tree at edges where distance > max_dist

    Parameters
    ----------
    parent: numpy.ndarray
        parent[i] is the index of the parent of node `i`. parent represents an in-tree.
    distance: numpy.ndarray
        dist[i] is the distance from node `i` to its parent, node parent[i].
    max_dist: float
        Distance threshold at which to cut edges in the tree. This converts the in-tree into an in-forest. The child node in the edge that was cut becomes a root.

    Returns
    -------
    parent: numpy.ndarray
        parent[i] is the index of the root of node `i`.
    distance: numpy.ndarray
        New array of distances. New root nodes have now distance zero to their parent (themselves).
    """

    too_far = distance > max_dist
    if not in_place:
        parent, distance = parent.copy(), distance.copy()

    parent[too_far] = np.arange(parent.size)[too_far]
    distance[too_far] = 0

    if not in_place:
        return parent, distance


def mark_with_root(parent):
    """ Mark each node in the forest with its root

    Parameters
    ----------
    parent: numpy.ndarray
        parent[i] is the index of the parent of node `i`. parent represents an in-forest.

    Returns
    -------
    parent: numpy.ndarray
        parent[i] is the index of the root of node `i`.

    References:
    /segmentation/_quickshift_cy.pyx in sckit-learn
    """
    old = np.zeros_like(parent)
    while(old != parent).any():
        old = parent
        parent = parent[parent]

    return parent


def merge_roots(roots, density, gap):
    """ Merge roots in `roots` that are close in time

    Parameters
    ---------
    roots: numpy.ndarray
        roots[i] is the root of node i. roots.shape=(N,)
    density: numpy.ndarray
        density[i] is the density estimate evaluated at node i.
        density.shape=(N,)
    gap: int
        If two roots are less than `gap` time samples away, they get merged.
        The root with higher density is the winner.

    Returns
    -------
    roots: numpy.ndarray
        The indices of the roots after the merge process. roots.shape=(N,).
    winning_roots: numpy.ndarray
        Array with winning modes, ordered in descending order of density.
    tree_size: numpy.ndarray
        tree_size[i] has the number of nodes whose root is winning_roots[i]
    """

    unique_roots, ind, num_nodes = np.unique(
        roots, return_index=True, return_counts=True)

    unique_roots = unique_roots[num_nodes > 1] #ignore orphan roots
    ind = ind[num_nodes > 1]
    num_nodes = num_nodes[num_nodes > 1]

    if unique_roots.size == 0:
        return roots, np.full(0, 0), np.full(0, 0)

    start_arr = np.asarray(
        np.r_[True, np.diff(unique_roots) > gap]).nonzero()[0]
    end_arr = np.r_[start_arr[1:], unique_roots.size]

    n_roots = start_arr.size
    winning_roots = np.zeros(n_roots, dtype=np.int64)
    tree_size = np.zeros(n_roots, dtype=np.int64)
    isort = np.argsort(roots)
    roots_sorted = roots[isort]
    for i, (start, end) in enumerate(zip(start_arr, end_arr)):
        imax = np.argmax(density[unique_roots[start:end]])
        winning_roots[i] = unique_roots[start + imax]
        idx = utils.where_equal(roots_sorted, unique_roots[start:end])
        roots[isort[idx]] = winning_roots[i]  #XXX: in-place bug?
        tree_size[i] = np.sum(num_nodes[start:end])

    isort = np.argsort(-density[winning_roots])
    winning_roots = winning_roots[isort]
    tree_size = tree_size[isort]

    return roots, winning_roots, tree_size


def drop_trivial_matches(nodes, density, gap):
    """Within ``nodes``, keep only the highest-density representative of each
    temporally-contiguous group.

    Two nodes are considered to belong to the same group if their start
    indices are within ``gap`` samples. Inside a group, the node with the
    largest density is kept and the rest are dropped. This is used by
    :func:`k_neighborhood` to avoid returning multiple shifted copies of the
    same waveform as the "neighbors" of a mode.

    Parameters
    ----------
    nodes : numpy.ndarray
        1D integer array of node (subsequence start) indices.
    density : numpy.ndarray
        Density estimate evaluated at every node (only ``density[nodes]`` is
        used here, but the full array is passed for indexing convenience).
    gap : int
        Minimum temporal separation between retained nodes, in samples. The
        paper's exclusion zone of ``m/4`` is the typical value.

    Returns
    -------
    winning_nodes : numpy.ndarray
        Sorted, deduplicated subset of ``nodes``.
    """
    nodes = np.sort(nodes)
    start_arr = np.asarray(
        np.r_[True, np.diff(nodes) > gap]).nonzero()[0]
    end_arr = np.r_[start_arr[1:], nodes.size]

    n_roots = start_arr.size
    winning_nodes = np.zeros(n_roots, dtype=np.int64)
    for i, (start, end) in enumerate(zip(start_arr, end_arr)):
        imax = np.argmax(density[nodes[start:end]])
        winning_nodes[i] = nodes[start + imax]

    return winning_nodes


def k_neighborhood(roots, k, NNdist, NNindex, density, gap):
    """ The k nearest neighbors of a set of roots

    NNdist.shape=(N,). NNdist[i] is the distance between node i
    and its nearest neighbor (NN). N is the number of nodes in the in-forest.
    NNindex.shape=(N,). NNindex[i] is the index of the NN of node i.
    roots.shape=(m,). roots[i] is the start index of the i-th root node.
    k is an int, the size of the neighborhood of each root.

    Return a vector `idx` with the indices of each root followed by the
    indices of the their k-NN, sorted in ascending order of distance. `idx`
    has shape (m*(k+1),). idx[0]=roots[0], idx[1] is the nearest neighbor of idx[0], excluding itselft, idx[2] is the second closest neighbor of idx[0],
    idx[k+1]=roots[1], and so on.

    The type of idx is float, as we might have NaN indices in cases where the neighborhood of a mode is too small, and NaN is represented as float.

    This function creates the indices that are then used to extract the waves and plot them.
    """

    n_roots = roots.size
    idx = [None] * n_roots  # np.zeros(n_roots*(k+1))

    for i, root in enumerate(roots):
        children = np.asarray(NNindex == root).nonzero()[0]
        children = drop_trivial_matches(children, density, gap)
        n_children = children.size - 1
        _k = min(k, n_children)
        ind_topk = np.argpartition(NNdist[children], _k)[:_k+1]
        children = children[ind_topk]
        isort = np.argsort(NNdist[children])
        idx[i] = children[isort]

    return idx


def tree2clusters(sublen, density, NNindex, NNdist, max_dist):
    """Cut the QSMP tree at ``max_dist`` and return the resulting clustering.

    Convenience wrapper that chains :func:`cut_tree` (cut edges longer than
    ``max_dist`` to convert the tree into a forest), :func:`mark_with_root`
    (label every node with the index of its root), and :func:`merge_roots`
    (merge roots that are within ``sublen/4`` samples of each other, keeping
    the densest one).

    Parameters
    ----------
    sublen : int
        Subsequence length ``m``. The exclusion-zone half-width
        ``sublen/4`` is used as the gap parameter for ``merge_roots``,
        matching the paper's exclusion zone (Definition 5 of [1]).
    density : numpy.ndarray
        Density estimate, shape ``(N,)`` (one of the columns of the QS-tuple
        if multiple kernel widths were used).
    NNindex : numpy.ndarray
        NN-index, shape ``(N,)``. ``NNindex[i]`` is the index of the
        higher-density nearest neighbor of subsequence ``i``.
    NNdist : numpy.ndarray
        NN-distance, shape ``(N,)``. ``NNdist[i]`` is the (shift-invariant)
        distance between subsequence ``i`` and its nearest higher-density
        neighbor.
    max_dist : float
        Distance threshold ``tau``. Edges longer than ``max_dist`` are cut.

    Returns
    -------
    NNdist : numpy.ndarray
        Per-node distances after the cut (root nodes have distance 0).
    NNindex : numpy.ndarray
        Per-node root indices.
    modes : numpy.ndarray
        Indices of the surviving roots, ordered by descending density.
    cluster_size : numpy.ndarray
        Number of subsequences in each cluster, in the same order as
        ``modes``.

    See Also
    --------
    find_tau : Binary-search the value of ``max_dist`` that yields exactly
        ``k`` modes.

    References
    ----------
    .. [1] Mendoza-Cardenas & Brockmeier, *QSMP: finding representative time
       series subsequences through Quick Shift + Matrix Profile* (under
       resubmission).
    """
    NNindex, NNdist = cut_tree(
        NNindex, NNdist, max_dist)

    # Assign each subsequence to its root: clustering
    NNindex = mark_with_root(NNindex)

    # Merge roots that are less than m/4 apart
    # The densest root wins. Ignore orphan roots.
    NNindex, modes, cluster_size = merge_roots(
        NNindex, density, sublen/4)

    return NNdist, NNindex, modes, cluster_size


def get_neighbors(
        T, sublen, density, NNindex, NNdist, modes, max_modes, n_neighbors):
    """Extract the actual subsequence values for the top modes and their
    neighbors.

    For each of the first ``max_modes`` entries of ``modes``, this looks up
    its ``n_neighbors`` nearest non-trivial children in the QS forest and
    materialises the time-series segments (the *waveforms*) at those start
    indices. Used by the reporting and visualisation utilities.

    Parameters
    ----------
    T : numpy.ndarray
        The 1D time series (used to slice waveforms out of).
    sublen : int
        Subsequence length ``m``.
    density, NNindex, NNdist : numpy.ndarray
        The per-sigma columns of the QS-tuple (see :func:`tree2clusters`).
    modes : numpy.ndarray
        Mode (root) indices, ordered by descending density (typically the
        output of :func:`tree2clusters`).
    max_modes : int
        How many of the leading modes to return waveforms for.
    n_neighbors : int
        How many nearest neighbors to extract per mode.

    Returns
    -------
    sample : list of numpy.ndarray
        ``sample[i]`` is a 2D array of shape
        ``(n_neighbors_i + 1, sublen)`` whose first row is the ``i``-th mode
        and the remaining rows are its nearest neighbors.
    idx_list : list of numpy.ndarray
        ``idx_list[i]`` are the start indices in ``T`` corresponding to
        ``sample[i]``.
    """
    # Find nearest neighbors
    idx_list = k_neighborhood(modes[:max_modes],
                                   n_neighbors, NNdist, NNindex, density, sublen/4)

    # Build sample with modes and
    # sample[i][0] is the i-th mode, and sample[i][1:] are its neighbors
    sample = [None] * len(idx_list)
    for i, idx in enumerate(idx_list):
        t = idx[:, None] + np.arange(sublen)[None, :]
        sample[i] = T[t]

    return sample, idx_list


def recompute_distances(NNindex, time_series, wave_length):
    """Recompute z-normalised Euclidean distances between every node and its
    cluster root, on raw (non-shift-invariant) waveforms.

    The QSMP NN-distances stored in ``profile`` are *shift-invariant*: each
    one is the minimum of a window of length ``B`` around the actual nearest
    neighbour (see Eq. 7 of the paper). This function replaces them with
    plain z-normalised Euclidean distances between each subsequence and the
    root of its cluster, which can be useful for sanity-checking or
    distortion-style metrics.

    The implementation chunks the per-cluster batched matmul to keep memory
    use under ``1 GiB`` per chunk; tune ``max_chunk_size`` in the source if
    needed.

    Parameters
    ----------
    NNindex : numpy.ndarray
        Per-node root indices, one entry per subsequence (typically the
        output of :func:`tree2clusters`).
    time_series : numpy.ndarray
        The 1D time series ``T``.
    wave_length : int
        Subsequence length ``m``.

    Returns
    -------
    NNdist : numpy.ndarray
        Per-node distance to the cluster root, shape
        ``(NNindex.shape[0],)``.
    """
    n_children = NNindex.shape[0]
    NNdist = np.zeros(n_children)

    roots_idx, cluster_size = np.unique(NNindex, return_counts=True)
    roots_idx = roots_idx[cluster_size > 1] # ignore orphan roots
    n_roots = roots_idx.size
    max_chunk_size = 1024**3   # in bytes
    nbytes_per_wave = (wave_length*time_series.dtype.alignment)
    max_num_of_rows = np.int(max_chunk_size/nbytes_per_wave)
    for i in range(n_roots):

        root_wave = utils.get_waves(roots_idx[i], time_series, wave_length)
        mu, std = np.mean(root_wave), np.std(root_wave)
        if std == 0: std = 1
        root_wave = (root_wave - mu)/std

        children_idx = np.asarray(NNindex == roots_idx[i]).nonzero()[0]
        n_children = children_idx.size
        if n_children > max_num_of_rows:
            n_chunks = np.int(n_children/max_num_of_rows) + 1
            start = 0
            for _ in np.arange(n_chunks):
                end = min(start+max_num_of_rows, n_children)
                children_waves = utils.get_waves(
                    children_idx[start:end], time_series, wave_length)
                mu = np.mean(children_waves, axis=1, keepdims=True)
                std = np.std(children_waves, axis=1, keepdims=True)
                std[std == 0] = 1
                children_waves = (children_waves - mu)/std
                sq_dist = 2*wave_length - 2*children_waves @ root_wave
                sq_dist[sq_dist < 1e-10] = 0
                NNdist[children_idx[start:end]] = np.sqrt(sq_dist)
                start = end
        else:
            children_waves = utils.get_waves(children_idx, time_series, wave_length)
            mu, std = np.mean(children_waves, axis=1), np.std(children_waves, axis=1)
            std[std==0] = 1
            children_waves = (children_waves - mu[:,None])/std[:,None]
            sq_dist = 2*wave_length - 2*children_waves @ root_wave
            sq_dist[sq_dist < 1e-10] = 0
            NNdist[children_idx] = np.sqrt(sq_dist)

    return NNdist


def find_tau(k, subseq_len, density, NNindex, NNdist):
    """Binary-search the distance threshold that yields exactly ``k`` modes.

    Implements the binary search over ``tau`` mentioned in §3.2 of the
    paper. For a fixed kernel width, the number of modes is a non-increasing
    step function of ``tau``: large ``tau`` yields few clusters, small
    ``tau`` over-segments the tree. This function bisects ``[min_tau,
    max_tau]`` (where ``min_tau`` is the smallest non-zero NN-distance and
    ``max_tau`` is the largest) until :func:`tree2clusters` returns exactly
    ``k`` modes.

    Some time series have no ``tau`` that gives an exact ``k``-mode
    clustering (e.g. the count jumps from 1 to 3 as ``tau`` decreases).
    In that case the function gives up and returns ``None`` once the
    bisection step shrinks below ``1e-5``.

    Parameters
    ----------
    k : int
        Desired number of modes.
    subseq_len : int
        Subsequence length ``m`` (forwarded to :func:`tree2clusters`).
    density, NNindex, NNdist : numpy.ndarray
        Per-sigma columns of the QS-tuple (see :func:`tree2clusters`).

    Returns
    -------
    tau : float or None
        A distance threshold such that
        ``tree2clusters(..., tau)`` returns exactly ``k`` modes, or ``None``
        if no such threshold exists in the searched range.
    """
    max_tau = np.max(NNdist)
    min_tau = np.min(NNdist[NNdist > 0])
    step = (max_tau - min_tau)/2
    tau = min_tau + step
    prev_left = min_tau
    prev_right = max_tau
    while True:
        NNd, NNi, modes, cluster_size = tree2clusters(
            subseq_len, density, NNindex, NNdist, tau)
        n_modes = modes.size
        # For S0599_V1m_S0573_V3m_2_9000_1000.txt, from the MixedBag dataset,
        # with m=500, there is no tau that yields exactly 2 modes. It jumps
        # from 1 to 3. So, exit if step becomes too small.
        if step < 1e-5:
            return None
        if n_modes == k:
            break
        elif n_modes > k:
            step = (prev_right - tau)/2
            prev_left = tau
            tau = tau + step
            if max_tau - tau < 1e-5:
                return None
        else:
            step = (tau - prev_left)/2
            prev_right = tau
            tau = tau - step
            if tau - min_tau < 1e-5:
                return None
    return tau
