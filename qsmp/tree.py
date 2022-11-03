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
        roots[isort[idx]] = winning_roots[i]
        tree_size[i] = np.sum(num_nodes[start:end])

    isort = np.argsort(-density[winning_roots])
    winning_roots = winning_roots[isort]
    tree_size = tree_size[isort]

    return roots, winning_roots, tree_size


def drop_trivial_matches(nodes, density, gap):

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
    idx = np.zeros(n_roots*(k+1))

    for i, root in enumerate(roots):
        children = np.asarray(NNindex == root).nonzero()[0]
        children = drop_trivial_matches(children, density, gap)
        n_children = children.size - 1
        if n_children <= k:
            isort = np.argsort(NNdist[children])
            idx[i*(k+1):(i+1)*(k+1)] = np.r_[
                children[isort], np.full(k-n_children, fill_value=np.nan)]
        else:
            ind_topk = np.argpartition(NNdist[children], k+1)[:k+1]
            children = children[ind_topk]
            isort = np.argsort(NNdist[children])        
            idx[i*(k+1):(i+1)*(k+1)] = children[isort]

    return idx


def recompute_distances(NNindex, time_series, wave_length):

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