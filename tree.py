import numpy as np
from time import perf_counter
import utils

class Mode:
    def __init__(self, index, distfunc, parent, dist2parent) -> None:
        self.index = index
        self.distfunc = getattr(self, distfunc)
        self.parent = parent
        self.ancestor = parent
        self.dist2parent = dist2parent
        self.dist2ancestor = dist2parent
        self.n_ancestors = 1

    def mean(self):
        parent = self.parent
        mu = parent.dist2ancestor
        x = self.dist2parent
        self.dist2ancestor = mu + (x - mu)/self.n_ancestors

    def add(self):
        parent = self.parent
        self.dist2ancestor = self.dist2parent + parent.dist2ancestor

    def max(self):
        parent = self.parent
        self.dist2ancestor = max(self.dist2parent, parent.dist2ancestor)

    def update_ancestor(self):
        parent = self.parent
        self.n_ancestors = parent.n_ancestors + 1
        self.ancestor = parent.ancestor
        self.distfunc()

def make_update_modes(distfunc, profile, neighbor, density):
    def update_modes(modes, parent, update_ancestor=False):
        children = np.asarray(neighbor == parent.index).nonzero()[0]
        modes_idxs = np.array([mode.index for mode in modes])
        for kid in children:
            i = np.searchsorted(-density[modes_idxs], -density[kid])
            modes_idxs = np.insert(modes_idxs, i, kid)
            kidObj = Mode(kid, distfunc, parent, profile[kid])
            if update_ancestor:
                kidObj.update_ancestor()

            modes.insert(i, kidObj)
        return modes
    return update_modes


def reduce_close_modes(modes, m):
    """ Merge modes that are close in time

    Modes that are withing ±m/2 of each other are considered to be close in
    time, with `m` being the subsequence length. Keep the first (highest
    density) mode and discard the others.

    modes[i] is the i-th mode with highest density.    
    """
    
    idx = np.array([mode.index for mode in modes])
    gap = int(np.ceil(m/2))
    i = 0    
    while i < len(modes):

        reject = np.asarray(np.abs(idx - idx[i]) < gap).nonzero()[0]
        if reject.size > 1:

            i_close = reject[1:]            
            for j in -np.sort(-i_close):
                modes.pop(j)

            idx = np.delete(idx, i_close)       
        
        i += 1
    
    return modes


def find_modes(qsmp, maxdist, distfunc='add'):
    """ Find distant modes

    Use a priority queue based on density: highest-density points are processed
    first. A point is removed from the queue if the distance to its neighbor is
    smaller than `maxdist`. For any point, their children are added to the
    queue. Since we only have access to the distance to the parent node
    (nearest and highest-density neighbor), we use `distfunc` to aggregate the
    distance from a node to its ancestor.

    Since subsequences that contain the splice don't have a nearest neighbor (negative index), they are ignored.
    """

    profile = qsmp[0]
    neighbor = qsmp[1].astype(np.int64)
    density = qsmp[2]

    update_modes = make_update_modes(distfunc, profile, neighbor, density)

    root = np.argmax(density)
    modes = [Mode(root, distfunc, root, 0)]
    modes = update_modes(modes, modes[0], update_ancestor=False)
    modes.pop(0) # root is a child of itself, remove one

    max_modes = len(modes)
    mode_cnt = 1
    while mode_cnt < max_modes:
        current_child = modes[mode_cnt]
        if current_child.dist2ancestor < maxdist:
            modes.pop(mode_cnt)
            modes = update_modes(modes, current_child, update_ancestor=True)
        else:
            mode_cnt += 1
            modes = update_modes(modes, current_child, update_ancestor=False)

        max_modes = len(modes)

    return modes


def _update_ancestor(modes, neighbor):
    # There are distant modes with close neighbors (distance in time).
    idx_modes = [mode.index for mode in modes]
    for i in range(1, len(modes)):
        previous_ancestor = modes[i].ancestor
        while previous_ancestor not in idx_modes:
            new_ancestor = neighbor[previous_ancestor]
            previous_ancestor = new_ancestor
        modes[i].ancestor = previous_ancestor

    return modes


def find_modes_no_exclusion_zone(qsmp, maxdist, distfunc):
    """ Find distant modes

    Nearest-neighbor (NN) distance and index are computed without using a
    exclusion zone. This makes the tree highly fragmented and more expensive to
    traverse using the method in find_modes().

    Here we order the density, NN-distance (`profile`), and NN-index
    (`neighbor`) in descending order of the density. Then, we pick as modes the
    points where the profile is bigger than the distance threshold (`maxdist`).

    qsmp[0]: profile
    qsmp[1]: neighbor
    qsmp[2]: density
    """

    isort = np.argsort(-qsmp[2])
    is_distant = qsmp[0][isort] > maxdist
    idx_modes = np.asarray(is_distant).nonzero()[0]
    idx_modes = np.r_[isort[0], isort[idx_modes]]

    profile = qsmp[0]
    neighbor = qsmp[1].astype(np.int64)
    modes = []
    for idx in idx_modes:
        modes.append(Mode(idx, distfunc, neighbor[idx], profile[idx]))

    return modes


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

    n_nodes = NNindex.shape[0]
    NNdist = np.zeros(n_nodes)

    roots, cluster_size = np.unique(NNindex, return_counts=True)
    roots = roots[cluster_size > 1] # ignore orphan roots
    n_roots = roots.size
    max_chunk_size = 1024**3   # in bytes
    nbytes_per_wave = (wave_length*time_series.dtype.alignment)
    max_num_of_rows = np.int(max_chunk_size/nbytes_per_wave)
    for i in range(n_roots):
        
        root = utils.get_waves(roots[i], time_series, wave_length)
        mu, std = np.mean(root), np.std(root)
        if std == 0: std = 1
        root = (root - mu)/std
        
        node_indices = np.asarray(NNindex == roots[i]).nonzero()[0]
        n_nodes = node_indices.size
        if n_nodes > max_num_of_rows:
            n_chunks = np.int(n_nodes/max_num_of_rows) + 1
            start = 0
            for _ in np.arange(n_chunks):
                end = min(start+max_num_of_rows, n_nodes)
                waves = utils.get_waves(
                    node_indices[start:end], time_series, wave_length)
                mu, std = np.mean(waves, axis=1), np.std(waves, axis=1)
                std[std == 0] = 1
                waves = (waves - mu[:, None])/std[:, None]
                sq_dist = 2*wave_length - 2*waves @ root
                sq_dist[sq_dist < 1e-10] = 0
                NNdist[node_indices[start:end]] = np.sqrt(sq_dist)
                start = end
        else:
            waves = utils.get_waves(node_indices, time_series, wave_length)
            mu, std = np.mean(waves, axis=1), np.std(waves, axis=1)
            std[std==0] = 1
            waves = (waves - mu[:,None])/std[:,None]
            sq_dist = 2*wave_length - 2*waves @ root
            sq_dist[sq_dist < 1e-10] = 0
            NNdist[node_indices] = np.sqrt(sq_dist)
    
    return NNdist