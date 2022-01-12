import numpy as np

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


def reduce_close_modes(modes, neighbor, m):
    """ Merge modes that are close in time

    Modes that are withing ±m/2 of each other are considered to be close in
    time, with `m` being the subsequence length. Keep the first (highest
    density) mode and discard the others. The nodes whose parent was one of the discarded modes now have the winnig mode as their parent.

    modes[i] is the i-th mode with highest density.
    neighbor[i] is the parent of node (subsquence at time) i.
    """

    modes = np.asarray(modes)
    idx = np.array([mode.index for mode in modes])
    gap = int(np.ceil(m/2))
    i = 0
    while i < modes.size:

        reject = np.asarray(np.abs(idx - idx[i]) < gap).nonzero()[0]
        if reject.size > 1:

            i_best = reject[0]
            i_close = reject[1:]
            best_mode = idx[i_best]
            close_modes = idx[i_close]

            for mode in close_modes:
                children = np.asarray(neighbor == mode).nonzero()[0]
                neighbor[children] = best_mode

            modes = np.delete(modes, i_close)
            idx = np.delete(idx, i_close)

        i += 1

    return modes, neighbor


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
