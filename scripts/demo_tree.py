#%%
import os
import numpy as np
import copy
import utils
from time import perf_counter
# %%
DPATH = "/home/cmendoza/Research/QSMP/data/Study019/preictal"
fname = 'qsmp_m350_snr0.0_2.0_4.0_8.0_10.0.npz'
fpath = os.path.join(DPATH, fname)

with np.load(fpath) as data:
    full_density = data['density']
    full_profile = data['profile']
    indices = data['indices']
# %%
snr = np.r_[0, 2, 4, 8, 10]
var_noise = 10 ** (-snr/10)
m = 350
ibw = 2
n_subseq = indices.shape[0]
#%%
inan = np.asarray(np.isnan(full_profile[:, ibw])).nonzero()[0]
is_mode = np.isinf(full_profile[:, ibw])
iinf = np.asarray(is_mode).nonzero()[0]

# %%
# If QSMP == inf, we hit a mode: the point doesn't move, its nearest neighbor
# is itself.
imax = np.argmax(full_density[:, ibw])
assert imax in iinf
indices[iinf, ibw] = iinf
full_profile[iinf, ibw] = 0

#%% Make a copy
subseq = np.array(range(n_subseq))
neighbor = copy.copy(indices[:, ibw])
profile = copy.copy(full_profile[:, ibw])
density = copy.copy(full_density[:, ibw])

#%%
dist_th = 4
max_modes = np.inf
labels = np.zeros_like(neighbor)
n_modes = 0
root = np.argmax(density)
children = np.asarray(neighbor == root).nonzero()[0]
isort = np.argsort(-density[children])
children = children[isort]
children = np.delete(children, 0)

dist2root = utils.make_dist2root('add', neighbor, profile, root)
update_children_labels = utils.make_update_children_labels(neighbor)

while n_modes < max_modes:
    current_child = children[n_modes]    
    if dist2root(current_child) < dist_th:
        children = np.delete(children, n_modes)
    else:        
        n_modes += 1
        #print(f'New mode: {current_child}')
        labels[current_child] = n_modes
        # labels = update_children_labels(current_child, labels)    
      
    # Update priority queue
    new_children = np.asarray(neighbor == current_child).nonzero()[0]
    for kid in new_children:        
        i = np.searchsorted(-density[children], -density[kid])
        assert(i >= n_modes)
        children = np.insert(children, i, kid)
    max_modes = children.size
# %% Merge modes that are close in time
t_start = perf_counter()
n_modes = children.size
new_neighbor = copy.copy(neighbor)
gap = int(np.ceil(m/2))
reduc_modes = -1 * np.ones_like(children)
done = np.zeros(children.shape, dtype=bool)
for i in np.arange(n_modes):
    if done[i]:
        continue
    
    i_local = np.asarray(np.abs(children - children[i]) < gap).nonzero()[0]
    if i_local.size == 1:
        reduc_modes[i] = children[i]
    else:
        close_modes = children[i_local]        
        best_mode = close_modes[0]
        reduc_modes[i] = best_mode        
        done[i_local[1:]] = True        
        for mode in close_modes[1:]:
            new_children = np.asarray(new_neighbor == mode).nonzero()[0]
            new_neighbor[new_children] = best_mode

reduc_modes = reduc_modes[reduc_modes >= 0]
t_stop = perf_counter()
print(f'Finished after {t_stop-t_start} seconds!')
#%%
fname = 'first56_segments_CSP1.npz'
fpath = os.path.join(DPATH, fname)
with np.load(fpath) as data:
    T = data['time_series']
    splice = data['splice']

# %%
