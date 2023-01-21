#%%
from pathlib import Path
import pickle
import re
from time import perf_counter
from numba import cuda

import numpy as np
from qsmp import tree
import qsmp.utils.utils as utils
from qsmp.gpu_density import gpu_density
from qsmp.gpu_qsmp import gpu_qsmp

from qsmp.utils import windows

#%%
class Args:
    root = '..'
args  = Args()
#%%
root = Path(args.root)
data_dir = root.joinpath('data/MixedBag')
results_dir = root.joinpath('results/MixedBag')
results_dir.mkdir(exist_ok=True)
expr = r'^.+_(?P<split_point>\d+)_(?P<sublen>\d+).txt$'
p = re.compile(expr)
#%%
k = 2 # Number of modes
device_ids = [device.id for device in cuda.list_devices()]
#%%
sigma_scale = np.r_[
    0.1, 0.5, 1, 1.5, 2.0, 2.5, 3, 3.5, 4.0, 4.5, 5.0]
m_scale = np.r_[1, 0.75, 0.5, 0.25]
n_sigma = len(sigma_scale)
n_subseq_len = len(m_scale)
file_list = list(data_dir.iterdir())
n_files = len(file_list)
successes = np.zeros((n_files, n_subseq_len, n_sigma))
m_arr = np.zeros((n_files, n_subseq_len))
split_point_ar = np.zeros(n_files)
T_len = np.zeros(n_files)
t_start = perf_counter()
for i_file, file in enumerate(file_list):

    print(f'=== {i_file+1}/{n_files} ===')

    try:
        T = np.loadtxt(file)
    except ValueError:
        T = np.loadtxt(file, delimiter=',')
    match = p.search(file.name)
    split_point = int(match.group('split_point'))
    subseq_len = int(match.group('sublen'))

    subseq_len = (subseq_len * m_scale).astype(int)
    m_arr[i_file] = subseq_len
    T_len[i_file] = T.size
    split_point_ar[i_file] = split_point

    for i_m, m in enumerate(subseq_len):

        minfilt_size = m // 4

        N = T.size - m + 1
        # sigma = N**(-1/(4 + subseq_len)) * sigma_scale
        sigma = 1 * sigma_scale

        win_fn = windows.get_window('rect')
        win = win_fn(m, 0.5)

        params_str = str(i_file)

        T, splice, density = gpu_density(
            T, m, sigma, root, params_str,
            None, splice=None, window=win, device_id=device_ids
        )

        profile, indices = gpu_qsmp(
            T, m, minfilt_size, density, root, params_str, splice=splice, device_id=device_ids
        )

        # Find global maxima (root), and fix neighbor and profile
        profile, indices, density = utils.fix_root((profile, indices, density))

        out_file = results_dir.joinpath(file.stem + '.npz')
        with out_file.open('wb') as f:
            np.savez(
                f, density=density, profile=profile, indices=indices,
                T=T, splice=splice
            )

        for i_sigma in range(len(sigma)):

            if all(profile.T[i_sigma] == 0):
                # print(f'Failed with {file.stem}, sigma={sigma[i_sigma]:.3f}')
                continue

            tau = tree.find_tau(
                k, m, density.T[i_sigma],
                indices.T[i_sigma], profile.T[i_sigma])

            if tau is None:
                # print(
                #     f'Not enough modes in {file.stem}, sigma={sigma[i_sigma]:.3f}')
                continue

            NNd, NNi, modes_idx, cluster_size = tree.tree2clusters(
                m, density.T[i_sigma], indices.T[i_sigma],
                profile.T[i_sigma], tau
            )

            first = min(modes_idx)
            second = max(modes_idx)
            hits = 0
            if first + 0.9*m <= split_point:
                hits = hits + 1
            if second + 0.1*m >= split_point:
                hits = hits + 1
            if hits == 2:
                successes[i_file, i_m, i_sigma] = 1
            # else:
            #     print(f'Failed with {file.stem}, sigma={sigma[i_sigma]:.3f}')
        print(f'file={file.stem}, m={m}:\nsuccesses=\n{successes[i_file, i_m]}')

t_stop = perf_counter()
print(f'Finished after {t_stop-t_start} seconds!')
fpath = results_dir.joinpath('sucess_rate.pickle')
results = dict(
    file_list=file_list,
    successes=successes,
    sigma=sigma_scale,
    m=m_arr,
    T_len=T_len,
    split_point=split_point_ar
)

with fpath.open('wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
#%%
# with fpath.open('rb') as f:
#     results = pickle.load(f)

# #%%
# results
# # %%
# successes = results['successes']
# n_succesful_sigmas = np.sum(successes, axis=2)
# n_succesful_sigmas.shape
# #%%
# best_m = np.argmax(n_succesful_sigmas, axis=1)
# best_successes = successes[np.arange(100)[:, None], best_m[:, None]].squeeze()
# best_sigma = np.argmax(best_successes)
# %%
# %%
