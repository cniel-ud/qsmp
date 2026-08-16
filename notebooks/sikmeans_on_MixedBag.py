#%%
import pickle
import re
from qsmp.shift_kmeans.shift_kmeans import shift_invariant_k_means
import numpy as np
from pathlib import Path
from time import perf_counter

root = Path(__file__).resolve().parent.parent
data_dir = root.joinpath('data/MixedBag')
results_dir = root.joinpath('results/MixedBag')
results_dir.mkdir(parents=True, exist_ok=True)
expr = r'^.+_(?P<split_point>\d+)_(?P<sublen>\d+).txt$'
p = re.compile(expr)
#%%
k = 2  # Number of clusters
metric, init = 'cosine', 'random'
n_runs, rng = 10, 13
w_scale = np.r_[1.1, 1.25, 1.5, 2.0]
n_w_scale = len(w_scale)
file_list = list(data_dir.iterdir())
n_files = len(file_list)
# Per (file, w): success indicator AND the *unsupervised* selection objective
# (mean per-window distortion = inertia / n_windows). w is chosen per file by
# MIN distortion -- ground-truth-free -- exactly as in the recovery experiment.
successes = np.zeros((n_files, n_w_scale))
distortion = np.full((n_files, n_w_scale), np.nan)
m_arr = np.zeros(n_files)
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
    m = int(match.group('sublen'))          # per-series default, eta=1

    m_arr[i_file] = m
    T_len[i_file] = T.size
    split_point_ar[i_file] = split_point

    for i_w, w in enumerate(w_scale):

        win_len = int(m * w)
        tot_win = np.sum(T.size//win_len)
        ind = np.arange(tot_win) * win_len
        ind = ind[:, None] + np.arange(win_len)[None, :]
        X = T[ind]

        centroids, labels, shifts, distances, inertia, _ = \
            shift_invariant_k_means(
                X, k, m, metric=metric, init=init,
                n_init=n_runs, rng=rng,  verbose=True
            )

        # Unsupervised selection objective: mean per-window distortion,
        # comparable across window counts (cosine metric, z-normalised windows).
        distortion[i_file, i_w] = float(inertia) / X.shape[0]

        first_cluster = (labels == 0).nonzero()[0]
        i_min = np.argmin(distances[first_cluster])
        i_min = first_cluster[i_min]
        first = i_min*win_len + shifts[i_min]

        second_cluster = (labels == 1).nonzero()[0]
        i_min = np.argmin(distances[second_cluster])
        i_min = second_cluster[i_min]
        second = i_min*win_len + shifts[i_min]

        hits = 0
        if first + 0.9*m <= split_point:
            hits = hits + 1
        if second + 0.1*m >= split_point:
            hits = hits + 1
        if hits == 2:
            successes[i_file, i_w] = 1

t_stop = perf_counter()
print(f'Finished after {t_stop-t_start} seconds!')

# --- Unsupervised window-scale selection + success rate --------------------- #
# For each file pick the w with the smallest distortion (ground-truth free),
# then read off whether that configuration was a success.
sel_success = np.zeros(n_files)
for i_file in range(n_files):
    dist = distortion[i_file]
    if np.all(np.isnan(dist)):
        continue
    i_sel = int(np.nanargmin(dist))
    sel_success[i_file] = successes[i_file, i_sel]
print(f'sikmeans unsupervised-w success rate: '
      f'{100 * sel_success.mean():.1f}% ({int(sel_success.sum())}/{n_files})')

fpath = results_dir.joinpath('sikmeans_sucess_rate.pickle')
results = dict(
    file_list=file_list,
    successes=successes,
    distortion=distortion,
    sel_success=sel_success,
    w_scale=w_scale,
    m=m_arr,
    T_len=T_len,
    split_point=split_point_ar
)

with fpath.open('wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
