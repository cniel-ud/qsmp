from argparse import ArgumentParser
import os
from time import perf_counter
os.unsetenv('OMP_THREAD_LIMIT')
from pathlib import Path
import numpy as np
from qsmp.shift_kmeans.shift_kmeans import shift_invariant_k_means

t_start = perf_counter()

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument('experiment', help='Experiment name')
parser.add_argument("--root", help="Path to root folder")
parser.add_argument("--centroid-len", type=int, default=512,
                    help="Centroid length")
parser.add_argument("--window-len", type=int, default=768,
                    help="Length of non-overlapping window length")
parser.add_argument('--num-clusters', type=int,
                    default=128, help='Number of clusters')

args = parser.parse_args()
win_len = args.window_len
root = Path(args.root)
data_dir = root.joinpath('data', args.experiment)
results_dir = root.joinpath('results', args.experiment)
data_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

fpath = list(data_dir.glob('*.npz'))[0]
with np.load(fpath, allow_pickle=True) as data:
    T = data['T']
    splice = data['splice']

tot_win = np.sum(np.diff(np.r_[0, splice, T.size])//win_len)
X = np.zeros((tot_win, win_len))
start_arr = np.r_[0, splice]
end_arr = np.r_[splice, T.size]
start_x = 0
for start, end in zip(start_arr, end_arr):
    segment = T[start:end]
    n_win = segment.size//win_len
    i_win = np.arange(0, n_win*win_len, win_len)
    i_win = i_win[:, None] + np.arange(win_len)[None, :]
    X[start_x:start_x+n_win] = segment[i_win]
    start_x = start_x + n_win

k, P = args.num_clusters, args.centroid_len
metric, init = 'cosine', 'random'
n_runs, rng = 30, 13
centroids, labels, shifts, distances, _, _ = shift_invariant_k_means(
    X, k, P, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)


out_file = f'sikmeans_k-{k}_P-{P}_wlen-{win_len}.npz'
out_file = results_dir.joinpath(out_file)
with out_file.open('wb') as f:
    np.savez(f, centroids=centroids, labels=labels,
             shifts=shifts, distances=distances)

t_stop = perf_counter()
print(f'Finished after {t_stop-t_start} seconds!')
