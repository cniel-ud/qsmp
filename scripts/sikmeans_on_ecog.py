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
parser.add_argument("-d", "--data-path", dest="dpath",
                    help="Path to folder with time series")
parser.add_argument("--centroid-len", type=int, default=512,
                    help="Centroid length")
parser.add_argument("--window-len", type=int, default=768,
                    help="Length of non-overlapping window length")
parser.add_argument('--num-clusters', type=int,
                    default=128, help='Number of clusters')

args = parser.parse_args()
win_len = args.window_len
dpath = Path(args.dpath)

in_file = dpath.joinpath('qsmp_T_splice.npz')
with np.load(in_file) as data:
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
n_runs, rng = 3, 13
centroids, labels, shifts, distances, _, _ = shift_invariant_k_means(
    X, k, P, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)


out_file = f'sikmeans_k-{k}_P-{P}_wlen-{win_len}.npz'
out_file = dpath.joinpath(out_file)
with out_file.open('wb') as f:
    np.savez(f, centroids=centroids, labels=labels,
             shifts=shifts, distances=distances)

t_stop = perf_counter()
print(f'Finished after {t_stop-t_start} seconds!')
