#%%
from pathlib import Path
import pickle
import plotly.express as px
import pandas as pd

import numpy as np
import seaborn as sns
sns.set(style="white")


class Args:
    root = '..'

args = Args()
#%%
root = Path(args.root)
data_dir = root.joinpath('data/MixedBag')
results_dir = root.joinpath('results/MixedBag')
fpath = results_dir.joinpath('sucess_rate.pickle')
with fpath.open('rb') as f:
    results = pickle.load(f)
# %%
successes = results['successes']
m = results['m']
sigma = results['sigma']
T_len = results['T_len']

n_succesful_sigmas = np.sum(successes, axis=2)
best_m = np.argmax(n_succesful_sigmas, axis=1)
best_successes = successes[np.arange(100)[:, None], best_m[:, None]].squeeze()
success_rate = np.sum(best_successes, axis=0)
best_m = m[np.arange(100)[:, None], best_m[:, None]].squeeze()

n_files = len(results['file_list'])
n_m = m.shape[1]
n_sigma = sigma.size
#%%
file_id = np.repeat(np.arange(n_files), repeats=n_m*n_sigma)
mr = np.repeat(m, repeats=11, axis=1).flatten()
sigmat = np.tile(sigma, reps=100*4)
data = {
    'file': file_id,
    'm': mr,
    'sigma': sigmat,
    'success': successes.flatten()
}
fig = px.parallel_coordinates(
    data, color='success',
    )
fig.show()
# %%
data = {
    'Sigma': sigma,
    'Sucess rate': success_rate
}
df = pd.DataFrame(data)
styler = pd.io.formats.style.Styler(df.T, precision=2)
styler.hide(axis=1)
# %%
file_id = np.repeat(np.arange(n_files), repeats=n_m)
mr = m / T_len[:, None]
mr = mr.flatten()
data = {
    'file': file_id,
    'm ratio': mr,
    'n succ. sigmas': n_succesful_sigmas.flatten()
}
fig = px.parallel_coordinates(
    data, color='n succ. sigmas',
    color_continuous_scale=px.colors.sequential.Viridis)
fig.show()
# %%
n_succesful_m = np.sum(successes, axis=1)
best_sigma = np.argmax(n_succesful_m, axis=1)
best_successes = successes[
    np.arange(n_files)[:, None],
    np.arange(n_m)[None, :], best_sigma[:, None]]
success_rate = np.sum(best_successes, axis=0)
# %%
data = {
    'm_frac': np.r_[1.0, 0.75, 0.5, 0.25],
    'Success rate': success_rate
}
df = pd.DataFrame(data)
styler = pd.io.formats.style.Styler(df.T, precision=2)
styler.hide(axis=1)
# %%
fpath = results_dir.joinpath('sikmeans_sucess_rate.pickle')
with fpath.open('rb') as f:
    results = pickle.load(f)

# %%
successes = results['successes']
n_succesful_w_scale = np.sum(successes, axis=2)
best_m = np.argmax(n_succesful_sigmas, axis=1)
best_successes = successes[np.arange(100)[:, None], best_m[:, None]].squeeze()
success_rate = np.sum(best_successes, axis=0)
best_m = m[np.arange(100)[:, None], best_m[:, None]].squeeze()
data = {
    'w_scale': np.r_[1.1, 1.25, 1.5, 2.0],
    'Success rate': success_rate
}
df = pd.DataFrame(data)
styler = pd.io.formats.style.Styler(df.T, precision=2)
styler.hide(axis=1)
# %%
n_succesful_m = np.sum(successes, axis=1)
best_sigma = np.argmax(n_succesful_m, axis=1)
best_successes = successes[
    np.arange(n_files)[:, None],
    np.arange(n_m)[None, :], best_sigma[:, None]]
success_rate = np.sum(best_successes, axis=0)
data = {
    'm_frac': np.r_[1.0, 0.75, 0.5, 0.25],
    'Success rate': success_rate
}
df = pd.DataFrame(data)
styler = pd.io.formats.style.Styler(df.T, precision=2)
styler.hide(axis=1)
# %%
