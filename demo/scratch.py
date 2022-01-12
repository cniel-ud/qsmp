#%%
import pltaux
#%%
idx = 0
folder = "/home/cmendoza/Research/QSMP/data/Study019/preictal"


grid = pltaux.built_grid_fixed_sigma(
    idx, folder, maxdist, distfunc, ts, m, max_modes)
