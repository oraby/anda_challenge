"""
Plot single trial raster for each data set and sort it based on the unit
correlation with behavioral events.
"""

import utils as ut
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.filters as sf
import copy

list_of_data_sets = ['blue', 'green', 'grey', 'orange', 'purple', 'red']
trial = 0
fs = 500
event = 'GO-ON'
f, ax = plt.subplots(2, 3)
axes = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
for i, c in enumerate(list_of_data_sets):
    print("loading / analyzing {}".format(c))
    block = ut.load_dataset(c)
    block = ut.sort_spiketrains(block)

    # get binned spikes
    R = ut.rasterize_data(block, sf=fs)
    R_raster = copy.deepcopy(R)
    # convert to firing rates
    R = ut.calculate_single_trial_PSTH(R, fs=fs, win_size=100)
    n_units = R.shape[1]
    R_flat = R.transpose(1, 0, 2).reshape(n_units, -1)

    # get event array and smooth
    ev = ut.get_event_dict(block, fs=fs)[event]
    ev_time = np.where(ev[trial, :])[0][0]
    ev_flat = sf.gaussian_filter1d(ev.reshape(1, -1), 10)
    cc = np.zeros(R_flat.shape[0])
    for u in range(n_units):
        cc[u] = np.corrcoef(R_flat[u, :], ev_flat[0, :])[0, 1]

    sorted = np.argsort(cc)

    units, time = np.where(R_raster[trial, sorted, :])

    ax[axes[i]].plot(time / fs, units, '.', color='r', markersize=1)
    ax[axes[i]].axvline(ev_time / fs, color='k', lw=3)

    ax[axes[i]].set_title(c)

f.tight_layout()

plt.show()
