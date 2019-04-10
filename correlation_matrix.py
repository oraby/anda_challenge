"""
Make a correlation matrix (neuron x neuron) for a range of time delays for
each data set
"""
import utils as ut
import numpy as np
import matplotlib.pyplot as plt

list_of_data_sets = ['blue', 'green', 'grey', 'orange', 'purple', 'red']

# load datasets
R = {}

for c in list_of_data_sets:
    block = ut.load_dataset(c)
    r = ut.rasterize_data(block, sf=100)
    r = ut.calculate_single_trial_PSTH(r, fs=100)
    n_units = r.shape[1]
    r = r.transpose(1, 0, 2).reshape(n_units, -1)

    R[c] = r

# compute correlation matrix for range of different time delays
f, axes = plt.subplots(2, 3)
ax = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
for c in list_of_data_sets:
    # sort the units according to electrode distance
    R_sorted = R[0]
    cc = np.corrcoef(R_sorted)
    axes[ax[i]].imshow(cc, aspect='auto', cmap='seismic')
    axes[ax[i]].set_title(c)
f.tight_layout()
