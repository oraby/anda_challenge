"""
Make a correlation matrix (neuron x neuron) for a range of time delays for
each data set
"""
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sf

list_of_data_sets = ['blue', 'green', 'grey', 'orange', 'purple', 'red']

# load datasets
R = {}
B = {}

for c in list_of_data_sets:
    print("preprocessing: {}".format(c))
    block = ut.load_dataset(c)
    B[c] = block
    r = ut.rasterize_data(block, sf=100)
    #r = ut.calculate_single_trial_PSTH(r, fs=100)
    n_units = r.shape[1]
    r = r.transpose(1, 0, 2).reshape(n_units, -1)

    R[c] = r

# compute correlation matrix for range of different time delays
f, axes = plt.subplots(2, 3)
ax = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
for i, c in enumerate(list_of_data_sets):
    # sort the units according to electrode distance
    _, d_mat = ut.array_locations(B[c])
    d_mat = d_mat[~np.eye(d_mat.shape[0],dtype=bool)]
    #d_mat = d_mat[0, :]
    d_flat = d_mat.flatten()
    sorted_args = np.argsort(d_flat)
    cc = np.corrcoef(R[c])
    cc = cc[~np.eye(cc.shape[0],dtype=bool)]
    cc_sorted = cc.flatten()[sorted_args]
    cc_bin = []
    # bin sorted cc
    for dist in np.unique(d_mat):
        args = (d_mat == dist)
        cc_bin.append(np.mean(cc_sorted[args]))
    #axes[ax[i]].imshow(cc_sorted, aspect='auto', cmap='seismic')
    axes[ax[i]].plot(cc_bin)
    axes[ax[i]].set_title(c)
f.tight_layout()

plt.show()
