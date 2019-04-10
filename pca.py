from sklearn.decomposition import PCA
import utils as ut
import numpy as np
import matplotlib.pyplot as plt

list_of_data_sets = ['blue', 'green', 'grey', 'orange', 'purple', 'red']

var_explained = {}

for c in list_of_data_sets:
    print("preprocessing {}".format(c))
    block = ut.load_dataset(c)
    R = ut.rasterize_data(block, sf=100)
    R = ut.calculate_single_trial_PSTH(R, fs=100)
    n_units = R.shape[1]
    R = R.transpose(1, 0, 2).reshape(n_units, -1)
    pca = PCA(n_components=n_units)
    pca.fit(R.T)

    var_explained[c] = 100 * np.cumsum(pca.explained_variance_) / np.sum(pca.explained_variance_)

# make the plot of variance explained_variance_

f, ax = plt.subplots(1, 1)
for c in list_of_data_sets:
    ax.plot(var_explained[c], '.-', color=c)

ax.set_xlabel("Number of PCs")
ax.set_ylabel("% Variance explained")
ax.legend(list_of_data_sets)
asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
ax.set_aspect(asp)

plt.show()
