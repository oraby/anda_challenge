"""
Make a correlation matrix (neuron x neuron) for a range of time delays for
each data set
"""
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sf

def computeCorrelationMatrix(colors=ut.COLORS):
    # compute correlation matrix for range of different time delays
    f, ax = plt.subplots(1, 1)
    for i, c in enumerate(colors):
        print("preprocessing: {}".format(c))
        block = ut.load_dataset(c)
        block = ut.sort_spiketrains(block)
        r = ut.rasterize_data(block, sf=50)
        r = ut.calculate_single_trial_PSTH(r, fs=50)
        n_units = r.shape[1]
        r = r.transpose(1, 0, 2).reshape(n_units, -1)
        print("Calculating...")
        # sort the units according to electrode distance
        sig_mat = ut.get_sig_corr_mask(r, r.shape[0], 142)
        sig = sig_mat < 1 #0.05
        print("% significant: {0} for c: {1}".format(sig.sum() / sig.shape[0], c))
        cc = np.corrcoef(r)
        cc = cc[~np.eye(cc.shape[0],dtype=bool)]
        sig = sig[~np.eye(sig.shape[0],dtype=bool)]
        cc = cc[sig]

        _, d_mat = ut.array_locations(block)
        d_mat = d_mat[~np.eye(d_mat.shape[0],dtype=bool)]
        d_mat = d_mat[sig]

        cc_bin = []
        distance = []
        # bin sorted cc
        for dist in np.unique(d_mat):
            args = (d_mat == dist)
            distance.append(dist)
            cc_bin.append(np.mean(cc[args]))
        #axes[ax[i]].imshow(cc_sorted, aspect='auto', cmap='seismic')
        ax.plot(distance, sf.gaussian_filter1d(cc_bin, 3), 'o-', color=c)

    f.tight_layout()
    plt.show()


if __name__ == "__main__":
    computeCorrelationMatrix()
