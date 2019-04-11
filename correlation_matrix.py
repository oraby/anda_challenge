"""
Make a correlation matrix (neuron x neuron) for a range of time delays for
each data set
"""
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sf
import scipy.cluster.hierarchy as sch

def computeCorrelationMatrix(colors=ut.COLORS, fs=50, win_size=None, sig=False):
    # compute correlation matrix for range of different time delays
    f, ax = plt.subplots(1, 1)
    corr_mat = {}
    for i, c in enumerate(colors):
        print("preprocessing: {}".format(c))
        block = ut.load_dataset(c)
        r = ut.rasterize_data(block, sf=fs)
        if win_size == 0:
            pass
        else:
            r = ut.calculate_single_trial_PSTH(r, fs=fs, win_size=win_size)
        n_units = r.shape[1]
        r = r.transpose(1, 0, 2).reshape(n_units, -1)
        print("Calculating...")
        # sort the units according to electrode distance
        if sig == True:
            sig_mat = ut.get_sig_corr_mask(r, r.shape[0], 142)
            sig = sig_mat < 0.05
            print("% significant: {0} for c: {1}".format(sig.sum() / sig.shape[0], c))
        cc = np.corrcoef(r)
        corr_mat[c] = cc.copy()
        cc = cc[~np.eye(cc.shape[0],dtype=bool)]
        if sig == True:
            sig = sig[~np.eye(sig.shape[0],dtype=bool)]
            cc = cc[sig]

        _, d_mat = ut.array_locations(block)
        d_mat = d_mat[~np.eye(d_mat.shape[0],dtype=bool)]
        if sig == True:
            d_mat = d_mat[sig]

        cc_bin = []
        cc_sem = []
        distance = []
        # bin sorted cc
        for dist in np.unique(d_mat):
            args = (d_mat == dist)
            distance.append(dist)
            cc_bin.append(np.mean(cc[args]))
            cc_sem.append(np.std(cc[args]) / np.sqrt(len(cc[args])))
        #axes[ax[i]].imshow(cc_sorted, aspect='auto', cmap='seismic')
        ax.plot(distance, cc_bin, 'o-', color=c)
        ax.errorbar(distance, cc_bin, cc_sem, color=c)
        ax.set_xlabel("Electrode distance")
        ax.set_ylabel("Pairwise correlation")

    f.tight_layout()


    return corr_mat


if __name__ == "__main__":
    corr_mat = computeCorrelationMatrix(fs=100, win_size=100)


    # plot sorted correlation matrix to look for higher order correlations
    axes = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
    colors = ut.COLORS[::-1]
    vmin = 0
    vmax = 0
    for c in colors:
        min = np.min(corr_mat[c][~np.eye(corr_mat[c].shape[0],dtype=bool)])
        max = np.max(corr_mat[c][~np.eye(corr_mat[c].shape[0],dtype=bool)])

        if min < vmin:
            vmin = min
        if max > vmax:
            vmax = max

    f1, ax1 = plt.subplots(2, 3)
    for i, a in enumerate(axes):
        cm = corr_mat[colors[i]]
        Y = sch.linkage((1 - cm).copy(), method='centroid')
        Z = sch.dendrogram(Y, orientation='right', no_plot=True)
        index = Z['leaves']
        cm = cm[:, index][index, :]
        # fill diagonal with 0s
        cm[np.diag_indices(cm.shape[0])] = 0
        im = ax1[a].imshow(cm, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax)
        ax1[a].set_title(colors[i])
    f1.colorbar(im)
    f1.tight_layout()
    plt.show()
