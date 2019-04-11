"""
Try to see if task condition separates any trial by trial features for
some datasets and not others
"""

from dPCA import dPCA # install from https://github.com/machenslab/dPCA/tree/master/python/dPCA
                      # Tested with commit 34d5397
import utils as ut
import numpy as np
import matplotlib.pyplot as plt

def compute_dPCA(colors=ut.list_of_data_sets, fs=100, win_size=200):
    print("Loading datasets")
    data = ut.load_dataset(colors)
    for color in colors:
        print("analyzing dataset {}".format(color))
        win = win_size
        block = data[color]
        #block = ut.sort_spiketrains(block)
        R = ut.rasterize_data(block, sf=fs)

        nunits = R.shape[1]
        ntrials = R.shape[0]
        trial_type = [block.segments[i].annotations['belongs_to_trialtype']  for i in range(0, ntrials)]
        un_types = np.unique(trial_type)

        R = ut.calculate_single_trial_PSTH(R, fs=fs, win_size=win)
        s = int(np.ceil((win / 2) * (1 / fs)))
        e = int(R.shape[-1] - (1 / (1 / fs))) - s
        time = np.linspace(0, 5, R.shape[-1])
        time = time[s:e]

        for i, t in enumerate(un_types):
            trials = np.argwhere(np.array(trial_type)==t)
            r_temp = np.mean(R[trials, :, s:e], 0)
            if i == 0:
                R_psth = r_temp
            else:
                R_psth = np.concatenate((R_psth, r_temp), axis=0)

        R_psth = R_psth.transpose(1, 0, 2)

        dpca = dPCA.dPCA(labels='st', join= {'st': ['s', 'st']}, n_components=nunits, regularizer=None)

        # fit and project the mean PSTHs back out
        R_transform = dpca.fit_transform(R_psth)

        # get event times
        ev_times = ut.get_event_dict(block, fs=fs)
        events ={}
        for ev in ut.trial_events:
            events[ev] = np.argwhere(ev_times[ev][0, :]==1)[0][0] / fs

        for comp in R_transform.keys():
            # plot first four PCs for each marginalized component
            f, ax = plt.subplots(1, 1)
            f.suptitle(comp+", "+color)

            ax.set_title(str(dpca.explained_variance_ratio_[comp][0]*100))
            ax.plot(time, R_transform[comp][0, :, :].T)
            ax.legend(un_types)
            for ev in ut.trial_events:
                ax.axvline(events[ev], lw=2, color='k')
            """
            ax[1].set_title(str(dpca.explained_variance_ratio_[comp][1]*100))
            ax[1].plot(time, R_transform[comp][1, :, :].T)
            for ev in ut.trial_events:
                ax[1].axvline(events[ev], lw=2, color='k')
            """
        f.tight_layout()

        l = []
        f, ax = plt.subplots(1, 2)
        for k in R_transform.keys():
            ax[0].plot(np.cumsum(np.array(dpca.explained_variance_ratio_[k])*100))
            l.append(k)
        ax[0].legend(l)
        ax[0].set_xlabel("PC")
        ax[0].set_ylabel("% variance explained")

        l = []
        for k in R_transform.keys():
            if k != 't':
                ax[1].plot(np.cumsum(np.array(dpca.explained_variance_ratio_[k])*100))
                l.append(k)
        ax[1].legend(l)
        ax[1].set_xlabel("PC")
        ax[1].set_ylabel("% variance explained")

        f.suptitle(comp+", "+color)
        f.tight_layout()


if __name__ == "__main__":
    compute_dPCA()
