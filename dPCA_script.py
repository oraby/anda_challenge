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

        time = np.linspace(0, 5, R.shape[-1])
        if win_size != 0:
            R = ut.calculate_single_trial_PSTH(R, fs=fs, win_size=win)
            win = int(np.ceil(fs * (win_size/1000)))
            s = int(np.ceil((win / 2) * (1 / fs)))
            e = int(R.shape[-1] - (1 / (1 / fs))) - s
        else:
            s = 0
            e = int(R.shape[-1] - (1 / (1 / fs)))

        time = time[s:e]

        for i, t in enumerate(un_types):
            trials = np.argwhere(np.array(trial_type)==t)
            r_temp = np.mean(R[trials, :, s:e], 0)
            if i == 0:
                R_psth = r_temp
            else:
                R_psth = np.concatenate((R_psth, r_temp), axis=0)

        R_psth = R_psth.transpose(1, 0, 2)

        dpca = dPCA.dPCA(labels='st', join= {'st': ['s', 'st']}, n_components=nunits-50, regularizer=None)

        # fit and project the mean PSTHs back out
        R_transform = dpca.fit_transform(R_psth)

        # get event times
        ev_times = ut.get_event_dict(block, fs=fs)
        events ={}
        for ev in ut.trial_events:
            events[ev] = np.argwhere(ev_times[ev][0, :]==1)[0][0] / fs


        for i, comp in enumerate(R_transform.keys()):
            # plot first PC for each marginalized component
            f, ax = plt.subplots(4, 1)
            for j in range(0, 4):
                ax[j].set_title(comp + ", var explained: " + str(round(dpca.explained_variance_ratio_[comp][j]*100, 2)))
                ax[j].plot(time, R_transform[comp][j, :, :].T)
                ax[j].legend(un_types)
                for ev in ut.trial_events:
                    ax[j].axvline(events[ev], lw=2, color='k')
            f.suptitle(color)
            f.tight_layout()

        f, ax = plt.subplots(1, 1)
        c = ['black', 'gold']
        for i, k in enumerate(R_transform.keys()):
            ax.plot(np.cumsum(np.array(dpca.explained_variance_ratio_[k])*100), 'o-', color=c[i])
        ax.legend(["Condition independent", "Condition dependent"])
        ax.set_xlabel("PC")
        ax.set_ylabel("% variance explained")
        ax.set_ylim((0, 100))
        ax.axhline(0, linestyle='--', color='k')
        ax.set_title(color)
        f.tight_layout()

    return dpca


if __name__ == "__main__":
    dpca = compute_dPCA(colors=['grey'], fs=4, win_size=0)
