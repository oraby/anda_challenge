"""
Try to see if task condition separates any trial by trial features for
some datasets and not others
"""

from dPCA import dPCA
import utils as ut
import numpy as np
import matplotlib.pyplot as plt

list_of_data_sets = ['blue', 'green', 'grey', 'orange', 'purple', 'red']

data = ut.load_dataset(list_of_data_sets)

for color in list_of_data_sets:
    print("analyzing dataset {}".format(color))
    win = 500
    fs = 50
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
    event_list = ['TS-ON', 'CUE-ON', 'GO-ON', 'SR', 'RW-ON']
    for ev in event_list:
        events[ev] = np.argwhere(ev_times[ev][0, :]==1)[0][0] / fs

    for comp in R_transform.keys():
        # plot first four PCs for each marginalized component
        f, ax = plt.subplots(2, 2, figsize=(8, 5))
        f.suptitle(comp+", "+color)

        ax[0, 0].set_title(str(dpca.explained_variance_ratio_[comp][0]*100))
        ax[0, 0].plot(time, R_transform[comp][0, :, :].T)
        ax[0, 0].legend(un_types)
        for ev in event_list:
            ax[0, 0].axvline(events[ev], lw=2, color='k')

        ax[0, 1].set_title(str(dpca.explained_variance_ratio_[comp][1]*100))
        ax[0, 1].plot(time, R_transform[comp][1, :, :].T)
        for ev in event_list:
            ax[0, 1].axvline(events[ev], lw=2, color='k')

        ax[1, 0].set_title(str(dpca.explained_variance_ratio_[comp][2]*100))
        ax[1, 0].plot(time, R_transform[comp][2, :, :].T)
        for ev in event_list:
            ax[1, 0].axvline(events[ev], lw=2, color='k')

        ax[1, 1].set_title(str(dpca.explained_variance_ratio_[comp][3]*100))
        ax[1, 1].plot(time, R_transform[comp][3, :, :].T)
        for ev in event_list:
            ax[1, 1].axvline(events[ev], lw=2, color='k')

    f.tight_layout()

plt.show()
