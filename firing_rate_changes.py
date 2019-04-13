import utils as ut
import numpy as np
import matplotlib.pyplot as plt

colors = ['red', 'grey']
data = ut.load_dataset(colors)

fs = 100
window_size = 100
end = (5 * fs) - (fs * 1)

R_red = ut.rasterize_data(data['red'], sf=fs)
R_grey = ut.rasterize_data(data['grey'], sf=fs)

R_red = ut.calculate_single_trial_PSTH(R_red, fs=fs, win_size=window_size)
R_grey = ut.calculate_single_trial_PSTH(R_grey, fs=fs, win_size=window_size)

# compute means for each trial type separately
ntrials = R_red.shape[0]
trial_type = [data['red'].segments[i].annotations['belongs_to_trialtype']  for i in range(0, ntrials)]
un_types = np.unique(trial_type)

for i, t in enumerate(un_types):
    idx = np.argwhere(np.array(trial_type)==t)
    red_psth = np.mean(R_red[idx, :, :end], 0).squeeze()
    grey_psth = np.mean(R_grey[idx, :, :end], 0).squeeze()

    f, ax = plt.subplots(1, 1)
    ax.set_title("{0} minus {1}, trialtype: {2}".format(colors[0], colors[1], t))
    vmax = np.max(abs(grey_psth - red_psth))
    im = ax.imshow(grey_psth - red_psth, aspect='auto', cmap='seismic', vmin=-vmax, vmax=vmax)
    plt.colorbar(im)


plt.show()
