"""
Plot the population PSTH for a couple of different bin sizes to differentiate
between the dithered data and real data
"""

import utils as ut
import numpy as np
import matplotlib.pyplot as plt

colors = ['red', 'green']

data = ut.load_dataset(colors)

for fs in [50, 5]:
    R_red = ut.rasterize_data(data['red'], sf=fs)
    R_red = R_red.mean(0).mean(0)

    R_green = ut.rasterize_data(data['green'], sf=fs)
    R_green = R_green.mean(0).mean(0)
    time = np.linspace(0, 5, R_red.shape[0])

    s = int(np.ceil(fs * (10 / 1000)))
    e = 4 * fs # int(np.ceil(fs * (1000 / 1000)))
    R_red = R_red[s:e]
    R_green = R_green[s:e]
    time = time[s:e]

    f, ax = plt.subplots(1, 1)

    ax.plot(time, R_red * fs, color='red')
    ax.plot(time, R_green * fs, color='green')
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_xlabel("Time (s)")

    ax.set_title("Bin size: {} ms".format((1/fs) * 1000))

    f.tight_layout()

abs_diff = []
FS = np.arange(10, 210, 10)
bin_size = (1 / FS) * 1000
for fs in FS:
    R_red = ut.rasterize_data(data['red'], sf=fs)
    R_red = R_red.mean(0).mean(0)

    R_green = ut.rasterize_data(data['green'], sf=fs)
    R_green = R_green.mean(0).mean(0)

    s = int(np.ceil(fs * (10 / 1000)))
    e = 4 * fs # int(np.ceil(fs * (1000 / 1000)))
    R_red = R_red[s:e]
    R_green = R_green[s:e]

    abs_diff.append(np.sum(R_red - R_green))

f, ax = plt.subplots(1, 1)
ax.plot(bin_size, abs_diff, 'o-', color='k')
ax.set_xlabel("Bin size (ms)")
ax.set_ylabel("Absolute difference (red minus green)")
