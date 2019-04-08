import utils as ut
import matplotlib.pyplot as plt
import logging
log = logging.getLogger(__name__)

list_of_data_sets = ['blue', 'green', 'grey', 'orange', 'purple', 'red']

f, ax = plt.subplots(2, 3)
axes = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
for i, color in enumerate(list_of_data_sets):
    log.info("loading and plotting ISI for {}".format(color))
    data_block = ut.load_dataset(color)
    isi = ut.make_lists_of_isi(data_block)

    ax[axes[i]].hist(isi, bins=100, color='grey', edgecolor='k')
    ax[axes[i]].set_xlabel("Time (s)")
    ax[axes[i]].set_title(color)

f.tight_layout()

plt.show()
