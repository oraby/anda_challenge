import utils as ut
import matplotlib.pyplot as plt

# load and calculate the ISIs
list_of_data_sets = ['blue', 'green', 'grey', 'orange', 'purple', 'red']
ISIs = {}
for i, color in enumerate(list_of_data_sets):
    log.info("loading and plotting ISI for {}".format(color))
    data_block = ut.load_dataset(color)
    isi = ut.make_lists_of_isi(data_block)
    ISIs[color] = isi


f, ax = plt.subplots(1, 2)
for d in list_of_data_sets:
    ax[0].hist(ISIs[d], histtype='step', density=1, cumulative=False, bins=10000, label=d)
    ax[1].hist(ISIs[d], histtype='step', density=1, cumulative=True, bins=100000, label=d)


for a in ax:
    a.semilogy()
    a.set_xlim(0, 0.05)

ax[1].legend(loc='lower right')

plt.show()
