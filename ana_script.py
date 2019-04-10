import pickle
import os
import utils as ut
from analysis_functions import compute_isi_fits

list_of_data_sets = ['blue', 'green', 'grey', 'orange', 'purple', 'red']

all_data = dict()

#Create a folder for temporary data
DUMP_DIR="dumps"
PLOTS_DIR="plots"
for dir in [DUMP_DIR, PLOTS_DIR]:
    if not os.path.exists(dir):
        print("Creating " + dir + " directory")
        os.mkdir(dir)

for color in list_of_data_sets:
    data_path = DUMP_DIR + "/" + color + ".pickle"
    if os.path.exists(data_path):
        print("Loading dumped data from", data_path)
        with open(data_path, 'rb') as f:
            all_data[color] = pickle.load(f)
    else:
        print('Processing '+str(color)+' data set')

        #Read in raw data
        individual_data_set = dict()
        data_block = ut.load_dataset(color, path=None)

        #Make rasta data
        individual_data_set['neo_block'] = data_block # this original data block from neo
        individual_data_set['spike_raster'] = ut.rasterize_data(data_block, sf = 1.e3)  # spikes in 1s and 0s/
        individual_data_set['spike_trains'] = ut.make_lists_of_spike_trains(data_block) # spikes as time (in seconds?)
        individual_data_set['isi'] = ut.make_lists_of_isi(data_block)

        all_data[color] = individual_data_set

        print("Dumping data to disk for future quick loading times")
        with open(data_path, 'wb') as f:
            pickle.dump(individual_data_set, f)

fns = [
  (compute_isi_fits, [all_data], {}),
]

if __name__ == "__main__":
    for fn, args, kargs in fns:
        fn(*args, **kargs)
