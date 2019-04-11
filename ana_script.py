import pickle
import os
import utils as ut
from analysis_functions import compute_isi_fits
from correlation_matrix import plotCorrelationMatrix, plotCorrelationByDistance
from complexity import complexity
from trff import ff_wrapper


all_data = ut._all_data
#Create a folder for temporary data
ut.createIfNotExistDirs([ut.DUMP_DIR, ut.PLOTS_DIR])

for color in ut.list_of_data_sets:
    data_path = ut.DUMP_DIR + "/" + color + ".pickle"
    if os.path.exists(data_path):
        print("Loading dumped data from", data_path)
        with open(data_path, 'rb') as f:
            all_data[color] = pickle.load(f)
    else:
        print('Processing '+str(color)+' data set')
        #Read in raw data
        data_block = ut.load_dataset(color, path=None)
        #ut.load_dataset() creates an entry in all_data for the color
        #Assigning neo_block here is redundant but left for readability
        all_data[color]['neo_block'] = data_block # this original data block from neo
        all_data[color]['spike_raster'] = ut.rasterize_data(data_block, sf = 1.e3)  # spikes in 1s and 0s/
        all_data[color]['spike_trains'] = ut.make_lists_of_spike_trains(data_block) # spikes as time (in seconds?)
        all_data[color]['isi'] = ut.make_lists_of_isi(data_block)

        print("Dumping data to disk for future quick loading times")
        with open(data_path, 'wb') as f:
            pickle.dump(all_data[color], f)

fns = [
  (compute_isi_fits, [all_data], {}),
  (plotCorrelationMatrix, [ut.list_of_data_sets], {'fs': 50, 'win_size': 0}),
  (plotCorrelationMatrix, [ut.list_of_data_sets], {'fs': 500, 'win_size': 0}),
  (plotCorrelationByDistance, [ut.list_of_data_sets], {'fs': 50, 'win_size': 0}),
  (plotCorrelationByDistance, [ut.list_of_data_sets], {'fs': 500, 'win_size': 0}),
  (ff_wrapper, [ut.list_of_data_sets,150, None],{}),
]
try: # Not all of us has dPCA installed, it needs manual installation
    from dPCA_script import compute_dPCA
except ImportError:
    pass
else:
    print("Imported dPCA")
    fns.append(
        (compute_dPCA, [ut.list_of_data_sets], {'fs': 100, 'win_size': 200}))

if __name__ == "__main__":
    for fn, args, kargs in fns:
        print("Running", fn.__name__)
        fn(*args, **kargs)
