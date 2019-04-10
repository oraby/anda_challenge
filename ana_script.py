import numpy as np
import matplotlib.pyplot as plt
import neo
import utils as ut

list_of_data_sets = ['blue', 'green', 'grey', 'orange', 'purple', 'red']

all_data = dict()

for color in list_of_data_sets:
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