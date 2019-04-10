import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from scipy.stats import gamma

import neo
import utils as ut

def spike_counts(all_data, color):
	ntrials = all_data[color]['spike_raster'].shape[0]
	nunits = all_data[color]['spike_raster'].shape[1]
	spikes_across_time =  np.sum(all_data[color]['spike_raster'],axis=2)
	spikes_across_units = np.sum(all_data[color]['spike_raster'],axis=1)/nunits
	return spikes_across_units, spikes_across_time
