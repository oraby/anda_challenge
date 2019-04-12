

colors = ['blue', 'green', 'grey', 'orange', 'purple', 'red']

color = colors[0]


ntrials = all_data[color]['spike_raster'].shape[0]
nunits = all_data[color]['spike_raster'].shape[1]


def spikes_over_time(spike_raster):
	return np.sum(spike_raster,axis=2)

def spikes_across_units(spike_raster):
	return np.sum(spike_raster,axis=1)/nunits


print("Condition: "+color+"; number of spikes over time: ")

