import numpy as np
import matplotlib.pyplot as plt
import neo
import quantities as pq

colors = ['red', 'blue', 'grey', 'green', 'orange', 'purple']

f = open("datapath.txt", "r")
        path = f.read()
        path = path.strip()

color = colors[0]

# Load the data
block = np.load(path + '{}.npy'.format(color)).item()

# Get all the idxs of the trials and select one
seg_idx = block.annotations['all_trial_ids'][0]

# Get the list of spiketrains of the selected trial
sts = block.filter(
    targdict={'trial_id': seg_idx}, objects=neo.Segment)[0].spiketrains

# Get the list of events of the trial
events = block.filter(
    targdict={'trial_id': seg_idx}, objects=neo.Segment)[0].events[0]
# List of events labels that we want to plot
eve_name_to_plot = ['TS-ON', 'CUE-ON', 'GO-ON', 'SR', 'RW-ON']

# Get the trial type
trial_type = block.filter(
    targdict={'trial_id': seg_idx}, objects=neo.Segment)[0].annotations[
    'belongs_to_trialtype']