import neo
import quantities as pq
import numpy as np
import matplotlib.pyplot as plt
import logging
log = logging.getLogger(__name__)


def load_dataset(color, path=None):
    """
    color: string, name of data set
    path: path to the data

    returns a neo block
    """

    if path is None:
        log.info("loading apth from datapath.txt")
        f = open("datapath.txt", "r")
        path = f.read()

    return np.load(path + '{}.npy'.format(color)).item()

def plot_trial_raster(block, trial_num=0):
    """
    Given a neo block, and trial number, plot the raster plot for all recorded
    units on this trial.

    Returns plot axis
    """

    log.info("Plotting raster for trial {}".format(trial_num))

    # Get all the idxs of the trials and select one
    seg_idx = block.annotations['all_trial_ids'][trial_num]

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

    # Raster plot
    f, ax = plt.subplots(1, 1)
    markersize = 1

    # Plotting the spikes
    for idx, st in enumerate(sts):
        plt.plot(st, [idx] * len(st), 'ko', markersize=markersize)

    # Plotting the events
    for ev_idx, ev in enumerate(events):
        # Getting the labe of the single event
        ev_label = events.annotations['trial_event_labels'][ev_idx]
        if ev_label in eve_name_to_plot:
            print(ev_label)
            ax.vlines(ev.rescale(pq.s), 0, len(sts))
            ax.text(ev.rescale(pq.s) - 0.01 * pq.s, len(sts) + 2, ev_label)

    # Additional plot parameter
    ax.set_xlim([sts[0].t_start - 0.2 * pq.s, sts[0].t_stop + 0.2 * pq.s])
    ax.set_ylim([-1, len(sts) + 5])
    ax.set_ylabel('Neu idx')
    ax.set_xlabel('Time ({})'.format(sts[0].units))
