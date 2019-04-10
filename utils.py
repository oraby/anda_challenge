import neo
import quantities as pq
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
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
        path = path.strip()

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

def make_lists_of_spike_trains(data_block):
    n_trials = len(data_block.segments)
    n_units = len(data_block.segments[0].spiketrains)

    output = list()
    for i in range(n_trials):
        l = list()
        for j in range(n_units):
            l.append(np.array(data_block.segments[i].spiketrains[j]))
        output.append(l)

    return output

def rasterize_data(data_block, sf = 1.e3):
    n_trials = len(data_block.segments)
    n_units = len(data_block.segments[0].spiketrains)
    time_bins = np.arange(0., 6., 1/sf)

    spike_matrix = np.zeros((n_trials, n_units, (len(time_bins))))

    for i in range(n_trials):
        for j in range(n_units):
            index = np.digitize(np.array(data_block.segments[i].spiketrains[j]), time_bins)
            spike_matrix[i,j,index] = 1

    return spike_matrix

def make_lists_of_isi(data_block):
    n_trials = len(data_block.segments)
    n_units = len(data_block.segments[0].spiketrains)

    for i in range(n_trials):
        for j in range(n_units):
            spike_times = data_block.segments[i].spiketrains[j].as_array()
            if i == j == 0:
                output = np.diff(spike_times)
            else:
                output = np.concatenate((output, np.diff(spike_times)))

    return output

def calculate_single_trial_PSTH(R, fs=None, win_size=None, window='triangle'):
    """
    Estimates single trial PSTH by convolving the binned spik train with a
    normalized window

    Input:
        R: Trials X Neuron X Time numpy array
        fs: Sampling rate (in Hz)
        win_size: Width of filter window in ms. If none, defaults to 10 times the
                bin size (1 / sampling rate)
        window: Type of filter window (triangle or boxcar). Default is triangle

    Return:
        R: Trial X Neuron X Time numpy array of firing rates (spk / s)
    """

    if fs is None:
        raise ValueError("Must specify a sampling rate")

    if win_size is None:
        win_size = 10
    else:
        samps_per_ms = int((1 / fs) * 1000)
        win_size = int(samps_per_ms * win_size)

    log.info("Using window size: {} ms".format(win_size))

    n_trials = R.shape[0]

    if window == 'triangle':
        window = ss.triang(M=win_size) / (win_size / 2)
    elif window == 'boxcar':
        window = ss.boxcar(M=win_size) / win_size
    else:
        "{} is not a valid type of window".format(window)

    for trial in range(n_trials):
        r = R[trial, :, :]
        r_psth = ss.filtfilt(window, 1, r, axis=-1)
        R[trial, :, :] = r_psth

    # convert to spike / sec
    R = R * fs

    return R

def get_event_time(neo_block, event_name=None):
    """
    Return array of time the event occured during the experiment
    """

    if event_name is None:
        raise ValueError("Must specify event name!")

    for j, trial in enumerate(range(len(neo_block.segments))):
        trial_events = neo_block.segments[trial].events[0]
        ev_names = np.array(trial_events.annotations['trial_event_labels'])
        ev_time = np.array([np.float(e) for e in trial_events.times])

        if j == 0:
            times = ev_time[ev_names==event_name]
        else:
            times = np.concatenate((times, ev_time[ev_names==event_name]))

    return times
