import os
import neo
import quantities as pq
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import cross_correlation_histogram
import elephant.unitary_event_analysis as ue
import copy as cp
import logging
import misc

log = logging.getLogger(__name__)

list_of_data_sets = ['blue', 'green', 'grey', 'orange', 'purple', 'red']
COLORS = list_of_data_sets # Creating an alias for clarity
DUMP_DIR="dumps"
PLOTS_DIR="plots"

_all_data = {}
def load_dataset(colors=list_of_data_sets, path=None):
    """
    color: string, name of data set
    path: path to the data

    returns a neo block
    """
    # Colors was initially used for a single color, convert to list to avoid
    # breaking code
    if type(colors) == str:
        return_single = True
        log.info('Use `load_dataset(["' + colors +'"])` instead of' + \
                 '`load_dataset("' + colors +'")`')
        colors = [colors]
    else:
        return_single = False

    return_list = []
    for color in colors:
        if color in _all_data:
            return_list.append(_all_data[color]['neo_block'])
        else:
            if path is None:
                log.info("loading path from datapath.txt")
                f = open("datapath.txt", "r")
                path = f.read()
                path = path.strip()
            neo_block = np.load(path + '{}.npy'.format(color)).item()
            _all_data[color] = {'neo_block': neo_block}
            return_list.append(neo_block)

    if return_single: # if the caller passed a non-list then return such
        return return_list[0]
    else:
        return return_list

def createIfNotExistDirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            print("Creating " + dir + " directory")
            os.mkdir(dir)


def plot_trial_raster(block, trial_num=0, ax=None):
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

    # Get the list of evens of the trial
    events = block.filter(
        targdict={'trial_id': seg_idx}, objects=neo.Segment)[0].events[0]
    # List of events labels that we want to plot
    eve_name_to_plot = ['TS-ON', 'CUE-ON', 'GO-ON', 'SR', 'RW-ON']

    # Get the trial type
    trial_type = block.filter(
        targdict={'trial_id': seg_idx}, objects=neo.Segment)[0].annotations[
        'belongs_to_trialtype']

    # Raster plot
    if ax is None:
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
    time_bins = np.arange(0., 5., 1/sf)

    spike_matrix = np.zeros((n_trials, n_units, (len(time_bins))))

    for i in range(n_trials):
        for j in range(n_units):
            index = np.digitize(np.array(data_block.segments[i].spiketrains[j]), time_bins)
            if len(index) > 0:
                if index[-1] >= spike_matrix.shape[-1]:
                    index = index[:-1]
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
        samps_per_ms = int(fs * 1000)
        win_size = int(np.ceil((1 / samps_per_ms) * win_size))

    if win_size == 1:
        win_size += 1

    print("window size {}".format(win_size))
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

def pairwise_cch(neo_block, unit1=0, unit2=1, binsize=5):
    """
    Return cross correlation histogram between unit1 and unit2 in the neoblock.
    Computes this over each trial independently, then takes the mean over trials
    """

    for i, trial in enumerate(range(len(neo_block.segments))):
        st1 = neo_block.segments[trial].spiketrains[unit1]
        st2 = neo_block.segments[trial].spiketrains[unit2]
        bst1 = BinnedSpikeTrain(st1, binsize=binsize*pq.ms)
        bst2 = BinnedSpikeTrain(st2, binsize=binsize*pq.ms)

        cch = cross_correlation_histogram(bst1, bst2, border_correction=True)
        times = cch[1]
        cch = cch[0].as_array()

        if i == 0:
            CCH = cch
        else:
            CCH = np.concatenate((CCH, cch), axis=-1)

    times = times

    return times, np.mean(CCH, axis=-1)


def array_locations(block):
    """
    Return a list of electrodes for each unit and a matrix of the
    distances between electrodes from which units were recorded
    """
    nunits = len(block.segments[0].spiketrains)
    ntrials = len(block.segments)
    # get array of unit location
    unit_location = np.zeros((nunits,2))
    for i in range(nunits):
        # first column: unit number
        unit_location[i,0] = i+1
        # electrode number
        unit_location[i,1] = block.segments[0].spiketrains[i].annotations['connector_aligned_id']

    # matrix of distances between units
    matrix = np.zeros((nunits,nunits))
    for i in range(nunits):
        for j in range(nunits):
            electrode1 = unit_location[i,1]
            electrode2 = unit_location[j,1]
            x_dist = abs(electrode1%10 - electrode2%10)
            y_dist = abs(np.floor(electrode2/10) - np.floor(electrode1/10))
            distance = np.sqrt(x_dist**2 + y_dist**2)
            matrix[i,j] = distance
            matrix[j,i] = distance

    return unit_location, matrix

def get_sig_corr_mask(R, n_neurons, n_trials):

    true_cc = np.corrcoef(R)

    # shuffle each unit
    count = np.zeros(true_cc.shape)
    niters = 100
    for i in range(0, niters):
        r_shuff = R.copy()
        r_shuff = r_shuff.reshape(n_neurons, n_trials, -1)
        for unit in range(R.shape[0]):
            trials = np.arange(0, n_trials)
            np.random.shuffle(trials)
            r_shuff[unit, :, :] = r_shuff[unit, trials, :]

        r_shuff = r_shuff.reshape(n_neurons, -1)
        cc = np.corrcoef(r_shuff)
        count = count + (cc > true_cc)

    return count / niters


def sort_spiketrains(block):
    sorted_block = cp.deepcopy(block)
    ntrials = block.size['segments']
    nunits = len(block.segments[0].spiketrains)

    for t in range(ntrials):
        order = np.zeros((nunits,1))
        for u in range(nunits):
            channel = block.segments[t].spiketrains[u].annotations['connector_aligned_id']
            unit_id = block.segments[t].spiketrains[u].annotations['unit_id']
            order[u] = int(channel)*100+int(unit_id)

        order = np.argsort(order.squeeze())


        sorted_block.segments[t].spiketrains = np.array(block.segments[t].spiketrains)[order].tolist()

    return sorted_block

def ue_analysis(block, unit1 = 0, unit2 = 1):
    """
    copied from Sonja's lecture
    """
    sts1 = [segment.spiketrains[unit1] for segment in block.segments]
    sts2 = [segment.spiketrains[unit2] for segment in block.segments]

    spiketrain = [[sts1[i], sts2[i]] for i in range(len(sts1))]

    num_trial, N = np.shape(spiketrain)[:2]

    # parameters for unitary events analysis
    winsize = 100*pq.ms
    binsize = 1*pq.ms
    winstep = 5*pq.ms
    pattern_hash = [3]
    significance_level = 0.01

    print('calculating UE ...')
    UE = ue.jointJ_window_analysis(spiketrain, binsize, winsize, winstep, pattern_hash)

    # plotting parameters
    Js_dict = {'events':{'':[]},
         'save_fig': False,
         'path_filename_format':'./UE.pdf',
         'showfig':True,
         'suptitle':True,
         'figsize':(10,12),
        'unit_ids':range(1,N+1,1),
        'fontsize':15,
        'linewidth':2}

    misc._plot_UE(spiketrain, UE, significance_level, binsize, winsize, winstep, pattern_hash, N, Js_dict)


def get_event_dict(block, fs):
    block = sort_spiketrains(block)

    events = ['TS-ON', 'CUE-ON', 'GO-ON', 'SR', 'RW-ON']

    ntrials = len(block.segments)
    time_bins = np.arange(0., 5., 1/fs)
    event_dict = {}

    for ev in events:
        event_dict[ev] = np.zeros((ntrials, len(time_bins)))
        for t in range(ntrials):
            idx = np.argwhere(np.array(block.segments[t].events[0].annotations['trial_event_labels'])==ev)[0][0]
            ev_time = np.float(block.segments[t].events[0].times[idx])
            indexes = np.digitize(ev_time, time_bins)
            event_dict[ev][t, indexes] = 1

    return event_dict
