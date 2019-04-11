import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
import quantities as pq
import neo
import utils as ut

def _subplotRastor(color, trial_index=0):
    '''Draw rastor plot for a given trial. The function adds one dataset to the
    current plot being plotted. The function doesn't attempt to manage the
    plotting. Call plt.subplot() (if needed) and plt.show() from the calling
    site.'''
    # Load the data
    block = ut.load_dataset(color)
    # Get the list of spiketrains of the selected trial
    sts = block.segments[trial_index].spiketrains
    # Get the list of time of the events of the trial
    events = block.segments[trial_index].events[0]

    # Plotting the spikes
    for idx, st in enumerate(sts):
        plt.plot(st, [idx] * len(st), 'o', markersize=1, color=color)
    # Plotting the events
    for ev_idx, ev in enumerate(events):
        # Getting the labe of the single event
        ev_label = events.annotations['trial_event_labels'][ev_idx]
        if ev_label in ut.trial_events:
            plt.vlines(ev.rescale(pq.s), 0, len(sts))
            plt.text(ev.rescale(pq.s) - 0.01 * pq.s, len(sts) + 9, ev_label,
                     ha='center', fontsize='small')
    # Additional plot parameter
    plt.xlim([sts[0].t_start - 0.2 * pq.s, sts[0].t_stop + 0.2 * pq.s])
    plt.ylim([-1, len(sts) + 5])
    plt.ylabel('Neu idx')
    plt.xlabel('Time ({})'.format(sts[0].units))

def _subplotPSTH(trial_type = None, win_size=25, trim_data=False):
    '''Plots the PSTHs for all the trial_types. As in _subplotRastor(), you
    need to call plt.show() and manage your subplots from where you call this
    function.'''
    window = ss.triang(M=win_size) / (win_size / 2)
    for i, color in enumerate(ut.COLORS):
        print("Processing dataset for", color)
        block = ut.load_dataset(color)
        time_ms = int((block.segments[0].t_stop - block.segments[0].t_start) *
                      1000) + 1
        xtrials = np.zeros([len(block.segments),
                            len(block.segments[0].spiketrains), time_ms])
        for trial_idx, segment in enumerate(block.segments):
            if trial_type != None and \
               trial_type not in segment.annotations['belongs_to_trialtype']:
                continue
            for neuron_idx, spiketrain in enumerate(segment.spiketrains):
                np_array = spiketrain.as_array()
                for spike_time_s in np_array:
                    spike_time_ms = int(spike_time_s*1000)
                    xtrials[trial_idx,neuron_idx,spike_time_ms] = 1

        trials_sum = xtrials.sum(axis=0).sum(axis=0)
        r_psth = ss.filtfilt(window, 1, trials_sum, axis=-1)
        if trim_data:
            r_psth = r_psth[win_size:-win_size]
        plt.ylim(bottom=0,top=80)
        plt.plot(np.arange(r_psth.shape[0]),r_psth, color=color)


    ## Draw events vertical lines
    # Get the list of events of the trial, they vary
    events = segment.events[0]
    # Plotting the events
    for ev_idx, ev in enumerate(events):
        # Getting the labels of the single event
        ev_label = events.annotations['trial_event_labels'][ev_idx]
        if ev_label in ut.trial_events:
            x_pos = ev.rescale(pq.ms)
            if trim_data:
                x_pos +=  pq.Quantity(win_size, 'ms')
            plt.vlines(x_pos, 0, np.amax(r_psth))
            plt.text(x_pos, np.amax(r_psth) + 10, ev_label,
                     horizontalalignment='center', fontsize='small')

def plotRastor():
    DPI=120
    plt.figure(figsize=(1024*1.5/DPI, 1024*1.5/DPI), dpi=DPI)
    for i, color in enumerate(ut.COLORS):
        print("Processing rastor for", color)
        plt.subplot(int((len(ut.COLORS)+1)/2), 2, i+1)
        _subplotRastor(color)

    plt.savefig("plots/rastor_plots.png", dpi=DPI)
    plt.show()

def plotPSTH():
    DPI=120
    plt.figure(figsize=(1024*1.5/DPI, 1024*1.5/DPI), dpi=DPI)
    #plotPSTH() # Use this to draw one PSTH for all trial types together
    for plot_idx, trial_type in enumerate(ut.trial_types):
        print("Processing for:", trial_type)
        plt.subplot(int((len(ut.trial_types)+1)/2), 2, plot_idx+1)
        _subplotPSTH(trial_type)
    plt.savefig("plots/PSTH_plots.png", dpi=DPI)
    plt.show()

if __name__ == "__main__":
    plotRastor()
    plotPSTH()
