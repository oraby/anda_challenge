import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import pdb


def trff(psth):
    """
    calculate time-resolved fano factor
    """
    trial_avg = np.mean(psth,axis=0)
    trial_var = np.var(psth,axis=0)
    FF = trial_var/trial_avg
    return FF


def population_trff(psth,fs = 1.e3, win_size = 10):
    win_size = int(np.ceil(fs * (win_size/1000)))
    ntrials = np.shape(psth)[0] 
    nunits = np.shape(psth)[1]
    ntimepoints = np.shape(psth)[2]
    ff = np.zeros((ntimepoints,nunits))
    # calculate time-resolved FF for each unit
    for i in range(nunits):
        ff[:,i] = trff(psth[:,i,:])
    
    # cut 500ms from beginning and 1500ms from end
    # to take care of edge effects
    ff = ff[int(win_size):int(-win_size)]
    # replace nans with 0
    #ff[np.isnan(ff)]=0 
    # average across units:
    pop_trff = np.nanmean(ff,axis=1)
    
    return pop_trff


def plot_pop_trff(pop_trff, events):
    
    plt.figure()  
    for c in pop_trff.keys():
        plt.plot(pop_trff[c],color=c)
        plt.title('Time Resolved Fano Factor')
        plt.xlabel('Time from trial start [ms]')
        plt.ylabel('Fano Factor')
    for key in events.keys():
        relevant_time = np.argwhere(events[key][0,:]==1)[0][0]
        plt.axvline(relevant_time,color="black")
    plt.tight_layout()


def ff_wrapper(colors=ut.COLORS, win_size = 150, trial_type=None):
    pop_trff = {}
    for color in colors:
        data = ut.load_dataset(color)
        events = ut.get_event_dict(data,fs=1000)
        R = ut.rasterize_data(data, sf = 1.e3)
        ntrials = np.shape(R)[0] 
        tt = [data.segments[i].annotations['belongs_to_trialtype']  for i in range(0, ntrials)]
        if trial_type is not None:
            trials = np.argwhere(np.array(tt) == trial_type).squeeze()
            R = R[trials, :, :]
        psth = ut.calculate_single_trial_PSTH(R, 1.e3, win_size, window='triangle')
        pop_trff[color] = population_trff(psth,1.e3, win_size)
    plot_pop_trff(pop_trff, events)
