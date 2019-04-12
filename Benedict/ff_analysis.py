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
#import misc
import utils as ut

def boxcar_kernel(width): 
    length = int(2*np.sqrt(3)*width)
    height = 1/length
    return np.ones(length)*height*1000

def apply_kernel(signal,kernel,offset=None): 
    """ if offset=None assume kernel is symmetric"""
    #offset = offset if offset is not None else int(len(kernel)/2)
    #zero_shape = np.zeros((offset,signal.shape[1]) if len(signal.shape)==2 else (offset,))
    #ns = np.concatenate((signal,zero_shape))
    filtered = scipy.signal.lfilter(kernel,1,signal,axis=0)[len(kernel):]
    offset = int(len(kernel)/2)
    return offset,filtered



def trff(psth):
	"""
	calculate time-resolved fano factor
	"""
	trial_avg = np.mean(psth,axis=0)
	trial_var = np.var(psth,axis=0)
	FF = trial_var/trial_avg
	return FF


def population_trff(psth):
	ntrials = np.shape(psth)[0] 
	nunits = np.shape(psth)[1]
	ntimepoints = np.shape(psth)[2]
	ff = np.zeros((ntimepoints,nunits))
	# calculate time-resolved FF for each unit
	for i in range(nunits):
		ff[:,i] = trff(psth[:,i,:])
	
	# cut 500ms from beginning and 1500ms from end
	# to take care of edge effects
	ff = ff[500:-1500]
	# replace nans with 0
	#ff[np.isnan(ff)]=0 
	# average across units:
	pop_trff = np.nanmean(ff,axis=1)
	

	# plotting
	
	plt.figure()
	plt.plot(range(500,3500),pop_trff)
	plt.title('Time Resolved Fano Factor')
	plt.xlabel('Time from trial start [ms]')
	plt.ylabel('Fano Factor')
	plt.show()

	return pop_trff, ff
	




#plt.plot(range(500,3500),ff[:,0])
#plt.show()
