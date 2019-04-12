import numpy as np
import quantities as pq
import elephant.unitary_event_analysis as ue
import misc

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