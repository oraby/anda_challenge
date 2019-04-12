import sys
import numpy as np
import quantities as pq
import elephant.spike_train_correlation as corr
import elephant.spike_train_surrogates as surr
import elephant.statistics as stats
import elephant.conversion as conv
import elephant.unitary_event_analysis as ue
import elephant.spike_train_generation as stocmod
import elephant.spade as spade
sys.path.append('/Users/Benedict/Dropbox/conferences_Summerschools/ANDA2019/Gruen')
import misc
import matplotlib.pyplot as plt
import neo

def cch(block, unit1 = 0, unit2 = 1):
	# Load the data

	sts1 = [segment.spiketrains[unit1] for segment in block.segments] 
	sts2 = [segment.spiketrains[unit2] for segment in block.segments] 

	# Compute the CCH
	w = 2*pq.ms # binsize for CCH
	maxlag = 200*pq.ms # maximum correlation lag
	Nlags = int((maxlag/w).simplified)
	CCH = neo.AnalogSignal(np.zeros(2 * Nlags + 1) * pq.dimensionless, 
	                       sampling_period = w, t_start=-maxlag)

	for st1,st2 in zip(sts1, sts2):
	    CCH += corr.cross_correlation_histogram(
	            conv.BinnedSpikeTrain(st1, w), conv.BinnedSpikeTrain(st2, w), 
	            window=[-Nlags, Nlags], border_correction=False, binary=True)[0]
	CCH = CCH/float(len(sts1))

	# Compute PSTH
	psth1 = stats.time_histogram(sts1, binsize = w, output='rate').rescale('Hz')
	psth2 = stats.time_histogram(sts2, binsize = w, output='rate').rescale('Hz')
	# Compute ISI dist
	ISIpdf1 = misc.isi_pdf(sts1, bins=w, rng=[0*pq.s, None], density=True)
	ISIpdf2 = misc.isi_pdf(sts2, bins=w, rng=[0*pq.s, None], density=True)

	# plot 
	plt.figure(figsize=(12,8))
	plt.subplots_adjust(top=.85, right=.85, left=.2, bottom=.1, hspace=.7 , wspace=.3)
	num_row,num_col = 4, 3
	ax = plt.subplot2grid((num_row,num_col),(0, 0), rowspan = 1, colspan = 2)
	misc.raster(sts1, ms = 2,xlabel='time', ylabel='trial id',color = 'r')
	

	plt.subplot2grid((num_row,num_col),(1, 0), rowspan = 1, colspan = 2,sharex = ax)
	misc.raster(sts2, ms = 2,xlabel='time', ylabel='trial id',color = 'b')
	

	plt.subplot2grid((num_row,num_col),(2, 0), rowspan = 2, colspan = 2,sharex = ax)
	plt.title('PSTH')
	plt.plot(psth1.times,psth1,color = 'r')
	plt.plot(psth2.times,psth2,color = 'b')
	ax.set_xlim(0,4010)#max(psth1.times.magnitude))

	plt.subplot2grid((num_row,num_col),(0, 2), rowspan = 1, colspan = 1)
	plt.plot([len(st) for st in sts1],range(1,len(sts1)+1),  '.-', ms=5,color = 'r')
	plt.title('Spike Count')
	plt.ylabel('trial id')
	plt.subplot2grid((num_row,num_col),(1, 2), rowspan = 1, colspan = 1)
	plt.plot([len(st) for st in sts2],range(1,len(sts2)+1),  '.-', ms=5,color = 'b')
	plt.ylabel('trial id')

	plt.subplot2grid((num_row,num_col),(2, 2), rowspan = 2, colspan = 1)
	plt.plot(ISIpdf1.times,ISIpdf1,color = 'r')
	plt.plot(ISIpdf2.times,ISIpdf2,color = 'b')
	plt.title('ISI distribution')

	# Plot the CCH
	plt.figure(figsize=(12,8))
	plt.subplot2grid((2,2),(1, 1), rowspan = 2, colspan = 2)
	plt.plot(CCH.times.rescale(pq.ms), CCH, 'k-', label='raw CCH')
	plt.title('CCH')
	plt.xlabel('time lag (%s)' %(CCH.times.units.__str__().split(' ')[-1]))
	plt.ylabel('counts')
	plt.xlim(-maxlag, maxlag)
	plt.legend()
	plt.show()
