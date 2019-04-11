import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from scipy.stats import gamma

import neo
import utils as ut

def spike_counts(all_data, color):
	ntrials = all_data[color]['spike_raster'].shape[0]
	nunits = all_data[color]['spike_raster'].shape[1]
	spikes_across_time =  np.sum(all_data[color]['spike_raster'],axis=2)
	spikes_across_units = np.sum(all_data[color]['spike_raster'],axis=1)/nunits
	return spikes_across_units, spikes_across_time


def exp_func(x, a, b, c):
	return a * np.exp(-b * x) + c

def gamma_func(x, a, b, c, d, e):
	res = e * gamma.pdf(x, a, loc=0, scale=c) + d
	return res


def compute_isi_fits(all_data):
	print('Performing the ISI fitting analysis')

	plt.xkcd()
	plt.figure(figsize=(10, 10))
	for d in all_data:
		bins = np.linspace(0, 0.05, 50)
		bins = 2500
		h = plt.hist(all_data[d]['isi'], histtype='step', density=1, cumulative=False, bins=bins, label=d, color=d)
		H = h[0]
		b = (h[1][:-1]+h[1][1:]) / 2.
		popt, pcov = curve_fit(exp_func, b, H)
		plt.plot(b, exp_func(b, *popt), color=d, ls=':')
		all_data[d]['exp_diff_fit'] = np.sum((H - exp_func(b, *popt))**2.)

		popt, pcov = curve_fit(gamma_func, b, H)
		plt.plot(b, gamma_func(b, *popt), color=d, ls='--')
		all_data[d]['gamma_diff_fit'] = np.sum((H - gamma_func(b, *popt))**2.)

	plt.xlim(0, 0.1)
	plt.ylim(1.e0, None)
	plt.legend(loc='best')
	plt.tight_layout()
	plt.savefig('plots/isi_fit.pdf')
	plt.savefig('plots/isi_fit.png')
	plt.show()



	plt.figure()
	for i, d in enumerate(all_data):
		plt.bar(i, all_data[d]['exp_diff_fit'], color=d)
	plt.ylabel(r'$L_2$ error')
	plt.xticks([])
	# plt.yticks([])
	plt.title('Exponential fit')
	plt.tight_layout()
	plt.savefig('plots/isi_poisson_fit.pdf')
	plt.savefig('plots/isi_poisson_fit.png')
	plt.show()

	plt.figure()
	for i, d in enumerate(all_data):
		plt.bar(i, all_data[d]['gamma_diff_fit'], color=d)
	plt.ylabel(r'$L_2$ error')
	plt.xticks([])
	# plt.yticks([])
	plt.title('Gamma fit')
	plt.tight_layout()
	plt.savefig('plots/isi_gamma_fit.pdf')
	plt.savefig('plots/isi_gamma_fit.png')
	plt.show()
