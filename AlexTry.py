import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from scipy.stats import chisquare
from scipy.stats import gamma


def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

def gamma_func(x, a, b, c, d, e):
    res = e * gamma.pdf(x, a, loc=0, scale=c) + d
    return res

def compute_isi_fits(all_data):
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
    plt.show()



    plt.figure()
    for i, d in enumerate(all_data):
        plt.bar(i, all_data[d]['exp_diff_fit'], color=d)
    plt.ylabel(r'$L_2$ error')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    plt.figure()
    for i, d in enumerate(all_data):
        plt.bar(i, all_data[d]['gamma_diff_fit'], color=d)
    plt.ylabel(r'$L_2$ error')
    plt.xticks([])
    plt.yticks([])
    plt.show()
