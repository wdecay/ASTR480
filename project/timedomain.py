import numpy as np
from astroML.time_series import lomb_scargle, lomb_scargle_BIC, lomb_scargle_bootstrap
from astroML.utils.exceptions import AstroMLDeprecationWarning
from matplotlib.colors import LogNorm
from scipy import fftpack
from scipy.fft import fft
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
from astropy.visualization import astropy_mpl_style
import warnings
import functools
import operator

plt.style.use(astropy_mpl_style) 
warnings.filterwarnings("ignore", category=AstroMLDeprecationWarning)

# plot phased light curve with times t, observations y_obs, and period P
def plot_phased(t, y_obs, sigma_y, P, ax = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(14,7))

    phase = t / P - np.floor(t / P)
    ax.errorbar(phase, y_obs, sigma_y, fmt='.k', lw=1, ecolor='gray')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Relative flux, arb.')
    ax.set_xlim(-0.05, 1.05)
    
# plot phased light curve and periodogram
# times: t, observations: y_obs, period: P_fit,
# 1%/5% false alarm probability thresholds for P_LS: sig1/sig5
def plot_LS(t, y_obs, sigma_y, period, PS, P_fit, sig1, sig5, saveas=None, noshow=False):
    #------------------------------------------------------------
    # Plot the results
    fig = plt.figure(figsize=(16, 10))
    fig.subplots_adjust(left=0.1, right=0.9, hspace=0.25)

    # First panel: the data
    ax = fig.add_subplot(211)
    plot_phased(t, y_obs, sigma_y, P_fit, ax=ax)

    # Second panel: the periodogram & significance levels
    ax1 = fig.add_subplot(212)
    ax1.plot(period, PS, '-', c='black', lw=1, zorder=1)
    ax1.plot([period[0], period[-1]], [sig1, sig1], ':', c='black', label="99% significance level")
    ax1.plot([period[0], period[-1]], [sig5, sig5], '-.', c='black', label="95% significance level")
    ax1.legend()

    max_y = np.max(PS)
    ax1.annotate("P={:.0f} s".format(P_fit), (P_fit, max_y), (P_fit + 20, max_y * 1.1))
    ax1.axvline(P_fit, ls='-', c='gray')

    ax1.set_xlim(period[0], period[-1])
    ax1.set_ylim(-0.01, max(max_y * 1.2, 1.2 * sig5))

    ax1.set_xlabel(r'Period, seconds')
    ax1.set_ylabel('Power')

    # Twin axis: label BIC on the right side
    ax2 = ax1.twinx()
    ax2.set_ylim(tuple(lomb_scargle_BIC(ax1.get_ylim(), y_obs, sigma_y)))
    # print(tuple(lomb_scargle_BIC(ax1.get_ylim(), y_obs, sigma_y)))
    ax2.set_ylabel(r'$\Delta BIC$')

    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax1.xaxis.set_minor_formatter(plt.FormatStrFormatter('%.1f'))
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))

    if saveas is not None:
        plt.savefig(saveas, bbox_inches='tight', dpi=150)
    
    if not noshow:
        plt.show()

def plot_ts(t, y_obs, sigma_y, t_imp, y_imp, saveas=None, noshow=False):
    if not noshow:
        plt.figure(figsize=(16, 6))
    plt.errorbar(t, y_obs - np.mean(y_obs), sigma_y, fmt='.k', lw=1, label='Observation')
    plt.scatter(t_imp, y_imp - np.mean(y_imp), marker='x', color='k', label='Imputed data')
    plt.xlabel('Time, seconds')
    plt.ylabel('Relative flux, arb.')
    plt.legend()
    if saveas is not None:
        plt.savefig(saveas, bbox_inches='tight', dpi=150)
    
    if not noshow:
        plt.show()
        
def plot_ps(t, y_obs, saveas=None, noshow=False, xlabel=True, ylabel=True):
    sig_fft = fftpack.fft(y_obs - y_obs.mean())
    power = np.abs(sig_fft)**2

    dt = np.median(t[1:] - t[:-1])
    sample_freq = fftpack.fftfreq(len(t), d=dt) * 1000 # mHz

    n = int(len(sample_freq)/2)
    X_Y_Spline = make_interp_spline(sample_freq[0:n], power[0:n])
 
    X_ = np.linspace(0, np.max(sample_freq), 400)
    Y_ = X_Y_Spline(X_)
    Y_[Y_ < 0] = 0

    if not noshow:
        plt.figure(figsize=(8, 5))

    plt.fill_between(X_, Y_, color='LightGrey')
    plt.plot(sample_freq[0:n], power[0:n], 'k.')
    
    n_max = np.argmax(power[0:n])
    f_max = sample_freq[0:n][n_max]
    p_max = power[0:n][n_max]

    step = sample_freq[n]-sample_freq[n-1]
    plt.annotate("$f={:.2f}\pm{:.2f}$ mHz".format(f_max, step/2), (f_max + 0.1, p_max))

    plt.xlim([0, 4])
    plt.ylim([0, 1.1 * np.max(power)])
    plt.tick_params(left = False, right = False , labelleft = True,
                    labelbottom = xlabel, bottom = True)
    if xlabel:
        plt.xlabel('Frequency [mHz]')
    if ylabel:
        plt.ylabel('Power, arb.')
    #plt.grid(False, axis='y')
    if saveas is not None:
        plt.savefig(saveas, bbox_inches='tight', dpi=150)
    
    if not noshow:
        plt.show()


def periodogram(t, y, sigma, saveas=None):
    #------------------------------------------------------------
    # Compute periodogram
    period = np.linspace(250, 1500, 10000)
    omega = 2 * np.pi / period
    PS = lomb_scargle(t, y, sigma, omega, generalized=True)

    # find the highest peak
    P_fit = period[PS.argmax()]

    #------------------------------------------------------------
    # Get significance via bootstrap
    D = lomb_scargle_bootstrap(t, y, sigma, omega, generalized=True,
                            N_bootstraps=500, random_state=0)
    sig1, sig5 = np.percentile(D, [99, 95])
    plot_LS(t, y, sigma, period, PS, P_fit, sig1, sig5)