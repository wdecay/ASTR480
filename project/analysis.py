import astroalign as aa
import astropy.constants
import astropy.units as u
import ccdproc
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import os
import pandas as pd
import signal
import warnings
from astroML.time_series import lomb_scargle, lomb_scargle_BIC, lomb_scargle_bootstrap
from astroML.utils.exceptions import AstroMLDeprecationWarning
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.visualization import astropy_mpl_style
from datetime import datetime
from matplotlib.colors import LogNorm
from multiprocessing import Pool, TimeoutError
from progress.bar import Bar
from scipy import fftpack
from scipy.fft import fft
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from scipy.optimize import minimize

plt.style.use(astropy_mpl_style) 
warnings.filterwarnings("ignore", category=AstroMLDeprecationWarning)

base = './data'
gain = 1.25*u.electron/u.adu
readnoise = 11.8*u.electron
master_bias = CCDData(fits.getdata(os.path.join(base, 'processed/master_bias.fits'), ext=0), unit=u.electron)
master_flat = CCDData(fits.getdata(os.path.join(base, 'processed/master_flat.fits'), ext=0), unit=u.electron)

n_ref = 83

star = (567, 483)

control = [(427, 537), (420, 584), (978, 68)]

refs = [(671, 486), (521, 509), (218, 794), (905, 530), 
        (722, 238), (688, 252), (464, 149), (559, 848), 
        (391, 387), (815, 784), (788, 836), (827, 859)]

refs = sorted(refs, key=lambda r: r[1]) # sorting refs by the y coordinate

def load_image(row):
    img = fits.getdata(row['Name'], ext=0)
    
    data = CCDData(img, unit=u.adu)
    data_with_deviation = ccdproc.create_deviation(
        data, gain=gain,
        readnoise=readnoise, disregard_nan=True)
    data_with_deviation.header['exposure'] = 100.0  # for dark subtraction
    gain_corrected = ccdproc.gain_correct(data_with_deviation, gain)
    # img_cleaned = ccdproc.cosmicray_lacosmic(gain_corrected)
    bias_subtracted = ccdproc.subtract_bias(gain_corrected, master_bias)
    reduced_image = ccdproc.flat_correct(bias_subtracted, master_flat)
    return reduced_image

def show_field(img, main, refs, control=None, shift=None, saveas=None):    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    im = ax.imshow(img, vmax=1000, cmap='binary')
    plt.grid(False)
    if shift is None:
        shift = np.array([0, 0])

    # correction
    main = main + shift
    refs = [ref + shift for ref in refs]

    c_main = patches.Circle(main, 6, linewidth=1, linestyle='-',
                                 edgecolor='r', facecolor="none")
    ax.add_patch(c_main)
    ax.annotate("J1903+6035", main, main + (0, -12), ha='center')
    
    for i, c in enumerate(refs):
        x, y = c

        rect = patches.Circle((x, y) + shift, 6, linewidth=1, linestyle='--',
                                 edgecolor='g', facecolor="none")
        ax.add_patch(rect)
        ax.annotate("#{:02d}".format(i + 1), c, c + (0, +24), ha='center')

    if control is not None:
        control = [c + shift for c in control]

        for i, c in enumerate(control):
            x, y = c

            rect = patches.Circle((x, y) + shift, 6, linewidth=1,
                                  linestyle='--', edgecolor='m',
                                  facecolor="none")
            ax.add_patch(rect)
            ax.annotate("C{}".format(i + 1), c, c + (10, 0))

             
    fig.colorbar(mappable = im)
    if saveas is not None:
        plt.savefig(saveas, bbox_inches='tight', dpi=150)
    else:
        plt.show()
    
def compute_transform(img_ref, img_other):
    T, matches = aa.find_transform(img_ref, img_other, detection_sigma=2)
    shift = np.array([(d[0]-s[0], d[1]-s[1]) for s, d in zip(matches[0], matches[1])]).mean(axis=0)
    return shift

def optimize_star(img, p, fwhm):
    xx = np.arange(img.shape[0])
    yy = np.arange(img.shape[1])

    def f(x):
        return -np.sum(img * np.exp(-4*np.log(2) * ((xx[None, :]-x[0])**2 + (yy[:, None]-x[1])**2) / fwhm**2))
    
    res = minimize(f, p, method='powell', options={'disp': False})
    return res

def new_optimize_star(img, p, fwhm):
    c = int(5*fwhm/2)
    d = 2 * c + 1
    xx = np.arange(d)
    yy = np.arange(d)

    p0 = int(p[0])
    p1 = int(p[1])

    sub_img = img[p1-c:p1+c+1, p0-c:p0+c+1]

    def f(x):
        return -np.sum(sub_img * np.exp(-4*np.log(2) * ((xx[None, :]-x[0])**2 + (yy[:, None]-x[1])**2) / fwhm**2))
    
    res = minimize(f, (c, c), method='powell', options={'disp': False})
    plt.imshow(sub_img, vmax=1000)
    return res.x - (c, c) # offset

def aperture(img, p, r):
    xx = np.arange(img.shape[0])
    yy = np.arange(img.shape[1])
    
    offset = new_optimize_star(img, p, r)
    y0, x0 = p + offset
    rpix = np.sqrt((xx[:, None]-x0)**2 + (yy[None, :]-y0)**2)
    apPhotImage = img.copy()
    apPhotImage[rpix > r] = 0
    return apPhotImage.sum()

# Similar to Figure 10.15 in the textbook

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
def plot_LS(t, y_obs, sigma_y, period, PS, P_fit, sig1, sig5, saveas=None):
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
    print(tuple(lomb_scargle_BIC(ax1.get_ylim(), y_obs, sigma_y)))
    ax2.set_ylabel(r'$\Delta BIC$')

    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax1.xaxis.set_minor_formatter(plt.FormatStrFormatter('%.1f'))
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))

    if saveas is not None:
        plt.savefig(saveas, bbox_inches='tight', dpi=150)
    else:
        plt.show()

def plot_ts(t, y_obs, sigma_y, saveas=None):
    plt.figure(figsize=(16, 6))
    plt.errorbar(t, y_obs, sigma_y, fmt='.k', lw=1)
    plt.xlabel('Time, seconds')
    plt.ylabel('Relative flux, arb.')
    if saveas is not None:
        plt.savefig(saveas, bbox_inches='tight', dpi=150)
    else:
        plt.show()
        
def plot_ps(t, y_obs, saveas=None):
    sig_fft = fftpack.fft(y_obs - y_obs.mean())
    power = np.abs(sig_fft)**2

    dt = np.median(t[1:] - t[:-1])
    print('Median interval: {}'.format(dt))
    sample_freq = fftpack.fftfreq(len(t), d=dt) * 1000 # mHz

    n = int(len(sample_freq)/2)
    X_Y_Spline = make_interp_spline(sample_freq[0:n], power[0:n])
 
    X_ = np.linspace(0, np.max(sample_freq), 400)
    Y_ = X_Y_Spline(X_)
    Y_[Y_ < 0] = 0

    plt.figure(figsize=(8, 5))

    plt.fill_between(X_, Y_, color='LightGrey')
    plt.plot(sample_freq[0:n], power[0:n], 'k.')
    # plt.axvline(1/760)
    plt.xlim([0, 4])
    plt.ylim([0, 1.1 * np.max(power)])
    plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = True, bottom = True)
    plt.xlabel('Frequency [mHz]')
    plt.ylabel('Power, arb.')
    plt.grid(False, axis='y')
    if saveas is not None:
        plt.savefig(saveas, bbox_inches='tight', dpi=150)
    else:
        plt.show()    

def compute_relative_flux(star, r = 6):
    ids = np.setdiff1d(np.arange(3, 100), range(45, 49))
    with Bar('Processing', max = ids.shape[0], suffix='%(percent).1f%%') as bar:
        for i in ids:
            img = load_image(data.iloc[i]).data
            shift = compute_transform(load_image(data.iloc[n_ref]), img)
            x = aperture(img, star + shift, r) / np.array([aperture(img, ref + shift, r) for ref in refs]).mean()
            yield datetime.fromisoformat(data.iloc[i]['DATE-OBS']).timestamp(), x
            bar.next()

def process_one(i):
    star = process_one.star
    r = process_one.r
    data=process_one.data
    n_ref = process_one.n_ref    

    img = load_image(data.iloc[i]).data
    shift = compute_transform(load_image(data.iloc[n_ref]), img)
    x = aperture(img, star + shift, r) / np.array([aperture(img, ref + shift, r) for ref in refs]).mean()

    return datetime.fromisoformat(data.iloc[i]['DATE-OBS']).timestamp(), x

def initializer(star,r,data,n_ref):
    process_one.star = star
    process_one.r = r
    process_one.data = data
    process_one.n_ref = n_ref
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def compute_relative_flux_parallel(star, r = 6):
    ids = np.setdiff1d(np.arange(3, 100), range(45, 49))
    with Bar('Processing', max = ids.shape[0], suffix='%(percent).1f%%') as bar:
        with Pool(processes = 8, initializer = initializer, initargs = (star, r, data, n_ref)) as pool:
            for res in pool.imap(process_one, ids):
                yield res
                bar.next()


def execute(star, r, suffix=""):
    fiter = compute_relative_flux_parallel(star, r)
    ts = list(fiter)

    tsa = np.array(ts)
    t = tsa[:, 0] - tsa[0, 0]
    y_obs = tsa[:, 1]
    sigma_y = 0.01 * np.ones(t.shape)

    plot_ts(t, y_obs, sigma_y, saveas='img/lightcurve{}.png'.format(suffix))
    plot_ps(t, y_obs, saveas='img/power{}.png'.format(suffix))

    #------------------------------------------------------------
    # Compute periodogram
    period = np.linspace(100, 1400, 10000)
    omega = 2 * np.pi / period
    PS = lomb_scargle(t, y_obs, sigma_y, omega, generalized=True)

    # find the highest peak
    P_fit = period[PS.argmax()]

    #------------------------------------------------------------
    # Get significance via bootstrap
    D = lomb_scargle_bootstrap(t, y_obs, sigma_y, omega, generalized=True,
                               N_bootstraps=500, random_state=0)
    sig1, sig5 = np.percentile(D, [99, 95])


    plot_LS(t, y_obs, sigma_y, period, PS, P_fit, sig1, sig5, saveas='img/ls_periodogram{}.png'.format(suffix))
    print(f"Location of highest periodogram peak, P_fit = {P_fit:.3f} seconds")


if __name__ == "__main__":
    files = [os.path.join(base, f) for f in os.listdir(base) if not os.path.isdir(os.path.join(base, f))]

    def extract_header(fname, fields):
        header = fits.getheader(fname, ext=0)
        return [fname] + [header[f] for f in fields]

    fields = ['EXPTIME', 'FILTER', 'IMAGETYP', 'BZERO', 'CCD-TEMP', 'INSTRUME', 'DATE-OBS']
    data = pd.DataFrame([extract_header(f, fields) for f in files], columns = ["Name"] + fields).sort_values('DATE-OBS')


    show_field(load_image(data.iloc[n_ref]), star, refs, control, saveas='img/field.png')


    n = 21
    img = load_image(data.iloc[n])
    shift = compute_transform(load_image(data.iloc[n_ref]), img)
    show_field(img, star, refs, shift, saveas='img/trans_field.png')


    # execute(star, r = 5)

    for i, c in enumerate(control):
        execute(c, r = 5, suffix="_C{}".format(i+1))
