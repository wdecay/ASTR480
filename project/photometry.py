import astroalign as aa
import astropy.constants
import astropy.units as u
import ccdproc
import logging
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import signal
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import SigmaClip
from astropy.visualization import astropy_mpl_style
from datetime import datetime
from matplotlib.colors import LogNorm
from multiprocessing import Pool, TimeoutError
from photutils.background import Background2D, MedianBackground
from progress.bar import Bar
from scipy.optimize import curve_fit
from scipy.optimize import minimize


from photutils.psf import IterativelySubtractedPSFPhotometry, BasicPSFPhotometry
from photutils.detection import IRAFStarFinder
from photutils.psf import IntegratedGaussianPRF, DAOGroup
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.table import Table

def load_frames(base_dir):
    files = [os.path.join(base_dir, f) for f in os.listdir(base_dir)
             if not os.path.isdir(os.path.join(base_dir, f))]

    def extract_header(fname, fields):
        header = fits.getheader(fname, ext=0)
        return [fname] + [header[f] for f in fields]

    fields = ['EXPTIME', 'FILTER', 'IMAGETYP', 'BZERO', 'CCD-TEMP',
              'INSTRUME', 'DATE-OBS']
    data = pd.DataFrame([extract_header(f, fields) for f in files],
                        columns = ["Name"] + fields).sort_values('DATE-OBS')
    return data

def take_frames(frames, ids):
    return frames.take(ids).reset_index()

def load_image(row, master_bias, master_flat, gain, readnoise, subtract_bkg=True):
    img = fits.getdata(row['Name'], ext=0)
    
    data = CCDData(img, unit=u.adu)
    data_with_deviation = ccdproc.create_deviation(
        data, gain=gain,
        readnoise=readnoise, disregard_nan=True)
    data_with_deviation.header['exposure'] = 100.0  # for dark subtraction
    gain_corrected = ccdproc.gain_correct(data_with_deviation, gain)
    # img_cleaned = ccdproc.cosmicray_lacosmic(gain_corrected)
    bias_subtracted = ccdproc.subtract_bias(gain_corrected, master_bias)
    flat_corrected = ccdproc.flat_correct(bias_subtracted, master_flat)

    if not subtract_bkg:
        return flat_corrected

    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = MedianBackground()
    bkg = Background2D(flat_corrected, (70, 70), filter_size=(3, 3),
                       sigma_clip=sigma_clip,
                       bkg_estimator=bkg_estimator)

    return ccdproc.subtract_bias(flat_corrected, bkg.background)

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
        x, y = c #optimize_star(img, c, 6)

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
    c = int(4*fwhm/2)
    d = 2 * c + 1
    xx = np.arange(d)
    yy = np.arange(d)

    p0 = int(p[0])
    p1 = int(p[1])

    sub_img = img[p1-c:p1+c+1, p0-c:p0+c+1]

    def f(x):
        return -np.sum(sub_img * np.exp(-4*np.log(2) * ((xx[None, :]-x[0])**2 + (yy[:, None]-x[1])**2) / fwhm**2))
    
    res = minimize(f, (c, c), method='powell', options={'disp': False})
    
    # plt.imshow(sub_img, vmax=2000)
    offset = res.x - (c, c)
    return p + offset, res.x, sub_img

def aperture(img, p, r):
    xx = np.arange(img.shape[0])
    yy = np.arange(img.shape[1])
    pos, _, _ = optimize_star(img, p, r)
    y0, x0 = pos
    rpix = np.sqrt((xx[:, None]-x0)**2 + (yy[None, :]-y0)**2)
    return np.sum(img[rpix <= r]), np.std(img[rpix <= r])


def psf(img, p, r):
    sigma_psf = 2.5
    
    _, pos, sub_img = optimize_star(img, p, r)

    
    bkgrms = MADStdBackgroundRMS()
    std = bkgrms(sub_img)
    daogroup = DAOGroup(2.0*sigma_psf*gaussian_sigma_to_fwhm)
    mmm_bkg = MMMBackground()
    fitter = LevMarLSQFitter()
    psf_model = IntegratedGaussianPRF(sigma=sigma_psf)

    psf_model.x_0.fixed = True
    psf_model.y_0.fixed = True
    # psf_model.sigma.fixed = False

    pos = Table(names=['x_0', 'y_0'], data=[[pos[0]],
                                            [pos[1]]])

    photometry = BasicPSFPhotometry(group_maker=daogroup,
                                bkg_estimator=mmm_bkg,
                                psf_model=psf_model,
                                fitter=LevMarLSQFitter(),
                                fitshape=(11,11))

    result_tab = photometry(image=sub_img, init_guesses=pos)
    return result_tab['flux_fit'][0], result_tab['flux_unc'][0]

def process_one(input):
    row, stars = input
    _, row = row
    
    frames = process_one.frames
    ref_stars = process_one.ref_stars
    master_bias = process_one.master_bias
    master_flat = process_one.master_flat

    gain = process_one.gain
    readnoise = process_one.readnoise
    r = process_one.r

    img = load_image(row, master_bias, master_flat, gain, readnoise).data
    shift = compute_transform(process_one.ref_img, img)

    pf = aperture
    
    ref_flux = np.array([pf(img, ref + shift, r) for ref in ref_stars])
    star_flux = np.array([pf(img, star + shift, r) for star in stars])
    
    denom = ref_flux[:, 0].mean()
    x = [pf(img, star + shift, r)[0] / denom for star in stars]
    stds = [0.01] * len(x)

    return datetime.fromisoformat(row['DATE-OBS']).timestamp(), *x, *stds

def initializer(frames, ref_frame_index, ref_stars, master_bias, master_flat, gain, readnoise, r):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    process_one.frames = frames
    process_one.ref_stars = ref_stars
    process_one.master_bias = master_bias
    process_one.master_flat = master_flat
    process_one.gain = gain
    process_one.readnoise = readnoise
    process_one.r = r
    process_one.ref_img = load_image(frames.iloc[ref_frame_index], master_bias, master_flat, gain, readnoise)


def compute_relative_flux_parallel(pool, frames, stars):
    with Bar('Processing', max = frames.shape[0], suffix='%(percent).1f%%') as bar:
        for res in pool.imap(process_one, zip(frames.iterrows(), frames.shape[0] * [stars])):
            yield res
            bar.next()


def extract_light_curves(pool, frames, ref_frame_index, stars):
    fiter = compute_relative_flux_parallel(pool, frames, stars)
    ts = list(fiter)

    tsa = np.array(ts)
    
    t = tsa[:, 0] - tsa[0, 0]
    tsa[:, 0] = t
    return tsa

def analysis_stub():
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
    # print(f"Location of highest periodogram peak, P_fit = {P_fit:.3f} seconds")


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    plt.style.use(astropy_mpl_style) 

    base = './data'

    gain = 1.25*u.electron/u.adu
    readnoise = 11.8*u.electron

    ref_frame_index = 76
    
    star = (567, 483)

    control = [(427, 537), (420, 584), (978, 68), (913, 313), (713, 873), (340, 947), (563, 86), (169, 320), (785, 947)]
    ref_stars = [(671, 486), (521, 509), (218, 794), (905, 530), 
                 (722, 238), (688, 252), (464, 149), (559, 848), 
                 (391, 387), (815, 784), (788, 836), (827, 859)]
    
    ref_stars = sorted(ref_stars, key=lambda r: r[1]) # sorting refs by the y coordinate

    master_bias = CCDData(fits.getdata(os.path.join(base, 'processed/master_bias.fits'), ext=0), unit=u.electron)
    master_flat = CCDData(fits.getdata(os.path.join(base, 'processed/master_flat.fits'), ext=0), unit=u.electron)
    # control, ref_stars = ref_stars, control
    
    all_frames = load_frames(base)
    good_ids = np.setdiff1d(np.arange(3, 100), range(45, 49))
    good_frames = take_frames(all_frames, good_ids)

    show_field(load_image(good_frames.iloc[ref_frame_index],
                          master_bias, master_flat, gain, readnoise), star, ref_stars,
               control, saveas='img/field.png')
    img = load_image(good_frames.iloc[18], master_bias, master_flat, gain, readnoise)
    shift = compute_transform(load_image(good_frames.iloc[ref_frame_index], master_bias, master_flat, gain, readnoise), img)
    show_field(img, star, ref_stars, shift, saveas='img/trans_field.png')

    r = 5
    logging.info('Initializing multiprocessing pool...')
    pool = Pool(processes = 8, initializer = initializer, initargs =
                (good_frames, ref_frame_index, ref_stars, master_bias,
                 master_flat, gain, readnoise, r))

    with pool:
        lcs = extract_light_curves(pool, good_frames, ref_frame_index, [star] + control)

    columns = ['t', 'main'] + ['C_{}'.format(i + 1) for i in
                               range(len(control))] + ['std_main'] + ['std_{}'.format(i + 1)
                                                                      for i in range(len(control))]

    df = pd.DataFrame(lcs, columns=columns)

    df.to_csv('data/processed/lcs.csv', index=False)

