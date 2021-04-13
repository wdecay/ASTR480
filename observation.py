# ICRS coordinates are roughly the same as equatorial coordinates:
# https://en.wikipedia.org/wiki/International_Celestial_Reference_System
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_moon, get_sun
from astroplan import moon_phase_angle, moon_illumination

class ObservationPlanner:
    def __init__(self, date_midnight_utc, utc_offset, loc):
        self.__midnight = Time(date_midnight_utc) - utc_offset
        self.__loc = loc
        
        self.__delta_midnight = np.linspace(-12, 12, 1000)*u.hour
        times = self.__midnight + self.__delta_midnight
        
        self.__altazframe = AltAz(obstime=times, location=loc)
        self.__sunaltazs = get_sun(times).transform_to(self.__altazframe)
        self.__moonaltazs = get_moon(times).transform_to(self.__altazframe)

        self.__astro_twilight = self.__sunaltazs.alt < -18*u.deg

        self.__dark_delta = self.__delta_midnight[self.__astro_twilight]
        self.__dark_hours = times[self.__astro_twilight]
        self.__dark_altazframe = AltAz(obstime=self.__dark_hours, location=loc)
        
    def print_moon_info(self):
        print("Phase angle: {:.2f}".format(moon_phase_angle(self.__midnight).to(u.degree)))
        print("Illumination: {:.2f}%".format(100*moon_illumination(self.__midnight)))

    def astro_twilight_altitudes_for(self, coords):
        for ra_str, dec_str in coords:
            yield SkyCoord(
                    ra=ra_str, dec=dec_str
                ).transform_to(self.__dark_altazframe).alt
            
    def plot_alt_vs_hours(self, extra_coords = None):
        if extra_coords is not None:
            for ra_str, dec_str in extra_coords:
                plt.plot(self.__dark_delta, SkyCoord(
                    ra=ra_str, dec=dec_str
                ).transform_to(self.__dark_altazframe).alt, 'c:')

        plt.plot(self.__delta_midnight, self.__sunaltazs.alt, color='y', label='Sun')
        plt.plot(self.__delta_midnight, self.__moonaltazs.alt, color='r', label='Moon')

        plt.fill_between(self.__delta_midnight, 0, 90, self.__sunaltazs.alt < -0*u.deg, color='0.5', zorder=0)
        plt.fill_between(self.__delta_midnight, 0, 90, self.__astro_twilight, color='k', zorder=0)

        plt.legend(loc='lower left')
        plt.xlim(-12, 12)
        plt.xticks(np.arange(13)*2 -12)
        plt.ylim(0, 90)
        plt.xlabel('Hours from midnight')
        plt.ylabel('Altitude [deg]')
        plt.show()