import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation
from regions import RectangleSkyRegion
from synphot import Empirical1D, SpectralElement

from ...constraints import (
    AtNightConstraint,
    MoonSeparationConstraint,
   AirmassConstraint,
)
from ...dynamics import EigenAxisSlew
from ...observer import EarthFixedObserverLocation
from ...skygrid import healpix
from ...synphot import Detector
from ...synphot.background import SkyBackground, ZodiacalBackground
from .._core import Mission
from .constraints import AvoidSkyRegionConstraint


def _gaussian_band(center_angstrom, sigma_angstrom, peak) -> SpectralElement:
    """Toy bandpass fora toy model."""
    wave = (
        np.arange(
            center_angstrom - 5 * sigma_angstrom,
            center_angstrom + 5 * sigma_angstrom,
            sigma_angstrom / 10,
        )
        * u.AA
    )
    trans = peak * np.exp(-0.5 * ((wave.value - center_angstrom) / sigma_angstrom) ** 2)
    return SpectralElement(Empirical1D, points=wave, lookup_table=trans)


toy_south = Mission(
    name = "toy_south",
    fov = RectangleSkyRegion(
        center=SkyCoord(0 * u.deg, 0 * u.deg), width= 2.5 * u.deg, height= 4 * u.deg
    ),
    constraints = (
        AtNightConstraint.twilight_astronomical()      # time-only: prunes the time axis first
        & AirmassConstraint(2.0)
        & MoonSeparationConstraint(30 * u.deg)
        & AvoidSkyRegionConstraint(SkyCoord(266.4 * u.deg, -29.0 * u.deg), 10 * u.deg) # avoid a random place in the sky (sgr A*)
    ),
    # Rubin like location (from rubin community)
    observer_location = EarthFixedObserverLocation(
        EarthLocation.from_geocentric(1818939, -5208471, -3195171.4, unit='m')
    ),
    slew = EigenAxisSlew(
            max_angular_velocity=2 * u.deg / u.s,
            max_angular_acceleration=0.4 * u.deg / u.s**2,
            settling_time=2 * u.s,
        ),
    # Tesselation skygrid: 10% overlap to include rough edges 
    skygrid=healpix(0.8 * 10 * u.deg**2),
    detector=Detector(
        area=np.pi * np.square(0.5 * 60 * u.cm), # trappist like
        plate_scale=(0.64 * u.arcsec) ** 2,  # trappist like
        bandpasses={
            "g": _gaussian_band(4800, 450, 0.55),
            "r": _gaussian_band(6300, 650, 0.60),
        },
        background=SkyBackground.medium() + ZodiacalBackground(),
        dark_noise=0.1 * u.Hz,
        read_noise=3.0,  
        npix=4 * np.pi, # adapted value for PSF photometry as per m4opt docstring
    ),
)
toy_south.__doc__ = r"""Toy mission with access to the southern sky. (Twin north mission with the same detector also avaiable)"""