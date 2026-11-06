"""Toy mission with access to the southern sky"""

from typing import override

from astropy import units as u
from astropy.coordinates import SkyCoord

from ...constraints import Constraint


class AvoidSkyRegionConstraint(Constraint):
    """Custom constraints (testing purposes): stay away off a certain `radius` of a certain sky loc (`SkyCoord`)"""

    def __init__(self, center: SkyCoord, radius: u.Quantity[u.physical.angle]):
        self.center = center
        self.radius = radius

    @override
    def __call__(self, observer_location, target_coord, obstime):
        return target_coord.separation(self.center) >= self.radius