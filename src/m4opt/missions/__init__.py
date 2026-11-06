"""This module contains builtin settings for supported missions."""

from ._core import Mission
from ._rubin import rubin
from ._ultrasat import ultrasat
from ._uvex import uvex, uvex_downlink_orientation
from ._ztf import ztf
from ._toy_north import toy_north
from ._toy_south import toy_south

__all__ = ("Mission", "rubin", "ultrasat", "uvex", "uvex_downlink_orientation", "ztf", "toy_north", "toy_south")
