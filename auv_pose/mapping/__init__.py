"""Bathymetric mapping.

Depths follow the survey convention: ``sonar_depth`` in the CSVs is a positive
range from the vehicle to the seabed, while the GP is fitted on **negated** depth
so that the surface it models increases upward. :mod:`auv_pose.io.soundings`
applies that sign flip in one place.
"""

from auv_pose.mapping.sonar import bottom_return_range, range_bins
from auv_pose.mapping.svgp import BathymetryMap, SVGPModel, fit_svgp

__all__ = [
    "BathymetryMap",
    "SVGPModel",
    "bottom_return_range",
    "fit_svgp",
    "range_bins",
]
