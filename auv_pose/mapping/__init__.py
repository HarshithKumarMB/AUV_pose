"""Bathymetric mapping.

Survey CSVs record ``x, y, z``: where a beam struck the seabed, in the world
frame, with ``z`` increasing upward. The GP models that elevation directly.
:mod:`auv_pose.io.soundings` owns the schema.

:mod:`~auv_pose.mapping.octree` reads the same seabed out of the simulator's own
cached octree, which makes it ground truth rather than a second estimate -- a
sounding that disagrees with it is a sensor defect, not terrain.
"""

from auv_pose.mapping.octree import load_surface, top_surface
from auv_pose.mapping.sonar import bottom_return_range, range_bins
from auv_pose.mapping.svgp import BathymetryMap, SVGPModel, fit_svgp

__all__ = [
  "BathymetryMap",
  "SVGPModel",
  "bottom_return_range",
  "fit_svgp",
  "load_surface",
  "range_bins",
  "top_surface",
]
