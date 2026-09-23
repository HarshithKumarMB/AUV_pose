"""Bathymetric mapping.

Survey CSVs record ``x, y, z``: where a beam struck the seabed, in the world
frame, with ``z`` increasing upward. The GP models that elevation directly.
:mod:`auv_pose.io.soundings` owns the schema.

The map is :class:`~auv_pose.mapping.vecchia.VecchiaMap`, a Vecchia-approximated
GP with a Matern-5/2 kernel (:mod:`~auv_pose.mapping.kernels`) conditioned on
maximin-ordered neighbours (:mod:`~auv_pose.mapping.ordering`). The SVGP in
:mod:`~auv_pose.mapping.svgp` is kept as the baseline it is scored against.

:mod:`~auv_pose.mapping.octree` reads the same seabed out of the simulator's own
cached octree, and :mod:`~auv_pose.mapping.raycast` traces a beam through it.

**The octree is a reference, not ground truth.** It is what the simulator's
sonar was thought to raycast against, and over bare seabed the two agree to
0.035 m. But it holds only landscape: the pipelines lying on the Dam seabed are
absent from it, so the sonar returns 4-5 m short of it over them and is right to.
Changing ``octree_max`` leaves the returns bit-identical, which says the sonar
does not consult it at all. A disagreement is a question, not a verdict.
"""

from auv_pose.mapping.octree import load_surface, top_surface
from auv_pose.mapping.raycast import Heightfield, raycast
from auv_pose.mapping.sonar import range_bins
from auv_pose.mapping.svgp import BathymetryMap, SVGPModel, fit_svgp
from auv_pose.mapping.vecchia import VecchiaMap, fit_vecchia

__all__ = [
  "BathymetryMap",
  "Heightfield",
  "SVGPModel",
  "VecchiaMap",
  "fit_svgp",
  "fit_vecchia",
  "load_surface",
  "range_bins",
  "raycast",
  "top_surface",
]
