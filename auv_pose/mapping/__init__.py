"""Bathymetric mapping.

Survey CSVs record ``x, y, z``: where a beam struck the seabed, in the world
frame, with ``z`` increasing upward. The GP models that elevation directly.
:mod:`auv_pose.io.soundings` owns the schema.

The map is :class:`~auv_pose.mapping.vecchia.VecchiaMap`, a Vecchia-approximated
GP with a Matern-5/2 kernel (:mod:`~auv_pose.mapping.kernels`) conditioned on
maximin-ordered neighbours (:mod:`~auv_pose.mapping.ordering`). The SVGP in
:mod:`~auv_pose.mapping.svgp` is kept as the baseline it is scored against.
"""
