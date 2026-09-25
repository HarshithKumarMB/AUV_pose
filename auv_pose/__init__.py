"""Terrain-aided pose estimation for an underwater vehicle.

Algorithms only -- nothing here reads files, plots, or talks to the simulator.
Runnable drivers live in ``experiments/``.

Subpackages:

``auv_pose.estimation``
    State estimation: quaternion algebra, the inertial navigation filter,
    and its smoother.
``auv_pose.mapping``
    Bathymetry: the Vecchia GP map, sonar geometry and pass registration.
``auv_pose.io``
    Extract/transform/load: soundings, raw survey logs, map checkpoints.

Nothing is re-exported, here or in the subpackages: import from the module
that defines it.
"""
