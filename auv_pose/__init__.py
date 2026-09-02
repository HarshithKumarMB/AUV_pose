"""Terrain-aided pose estimation for an underwater vehicle.

Algorithms only -- nothing here reads files, plots, or talks to the simulator.
Runnable drivers live in ``experiments/``.

Subpackages:

``auv_pose.estimation``
    State estimation: quaternion algebra, attitude determination, and the
    constant-velocity EKF with an RTS smoother.
``auv_pose.mapping``
    Bathymetry: the sparse variational GP surrogate and sonar range extraction.
``auv_pose.io``
    Extract/transform/load: soundings, model checkpoints, run logs.

Subpackages are deliberately not imported here: ``auv_pose.mapping`` pulls in torch
and gpytorch, and ``auv_pose.io`` pulls in pandas. Importing ``auv_pose.estimation``
should not cost either. Import what you need explicitly.
"""
