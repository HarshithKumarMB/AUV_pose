"""State estimation.

Organised by causality:

``quaternion``
    Rotation algebra.
``strapdown``
    Open-loop inertial propagation. No correction, no uncertainty.
``filters``
    Causal recursive estimators. Consume observations in order, never look
    ahead, and record each cycle.
``smoothers``
    Non-causal. Consume a completed record and use the future to improve the
    past.

Frame and sign conventions, assumed throughout:

Frames
    The world frame is whatever the caller's attitude quaternion rotates into,
    and gravity is a parameter rather than a global. For HoloOcean that world
    is **z-up** (NWU): its PoseSensor reports z increasing upward, so a vehicle
    65 m down sits at ``z = -65``. Body is the vehicle frame, and a sensor in
    the ``IMUSocket`` reports in a z-down body frame -- which is why the
    orientation used to rotate IMU readings must come from that same socket.

Gravity
    Pass ``GRAVITY_NWU = [0, 0, -9.81]`` for a z-up world, ``GRAVITY_NED =
    [0, 0, +9.81]`` for a z-down one. An accelerometer measures specific force
    ``f = a - g``, so at rest it reads ``-g`` and kinematic acceleration is
    recovered by adding gravity back: ``a = R @ f_body + g``.

    Choosing wrongly is silent. A z-down body reading cancels a z-down gravity
    vector exactly, so the vehicle does not fall out of the sky at rest; what
    happens instead is that the recovered acceleration comes out mirrored in y
    and z, and dead reckoning integrates into a frame the map does not share.

Quaternions
    Scalar-first, ``[w, x, y, z]``, unit norm, rotating body vectors into
    world: ``v_world = R(q) @ v_body``.
"""

from auv_pose.estimation.filters import (
  ConstantVelocityEKF,
  Filter,
  position,
  velocity,
)
from auv_pose.estimation.quaternion import (
  GRAVITY,
  GRAVITY_NED,
  GRAVITY_NWU,
  quat_angle,
  quat_conjugate,
  quat_from_gyro,
  quat_multiply,
  quat_normalize,
  quat_to_rotmat,
  rotmat_to_quat,
  skew,
)
from auv_pose.estimation.smoothers import rts_smooth
from auv_pose.estimation.strapdown import StrapdownIntegrator
from auv_pose.estimation.typing import GaussianState, Measurement, Step

__all__ = [
  "GRAVITY",
  "GRAVITY_NED",
  "GRAVITY_NWU",
  "ConstantVelocityEKF",
  "Filter",
  "GaussianState",
  "Measurement",
  "Step",
  "StrapdownIntegrator",
  "gravity_trust",
  "position",
  "quat_angle",
  "quat_conjugate",
  "quat_from_gyro",
  "quat_multiply",
  "quat_normalize",
  "quat_to_rotmat",
  "rotmat_to_quat",
  "rts_smooth",
  "skew",
  "velocity",
]
