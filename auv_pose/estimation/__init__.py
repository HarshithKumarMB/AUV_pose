"""State estimation.

Organised by causality:

``quaternion``
    Rotation algebra.
``wahba``
    Static attitude determination -- observations in, attitude out, no state.
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
    World is **NED** (x north, y east, z down), matching HoloOcean's sensors
    when mounted in the ``IMUSocket``. Body is the vehicle frame.

Gravity
    ``G_NED = [0, 0, +9.81]`` -- positive, because z points down. An
    accelerometer measures specific force ``f = a - g``, so at rest it reads
    ``-G_NED`` and kinematic acceleration is recovered by adding gravity back:
    ``a = R @ f_body + G_NED``.

Quaternions
    Scalar-first, ``[w, x, y, z]``, unit norm, rotating body vectors into
    world: ``v_world = R(q) @ v_body``.
"""

from auv_pose.estimation.filters import (
  AttitudeFilter,
  ConstantVelocityEKF,
  Filter,
  position,
  velocity,
)
from auv_pose.estimation.quaternion import (
  G_NED,
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
from auv_pose.estimation.wahba import (
  accel_weight,
  wahba_davenport,
  wahba_svd,
)

__all__ = [
  "G_NED",
  "AttitudeFilter",
  "ConstantVelocityEKF",
  "Filter",
  "GaussianState",
  "Measurement",
  "Step",
  "StrapdownIntegrator",
  "accel_weight",
  "position",
  "quat_conjugate",
  "quat_from_gyro",
  "quat_multiply",
  "quat_normalize",
  "quat_to_rotmat",
  "rotmat_to_quat",
  "rts_smooth",
  "skew",
  "velocity",
  "wahba_davenport",
  "wahba_svd",
]
