"""State estimation.

Frame and sign conventions, fixed here and assumed by every module in this
subpackage:

Frames
    World is **NED** (x north, y east, z down), matching HoloOcean's sensors when
    mounted in the ``IMUSocket``. Body is the vehicle frame.

Gravity
    ``G_NED = [0, 0, +9.81]`` -- positive, because z points down. An accelerometer
    at rest therefore reads ``-G_NED`` in the world frame (specific force), so
    linear acceleration is recovered as ``R @ a_body - G_NED``.

Quaternions
    Scalar-first, ``[w, x, y, z]``, unit norm, rotating body vectors into world:
    ``v_world = R(q) @ v_body``.
"""

from auv_pose.smoothing.attitude import (
    AttitudeFilter,
    accel_weight,
    wahba_davenport,
    wahba_svd,
)
from auv_pose.smoothing.ekf import ConstantVelocityEKF
from auv_pose.smoothing.quaternion import (
    G_NED,
    quat_conjugate,
    quat_from_gyro,
    quat_multiply,
    quat_normalize,
    quat_to_rotmat,
    rotmat_to_quat,
    skew,
)
from auv_pose.smoothing.strapdown import StrapdownIntegrator

__all__ = [
    "G_NED",
    "AttitudeFilter",
    "ConstantVelocityEKF",
    "StrapdownIntegrator",
    "accel_weight",
    "quat_conjugate",
    "quat_from_gyro",
    "quat_multiply",
    "quat_normalize",
    "quat_to_rotmat",
    "rotmat_to_quat",
    "skew",
    "wahba_davenport",
    "wahba_svd",
]
