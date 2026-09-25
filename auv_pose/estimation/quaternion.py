"""Quaternion algebra and small rotation helpers.

Quaternions are scalar-first ``[w, x, y, z]``, unit norm, and rotate body vectors
into the world frame. See :mod:`auv_pose.estimation` for the frame conventions.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
  "quat_angle",
  "quat_conjugate",
  "quat_exp",
  "quat_from_gyro",
  "quat_log",
  "quat_multiply",
  "quat_normalize",
  "quat_to_rotmat",
  "rotmat_to_quat",
  "skew",
  "so3_right_jacobian",
]


def quat_normalize(q: ArrayLike) -> NDArray[np.float64]:
  """Scale ``q`` to unit norm."""
  q = np.asarray(q, dtype=float)
  norm = np.linalg.norm(q)
  if norm == 0.0:
    raise ValueError("cannot normalize a zero quaternion")
  return q / norm


def quat_multiply(q: ArrayLike, r: ArrayLike) -> NDArray[np.float64]:
  """Hamilton product ``q * r``.

  Composition is left-to-right in the body frame: ``quat_multiply(q, dq)``
  applies ``dq`` in the frame that ``q`` already describes.
  """
  w0, x0, y0, z0 = np.asarray(q, dtype=float)
  w1, x1, y1, z1 = np.asarray(r, dtype=float)
  return np.array(
    [
      w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
      w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
      w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
      w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ]
  )


def quat_conjugate(q: ArrayLike) -> NDArray[np.float64]:
  """Conjugate of ``q``; the inverse for unit quaternions."""
  q = np.asarray(q, dtype=float)
  return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_exp(rotvec: ArrayLike) -> NDArray[np.float64]:
  """Exponential map: a rotation vector to the quaternion it names.

  The rotation is through ``|rotvec|`` radians about ``rotvec / |rotvec|``, so
  this is ``exp`` on :math:`\\mathfrak{so}(3)` written in quaternions.

  Args:
      rotvec: Rotation vector, radians, shape ``(3,)``.

  Returns:
      Unit quaternion, scalar-first.

  Note:
      Below ``1e-6`` radians the half-angle sinc is taken from its series
      rather than as ``sin(theta/2)/theta``. The quotient is 0/0 at the
      identity, and returning the identity wholesale -- as this function's
      predecessor did below ``1e-8`` -- puts a step in the derivative that a
      sigma-point rule walks straight into.
  """
  rotvec = np.asarray(rotvec, dtype=float)
  theta = float(np.linalg.norm(rotvec))

  if theta < 1e-6:
    # sin(theta/2)/theta, expanded. Exact to double precision well past 1e-6.
    half_sinc = 0.5 - theta**2 / 48.0 + theta**4 / 3840.0
  else:
    half_sinc = np.sin(theta / 2.0) / theta

  return np.concatenate(([np.cos(theta / 2.0)], rotvec * half_sinc))


def quat_log(q: ArrayLike) -> NDArray[np.float64]:
  """Logarithmic map: the rotation vector a quaternion names.

  Inverse of :func:`quat_exp` on rotations of less than a half turn.

  Args:
      q: Unit quaternion, scalar-first.

  Returns:
      Rotation vector, radians, shape ``(3,)``, with norm in ``[0, pi]``.

  Note:
      ``q`` and ``-q`` are the same rotation, so the sign is flipped when
      ``w < 0``. Without that a rotation just past a half turn comes back as
      nearly ``2 pi`` about the opposite axis -- the same orientation, but a
      tangent vector far outside the range a covariance in this chart can
      describe.
  """
  q = quat_normalize(q)
  if q[0] < 0.0:
    q = -q

  vec = q[1:]
  norm = float(np.linalg.norm(vec))

  if norm < 1e-10:
    # theta = 2 atan2(norm, w) -> 2 norm / w, so theta/norm -> 2/w.
    return 2.0 * vec / q[0]

  return vec * (2.0 * np.arctan2(norm, q[0]) / norm)


def quat_from_gyro(omega: ArrayLike, dt: float) -> NDArray[np.float64]:
  """Rotation increment from an angular rate held over ``dt``.

  Args:
      omega: Body angular rate, rad/s.
      dt: Interval, seconds.

  Returns:
      Unit quaternion for the rotation through ``|omega| * dt`` about
      ``omega / |omega|``. Identity when the rotation is negligible.
  """
  return quat_exp(np.asarray(omega, dtype=float) * dt)


def so3_right_jacobian(rotvec: ArrayLike) -> NDArray[np.float64]:
  """Right Jacobian of the exponential map at ``rotvec``.

  Relates a perturbation of the rotation vector to the body-frame rotation it
  produces: ``exp(phi + dphi) ~= exp(phi) exp(Jr(phi) dphi)``. This is what
  carries gyro noise and gyro-bias error into the attitude block of the
  propagated covariance, and what transports a covariance across a
  :func:`~auv_pose.estimation.manifold.boxplus` correction.

  Args:
      rotvec: Rotation vector, radians, shape ``(3,)``.

  Returns:
      ``(3, 3)`` matrix; the identity at zero.

  Note:
      The series branch cuts in at ``1e-2``, which looks generous and is not.
      Both coefficients are differences of nearly equal numbers -- ``1 - cos``
      and ``theta - sin`` -- and lose precision to cancellation far earlier
      than the usual small-angle intuition suggests. Measured relative error of
      the closed form: 3e-13 at ``theta = 1e-2``, but 3e-8 at ``1e-4`` and
      **9e-5 at 1e-6**. A threshold placed at ``1e-6`` would hand back four
      correct digits just above it while the series below it was exact, and put
      a step of that size into the middle of the propagated covariance.

  See also:
      Barfoot, *State Estimation for Robotics* (2024), eq. 8.82a, which writes
      the same matrix in axis-angle form. Note his convention is the **left**
      Jacobian, with ``J_l(-phi) = J_r(phi)`` (eq. 8.85).
  """
  rotvec = np.asarray(rotvec, dtype=float)
  theta = float(np.linalg.norm(rotvec))
  W = skew(rotvec)

  if theta < 1e-2:
    # (1 - cos t)/t^2 and (t - sin t)/t^3, expanded. Truncation here is below
    # 1e-16 at the threshold, against the 3e-13 of the closed form above it.
    coeff_w = 0.5 - theta**2 / 24.0 + theta**4 / 720.0
    coeff_ww = 1.0 / 6.0 - theta**2 / 120.0 + theta**4 / 5040.0
  else:
    coeff_w = (1.0 - np.cos(theta)) / theta**2
    coeff_ww = (theta - np.sin(theta)) / theta**3

  return np.eye(3) - coeff_w * W + coeff_ww * (W @ W)


def quat_to_rotmat(q: ArrayLike) -> NDArray[np.float64]:
  """Rotation matrix ``R`` such that ``v_world = R @ v_body``."""
  w, x, y, z = quat_normalize(q)
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
  )


def rotmat_to_quat(R: ArrayLike) -> NDArray[np.float64]:
  """Inverse of :func:`quat_to_rotmat`.

  Uses Shepperd's method: pick the branch with the largest divisor so the square
  root never loses precision near a 180-degree rotation.
  """
  R = np.asarray(R, dtype=float)
  q = np.zeros(4)
  trace = np.trace(R)

  if trace > 0:
    s = np.sqrt(trace + 1.0) * 2
    q[0] = 0.25 * s
    q[1] = (R[2, 1] - R[1, 2]) / s
    q[2] = (R[0, 2] - R[2, 0]) / s
    q[3] = (R[1, 0] - R[0, 1]) / s
  else:
    i = int(np.argmax(np.diag(R)))
    if i == 0:
      s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
      q[0] = (R[2, 1] - R[1, 2]) / s
      q[1] = 0.25 * s
      q[2] = (R[0, 1] + R[1, 0]) / s
      q[3] = (R[0, 2] + R[2, 0]) / s
    elif i == 1:
      s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
      q[0] = (R[0, 2] - R[2, 0]) / s
      q[1] = (R[0, 1] + R[1, 0]) / s
      q[2] = 0.25 * s
      q[3] = (R[1, 2] + R[2, 1]) / s
    else:
      s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
      q[0] = (R[1, 0] - R[0, 1]) / s
      q[1] = (R[0, 2] + R[2, 0]) / s
      q[2] = (R[1, 2] + R[2, 1]) / s
      q[3] = 0.25 * s

  return quat_normalize(q)


def quat_angle(q: ArrayLike, r: ArrayLike) -> float:
  """Smallest rotation angle between two orientations, in radians.

  ``q`` and ``-q`` denote the same rotation, so the dot product is taken in
  absolute value -- without that, identical orientations of opposite sign would
  read as a half turn.

  :param q: First orientation.
  :param r: Second orientation.
  :return: Angle in ``[0, pi]``.
  """
  dot = abs(float(np.dot(quat_normalize(q), quat_normalize(r))))
  return 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))


def skew(w: ArrayLike) -> NDArray[np.float64]:
  """Skew-symmetric matrix with ``skew(w) @ v == np.cross(w, v)``."""
  wx, wy, wz = np.asarray(w, dtype=float)
  return np.array(
    [
      [0.0, -wz, wy],
      [wz, 0.0, -wx],
      [-wy, wx, 0.0],
    ]
  )
