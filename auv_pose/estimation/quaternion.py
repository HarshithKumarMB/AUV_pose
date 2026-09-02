"""Quaternion algebra and small rotation helpers.

Quaternions are scalar-first ``[w, x, y, z]``, unit norm, and rotate body vectors
into the world frame. See :mod:`auv_pose.estimation` for the frame conventions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

GRAVITY = 9.81

#: Gravity in a **z-down** world frame such as NED. Positive third component.
GRAVITY_NED: NDArray[np.float64] = np.array([0.0, 0.0, GRAVITY])

#: Gravity in a **z-up** world frame such as NWU or ENU. This is the one
#: HoloOcean's world uses -- its PoseSensor reports z increasing upward, so a
#: vehicle 65 m under the surface sits at ``z = -65``. Getting this backwards
#: does not blow up at rest, because an accelerometer read in a z-down *body*
#: frame cancels a z-down gravity vector exactly; it silently mirrors the
#: recovered acceleration in y and z instead.
GRAVITY_NWU: NDArray[np.float64] = np.array([0.0, 0.0, -GRAVITY])

__all__ = [
  "GRAVITY",
  "GRAVITY_NED",
  "GRAVITY_NWU",
  "quat_angle",
  "quat_conjugate",
  "quat_from_gyro",
  "quat_multiply",
  "quat_normalize",
  "quat_to_rotmat",
  "rotmat_to_quat",
  "skew",
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


def quat_from_gyro(omega: ArrayLike, dt: float) -> NDArray[np.float64]:
  """Rotation increment from an angular rate held over ``dt``.

  Args:
      omega: Body angular rate, rad/s.
      dt: Interval, seconds.

  Returns:
      Unit quaternion for the rotation through ``|omega| * dt`` about
      ``omega / |omega|``. Identity when the rotation is negligible.
  """
  omega = np.asarray(omega, dtype=float)
  rate = np.linalg.norm(omega)
  theta = rate * dt
  if abs(theta) < 1e-8:
    return np.array([1.0, 0.0, 0.0, 0.0])

  axis = omega / rate
  return np.concatenate(([np.cos(theta / 2.0)], axis * np.sin(theta / 2.0)))


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
