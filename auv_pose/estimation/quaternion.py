"""Quaternion algebra and SO(3) helpers.

Quaternions are unit, scalar-first ``[w, x, y, z]``, rotating body into world.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def quat_normalize(q: ArrayLike) -> NDArray[np.float64]:
  """Scale ``q`` to unit norm."""
  q = np.asarray(q, dtype=float)
  norm = np.linalg.norm(q)
  if norm == 0.0:
    raise ValueError("cannot normalize a zero quaternion")
  return q / norm


def quat_multiply(q: ArrayLike, r: ArrayLike) -> NDArray[np.float64]:
  """Hamilton product; ``quat_multiply(q, dq)`` applies ``dq`` in body."""
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
  """Exponential map: rotation vector (radians) to unit quaternion.

  Near zero the half-angle sinc uses its series, keeping the derivative smooth
  at the identity for sigma points.
  """
  rotvec = np.asarray(rotvec, dtype=float)
  theta = float(np.linalg.norm(rotvec))

  if theta < 1e-6:
    # sin(theta/2)/theta, expanded.
    half_sinc = 0.5 - theta**2 / 48.0 + theta**4 / 3840.0
  else:
    half_sinc = np.sin(theta / 2.0) / theta

  return np.concatenate(([np.cos(theta / 2.0)], rotvec * half_sinc))


def quat_log(q: ArrayLike) -> NDArray[np.float64]:
  """Logarithmic map: unit quaternion to rotation vector, norm in ``[0, pi]``.

  ``q`` is flipped to ``w >= 0`` so rotations past a half turn do not come back
  as nearly ``2 pi`` about the opposite axis.
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


def so3_right_jacobian(rotvec: ArrayLike) -> NDArray[np.float64]:
  """Right Jacobian: ``exp(phi + dphi) ~= exp(phi) exp(Jr(phi) dphi)``.

  The series threshold of ``1e-2`` is deliberate: the closed form loses
  precision to cancellation well above ``1e-6``.
  """
  rotvec = np.asarray(rotvec, dtype=float)
  theta = float(np.linalg.norm(rotvec))
  W = skew(rotvec)

  if theta < 1e-2:
    # (1 - cos t)/t^2 and (t - sin t)/t^3, expanded.
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
  """Inverse of :func:`quat_to_rotmat`, by Shepperd's method."""
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
  """Smallest angle between two orientations, radians in ``[0, pi]``."""
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
