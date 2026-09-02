"""Attitude determination from gyro propagation plus vector observations.

Solves Wahba's problem two independent ways -- SVD and Davenport's q-method -- so
the pair can be cross-checked. Disagreement between them signals a degenerate
observation set, such as all reference vectors being parallel.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from auv_pose.smoothing.quaternion import (
  quat_conjugate,
  quat_from_gyro,
  quat_multiply,
  quat_normalize,
  quat_to_rotmat,
  rotmat_to_quat,
)

__all__ = [
  "AttitudeFilter",
  "accel_weight",
  "wahba_davenport",
  "wahba_svd",
]


def _attitude_profile_matrix(
  v_body: Sequence[ArrayLike],
  v_ref: Sequence[ArrayLike],
  weights: ArrayLike,
) -> NDArray[np.float64]:
  """Weighted attitude profile matrix ``B = sum_i w_i * v_body_i v_ref_i^T``.

  The classical Wahba form: its SVD solution is the rotation taking *reference to
  body*, while Davenport's q-method on the same ``B`` yields the *inverse* under
  the scalar-first convention used here. :func:`wahba_svd` transposes to match.
  """
  B = np.zeros((3, 3))
  for vb, vr, w in zip(v_body, v_ref, np.asarray(weights, dtype=float)):
    B += w * np.outer(np.asarray(vb, dtype=float), np.asarray(vr, dtype=float))
  return B


def wahba_svd(
  v_body: Sequence[ArrayLike],
  v_ref: Sequence[ArrayLike],
  weights: ArrayLike,
) -> NDArray[np.float64]:
  """Solve Wahba's problem by singular value decomposition.

  Returns the **body-to-world** rotation best aligning ``v_ref`` with ``v_body``
  in a weighted least-squares sense. The determinant correction keeps the result
  a proper rotation rather than a reflection.

  The SVD of the classical profile matrix yields the world-to-body rotation, so
  it is transposed here to match :func:`wahba_davenport`.
  """
  B = _attitude_profile_matrix(v_body, v_ref, weights)

  U, _, Vt = np.linalg.svd(B)
  R = U @ Vt
  if np.linalg.det(R) < 0:
    Vt = Vt.copy()
    Vt[-1, :] *= -1
    R = U @ Vt

  return rotmat_to_quat(R.T)


def wahba_davenport(
  v_body: Sequence[ArrayLike],
  v_ref: Sequence[ArrayLike],
  weights: ArrayLike,
) -> NDArray[np.float64]:
  """Solve Wahba's problem by Davenport's q-method.

  Returns the **body-to-world** rotation, matching :func:`wahba_svd`.

  The optimal quaternion is the eigenvector of the 4x4 K matrix with the largest
  eigenvalue. Independent of :func:`wahba_svd`, so the two agreeing is meaningful
  evidence that the observation set is well conditioned.
  """
  B = _attitude_profile_matrix(v_body, v_ref, weights)

  S = B + B.T
  sigma = np.trace(B)
  Z = np.array(
    [
      B[1, 2] - B[2, 1],
      B[2, 0] - B[0, 2],
      B[0, 1] - B[1, 0],
    ]
  )

  K = np.zeros((4, 4))
  K[0, 0] = sigma
  K[0, 1:] = Z
  K[1:, 0] = Z
  K[1:, 1:] = S - sigma * np.eye(3)

  eigvals, eigvecs = np.linalg.eigh(K)
  return quat_normalize(eigvecs[:, int(np.argmax(eigvals))])


def accel_weight(acc: ArrayLike) -> float:
  """Trust an accelerometer reading according to how close it is to 1 g.

  A reading far from gravity's magnitude means the vehicle is accelerating, so it
  is a poor vertical reference. ``acc`` is expected normalised to g.
  """
  error = abs(float(np.linalg.norm(acc)) - 1.0)
  return float(np.exp(-5.0 * error))


class AttitudeFilter:
  """Complementary attitude filter over one or more accelerometers.

  Propagates orientation from the gyro, then nudges it toward the attitude that
  Wahba's problem says the accelerometers imply, accumulating a gyro bias estimate
  from the residual.

  Args:
      kp: Proportional gain on the attitude error.
      ki: Integral gain feeding the gyro bias estimate.
      q0: Initial orientation; identity if omitted.
      cross_check: Also solve with :func:`wahba_davenport` and record the
          disagreement in :attr:`solver_disagreement`. Off by default -- it
          doubles the per-update solve cost, including a 4x4 eigendecomposition,
          for a diagnostic rather than a result.
  """

  def __init__(
    self,
    kp: float = 2.0,
    ki: float = 0.05,
    q0: ArrayLike | None = None,
    cross_check: bool = False,
  ) -> None:
    self.q = (
      np.array([1.0, 0.0, 0.0, 0.0]) if q0 is None else quat_normalize(q0)
    )
    self.bias = np.zeros(3)
    self.kp = kp
    self.ki = ki
    self.cross_check = cross_check
    # Gravity direction in NED as a unit vector: down.
    self.g_ref = np.array([0.0, 0.0, 1.0])
    # |q_svd - q_davenport| from the most recent update; only meaningful when
    # cross_check is on. Large values mean a degenerate observation set.
    self.solver_disagreement = 0.0

  def update(
    self,
    gyro: ArrayLike,
    accels: Sequence[ArrayLike],
    dt: float,
  ) -> NDArray[np.float64]:
    """Advance the estimate by one step and return the orientation.

    Args:
        gyro: Body angular rate, rad/s. Bias-corrected internally.
        accels: One reading per accelerometer, in the body frame.
        dt: Interval since the previous call, in **seconds**.
    """
    omega = np.asarray(gyro, dtype=float) - self.bias
    q_pred = quat_normalize(quat_multiply(self.q, quat_from_gyro(omega, dt)))

    v_body: list[NDArray[np.float64]] = []
    v_ref: list[NDArray[np.float64]] = []
    weights: list[float] = []

    for acc in accels:
      acc = np.asarray(acc, dtype=float)
      norm = np.linalg.norm(acc)
      if norm > 1e-6:
        v_body.append(acc / norm)
        v_ref.append(self.g_ref)
        weights.append(accel_weight(acc))

    if not v_body:
      self.q = q_pred
      return self.q

    w = np.asarray(weights, dtype=float)
    w = w / (w.sum() + 1e-6)

    q_svd = wahba_svd(v_body, v_ref, w)

    if self.cross_check:
      q_dav = wahba_davenport(v_body, v_ref, w)
      if np.dot(q_svd, q_dav) < 0:
        q_dav = -q_dav
      self.solver_disagreement = float(np.linalg.norm(q_svd - q_dav))

    # Vector part of the error quaternion is a small-angle attitude error.
    error = quat_multiply(quat_conjugate(q_pred), q_svd)[1:]

    self.bias = self.bias + self.ki * error * dt

    correction = quat_normalize(np.concatenate(([1.0], self.kp * error)))
    self.q = quat_normalize(quat_multiply(q_pred, correction))
    return self.q

  @property
  def rotation(self) -> NDArray[np.float64]:
    """Current orientation as a rotation matrix, body to world."""
    return quat_to_rotmat(self.q)
