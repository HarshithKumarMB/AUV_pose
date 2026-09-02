"""Static attitude determination from vector observations.

Solves Wahba's problem two independent ways -- SVD and Davenport's q-method --
so the pair can be cross-checked. Disagreement between them signals a degenerate
observation set, such as all reference vectors being parallel.

Memoryless: these take observations and return an attitude, with no state
carried between calls. The recursive estimator that uses them is
:class:`auv_pose.smoothing.filters.AttitudeFilter`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from auv_pose.smoothing.quaternion import quat_normalize, rotmat_to_quat

__all__ = [
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
