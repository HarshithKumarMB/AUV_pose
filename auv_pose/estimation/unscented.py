"""Sigma points and the moments they carry across a nonlinearity.

Generic in the chart, so the same rule runs on the manifold and on ``R^n``.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike

from auv_pose.estimation.typing import NumpyArray

T = TypeVar("T")


@dataclass(frozen=True)
class SigmaRule:
  """Placement and weighting of a scaled symmetric sigma-point set.

  :param alpha: Spread; points sit ``alpha * sqrt(n + kappa)`` standard
      deviations out.
  :param beta: ``2`` is optimal for a Gaussian.
  :param kappa: Secondary scaling.

  The default ``alpha = 1`` (not the textbook ``1e-3``) is deliberate: a tiny
  cloud queries one patch of seabed, so the map-residual spread cancels and the
  update becomes overconfident.
  """

  alpha: float = 1.0
  beta: float = 2.0
  kappa: float = 0.0

  def scaling(self, n: int) -> float:
    """The scaled spread parameter ``lambda``."""
    return self.alpha**2 * (n + self.kappa) - n

  def weights(self, n: int) -> tuple[NumpyArray, NumpyArray]:
    """``(weights_mean, weights_cov)`` for ``2n + 1`` points."""
    lambda_ = self.scaling(n)
    denominator = n + lambda_
    if denominator == 0.0:
      raise ValueError(
        f"degenerate sigma rule: n + lambda is zero for n={n}, {self}"
      )

    weights_mean = np.full(2 * n + 1, 1.0 / (2.0 * denominator))
    weights_cov = weights_mean.copy()
    weights_mean[0] = lambda_ / denominator
    weights_cov[0] = weights_mean[0] + (1.0 - self.alpha**2 + self.beta)

    return weights_mean, weights_cov


DEFAULT_RULE = SigmaRule()


def matrix_sqrt(cov: ArrayLike) -> NumpyArray:
  """A factor ``L`` with ``L @ L.T == cov``.

  Falls back from Cholesky to jittered Cholesky to a clipped eigendecomposition,
  because covariances here are legitimately singular (e.g. a pinned bias).
  """
  cov = np.asarray(cov, dtype=float)
  if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
    raise ValueError(f"expected a square matrix, got {cov.shape}")

  symmetric = 0.5 * (cov + cov.T)

  try:
    return np.linalg.cholesky(symmetric)
  except np.linalg.LinAlgError:
    pass

  scale = np.trace(symmetric) / symmetric.shape[0]
  if scale > 0.0:
    for epsilon in (1e-12, 1e-10, 1e-8, 1e-6):
      try:
        return np.linalg.cholesky(
          symmetric + epsilon * scale * np.eye(symmetric.shape[0])
        )
      except np.linalg.LinAlgError:
        continue

  eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
  return eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))


def sigma_offsets(cov: ArrayLike, rule: SigmaRule = DEFAULT_RULE) -> NumpyArray:
  """Sigma-point offsets about the mean, ``(2n + 1, n)``; row zero is the mean.

  These are exactly ``point - mean``; reuse them rather than recomputing.
  """
  cov = np.asarray(cov, dtype=float)
  n = cov.shape[0]

  spread = np.sqrt(n + rule.scaling(n))
  factor = matrix_sqrt(cov) * spread

  return np.vstack([np.zeros((1, n)), factor.T, -factor.T])


def weighted_mean(
  points: Sequence[T],
  weights: ArrayLike,
  boxplus: Callable[[T, NumpyArray], T],
  boxminus: Callable[[T, T], NumpyArray],
  initial: T | None = None,
  max_iter: int = 10,
  tol: float = 1e-12,
) -> T:
  """Intrinsic weighted mean under the chart ``boxplus``/``boxminus``.

  :param initial: Starting point; defaults to ``points[0]``.
  :param tol: Stop once the largest increment component is below this.
  """
  weights = np.asarray(weights, dtype=float)
  mean = points[0] if initial is None else initial

  for _ in range(max_iter):
    deltas = np.stack([boxminus(point, mean) for point in points])
    delta = weights @ deltas

    mean = boxplus(mean, delta)
    if np.max(np.abs(delta)) < tol:
      break

  return mean


def tangent_moments(offsets: ArrayLike, weights_cov: ArrayLike) -> NumpyArray:
  """Weighted covariance ``(n, n)`` of ``(m, n)`` offsets about the mean."""
  offsets = np.asarray(offsets, dtype=float)
  weights_cov = np.asarray(weights_cov, dtype=float)

  cov = (offsets * weights_cov[:, None]).T @ offsets
  return 0.5 * (cov + cov.T)


def cross_moments(
  left: ArrayLike, right: ArrayLike, weights_cov: ArrayLike
) -> NumpyArray:
  """Weighted cross-covariance ``(a, b)`` of centred ``(m, a)``, ``(m, b)``."""
  left = np.asarray(left, dtype=float)
  right = np.asarray(right, dtype=float)
  weights_cov = np.asarray(weights_cov, dtype=float)

  return (left * weights_cov[:, None]).T @ right
