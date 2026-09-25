"""Sigma points, and the moments they carry across a nonlinearity.

The unscented transform represents a Gaussian by a small set of weighted points,
pushes those through the nonlinearity, and reads the moments back off the
result. Unlike linearising, it never needs a Jacobian -- which is what makes it
usable against a map whose mean is a Gaussian process, where no Jacobian is
available in closed form.

Nothing here knows about the state manifold beyond the ``boxplus``/``boxminus``
pair it is handed, so the same rule runs on a plain vector space. That is
deliberate: the sharpest available test of a sigma-point rule is that it
reproduces the *exact* moments of an affine map, and that test only exists in a
vector space.

See Hauberg et al. for the transform on a manifold, and Barfoot section 4.2.9
for the filter built on it.
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

  :param alpha: Spread. The points sit at ``alpha * sqrt(n + kappa)`` standard
      deviations.
  :param beta: Prior on the shape of the distribution; ``2`` is optimal for a
      Gaussian and only affects the zeroth covariance weight.
  :param kappa: Secondary scaling, conventionally ``0`` or ``3 - n``.

  Note:
      **The default is ``alpha = 1``, not the textbook ``1e-3``**, and the
      difference matters here more than it usually does. The outer points are
      what query the bathymetry map, and the spread of the residual they
      produce is the term that tells the update how curved the seabed is within
      the pose uncertainty. At ``alpha = 1e-3`` every point lands on the same
      patch of seabed to within floating point, that spread cancels to zero,
      and the update credits each sounding with precision it does not have --
      which is the exact overconfidence the method exists to avoid.

      The cost is a wide cloud: at ``n = 15`` the points sit ``sqrt(15)``, or
      about 3.9, standard deviations out. If the attitude covariance grows
      enough that outer soundings leave the surveyed map, lower ``alpha`` --
      but then check that the residual spread has not collapsed with it.
  """

  alpha: float = 1.0
  beta: float = 2.0
  kappa: float = 0.0

  def scaling(self, n: int) -> float:
    """``lambda``, the scaled spread parameter, for an ``n``-dimensional state."""
    return self.alpha**2 * (n + self.kappa) - n

  def weights(self, n: int) -> tuple[NumpyArray, NumpyArray]:
    """Mean and covariance weights for ``2n + 1`` points.

    :param n: State dimension.
    :return: ``(weights_mean, weights_cov)``, each shape ``(2n + 1,)``.
    """
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


#: The rule every caller gets unless it says otherwise. A module-level
#: singleton rather than a default argument, since a call in a signature is
#: evaluated once at import and is easy to mistake for a fresh value.
DEFAULT_RULE = SigmaRule()


def matrix_sqrt(cov: ArrayLike) -> NumpyArray:
  """A factor ``L`` with ``L @ L.T`` equal to ``cov``.

  Tries a Cholesky factorisation, then the same with growing jitter, then falls
  back to an eigendecomposition with the eigenvalues clipped at zero.

  The fallback is not just belt and braces. A covariance here is legitimately
  singular whenever a direction is perfectly known -- a bias pinned in a test,
  or a state initialised from truth -- and Cholesky refuses those outright.
  Clipping before the square root also means no negative ever reaches ``sqrt``,
  which matters because this package runs with ``RuntimeWarning`` as an error.

  :param cov: Symmetric positive semi-definite matrix, shape ``(n, n)``.
  :return: Lower-triangular or symmetric factor, shape ``(n, n)``.
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
  """Tangent-space offsets of the sigma points about a mean.

  The mean itself is row zero, so the offsets can be reused directly as
  ``boxminus(point, mean)`` rather than recovered afterwards -- which is both
  cheaper and exact, where a recovered value would carry the rounding of a
  quaternion round trip.

  :param cov: Error-state covariance, shape ``(n, n)``.
  :param rule: Placement and weighting.
  :return: Offsets, shape ``(2n + 1, n)``, row zero all zeros.
  """
  cov = np.asarray(cov, dtype=float)
  n = cov.shape[0]

  spread = np.sqrt(n + rule.scaling(n))
  factor = matrix_sqrt(cov) * spread

  # Columns of the factor are the perturbation directions, so transpose to get
  # one offset per row.
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
  """Intrinsic weighted mean under an arbitrary chart.

  The vector-space special case of
  :func:`~auv_pose.estimation.manifold.manifold_mean`, generic in its chart so
  the same code serves both the state manifold and a plain ``R^n`` under
  addition and subtraction.

  :param points: Points to average.
  :param weights: Weight per point.
  :param boxplus: Applies an increment to a point.
  :param boxminus: The increment between two points.
  :param initial: Starting point; defaults to the first.
  :param max_iter: Cap on iterations.
  :param tol: Stop once the largest increment component falls below this.
  :return: The weighted mean.
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
  """Covariance of a set of tangent vectors.

  :param offsets: One tangent vector per row, shape ``(m, n)``. These must
      already be taken about the mean.
  :param weights_cov: Covariance weight per row, shape ``(m,)``.
  :return: Covariance, shape ``(n, n)``, symmetrised.
  """
  offsets = np.asarray(offsets, dtype=float)
  weights_cov = np.asarray(weights_cov, dtype=float)

  cov = (offsets * weights_cov[:, None]).T @ offsets
  return 0.5 * (cov + cov.T)


def cross_moments(
  left: ArrayLike, right: ArrayLike, weights_cov: ArrayLike
) -> NumpyArray:
  """Cross-covariance between two sets of tangent vectors.

  :param left: One vector per row, shape ``(m, a)``, taken about their mean.
  :param right: One vector per row, shape ``(m, b)``, taken about theirs.
  :param weights_cov: Covariance weight per row, shape ``(m,)``.
  :return: Cross-covariance, shape ``(a, b)``.
  """
  left = np.asarray(left, dtype=float)
  right = np.asarray(right, dtype=float)
  weights_cov = np.asarray(weights_cov, dtype=float)

  return (left * weights_cov[:, None]).T @ right
