"""The navigation state on ``(R^3)^4 x SO(3)`` and its tangent-space chart.

The rotation chart is a right perturbation: ``x + xi = (p + xi_p, R exp(xi_R))``
and ``x - x' = (p - p', log(R'^T R))``.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from auv_pose.estimation.quaternion import (
  quat_conjugate,
  quat_exp,
  quat_log,
  quat_multiply,
  quat_normalize,
  quat_to_rotmat,
  so3_right_jacobian,
)
from auv_pose.estimation.typing import NumpyArray

#: Tangent-vector layout, ``xi = (xi_p, xi_R, xi_v, xi_g, xi_a)``; every
#: covariance in this package is blocked this way.
POSITION = slice(0, 3)
ROTATION = slice(3, 6)
VELOCITY = slice(6, 9)
GYRO_BIAS = slice(9, 12)
ACCEL_BIAS = slice(12, 15)

DOF = 15

_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


@dataclass(frozen=True, eq=False)
class NavState:
  """A point on the state manifold.

  :param position: World position, metres, z up (depth is negative).
  :param attitude: Scalar-first unit quaternion rotating body into world.
  :param velocity: World velocity, m/s.
  :param gyro_bias: Body-frame gyroscope bias, rad/s.
  :param accel_bias: Body-frame accelerometer bias, m/s^2.

  ``state + xi`` applies a ``(15,)`` tangent increment and ``state - other``
  is the increment taking ``other`` to ``state``.
  """

  # Refuse NumPy's broadcasting, so ``xi + state`` fails rather than looping.
  __array_ufunc__ = None

  position: NumpyArray
  attitude: NumpyArray
  velocity: NumpyArray
  gyro_bias: NumpyArray
  accel_bias: NumpyArray

  @property
  def rotation(self) -> NumpyArray:
    """Body-to-world rotation matrix for :attr:`attitude`."""
    return quat_to_rotmat(self.attitude)

  @classmethod
  def at_rest(
    cls,
    position: ArrayLike = (0.0, 0.0, 0.0),
    attitude: ArrayLike = _IDENTITY_QUAT,
    velocity: ArrayLike = (0.0, 0.0, 0.0),
  ) -> Self:
    """A state with both biases zero, for starting a run or a test."""
    return cls(
      position=np.asarray(position, dtype=float),
      attitude=quat_normalize(attitude),
      velocity=np.asarray(velocity, dtype=float),
      gyro_bias=np.zeros(3),
      accel_bias=np.zeros(3),
    )

  def __add__(self, xi: ArrayLike) -> Self:
    xi = np.asarray(xi, dtype=float)
    if xi.shape != (DOF,):
      raise ValueError(f"expected a ({DOF},) increment, got {xi.shape}")
    return type(self)(
      position=self.position + xi[POSITION],
      attitude=quat_normalize(
        quat_multiply(self.attitude, quat_exp(xi[ROTATION]))
      ),
      velocity=self.velocity + xi[VELOCITY],
      gyro_bias=self.gyro_bias + xi[GYRO_BIAS],
      accel_bias=self.accel_bias + xi[ACCEL_BIAS],
    )

  def __sub__(self, other: "NavState") -> NumpyArray:
    """Single-valued while the attitudes are less than a half turn apart."""
    return np.concatenate(
      [
        self.position - other.position,
        quat_log(quat_multiply(quat_conjugate(other.attitude), self.attitude)),
        self.velocity - other.velocity,
        self.gyro_bias - other.gyro_bias,
        self.accel_bias - other.accel_bias,
      ]
    )


@dataclass(frozen=True, eq=False)
class ManifoldGaussian:
  """A Gaussian belief over :class:`NavState`.

  :param cov: ``(15, 15)`` covariance in the tangent space at ``mean``; it is
      meaningless with any other mean.
  """

  mean: NavState
  cov: NumpyArray


def covariance_transport(xi: ArrayLike) -> NumpyArray:
  """Jacobian moving a covariance from the chart at ``m`` to ``m + xi``.

  Only the rotation block differs from identity (the SO(3) right Jacobian);
  it matters once corrections reach degrees.
  """
  xi = np.asarray(xi, dtype=float)
  jacobian = np.eye(DOF)
  jacobian[ROTATION, ROTATION] = so3_right_jacobian(xi[ROTATION])
  return jacobian


def manifold_mean(
  states: Sequence[NavState],
  weights: ArrayLike,
  initial: NavState | None = None,
  max_iter: int = 10,
  tol: float = 1e-12,
) -> NavState:
  """Weighted intrinsic mean: where weighted tangent vectors sum to zero.

  :param weights: Weight per state; should sum to one.
  :param initial: Starting point; defaults to ``states[0]``.
  :param tol: Stop once the largest increment component is below this.
  :raises ValueError: If the attitude has not converged by ``max_iter``.
  """
  weights = np.asarray(weights, dtype=float)
  if len(states) != len(weights):
    raise ValueError(f"{len(states)} states against {len(weights)} weights")
  if len(states) == 0:
    raise ValueError("cannot average an empty set of states")

  mean = states[0] if initial is None else initial
  delta = np.zeros(DOF)

  for _ in range(max_iter):
    delta = np.zeros(DOF)
    for state, weight in zip(states, weights):
      delta += weight * (state - mean)

    mean = mean + delta

    if np.max(np.abs(delta)) < tol:
      return mean

  if np.max(np.abs(delta[ROTATION])) > 1e-6:
    raise ValueError(
      f"manifold mean did not converge in {max_iter} iterations; "
      f"attitude increment still {np.max(np.abs(delta[ROTATION])):.3e} rad"
    )

  return mean
