"""The navigation state and the chart the estimators work in.

The state is a position, an orientation, a velocity and the two IMU biases, so
it lives on ``(R^3)^4 x SO(3)`` rather than in a vector space. A Gaussian belief
over it is a mean *on* that manifold plus an error state in the tangent space at
the mean -- ``chi = m [+] xi``, ``xi ~ N(0, P)`` -- following Hertzberg et al.,
who introduce exactly this encapsulation and whose worked example is this state.

:func:`boxplus` and :func:`boxminus` are the encapsulation. On velocity and the
biases they are ordinary addition and subtraction; on the pose they are

    x [+] xi  =  (p + xi_p,  R exp(xi_R))
    x [-] x'  =  (p - p',    log(R'^T R))

which is a **right** perturbation -- the rotation increment is applied in the
frame the orientation already describes. That matches
:func:`~auv_pose.estimation.quaternion.quat_multiply`, whose Hamilton product
composes left-to-right in the body frame, so ``boxplus`` is one multiply with no
transposes to get backwards.

Frames and signs are as :mod:`auv_pose.estimation` documents them: a z-up world,
attitude as a scalar-first quaternion rotating body into world.
"""

from collections.abc import Sequence
from typing import NamedTuple, Self

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

__all__ = [
  "ACCEL_BIAS",
  "DOF",
  "GYRO_BIAS",
  "POSITION",
  "ROTATION",
  "VELOCITY",
  "ManifoldGaussian",
  "NavState",
  "boxminus",
  "boxplus",
  "covariance_transport",
  "manifold_mean",
]

#: Where each component lives in a tangent vector. The order is the paper's,
#: ``xi = (xi_p, xi_R, xi_v, xi_g, xi_a)``, and every covariance in this package
#: is blocked this way.
POSITION = slice(0, 3)
ROTATION = slice(3, 6)
VELOCITY = slice(6, 9)
GYRO_BIAS = slice(9, 12)
ACCEL_BIAS = slice(12, 15)

#: Degrees of freedom of the state, and so the side of every covariance here.
DOF = 15

_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


class NavState(NamedTuple):
  """A point on the state manifold.

  :param position: World position, metres, shape ``(3,)``. The world is z-up,
      so a vehicle 65 m down has ``position[2] == -65``.
  :param attitude: Scalar-first unit quaternion rotating body into world.
  :param velocity: World velocity, m/s, shape ``(3,)``.
  :param gyro_bias: Gyroscope bias in the body frame, rad/s, shape ``(3,)``.
  :param accel_bias: Accelerometer bias in the body frame, m/s^2, shape ``(3,)``.
  """

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


class ManifoldGaussian(NamedTuple):
  """A Gaussian belief over :class:`NavState`.

  The counterpart of :class:`~auv_pose.estimation.typing.GaussianState` for a
  state that is not a vector: the mean is a point on the manifold and the
  covariance lives in the tangent space **at that mean**, so a covariance is
  only meaningful alongside the mean it was built at.

  :param mean: Belief mean.
  :param cov: Error-state covariance, shape ``(15, 15)``, blocked by the
      module-level slices.
  """

  mean: NavState
  cov: NumpyArray


def boxplus(state: NavState, xi: ArrayLike) -> NavState:
  """Apply a tangent-space increment to a state.

  :param state: Point to perturb.
  :param xi: Increment, shape ``(15,)``, blocked by the module-level slices.
  :return: The perturbed state.
  """
  xi = np.asarray(xi, dtype=float)
  if xi.shape != (DOF,):
    raise ValueError(f"expected a ({DOF},) increment, got {xi.shape}")

  return NavState(
    position=state.position + xi[POSITION],
    attitude=quat_normalize(
      quat_multiply(state.attitude, quat_exp(xi[ROTATION]))
    ),
    velocity=state.velocity + xi[VELOCITY],
    gyro_bias=state.gyro_bias + xi[GYRO_BIAS],
    accel_bias=state.accel_bias + xi[ACCEL_BIAS],
  )


def boxminus(state: NavState, reference: NavState) -> NumpyArray:
  """The tangent-space increment taking ``reference`` to ``state``.

  Inverse of :func:`boxplus`: ``boxplus(m, boxminus(x, m))`` is ``x`` and
  ``boxminus(boxplus(m, xi), m)`` is ``xi``.

  :param state: The state to express.
  :param reference: The point whose tangent space to express it in.
  :return: Increment, shape ``(15,)``.

  Note:
      Single-valued only while the two attitudes are less than a half turn
      apart, which is what :func:`~auv_pose.estimation.quaternion.quat_log`
      can invert. For a sigma-point cloud that holds as long as the attitude
      covariance is sane; when it stops holding, the belief is already broken
      and the caller should say so rather than smooth over it.
  """
  return np.concatenate(
    [
      state.position - reference.position,
      quat_log(
        quat_multiply(quat_conjugate(reference.attitude), state.attitude)
      ),
      state.velocity - reference.velocity,
      state.gyro_bias - reference.gyro_bias,
      state.accel_bias - reference.accel_bias,
    ]
  )


def covariance_transport(xi: ArrayLike) -> NumpyArray:
  """Jacobian carrying a covariance along a :func:`boxplus` correction.

  A covariance computed in the tangent space at ``m`` is not the covariance in
  the tangent space at ``m [+] xi``; the two charts differ by the right
  Jacobian of the exponential map. Conjugating by this matrix moves it across,
  which matters once a correction is degrees rather than milliradians -- the
  first few updates of a run usually are.

  Only the rotation block is affected. The four vector blocks are flat, so
  their charts coincide everywhere.

  :param xi: The increment the state was moved by, shape ``(15,)``.
  :return: ``(15, 15)`` block-diagonal Jacobian.
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
  """Weighted intrinsic mean of points on the manifold.

  The mean of a set of orientations is not the mean of their coordinates, so
  this iterates to the point whose weighted tangent vectors sum to zero --
  Hauberg et al.'s unscented mean, and the ``E[chi]`` the prediction step needs.

  :param states: Points to average.
  :param weights: Weight per point; should sum to one.
  :param initial: Where to start. Defaults to the first state, which for a
      sigma-point set is the propagated previous mean and so is already within
      the nonlinearity of the answer.
  :param max_iter: Cap on iterations.
  :param tol: Stop once the largest component of the increment is below this.
  :return: The weighted mean.
  :raises ValueError: If the iteration has not converged in the rotation block
      by ``max_iter``, which means the cloud is wider than the chart can
      describe.

  Note:
      The twelve vector components are exact after one iteration, since their
      update is linear; only the attitude actually iterates, and it typically
      takes two to four passes. That is why an iterative mean costs almost
      nothing here.
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
      delta += weight * boxminus(state, mean)

    mean = boxplus(mean, delta)

    if np.max(np.abs(delta)) < tol:
      return mean

  if np.max(np.abs(delta[ROTATION])) > 1e-6:
    raise ValueError(
      f"manifold mean did not converge in {max_iter} iterations; "
      f"attitude increment still {np.max(np.abs(delta[ROTATION])):.3e} rad"
    )

  return mean
