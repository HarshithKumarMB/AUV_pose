"""Causal recursive estimators.

A filter is causal: it consumes observations in order and never looks ahead.
The non-causal counterpart lives in :mod:`auv_pose.estimation.smoothers`, which
consumes the history a filter records here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from auv_pose.estimation.quaternion import (
  quat_conjugate,
  quat_from_gyro,
  quat_multiply,
  quat_normalize,
  quat_to_rotmat,
)
from auv_pose.estimation.typing import (
  GaussianState,
  Measurement,
  NumpyArray,
  Step,
)
from auv_pose.estimation.wahba import accel_weight, wahba_davenport, wahba_svd

__all__ = [
  "AttitudeFilter",
  "ConstantVelocityEKF",
  "Filter",
  "position",
  "velocity",
]

_STATE_DIM = 6


class Filter(ABC):
  """Abstract base class for causal recursive estimators.

  Defines the predict/condition interface and records each completed cycle in
  :attr:`history`, which is what :func:`auv_pose.estimation.smoothers.rts_smooth`
  consumes.
  """

  def __init__(self) -> None:
    self.history: list[Step] = []

  @abstractmethod
  def predict(
    self, state: GaussianState, control: ArrayLike, dt: float
  ) -> GaussianState:
    """Propagate the belief forward through the motion model.

    :param state: Current belief.
    :param control: Control input, filter-specific.
    :param dt: Interval in seconds.
    :return: Predicted belief.
    """
    ...

  @abstractmethod
  def condition(self, state: GaussianState, obs: Measurement) -> GaussianState:
    """Update the belief given an observation.

    :param state: Predicted belief.
    :param obs: Observation to condition on.
    :return: Updated belief.
    """
    ...

  @abstractmethod
  def transition(self, dt: float) -> NumpyArray:
    """State transition matrix for a step of length ``dt``.

    :param dt: Interval in seconds.
    :return: Transition matrix, shape ``(n, n)``.
    """
    ...

  def step(
    self,
    state: GaussianState,
    control: ArrayLike,
    dt: float,
    observations: Iterable[Measurement] = (),
  ) -> GaussianState:
    """Run one predict/condition cycle and record it.

    Every observation for the step is folded into a single posterior, which is
    what a tick with a varying number of sensors needs -- and it keeps the
    recorded history one entry per step, as the backward pass requires.

    :param state: Current belief.
    :param control: Control input for the motion model.
    :param dt: Interval in seconds.
    :param observations: Observations available this step; may be empty.
    :return: Belief after the cycle.
    """
    prior = self.predict(state, control, dt)

    posterior = prior
    for obs in observations:
      posterior = self.condition(posterior, obs)

    self.history.append(Step(prior, posterior, self.transition(dt)))
    return posterior


def position(state: GaussianState) -> NumpyArray:
  """Position component of a constant-velocity state.

  :param state: Belief with layout ``[x, y, z, vx, vy, vz]``.
  :return: Position, shape ``(3,)``.
  """
  return state.mean[:3]


def velocity(state: GaussianState) -> NumpyArray:
  """Velocity component of a constant-velocity state.

  :param state: Belief with layout ``[x, y, z, vx, vy, vz]``.
  :return: Velocity, shape ``(3,)``.
  """
  return state.mean[3:]


class ConstantVelocityEKF(Filter):
  """Position/velocity filter with acceleration as a control input.

  State is ``[x, y, z, vx, vy, vz]`` in the world frame. The dynamics are
  linear, so this is an ordinary Kalman filter; the name reflects its role in a
  pipeline where the nonlinearity lives in constructing the measurement rather
  than in the filter.

  :param accel_process_sigma: Standard deviation of unmodelled acceleration in
      m/s^2. Shapes the process noise as ``sigma^2 B B^T``, so it enters
      position and velocity consistently for the timestep.
  """

  def __init__(self, accel_process_sigma: float = 0.5) -> None:
    super().__init__()
    self.accel_process_sigma = accel_process_sigma

    # dt is constant across a run, so F, B and Q are built once and reused.
    self._cached_dt: float | None = None
    self._cached: tuple[NumpyArray, NumpyArray, NumpyArray] | None = None

  @staticmethod
  def initial(
    position: ArrayLike,
    velocity: ArrayLike | None = None,
    cov: ArrayLike | None = None,
  ) -> GaussianState:
    """Build a starting belief.

    :param position: Initial world position, shape ``(3,)``.
    :param velocity: Initial world velocity; at rest if omitted.
    :param cov: Initial covariance; identity if omitted.
    :return: Initial belief.
    """
    mean = np.concatenate(
      [
        np.asarray(position, dtype=float),
        np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float),
      ]
    )
    return GaussianState(
      mean=mean,
      cov=np.eye(_STATE_DIM) if cov is None else np.asarray(cov, dtype=float),
    )

  def _matrices(self, dt: float) -> tuple[NumpyArray, NumpyArray, NumpyArray]:
    """``(F, B, Q)`` for a constant-velocity step of length ``dt``."""
    if dt != self._cached_dt:
      F = np.eye(_STATE_DIM)
      F[:3, 3:] = np.eye(3) * dt

      B = np.zeros((_STATE_DIM, 3))
      B[:3, :] = np.eye(3) * (0.5 * dt**2)
      B[3:, :] = np.eye(3) * dt

      self._cached_dt = dt
      self._cached = (F, B, self.accel_process_sigma**2 * (B @ B.T))

    assert self._cached is not None
    return self._cached

  def transition(self, dt: float) -> NumpyArray:
    """State transition matrix for a step of length ``dt``.

    :param dt: Interval in seconds.
    :return: Transition matrix, shape ``(6, 6)``.
    """
    return self._matrices(dt)[0]

  def predict(
    self, state: GaussianState, control: ArrayLike, dt: float
  ) -> GaussianState:
    """Propagate under world-frame acceleration.

    :param state: Current belief.
    :param control: Linear acceleration in the world frame in m/s^2, with
        gravity already removed. See :mod:`auv_pose.estimation` for the
        convention.
    :param dt: Interval in seconds.
    :return: Predicted belief.
    """
    F, B, Q = self._matrices(dt)
    u = np.asarray(control, dtype=float)

    return GaussianState(
      mean=F @ state.mean + B @ u,
      cov=F @ state.cov @ F.T + Q,
    )

  def condition(self, state: GaussianState, obs: Measurement) -> GaussianState:
    """Correct the belief with a linear-Gaussian observation.

    Stack simultaneous observations into one :class:`Measurement` rather than
    conditioning twice. For independent measurements the two are algebraically
    equivalent, but the stacked form makes the whole measurement model visible
    in one place.

    :param state: Predicted belief.
    :param obs: Observation to condition on.
    :return: Updated belief.
    """
    H = np.atleast_2d(np.asarray(obs.H, dtype=float))
    R = np.atleast_2d(np.asarray(obs.R, dtype=float))
    z = np.asarray(obs.z, dtype=float).reshape(H.shape[0])

    innovation = z - H @ state.mean
    S = H @ state.cov @ H.T + R
    K = state.cov @ H.T @ np.linalg.inv(S)

    # Joseph form: stays symmetric positive-definite even when K is not the
    # exact optimal gain, which matters once R is tuned by hand.
    I_KH = np.eye(_STATE_DIM) - K @ H

    return GaussianState(
      mean=state.mean + K @ innovation,
      cov=I_KH @ state.cov @ I_KH.T + K @ R @ K.T,
    )


class AttitudeFilter:
  """Complementary attitude filter over one or more accelerometers.

  Propagates orientation from the gyro, then nudges it toward the attitude that
  Wahba's problem says the accelerometers imply, accumulating a gyro bias
  estimate from the residual.

  Deliberately not a :class:`Filter`: its state is a quaternion plus a bias
  rather than a Gaussian, and its correction is a Wahba solve rather than a
  linear-Gaussian update. Forcing one hierarchy over both would misrepresent it.

  :param kp: Proportional gain on the attitude error.
  :param ki: Integral gain feeding the gyro bias estimate.
  :param q0: Initial orientation; identity if omitted.
  :param cross_check: Also solve with :func:`wahba_davenport` and record the
      disagreement in :attr:`solver_disagreement`. Off by default -- it doubles
      the per-update solve cost, including a 4x4 eigendecomposition, for a
      diagnostic rather than a result.
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
  ) -> NumpyArray:
    """Advance the estimate by one step.

    :param gyro: Body angular rate in rad/s. Bias-corrected internally.
    :param accels: One reading per accelerometer, in the body frame.
    :param dt: Interval since the previous call, in seconds.
    :return: Updated orientation quaternion.
    """
    omega = np.asarray(gyro, dtype=float) - self.bias
    q_pred = quat_normalize(quat_multiply(self.q, quat_from_gyro(omega, dt)))

    v_body: list[NumpyArray] = []
    v_ref: list[NumpyArray] = []
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
  def rotation(self) -> NumpyArray:
    """Current orientation as a rotation matrix, body to world."""
    return quat_to_rotmat(self.q)
