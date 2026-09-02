"""Causal recursive estimators.

A filter is causal: it consumes observations in order and never looks ahead.
The non-causal counterpart lives in :mod:`auv_pose.estimation.smoothers`, which
consumes the history a filter records here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike

from auv_pose.estimation.quaternion import (
  G_NED,
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

__all__ = [
  "AttitudeFilter",
  "ConstantVelocityEKF",
  "Filter",
  "gravity_trust",
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


def gravity_trust(accel_body: ArrayLike, sharpness: float = 5.0) -> float:
  """How far to trust an accelerometer as a gravity reference.

  An accelerometer measures specific force, so it points along gravity only
  while the vehicle is unaccelerated. The further its magnitude departs from
  ``g``, the more of what it reads is manoeuvre rather than gravity.

  :param accel_body: Body-frame specific force in m/s^2.
  :param sharpness: How quickly trust falls away from 1 g.
  :return: Weight in ``(0, 1]``, exactly 1 at ``|a| = g``.
  """
  magnitude = float(np.linalg.norm(accel_body))
  gravity = float(np.linalg.norm(G_NED))
  return float(np.exp(-sharpness * abs(magnitude / gravity - 1.0)))


class AttitudeFilter:
  """Complementary filter correcting gyro drift against gravity.

  Propagates orientation from the gyro and nudges it so that the accelerometer
  agrees with where the filter thinks "down" is, accumulating the residual into
  a gyro bias estimate.

  **Tilt only, by construction.** The correction is the cross product of the
  measured and predicted gravity directions in the body frame, which is
  orthogonal to gravity and so has no component about the vertical. Gravity
  carries no heading information, and a correction that pretended otherwise
  would corrupt yaw: solving Wahba's problem with a single gravity observation
  returns an attitude whose yaw error equals the true yaw exactly, because it
  always answers "zero heading". Roll and pitch are bounded here; heading still
  drifts with the gyro and needs a magnetometer, a DVL, or terrain to bound it.

  :param kp: Proportional gain on the tilt error, 1/s.
  :param ki: Integral gain feeding the gyro bias estimate, 1/s^2.
  :param q0: Initial orientation; identity if omitted.
  """

  def __init__(
    self,
    kp: float = 1.0,
    ki: float = 0.05,
    q0: ArrayLike | None = None,
  ) -> None:
    self.q = (
      np.array([1.0, 0.0, 0.0, 0.0]) if q0 is None else quat_normalize(q0)
    )
    self.bias = np.zeros(3)
    self.kp = kp
    self.ki = ki
    # Trust placed in the accelerometer on the most recent update, for
    # diagnostics: near 0 means the vehicle was manoeuvring and the correction
    # was effectively switched off.
    self.trust = 0.0

  def update(
    self, gyro: ArrayLike, accel_body: ArrayLike, dt: float
  ) -> NumpyArray:
    """Advance the estimate by one IMU sample.

    :param gyro: Body angular rate in rad/s.
    :param accel_body: Body-frame specific force in m/s^2.
    :param dt: Interval in seconds.
    :return: Updated orientation quaternion.
    """
    accel_body = np.asarray(accel_body, dtype=float)
    omega = np.asarray(gyro, dtype=float) - self.bias

    magnitude = np.linalg.norm(accel_body)
    if magnitude > 1e-6:
      # At rest the accelerometer reads -g, so measured "down" is -a.
      measured_down = -accel_body / magnitude
      # Where the current estimate says "down" is, in the body frame.
      predicted_down = quat_to_rotmat(self.q).T @ (
        G_NED / np.linalg.norm(G_NED)
      )

      self.trust = gravity_trust(accel_body)
      error = self.trust * np.cross(measured_down, predicted_down)

      self.bias = self.bias - self.ki * error * dt
      omega = omega + self.kp * error
    else:
      self.trust = 0.0

    self.q = quat_normalize(quat_multiply(self.q, quat_from_gyro(omega, dt)))
    return self.q

  @property
  def rotation(self) -> NumpyArray:
    """Current orientation as a rotation matrix, body to world."""
    return quat_to_rotmat(self.q)
