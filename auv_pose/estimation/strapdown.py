"""Strapdown inertial dead reckoning.

Propagate attitude from the gyro, rotate the accelerometer's specific force into
the world frame, remove gravity, and integrate twice. See :mod:`auv_pose.estimation`
for the frame conventions.

This is the open-loop baseline that terrain-aided navigation corrects: it drifts,
because integrating noisy acceleration twice accumulates error quadratically.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from auv_pose.estimation.quaternion import (
  GRAVITY_NWU,
  quat_from_gyro,
  quat_multiply,
  quat_normalize,
  quat_to_rotmat,
)

__all__ = ["StrapdownIntegrator"]


class StrapdownIntegrator:
  """Attitude, velocity and position integrated from IMU readings.

  :param position: Initial world position.
  :param attitude: Initial orientation as a scalar-first quaternion, body to
      world.
  :param velocity: Initial world velocity; at rest if omitted.
  :param attitude_filter: Optional correction for the attitude channel. Without
      one, attitude comes from integrating the gyro alone and drifts without
      bound -- and because gravity is removed using that attitude, the error
      leaks straight into acceleration as ``g sin(theta)``. Pass an
      :class:`auv_pose.estimation.filters.AttitudeFilter` to bound roll and
      pitch against the accelerometer. Leaving it ``None`` is the uncorrected
      baseline, useful for measuring what the correction buys.
  :param gravity: World-frame gravity vector. Must match the frame the attitude
      quaternion rotates into: ``GRAVITY_NWU`` for HoloOcean's z-up world,
      ``GRAVITY_NED`` for a z-down one. The wrong choice does not announce
      itself -- gravity still cancels exactly at rest, because an accelerometer
      read in a z-down body frame cancels a z-down gravity vector. What it does
      instead is mirror the recovered acceleration in y and z.
  """

  def __init__(
    self,
    position: ArrayLike,
    attitude: ArrayLike,
    velocity: ArrayLike | None = None,
    attitude_filter=None,
    gravity: ArrayLike = GRAVITY_NWU,
  ) -> None:
    self.position = np.asarray(position, dtype=float).copy()
    self.attitude = quat_normalize(attitude)
    self.gravity = np.asarray(gravity, dtype=float).copy()
    self.velocity = (
      np.zeros(3)
      if velocity is None
      else np.asarray(velocity, dtype=float).copy()
    )
    self.attitude_filter = attitude_filter
    if attitude_filter is not None:
      attitude_filter.q = self.attitude

  def step(
    self,
    gyro: ArrayLike,
    accel_body: ArrayLike,
    dt: float,
    kinematic_accel: ArrayLike | None = None,
  ) -> NDArray[np.float64]:
    """Advance by one IMU sample.

    An accelerometer measures specific force ``f = a - g``, so at rest it reads
    ``-g`` and the kinematic acceleration is recovered by adding gravity back:
    ``a = R f + g``. In a z-up world ``g`` is negative in its third component,
    so this subtracts 9.81 where an NED convention would add it.

    :param gyro: Body angular rate in rad/s.
    :param accel_body: Body specific force in m/s^2, as an accelerometer
        reports it.
    :param dt: Interval in seconds.
    :param kinematic_accel: The vehicle's own body-frame acceleration, if some
        other sensor measures it. Used only to give the attitude filter a
        gravity reference that survives a manoeuvre -- see
        :meth:`auv_pose.estimation.filters.AttitudeFilter.update`. Gravity is
        still removed from the raw ``accel_body`` below, because that is the
        reading being integrated; subtracting it here as well would integrate
        the DVL and call the result inertial.
    :return: World-frame linear acceleration with gravity removed -- the
        quantity a position filter wants as its control input.
    """
    if self.attitude_filter is not None:
      self.attitude = self.attitude_filter.update(
        gyro, accel_body, dt, kinematic_accel
      )
    else:
      self.attitude = quat_normalize(
        quat_multiply(self.attitude, quat_from_gyro(gyro, dt))
      )

    accel_world = quat_to_rotmat(self.attitude) @ np.asarray(
      accel_body, dtype=float
    )
    accel_world = accel_world + self.gravity

    self.velocity = self.velocity + accel_world * dt
    self.position = self.position + self.velocity * dt

    return accel_world

  def set_attitude(self, attitude: ArrayLike) -> None:
    """Override the current orientation, filter included.

    Assigning :attr:`attitude` directly is not enough when an attitude filter
    is attached: :meth:`step` takes its next attitude from the filter's own
    state, so a bare assignment is discarded on the following sample. Use this
    for diagnostics that inject a known attitude.
    """
    self.attitude = quat_normalize(attitude)
    if self.attitude_filter is not None:
      self.attitude_filter.q = self.attitude

  @property
  def rotation(self) -> NDArray[np.float64]:
    """Current orientation as a rotation matrix, body to world."""
    return quat_to_rotmat(self.attitude)
