"""Strapdown inertial dead reckoning.

Propagate attitude from the gyro, rotate the accelerometer's specific force into
the world frame, remove gravity, and integrate twice. See :mod:`auv_pose.smoothing`
for the frame conventions.

This is the open-loop baseline that terrain-aided navigation corrects: it drifts,
because integrating noisy acceleration twice accumulates error quadratically.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from auv_pose.smoothing.quaternion import (
    G_NED,
    quat_from_gyro,
    quat_multiply,
    quat_normalize,
    quat_to_rotmat,
)

__all__ = ["StrapdownIntegrator"]


class StrapdownIntegrator:
    """Attitude, velocity and position integrated from IMU readings.

    Args:
        position: Initial world position.
        attitude: Initial orientation as a scalar-first quaternion, body to world.
        velocity: Initial world velocity; at rest if omitted.
    """

    def __init__(
        self,
        position: ArrayLike,
        attitude: ArrayLike,
        velocity: ArrayLike | None = None,
    ) -> None:
        self.position = np.asarray(position, dtype=float).copy()
        self.attitude = quat_normalize(attitude)
        self.velocity = (
            np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float).copy()
        )

    def step(self, gyro: ArrayLike, accel_body: ArrayLike, dt: float) -> NDArray[np.float64]:
        """Advance by one IMU sample.

        Args:
            gyro: Body angular rate, rad/s.
            accel_body: Body specific force, m/s^2, as an accelerometer reports it.
            dt: Interval, seconds.

        Returns:
            World-frame linear acceleration, gravity removed -- the quantity a
            position filter wants as its control input.
        """
        self.attitude = quat_normalize(
            quat_multiply(self.attitude, quat_from_gyro(gyro, dt))
        )

        accel_world = quat_to_rotmat(self.attitude) @ np.asarray(accel_body, dtype=float)
        accel_world = accel_world - G_NED

        self.velocity = self.velocity + accel_world * dt
        self.position = self.position + self.velocity * dt

        return accel_world

    @property
    def rotation(self) -> NDArray[np.float64]:
        """Current orientation as a rotation matrix, body to world."""
        return quat_to_rotmat(self.attitude)
