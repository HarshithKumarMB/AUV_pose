"""The IMU motion model and the covariance its own noise injects.

:func:`propagate` is deterministic so sigma points can be pushed through it;
the IMU noise is accumulated separately by :func:`imu_noise_covariance`.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from auv_pose.estimation.manifold import (
  ACCEL_BIAS,
  DOF,
  GYRO_BIAS,
  POSITION,
  ROTATION,
  VELOCITY,
  NavState,
)
from auv_pose.estimation.quaternion import (
  quat_exp,
  quat_multiply,
  quat_normalize,
  quat_to_rotmat,
  skew,
  so3_right_jacobian,
)
from auv_pose.estimation.typing import NumpyArray

#: World-frame gravity. The world is z up, so it points down.
GRAVITY = np.array([0.0, 0.0, -9.81])


#: Where each noise source sits in the ``(12,)`` driving-noise vector.
_GYRO = slice(0, 3)
_ACCEL = slice(3, 6)
_GYRO_BIAS_WALK = slice(6, 9)
_ACCEL_BIAS_WALK = slice(9, 12)
_NOISE_DOF = 12


@dataclass(frozen=True, eq=False)
class ImuSamples:
  """A run of IMU samples, as one smoother step consumes them.

  :param gyro: Body angular rate per sample, rad/s, shape ``(k, 3)``.
  :param accel: Body specific force ``f = a - g`` per sample, m/s^2, shape
      ``(k, 3)``; at rest it reads ``-g``, not zero.
  :param dt: Interval per sample, seconds, shape ``(k,)``.
  """

  gyro: NumpyArray
  accel: NumpyArray
  dt: NumpyArray

  @classmethod
  def uniform(cls, gyro: ArrayLike, accel: ArrayLike, dt: float) -> Self:
    """Samples at a fixed interval ``dt``; ``(3,)`` inputs are one sample."""
    gyro = np.atleast_2d(np.asarray(gyro, dtype=float))
    accel = np.atleast_2d(np.asarray(accel, dtype=float))
    if gyro.shape != accel.shape:
      raise ValueError(
        f"gyro and accel must match: {gyro.shape} vs {accel.shape}"
      )
    return cls(gyro=gyro, accel=accel, dt=np.full(len(gyro), float(dt)))

  def __len__(self) -> int:
    return len(self.dt)


@dataclass(frozen=True)
class ImuNoise:
  """IMU noise as a datasheet states it: continuous-time densities.

  :param gyro: Angle random walk, rad/s/sqrt(Hz).
  :param accel: Velocity random walk, m/s^2/sqrt(Hz).
  :param gyro_bias: Gyro bias random walk, rad/s^2/sqrt(Hz).
  :param accel_bias: Accelerometer bias random walk, m/s^3/sqrt(Hz).
  """

  gyro: float = 1.8e-3
  accel: float = 9e-3
  gyro_bias: float = 2.7e-4
  accel_bias: float = 3.3e-4

  def covariance(self, dt: float) -> NumpyArray:
    """Driving-noise covariance of one sample of length ``dt``, ``(12, 12)``.

    White noise gives ``density / sqrt(dt)``; a random walk
    ``density * sqrt(dt)``.
    """
    return np.diag(
      np.repeat(
        [
          self.gyro**2 / dt,
          self.accel**2 / dt,
          self.gyro_bias**2 * dt,
          self.accel_bias**2 * dt,
        ],
        3,
      )
    )


DEFAULT_NOISE = ImuNoise()


def propagate(
  state: NavState,
  samples: ImuSamples,
) -> NavState:
  """Advance a state through a run of IMU samples, noise-free::

      R' = R exp(dt (w - b_g))
      v' = v + dt (R (a - b_a) + g)
      p' = p + dt v + (dt^2 / 2) (R (a - b_a) + g)

  The biases are held constant. The second-order position term is required:
  it is what :func:`imu_noise_covariance` is derived for.
  """
  position = np.asarray(state.position, dtype=float)
  attitude = np.asarray(state.attitude, dtype=float)
  velocity = np.asarray(state.velocity, dtype=float)

  for gyro, accel, dt in zip(samples.gyro, samples.accel, samples.dt):
    rotation = quat_to_rotmat(attitude)
    acceleration = rotation @ (accel - state.accel_bias) + GRAVITY

    position = position + dt * velocity + 0.5 * dt**2 * acceleration
    velocity = velocity + dt * acceleration
    attitude = quat_normalize(
      quat_multiply(attitude, quat_exp((gyro - state.gyro_bias) * dt))
    )

  return NavState(
    position=position,
    attitude=attitude,
    velocity=velocity,
    gyro_bias=state.gyro_bias,
    accel_bias=state.accel_bias,
  )


def imu_noise_covariance(
  state: NavState, samples: ImuSamples, noise: ImuNoise = DEFAULT_NOISE
) -> NumpyArray:
  """Covariance the IMU's own noise adds over a run of samples.

  Runs ``S <- F S F^T + G Q G^T`` along the state's trajectory. The rotation
  error is a right perturbation, ``R = R_hat exp(xi_R)``, matching
  ``NavState.__add__``, so the result adds directly to a sigma-point covariance.
  Bias uncertainty is not linearised here: each sigma point propagates under
  its own bias.

  :param state: State to propagate from; supplies attitude and bias estimates.
  :return: Covariance contribution, ``(15, 15)``, symmetric PSD.
  """
  covariance = np.zeros((DOF, DOF))
  attitude = np.asarray(state.attitude, dtype=float)

  for gyro, accel, dt in zip(samples.gyro, samples.accel, samples.dt):
    rotation = quat_to_rotmat(attitude)
    rate = (gyro - state.gyro_bias) * dt
    force = accel - state.accel_bias

    right_jacobian = so3_right_jacobian(rate)
    increment = quat_to_rotmat(quat_exp(rate))
    lever = rotation @ skew(force)

    transition = np.eye(DOF)
    transition[POSITION, VELOCITY] = dt * np.eye(3)
    transition[POSITION, ROTATION] = -0.5 * dt**2 * lever
    transition[POSITION, ACCEL_BIAS] = -0.5 * dt**2 * rotation
    transition[ROTATION, ROTATION] = increment.T
    transition[ROTATION, GYRO_BIAS] = -dt * right_jacobian
    transition[VELOCITY, ROTATION] = -dt * lever
    transition[VELOCITY, ACCEL_BIAS] = -dt * rotation

    gain = np.zeros((DOF, _NOISE_DOF))
    gain[POSITION, _ACCEL] = -0.5 * dt**2 * rotation
    gain[ROTATION, _GYRO] = -dt * right_jacobian
    gain[VELOCITY, _ACCEL] = -dt * rotation
    gain[GYRO_BIAS, _GYRO_BIAS_WALK] = np.eye(3)
    gain[ACCEL_BIAS, _ACCEL_BIAS_WALK] = np.eye(3)

    covariance = (
      transition @ covariance @ transition.T
      + gain @ noise.covariance(dt) @ gain.T
    )

    attitude = quat_normalize(quat_multiply(attitude, quat_exp(rate)))

  return 0.5 * (covariance + covariance.T)
