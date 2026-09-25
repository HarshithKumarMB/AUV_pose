"""The IMU motion model, and the uncertainty it injects.

Two things live here, deliberately apart. :func:`propagate` is the deterministic
motion model -- a pure function of a state and a run of IMU samples, which is
what lets a sigma point be pushed through it without the propagation carrying
any state of its own. :func:`imu_noise_covariance` is everything the sensor's
own noise adds, accumulated separately and added to the predicted covariance
afterwards.

That split is the whole reason the prediction step works: the unscented
transform carries the *prior's* uncertainty through the true nonlinear model,
while the IMU's own noise -- which is independent of the prior -- is accumulated
linearly alongside and summed in at the end.

The propagation equations are the paper's, and the second-order position update
is not optional: it is the form Forster's covariance recursion is derived for,
and it is what puts the accelerometer's noise into the position block at all.

See :mod:`auv_pose.estimation` for frames and signs. The world is z-up, so
``gravity`` defaults to :data:`~auv_pose.estimation.quaternion.GRAVITY_NWU`.
"""

from typing import NamedTuple, Self

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
  GRAVITY_NWU,
  quat_exp,
  quat_multiply,
  quat_normalize,
  quat_to_rotmat,
  skew,
  so3_right_jacobian,
)
from auv_pose.estimation.typing import NumpyArray

__all__ = [
  "DEFAULT_NOISE",
  "ImuNoise",
  "ImuSamples",
  "imu_noise_covariance",
  "propagate",
]

#: Where each noise source sits in the ``(12,)`` driving-noise vector.
_GYRO = slice(0, 3)
_ACCEL = slice(3, 6)
_GYRO_BIAS_WALK = slice(6, 9)
_ACCEL_BIAS_WALK = slice(9, 12)
_NOISE_DOF = 12


class ImuSamples(NamedTuple):
  """A run of IMU samples, as one smoother step consumes them.

  :param gyro: Body angular rate per sample, rad/s, shape ``(k, 3)``.
  :param accel: Body specific force per sample, m/s^2, shape ``(k, 3)``. This
      is what an accelerometer reports: ``f = a - g``, so at rest it reads
      ``-g`` rather than zero.
  :param dt: Interval per sample, seconds, shape ``(k,)``.
  """

  gyro: NumpyArray
  accel: NumpyArray
  dt: NumpyArray

  @classmethod
  def uniform(cls, gyro: ArrayLike, accel: ArrayLike, dt: float) -> Self:
    """Samples at a fixed rate, which is the usual case.

    :param gyro: Shape ``(k, 3)``, or ``(3,)`` for a single sample.
    :param accel: Shape ``(k, 3)``, or ``(3,)`` for a single sample.
    :param dt: Interval, seconds.
    """
    gyro = np.atleast_2d(np.asarray(gyro, dtype=float))
    accel = np.atleast_2d(np.asarray(accel, dtype=float))
    if gyro.shape != accel.shape:
      raise ValueError(
        f"gyro and accel must match: {gyro.shape} vs {accel.shape}"
      )
    return cls(gyro=gyro, accel=accel, dt=np.full(len(gyro), float(dt)))

  def __len__(self) -> int:
    return len(self.dt)


class ImuNoise(NamedTuple):
  """Per-sample noise of the IMU.

  .. warning::

     Every one of these is a **per-sample** standard deviation, not a
     continuous-time density, and the two bias terms are per-sample *increments
     of a random walk* rather than the size of the bias itself. They are
     written this way because that is what HoloOcean's ``IMUSensor`` takes --
     see :func:`experiments.scenarios.imu_sensor`, whose warning is entirely
     about this -- and because it keeps the recursion below free of any ``dt``
     rescaling, which is the step most easily got wrong in either direction.

     The consequence is that a bias grows as ``sigma * sqrt(k)`` over ``k``
     samples. Size these backwards from the bias you want at the end of a run.

  :param gyro: Angular-rate white noise, rad/s.
  :param accel: Specific-force white noise, m/s^2.
  :param gyro_bias: Gyro bias random-walk increment, rad/s per sample.
  :param accel_bias: Accelerometer bias random-walk increment, m/s^2 per sample.
  """

  gyro: float = 0.01
  accel: float = 0.05
  gyro_bias: float = 5e-5
  accel_bias: float = 6e-5

  def covariance(self) -> NumpyArray:
    """The driving-noise covariance ``Q``, shape ``(12, 12)``."""
    return np.diag(
      np.concatenate(
        [
          np.full(3, self.gyro**2),
          np.full(3, self.accel**2),
          np.full(3, self.gyro_bias**2),
          np.full(3, self.accel_bias**2),
        ]
      )
    )


#: The IMU the experiments are configured with -- see
#: :func:`experiments.scenarios.imu_sensor`. A module-level singleton rather
#: than a default argument, which is evaluated once at import and reads like a
#: fresh value.
DEFAULT_NOISE = ImuNoise()


def propagate(
  state: NavState,
  samples: ImuSamples,
  gravity: ArrayLike = GRAVITY_NWU,
) -> NavState:
  """Advance a state through a run of IMU samples.

  Composes the paper's propagation equations, noise-free, over every sample::

      R' = R exp(dt (w - b_g))
      v' = v + dt (R (a - b_a) + g)
      p' = p + dt v + (dt^2 / 2) (R (a - b_a) + g)

  The biases are held: their random walk has no mean, so it belongs in
  :func:`imu_noise_covariance` and not here.

  :param state: State to advance.
  :param samples: IMU samples for the step.
  :param gravity: World-frame gravity. Must match the frame the attitude
      rotates into -- see :mod:`auv_pose.estimation`, where getting this
      backwards is documented as silent.
  :return: The advanced state.

  Note:
      The position update is second order in ``dt``, unlike
      :meth:`~auv_pose.estimation.strapdown.StrapdownIntegrator.step`, which
      is semi-implicit Euler (``p += v' dt``). The two differ by
      ``dt^2 a / 2`` per sample. This form is the one Forster's covariance
      recursion assumes, so using the other here would mean citing a
      derivation for an integrator it was not written for.
  """
  gravity = np.asarray(gravity, dtype=float)

  position = np.asarray(state.position, dtype=float)
  attitude = np.asarray(state.attitude, dtype=float)
  velocity = np.asarray(state.velocity, dtype=float)

  for gyro, accel, dt in zip(samples.gyro, samples.accel, samples.dt):
    rotation = quat_to_rotmat(attitude)
    acceleration = rotation @ (accel - state.accel_bias) + gravity

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

  Runs the error-state recursion ``S <- F S F^T + G Q G^T`` along the state's
  own trajectory, which is Forster's preintegrated noise written as a filter
  recursion rather than as a factor.

  The error state is the chart of :mod:`auv_pose.estimation.manifold`, so the
  rotation error is a **right** perturbation, ``R = R_hat exp(xi_R)``. That is
  what makes the rotation row come out as ``dR^T xi_R - Jr dt (xi_g + eta_g)``
  with no adjoint left over, and it is why this covariance can be added
  directly to one produced by :func:`~auv_pose.estimation.manifold.boxplus`.

  Gravity does not appear: it is deterministic, so it cancels from every
  Jacobian.

  :param state: State the samples are propagated from; supplies the attitude
      the trajectory is linearised about, and the bias estimates.
  :param samples: IMU samples for the step.
  :param noise: Per-sample sensor noise.
  :return: Covariance contribution, shape ``(15, 15)``, symmetric PSD.

  Note:
      Approximated relative to full preintegration, and deliberately:

      * The Jacobians are linearised about this state's trajectory. That is
          Forster's own approximation, not an extra one.
      * **There are no bias Jacobians and no repropagation.** Preintegration
          carries ``d(dR)/d(b_g)`` so a factor need not be re-integrated when
          an optimiser moves the bias estimate. A filter re-integrates every
          step regardless, and each sigma point propagates under *its own*
          bias, so bias uncertainty goes through the true nonlinear model --
          which is stronger than the first-order correction it replaces.
      * ``Q`` is additive, and each interval is a zero-order hold. No coning
          or sculling compensation, which at 30 Hz on a slow vehicle is far
          below the accelerometer noise.
  """
  covariance = np.zeros((DOF, DOF))
  driving = noise.covariance()
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
      transition @ covariance @ transition.T + gain @ driving @ gain.T
    )

    attitude = quat_normalize(quat_multiply(attitude, quat_exp(rate)))

  return 0.5 * (covariance + covariance.T)
