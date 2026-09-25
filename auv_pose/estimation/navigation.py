"""The unscented forward pass: inertial prediction and aiding updates.

This is the causal half of the smoother in
:func:`~auv_pose.estimation.smoothers.unscented_rts_smooth`. Each cycle pushes
sigma points through the motion model, conditions on whatever aiding arrived,
and records a :class:`~auv_pose.estimation.typing.SmootherStep` carrying the
cross-covariance the backward pass needs. Nothing here forms a Jacobian of the
motion or of a measurement.

:func:`unscented_predict` and :func:`unscented_update` are generic in the
chart, as the backward pass is: handed ``operator.add`` and ``operator.sub``
they run in a vector space, where on a linear-Gaussian problem they must equal
the Kalman filter exactly. That is the test that pins them, for the same reason
it pins the backward pass.

The measurement models are the aiding a survey vehicle carries without any
absolute position fix -- a DVL, a pressure sensor and a magnetometer. All three
read in the body frame of HoloOcean's ``IMUSocket``, which is the body frame
:class:`~auv_pose.estimation.manifold.NavState`'s attitude rotates out of, so no
lever arm or mounting rotation appears.

**Horizontal position is never observed.** The DVL bounds velocity, so
position drifts linearly rather than quadratically, but it drifts. The
magnetometer is what keeps that drift from being dominated by heading error:
with the vehicle holding one attitude for a whole survey, the gyro alone gives
heading no reference at all.
"""

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Generic

import numpy as np
from numpy.typing import ArrayLike

from auv_pose.estimation.inertial import (
  DEFAULT_NOISE,
  ImuNoise,
  ImuSamples,
  imu_noise_covariance,
  propagate,
)
from auv_pose.estimation.manifold import (
  ManifoldGaussian,
  NavState,
  covariance_transport,
  manifold_mean,
)
from auv_pose.estimation.typing import Belief, NumpyArray, SmootherStep
from auv_pose.estimation.unscented import (
  DEFAULT_RULE,
  SigmaRule,
  cross_moments,
  sigma_offsets,
  tangent_moments,
  weighted_mean,
)

#: What HoloOcean's ``MagnetometerSensor`` measures unless configured otherwise:
#: the world x axis, expressed in the sensor frame. Unit length, so the reading
#: carries direction only.
MAGNETIC_NORTH = np.array([1.0, 0.0, 0.0])


@dataclass(frozen=True, eq=False)
class Update(Generic[Belief]):
  """The result of conditioning on one measurement.

  :param posterior: Belief after the update.
  :param innovation: ``z - E[h(x)]``, shape ``(m,)``.
  :param innovation_cov: ``S``, shape ``(m, m)``, including the measurement
      noise. ``innovation @ solve(S, innovation)`` is the normalised
      innovation squared, which is the run-time check that the filter is
      honest about itself.
  """

  posterior: Belief
  innovation: NumpyArray
  innovation_cov: NumpyArray


@dataclass(frozen=True, eq=False)
class Aiding:
  """One measurement to condition on: a reading, its model and its noise.

  :param z: The reading, shape ``(m,)``.
  :param observe: The measurement model, ``h(state) -> (m,)``.
  :param noise: Measurement noise covariance, shape ``(m, m)``.
  """

  z: NumpyArray
  observe: Callable[[NavState], NumpyArray]
  noise: NumpyArray


def _mean_of(points, weights_mean, chart_plus, chart_minus, average):
  if average is not None:
    return average(points, weights_mean)
  return weighted_mean(points, weights_mean, chart_plus, chart_minus)


def unscented_predict(
  belief: Belief,
  motion: Callable[[Any], Any],
  process_cov: ArrayLike,
  chart_plus: Callable = operator.add,
  chart_minus: Callable = operator.sub,
  average: Callable | None = None,
  rule: SigmaRule = DEFAULT_RULE,
) -> tuple[Belief, NumpyArray]:
  """Push a belief through a motion model by sigma points.

  :param belief: Posterior at the previous step.
  :param motion: The deterministic motion model, applied to each sigma point.
  :param process_cov: Covariance the motion adds that is independent of the
      prior -- the IMU's own noise, for the inertial model. Summed in after the
      transform, in the tangent space at the predicted mean.
  :param chart_plus: Applies a tangent increment to a mean.
  :param chart_minus: The tangent increment between two points.
  :param average: Weighted mean of the propagated points,
      ``(points, weights) -> mean``. Defaults to :func:`weighted_mean` under
      the given chart; pass :func:`manifold_mean` for the navigation state,
      which reports a cloud too wide for its chart rather than returning a
      mean that means nothing.
  :param rule: Sigma-point placement and weighting.
  :return: ``(prior, cross_cov)``: the predicted belief, and
      ``Cov[xi_previous, xi_predicted]`` as
      :class:`~auv_pose.estimation.typing.SmootherStep` records it.
  """
  cov = np.asarray(belief.cov, dtype=float)
  n = cov.shape[0]
  weights_mean, weights_cov = rule.weights(n)

  offsets = sigma_offsets(cov, rule)
  moved = [motion(chart_plus(belief.mean, offset)) for offset in offsets]

  mean = _mean_of(moved, weights_mean, chart_plus, chart_minus, average)
  spread = np.stack([chart_minus(point, mean) for point in moved])

  predicted = tangent_moments(spread, weights_cov) + np.asarray(
    process_cov, dtype=float
  )
  cross_cov = cross_moments(offsets, spread, weights_cov)

  prior = replace(belief, mean=mean, cov=0.5 * (predicted + predicted.T))
  return prior, cross_cov


def unscented_update(
  belief: Belief,
  observe: Callable[[Any], NumpyArray],
  z: ArrayLike,
  noise: ArrayLike,
  chart_plus: Callable = operator.add,
  transport: Callable[[NumpyArray], NumpyArray] | None = covariance_transport,
  rule: SigmaRule = DEFAULT_RULE,
) -> Update[Belief]:
  """Condition a belief on one measurement, by sigma points.

  The update is solved in the tangent space at the prior mean and applied with
  ``chart_plus``; the covariance is then carried across to the tangent space
  at the new mean, which is where a belief's covariance has to live.

  :param belief: Belief to condition.
  :param observe: The measurement model, applied to each sigma point. Must
      return a vector in a flat space -- a reading, not a rotation.
  :param z: The measurement, shape ``(m,)``.
  :param noise: Measurement noise covariance, shape ``(m, m)``.
  :param chart_plus: Applies a tangent increment to a mean.
  :param transport: Jacobian carrying a covariance along an increment, or
      ``None`` in a flat chart.
  :param rule: Sigma-point placement and weighting.
  :return: The posterior, with the innovation and its covariance.
  """
  cov = np.asarray(belief.cov, dtype=float)
  n = cov.shape[0]
  weights_mean, weights_cov = rule.weights(n)

  offsets = sigma_offsets(cov, rule)
  images = np.stack(
    [
      np.atleast_1d(observe(chart_plus(belief.mean, offset)))
      for offset in offsets
    ]
  )

  expected = weights_mean @ images
  centred = images - expected

  innovation_cov = tangent_moments(centred, weights_cov) + np.atleast_2d(
    np.asarray(noise, dtype=float)
  )
  cross_cov = cross_moments(offsets, centred, weights_cov)

  # K = C S^-1, solved rather than inverted: S^T K^T = C^T.
  gain = np.linalg.solve(innovation_cov.T, cross_cov.T).T
  innovation = np.atleast_1d(np.asarray(z, dtype=float)) - expected
  correction = gain @ innovation

  updated = cov - gain @ innovation_cov @ gain.T
  if transport is not None:
    jacobian = transport(correction)
    updated = jacobian @ updated @ jacobian.T

  posterior = replace(
    belief,
    mean=chart_plus(belief.mean, correction),
    cov=0.5 * (updated + updated.T),
  )
  return Update(posterior, innovation, innovation_cov)


def dvl_reading(state: NavState) -> NumpyArray:
  """What a DVL in the ``IMUSocket`` reads: velocity over ground, body frame."""
  return state.rotation.T @ state.velocity


def depth_reading(state: NavState) -> NumpyArray:
  """What HoloOcean's ``DepthSensor`` reads: world ``z``, increasing upward."""
  return state.position[2:3]


def magnetometer_reading(
  state: NavState, field: ArrayLike = MAGNETIC_NORTH
) -> NumpyArray:
  """What a magnetometer in the ``IMUSocket`` reads: the world field, in body.

  :param state: State to predict the reading at.
  :param field: The world-frame vector the sensor measures. HoloOcean's
      default is the world x axis.
  """
  return state.rotation.T @ np.asarray(field, dtype=float)


def dvl_noise_covariance(beam_sigma: float, elevation_deg: float) -> NumpyArray:
  """Body-velocity noise of a four-beam Janus DVL, from its per-beam noise.

  HoloOcean's ``VelSigma`` is applied to each beam's radial velocity. The four
  beams sit ``elevation`` off the body z axis, 90 degrees apart, so a
  least-squares solve for the velocity gives the two horizontal axes a variance
  of ``sigma^2 / (2 sin^2 e)`` and the vertical ``sigma^2 / (4 cos^2 e)``.

  :param beam_sigma: Standard deviation per beam, m/s.
  :param elevation_deg: Beam angle off the body z axis, degrees.
  :return: ``(3, 3)`` diagonal covariance, body frame.
  """
  elevation = np.radians(elevation_deg)
  return np.diag(
    [
      beam_sigma**2 / (2.0 * np.sin(elevation) ** 2),
      beam_sigma**2 / (2.0 * np.sin(elevation) ** 2),
      beam_sigma**2 / (4.0 * np.cos(elevation) ** 2),
    ]
  )


def inertial_step(
  belief: ManifoldGaussian,
  samples: ImuSamples,
  aiding: Sequence[Aiding] = (),
  noise: ImuNoise = DEFAULT_NOISE,
  rule: SigmaRule = DEFAULT_RULE,
) -> tuple[SmootherStep[ManifoldGaussian], list[Update[ManifoldGaussian]]]:
  """One cycle of the inertial filter: predict through the IMU, then aid.

  :param belief: Posterior at the end of the previous cycle.
  :param samples: The IMU samples covering this cycle.
  :param aiding: Measurements taken at the end of the cycle, applied in order.
  :param noise: The IMU's noise densities.
  :param rule: Sigma-point placement and weighting.
  :return: The recorded step, and one :class:`Update` per aiding measurement.
  """
  prior, cross_cov = unscented_predict(
    belief,
    motion=lambda state: propagate(state, samples),
    process_cov=imu_noise_covariance(belief.mean, samples, noise),
    average=lambda points, weights: manifold_mean(points, weights),
    rule=rule,
  )

  posterior = prior
  updates = []
  for measurement in aiding:
    update = unscented_update(
      posterior,
      measurement.observe,
      measurement.z,
      measurement.noise,
      rule=rule,
    )
    updates.append(update)
    posterior = update.posterior

  step = SmootherStep(prior=prior, posterior=posterior, cross_cov=cross_cov)
  return step, updates


@dataclass(frozen=True, eq=False)
class AidingNoise:
  """Measurement noise of the three aiding sensors.

  :param dvl: Body-velocity covariance, ``(3, 3)``; see
      :func:`dvl_noise_covariance`.
  :param depth: Pressure-depth standard deviation, metres.
  :param magnetometer: Per-axis standard deviation of the unit field reading.
      At small angles this is the heading error in radians, so 0.03 is about
      1.7 degrees.
  """

  dvl: NumpyArray
  depth: float
  magnetometer: float


class InertialNavigator:
  """The forward pass driven one IMU tick at a time.

  The same object serves a vehicle navigating live and a log being replayed,
  so the two give the same record by construction. IMU samples are buffered
  until a tick closes a cycle -- any tick carrying an aiding reading, or one
  the caller closes explicitly because a ping was taken there -- and each
  closed cycle is one :func:`inertial_step`. Cycle boundaries therefore land
  exactly on the ticks a sounding needs a pose for.

  :param initial: Belief at the tick before the first sample.
  :param dt: IMU sample interval, seconds.
  :param aiding_noise: Noise of the DVL, depth and magnetometer readings.
  :param imu_noise: The IMU's noise densities.
  :param field: World vector the magnetometer measures.
  """

  def __init__(
    self,
    initial: ManifoldGaussian,
    dt: float,
    aiding_noise: AidingNoise,
    imu_noise: ImuNoise = DEFAULT_NOISE,
    field: ArrayLike = MAGNETIC_NORTH,
  ) -> None:
    self.initial = initial
    self.belief = initial
    self.dt = dt
    self.aiding_noise = aiding_noise
    self.imu_noise = imu_noise
    self.field = np.asarray(field, dtype=float)

    #: One recorded step per closed cycle, and the tick that closed it.
    self.history: list[SmootherStep[ManifoldGaussian]] = []
    self.cycle_ticks: list[int] = []
    #: Normalised innovation squared per update, keyed by sensor name.
    self.nis: dict[str, list[float]] = {"dvl": [], "depth": [], "compass": []}

    self._gyro: list[NumpyArray] = []
    self._accel: list[NumpyArray] = []

  def _aiding(self, dvl, depth, magnetometer) -> tuple[list[str], list[Aiding]]:
    names, aiding = [], []
    if dvl is not None:
      names.append("dvl")
      aiding.append(
        Aiding(np.asarray(dvl, float), dvl_reading, self.aiding_noise.dvl)
      )
    if depth is not None:
      names.append("depth")
      aiding.append(
        Aiding(
          np.atleast_1d(np.asarray(depth, float)),
          depth_reading,
          np.eye(1) * self.aiding_noise.depth**2,
        )
      )
    if magnetometer is not None:
      names.append("compass")
      aiding.append(
        Aiding(
          np.asarray(magnetometer, float),
          lambda state: magnetometer_reading(state, self.field),
          np.eye(3) * self.aiding_noise.magnetometer**2,
        )
      )
    return names, aiding

  def tick(
    self,
    index: int,
    gyro: ArrayLike,
    accel: ArrayLike,
    dvl: ArrayLike | None = None,
    depth: float | None = None,
    magnetometer: ArrayLike | None = None,
    close: bool = False,
  ) -> ManifoldGaussian | None:
    """Take one IMU sample, and close the cycle if this tick ends one.

    :param index: The tick's index, recorded against the cycle it closes.
    :param gyro: Body angular rate, rad/s, ``(3,)``.
    :param accel: Body specific force, m/s^2, ``(3,)``.
    :param dvl: Body velocity reading, if one arrived this tick.
    :param depth: Pressure depth, if one arrived this tick.
    :param magnetometer: Field reading, if one arrived this tick.
    :param close: End the cycle here even without aiding.
    :return: The new posterior if a cycle closed, else ``None``.
    """
    self._gyro.append(np.asarray(gyro, dtype=float))
    self._accel.append(np.asarray(accel, dtype=float))

    names, aiding = self._aiding(dvl, depth, magnetometer)
    if not aiding and not close:
      return None

    samples = ImuSamples.uniform(self._gyro, self._accel, self.dt)
    self._gyro, self._accel = [], []

    step, updates = inertial_step(
      self.belief,
      samples,
      aiding,
      noise=self.imu_noise,
    )
    for name, update in zip(names, updates):
      innovation = update.innovation
      self.nis[name].append(
        float(innovation @ np.linalg.solve(update.innovation_cov, innovation))
      )

    self.history.append(step)
    self.cycle_ticks.append(index)
    self.belief = step.posterior
    return self.belief
