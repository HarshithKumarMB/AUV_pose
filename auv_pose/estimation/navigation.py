"""The unscented forward pass: IMU prediction; DVL, depth and compass aiding.

Aiding sensors read in the ``IMUSocket`` body frame, so there is no lever arm.
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

#: The world field HoloOcean's ``MagnetometerSensor`` measures by default.
MAGNETIC_NORTH = np.array([1.0, 0.0, 0.0])


@dataclass(frozen=True, eq=False)
class Update(Generic[Belief]):
  """The result of conditioning on one measurement.

  :param posterior: Belief after the update.
  :param innovation: ``z - E[h(x)]``, shape ``(m,)``.
  :param innovation_cov: ``S``, shape ``(m, m)``, including measurement noise.
  """

  posterior: Belief
  innovation: NumpyArray
  innovation_cov: NumpyArray


@dataclass(frozen=True, eq=False)
class Aiding:
  """One measurement: reading ``z``, model ``observe(state)`` and its noise."""

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
  """Push a belief through a deterministic motion model by sigma points.

  :param process_cov: Prior-independent covariance the motion adds, summed in
      at the predicted mean.
  :param average: ``(points, weights) -> mean``; defaults to
      :func:`weighted_mean` under the chart.
  :return: ``(prior, cross_cov)`` with ``cross_cov = Cov[xi_previous,
      xi_predicted]``.
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
  """Condition a belief on one measurement by sigma points.

  :param observe: Measurement model; must return a flat vector, not a rotation.
  :param transport: Carries the covariance to the tangent space at the new
      mean; ``None`` in a flat chart.
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

  # K = C S^-1 via S^T K^T = C^T.
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
  """What a magnetometer in the ``IMUSocket`` reads: world ``field`` in body."""
  return state.rotation.T @ np.asarray(field, dtype=float)


def dvl_noise_covariance(beam_sigma: float, elevation_deg: float) -> NumpyArray:
  """Body-velocity covariance of a four-beam Janus DVL, ``(3, 3)``.

  :param beam_sigma: Per-beam radial standard deviation (HoloOcean's
      ``VelSigma``), m/s.
  :param elevation_deg: Beam angle off the body z axis, degrees.
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
  """One filter cycle: predict through the IMU, then apply ``aiding`` in order.

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

  :param dvl: Body-velocity covariance, ``(3, 3)``.
  :param depth: Depth standard deviation, metres.
  :param magnetometer: Per-axis standard deviation of the unit field reading;
      roughly the heading error in radians.
  """

  dvl: NumpyArray
  depth: float
  magnetometer: float


class InertialNavigator:
  """The forward pass driven one IMU tick at a time, live or on replay.

  Samples are buffered until a tick carries aiding or is closed explicitly;
  each closed cycle is one :func:`inertial_step`.

  :param initial: Belief at the tick before the first sample.
  :param dt: IMU sample interval, seconds.
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

    #: One step per closed cycle, and the tick that closed it.
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
    """Take one IMU sample; close the cycle if aiding arrived or ``close``.

    :param dvl: Body-velocity reading, or ``None`` if none this tick; likewise
        ``depth`` and ``magnetometer``.
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
