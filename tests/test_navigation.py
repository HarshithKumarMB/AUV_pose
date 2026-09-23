"""The unscented forward pass.

The first test is the one that pins it. On a linear-Gaussian problem the
unscented predict and update are exact, so run in a vector chart they must
reproduce ``ConstantVelocityEKF`` -- and the smoother fed their record must
reproduce ``rts_smooth`` fed the EKF's. That checks the sigma-point moments, the
cross-covariance the backward pass consumes, and the gain, against an
implementation that was already trusted.

After that, the manifold: the measurement models against hand-derived readings,
then a Monte-Carlo consistency check of the whole inertial filter against
simulated truth.
"""

import operator

import numpy as np
import pytest
from scipy.stats import chi2

from auv_pose.estimation.filters import ConstantVelocityEKF
from auv_pose.estimation.inertial import ImuNoise, ImuSamples, propagate
from auv_pose.estimation.manifold import (
  DOF,
  ROTATION,
  ManifoldGaussian,
  NavState,
  boxminus,
  boxplus,
)
from auv_pose.estimation.navigation import (
  MAGNETIC_NORTH,
  Aiding,
  AidingNoise,
  InertialNavigator,
  depth_reading,
  dvl_noise_covariance,
  dvl_reading,
  inertial_step,
  magnetometer_reading,
  unscented_predict,
  unscented_update,
)
from auv_pose.estimation.quaternion import GRAVITY_NWU, quat_exp
from auv_pose.estimation.smoothers import rts_smooth, unscented_rts_smooth
from auv_pose.estimation.typing import GaussianState, Measurement, SmootherStep

POSITION_H = np.hstack([np.eye(3), np.zeros((3, 3))])


# -- exactness against the Kalman filter ------------------------------------


def linear_runs(n=25, seed=0):
  """The EKF's record, and the unscented pass's on the same data."""
  rng = np.random.default_rng(seed)
  dt = 0.1

  ekf = ConstantVelocityEKF(accel_process_sigma=0.5)
  F, B, Q = ekf._matrices(dt)
  initial = ConstantVelocityEKF.initial(
    np.zeros(3), cov=np.diag([1.0, 2.0, 0.5, 0.3, 0.2, 0.1])
  )

  state = initial
  belief = initial
  steps = []
  position, velocity = np.zeros(3), np.zeros(3)

  for _ in range(n):
    accel = rng.normal(scale=0.5, size=3)
    velocity = velocity + accel * dt
    position = position + velocity * dt
    z = position + rng.normal(scale=0.3, size=3)
    R = np.eye(3) * 0.09

    state = ekf.step(state, accel, dt, [Measurement(z, POSITION_H, R)])

    prior, cross_cov = unscented_predict(
      belief,
      motion=lambda x, accel=accel: F @ x + B @ accel,
      process_cov=Q,
      chart_plus=operator.add,
      chart_minus=operator.sub,
    )
    update = unscented_update(
      prior, lambda x: POSITION_H @ x, z, R, operator.add, transport=None
    )
    belief = update.posterior
    steps.append(SmootherStep(prior, belief, cross_cov))

  return ekf, initial, steps


def test_the_forward_pass_reproduces_the_kalman_filter():
  ekf, _, steps = linear_runs()

  for mine, theirs in zip(steps, ekf.history):
    np.testing.assert_allclose(mine.prior.mean, theirs.prior.mean, atol=1e-10)
    np.testing.assert_allclose(mine.prior.cov, theirs.prior.cov, atol=1e-10)
    np.testing.assert_allclose(
      mine.posterior.mean, theirs.posterior.mean, atol=1e-10
    )
    np.testing.assert_allclose(
      mine.posterior.cov, theirs.posterior.cov, atol=1e-10
    )


def test_its_record_smooths_to_the_linear_smoother():
  """The cross-covariance is what the backward pass needs, so check it there."""
  ekf, initial, steps = linear_runs(seed=3)

  theirs = rts_smooth(initial, ekf.history)
  mine = unscented_rts_smooth(
    initial, steps, operator.add, operator.sub, transport=None
  )

  for a, b in zip(mine, theirs):
    np.testing.assert_allclose(a.mean, b.mean, atol=1e-9)
    np.testing.assert_allclose(a.cov, b.cov, atol=1e-9)


def test_an_update_never_loosens_the_belief():
  rng = np.random.default_rng(1)
  root = rng.normal(size=(6, 6))
  belief = GaussianState(rng.normal(size=6), root @ root.T + np.eye(6))
  H = rng.normal(size=(2, 6))

  update = unscented_update(
    belief, lambda x: H @ x, np.zeros(2), np.eye(2), operator.add, None
  )

  assert np.linalg.eigvalsh(belief.cov - update.posterior.cov).min() > -1e-10


# -- measurement models -----------------------------------------------------


def yawed(angle, velocity=(0.0, 0.0, 0.0), position=(0.0, 0.0, -60.0)):
  return NavState.at_rest(
    position=position,
    attitude=quat_exp(np.array([0.0, 0.0, angle])),
    velocity=velocity,
  )


def test_the_dvl_reads_velocity_in_the_body_frame():
  state = yawed(np.pi / 2, velocity=(0.0, 1.5, 0.0))
  np.testing.assert_allclose(dvl_reading(state), [1.5, 0.0, 0.0], atol=1e-12)


def test_the_depth_sensor_reads_world_z():
  np.testing.assert_allclose(depth_reading(yawed(0.3)), [-60.0])


def test_the_magnetometer_turns_against_the_vehicle():
  """Yawing the vehicle left swings north to its right."""
  angle = np.radians(30.0)
  np.testing.assert_allclose(
    magnetometer_reading(yawed(angle)),
    [np.cos(angle), -np.sin(angle), 0.0],
    atol=1e-12,
  )


def test_the_dvl_noise_matches_a_least_squares_solve_over_the_beams():
  elevation = np.radians(22.5)
  beams = np.array(
    [
      [np.sin(elevation) * np.cos(a), np.sin(elevation) * np.sin(a), 1.0]
      for a in np.radians([0.0, 90.0, 180.0, 270.0])
    ]
  )
  beams[:, 2] = np.cos(elevation)
  sigma = 0.02
  expected = sigma**2 * np.linalg.inv(beams.T @ beams)

  np.testing.assert_allclose(
    dvl_noise_covariance(sigma, 22.5), expected, atol=1e-15
  )


def test_the_magnetometer_bounds_heading_and_the_gyro_does_not():
  """At rest, with a 10 degree heading prior, only the compass tightens it."""
  mean = NavState.at_rest(position=(0.0, 0.0, -60.0))
  cov = np.eye(DOF) * 1e-6
  cov[ROTATION, ROTATION] = np.diag([1e-6, 1e-6, np.radians(10.0) ** 2])
  belief = ManifoldGaussian(mean, cov)

  still = ImuSamples.uniform(np.zeros(3), -GRAVITY_NWU, 1.0 / 30.0)
  compass = Aiding(
    magnetometer_reading(mean), magnetometer_reading, np.eye(3) * 1e-4
  )

  gyro_only = compassed = belief
  for _ in range(20):
    gyro_only, _ = inertial_step(gyro_only, still)
    compassed, _ = inertial_step(compassed, still, [compass])
    gyro_only, compassed = gyro_only.posterior, compassed.posterior

  assert np.sqrt(gyro_only.cov[5, 5]) > np.radians(9.9)
  assert np.sqrt(compassed.cov[5, 5]) < np.radians(1.0)


# -- consistency against simulated truth ------------------------------------


NOISE = ImuNoise()
DT = 1.0 / 30.0
SAMPLES_PER_STEP = 6
DVL_NOISE = dvl_noise_covariance(0.02, 22.5)
DEPTH_NOISE = np.eye(1) * 0.05**2
COMPASS_NOISE = np.eye(3) * 0.02**2


def simulate(rng, prior, steps):
  """Truth drawn from the prior, and the noisy sensors a filter would see.

  The vehicle turns slowly and accelerates gently, so every state is excited,
  with biases walking as :class:`ImuNoise` says they do.
  """
  truth = boxplus(
    prior.mean, np.linalg.cholesky(prior.cov) @ rng.normal(size=DOF)
  )
  cycles = []

  for _ in range(steps):
    gyro, accel = [], []
    for _ in range(SAMPLES_PER_STEP):
      rate = np.array([0.01, -0.02, 0.05])
      force = truth.rotation.T @ (np.array([0.05, 0.0, 0.0]) - GRAVITY_NWU)
      clean = ImuSamples.uniform(
        rate + truth.gyro_bias, force + truth.accel_bias, DT
      )
      truth = propagate(truth, clean)
      truth = truth._replace(
        gyro_bias=truth.gyro_bias + rng.normal(scale=NOISE.gyro_bias, size=3),
        accel_bias=truth.accel_bias
        + rng.normal(scale=NOISE.accel_bias, size=3),
      )
      gyro.append(clean.gyro[0] + rng.normal(scale=NOISE.gyro, size=3))
      accel.append(clean.accel[0] + rng.normal(scale=NOISE.accel, size=3))

    aiding = [
      Aiding(
        dvl_reading(truth) + np.linalg.cholesky(DVL_NOISE) @ rng.normal(size=3),
        dvl_reading,
        DVL_NOISE,
      ),
      Aiding(
        depth_reading(truth) + rng.normal(scale=0.05, size=1),
        depth_reading,
        DEPTH_NOISE,
      ),
      Aiding(
        magnetometer_reading(truth) + rng.normal(scale=0.02, size=3),
        magnetometer_reading,
        COMPASS_NOISE,
      ),
    ]
    cycles.append((ImuSamples.uniform(gyro, accel, DT), aiding, truth))

  return cycles


@pytest.fixture(scope="module")
def monte_carlo():
  """NEES of the filter and the smoother at every step, over independent runs.

  One simulation serves both tests: the runs are the expensive part.
  """
  rng = np.random.default_rng(5)
  trials, steps = 40, 15

  cov = np.diag(
    np.concatenate(
      [
        np.full(3, 0.5**2),
        np.radians([2.0, 2.0, 5.0]) ** 2,
        np.full(3, 0.1**2),
        np.full(3, 1e-3**2),
        np.full(3, 1e-2**2),
      ]
    )
  )
  prior = ManifoldGaussian(
    NavState.at_rest(position=(0.0, 0.0, -60.0), velocity=(1.0, 0.0, 0.0)),
    cov,
  )

  def nees(truth, belief):
    error = boxminus(truth, belief.mean)
    return error @ np.linalg.solve(belief.cov, error)

  filtered = np.zeros((trials, steps))
  smoothed = np.zeros((trials, steps))
  for trial in range(trials):
    belief, history, truths = prior, [], []
    for samples, aiding, truth in simulate(rng, prior, steps):
      step, _ = inertial_step(belief, samples, aiding, noise=NOISE)
      history.append(step)
      truths.append(truth)
      belief = step.posterior

    smooth = unscented_rts_smooth(prior, history)[1:]
    for k, truth in enumerate(truths):
      filtered[trial, k] = nees(truth, history[k].posterior)
      smoothed[trial, k] = nees(truth, smooth[k])

  return filtered, smoothed


def consistent_band(trials, steps):
  """Where a step's mean NEES over ``trials`` runs should sit.

  The sum of ``trials`` chi-squared(15) draws is chi-squared(15 * trials).
  The band is 99% over all ``steps`` together, Bonferroni-corrected.
  """
  tail = 0.005 / steps
  return chi2.ppf([tail, 1.0 - tail], DOF * trials) / trials


def test_the_filter_is_consistent_with_its_own_covariance(monte_carlo):
  """An overconfident filter lands above the band.

  That is the failure that matters here, since the smoothed covariance becomes
  the soundings' ``Sigma_q``.
  """
  filtered, _ = monte_carlo
  low, high = consistent_band(*filtered.shape)
  mean = filtered.mean(axis=0)
  assert np.all((low < mean) & (mean < high)), mean


def test_the_smoother_is_consistent_with_its_own_covariance(monte_carlo):
  """The smoothed covariance is what places a sounding, so it must be honest.

  This replaces a "smoothing never loosens the belief" check, which holds in a
  vector space and not here. The smoothed and filtered covariances live in the
  tangent spaces at different means, so comparing them compares two charts --
  and the recursion itself differences covariances from different charts. The
  gap measured 6-31% of the largest eigenvalue, while the NEES stayed inside
  its band at every step.
  """
  _, smoothed = monte_carlo
  low, high = consistent_band(*smoothed.shape)
  mean = smoothed.mean(axis=0)
  assert np.all((low < mean) & (mean < high)), mean


# -- the tick-driven navigator ----------------------------------------------


AIDING_NOISE = AidingNoise(DVL_NOISE, depth=0.05, magnetometer=0.02)


def navigator():
  prior = ManifoldGaussian(
    NavState.at_rest(position=(0.0, 0.0, -60.0), velocity=(1.0, 0.0, 0.0)),
    np.eye(DOF) * 1e-4,
  )
  return InertialNavigator(prior, DT, AIDING_NOISE, imu_noise=NOISE)


def ticks(rng, n):
  """``n`` noisy still-water IMU ticks with aiding on every sixth."""
  for index in range(n):
    aided = index % SAMPLES_PER_STEP == SAMPLES_PER_STEP - 1
    yield {
      "index": index,
      "gyro": rng.normal(scale=NOISE.gyro, size=3),
      "accel": -GRAVITY_NWU + rng.normal(scale=NOISE.accel, size=3),
      "dvl": np.array([1.0, 0.0, 0.0]) if aided else None,
      "depth": -60.0 if aided else None,
      "magnetometer": MAGNETIC_NORTH if aided else None,
    }


def test_a_cycle_closes_on_aiding_and_carries_every_sample_since():
  nav = navigator()
  for reading in ticks(np.random.default_rng(0), 18):
    nav.tick(**reading)

  assert nav.cycle_ticks == [5, 11, 17]
  assert all(len(v) == 3 for v in nav.nis.values())


def test_a_ping_closes_a_cycle_without_aiding():
  nav = navigator()
  still = {"gyro": np.zeros(3), "accel": -GRAVITY_NWU}
  assert nav.tick(0, **still) is None
  assert nav.tick(1, **still, close=True) is not None
  assert nav.cycle_ticks == [1]


def test_the_navigator_matches_calling_the_step_directly():
  """Buffering is bookkeeping only: the record is inertial_step's."""
  rng = np.random.default_rng(1)
  readings = list(ticks(rng, 6))

  nav = navigator()
  for reading in readings:
    nav.tick(**reading)

  samples = ImuSamples.uniform(
    [r["gyro"] for r in readings], [r["accel"] for r in readings], DT
  )
  last = readings[-1]
  step, _ = inertial_step(
    navigator().initial,
    samples,
    [
      Aiding(last["dvl"], dvl_reading, DVL_NOISE),
      Aiding(np.array([last["depth"]]), depth_reading, np.eye(1) * 0.05**2),
      Aiding(last["magnetometer"], magnetometer_reading, np.eye(3) * 0.02**2),
    ],
    noise=NOISE,
  )

  np.testing.assert_allclose(nav.belief.cov, step.posterior.cov, atol=1e-15)
  np.testing.assert_allclose(
    boxminus(nav.belief.mean, step.posterior.mean), 0.0, atol=1e-15
  )


def test_replaying_the_same_ticks_gives_the_same_record():
  readings = list(ticks(np.random.default_rng(2), 24))
  first, second = navigator(), navigator()
  for reading in readings:
    first.tick(**reading)
  for reading in readings:
    second.tick(**reading)

  for a, b in zip(first.history, second.history):
    np.testing.assert_array_equal(a.posterior.cov, b.posterior.cov)
    np.testing.assert_array_equal(a.cross_cov, b.cross_cov)
