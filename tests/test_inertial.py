"""The IMU motion model and its noise covariance, checked by Monte Carlo."""

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from auv_pose.estimation.inertial import (
  GRAVITY,
  ImuNoise,
  ImuSamples,
  imu_noise_covariance,
  propagate,
)
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
  quat_angle,
  quat_exp,
  quat_normalize,
  quat_to_rotmat,
)

RATE = 30.0
DT = 1.0 / RATE


def resting_samples(state, k=6):
  """A motionless vehicle's IMU: zero rate, specific force ``-g`` in body."""
  specific_force = -quat_to_rotmat(state.attitude).T @ GRAVITY
  return ImuSamples.uniform(
    gyro=np.zeros((k, 3)), accel=np.tile(specific_force, (k, 1)), dt=DT
  )


# -- propagation ------------------------------------------------------------


def test_a_resting_vehicle_stays_at_rest():
  """Catches a gravity sign slip, which is otherwise silent."""
  start = NavState.at_rest(position=np.array([10.0, -5.0, -65.0]))
  moved = propagate(start, resting_samples(start, k=30))

  np.testing.assert_allclose(moved.position, start.position, atol=1e-12)
  np.testing.assert_allclose(moved.velocity, np.zeros(3), atol=1e-12)
  assert quat_angle(moved.attitude, start.attitude) < 1e-15


def test_a_resting_tilted_vehicle_also_stays_at_rest():
  start = NavState.at_rest(attitude=quat_exp(np.array([0.3, -0.2, 1.1])))
  moved = propagate(start, resting_samples(start, k=30))

  np.testing.assert_allclose(moved.velocity, np.zeros(3), atol=1e-12)
  np.testing.assert_allclose(moved.position, start.position, atol=1e-12)


def test_free_fall_accelerates_at_gravity():
  """A zero accelerometer reading is free fall, not rest."""
  start = NavState.at_rest()
  samples = ImuSamples.uniform(
    gyro=np.zeros((30, 3)), accel=np.zeros((30, 3)), dt=DT
  )
  moved = propagate(start, samples)

  np.testing.assert_allclose(moved.velocity, GRAVITY * 1.0, atol=1e-12)


def test_a_constant_body_rate_integrates_to_the_closed_form_angle():
  rate = np.array([0.0, 0.0, 0.7])
  start = NavState.at_rest()
  samples = ImuSamples.uniform(
    gyro=np.tile(rate, (30, 1)), accel=np.zeros((30, 3)), dt=DT
  )

  moved = propagate(start, samples)
  assert quat_angle(moved.attitude, quat_exp(rate * 1.0)) < 1e-12


def test_the_gyro_bias_is_subtracted_from_the_rate():
  bias = np.array([0.0, 0.0, 0.2])
  start = replace(NavState.at_rest(), gyro_bias=bias)
  samples = ImuSamples.uniform(
    gyro=np.tile(bias, (30, 1)), accel=np.zeros((30, 3)), dt=DT
  )

  moved = propagate(start, samples)
  assert quat_angle(moved.attitude, start.attitude) < 1e-14


def test_the_accel_bias_is_subtracted_from_the_specific_force():
  bias = np.array([0.1, -0.2, 0.3])
  start = replace(NavState.at_rest(), accel_bias=bias)
  reading = -quat_to_rotmat(start.attitude).T @ GRAVITY + bias

  samples = ImuSamples.uniform(
    gyro=np.zeros((30, 3)), accel=np.tile(reading, (30, 1)), dt=DT
  )
  moved = propagate(start, samples)
  np.testing.assert_allclose(moved.velocity, np.zeros(3), atol=1e-12)


def test_the_biases_are_held_through_propagation():
  start = replace(
    NavState.at_rest(),
    gyro_bias=np.array([1.0, 2.0, 3.0]),
    accel_bias=np.array([4.0, 5.0, 6.0]),
  )
  moved = propagate(start, resting_samples(start))

  np.testing.assert_array_equal(moved.gyro_bias, start.gyro_bias)
  np.testing.assert_array_equal(moved.accel_bias, start.accel_bias)


def test_propagating_together_matches_propagating_one_at_a_time():
  rng = np.random.default_rng(0)
  start = NavState.at_rest(velocity=np.array([1.0, 0.5, -0.2]))
  samples = ImuSamples.uniform(
    gyro=rng.normal(size=(8, 3)) * 0.2,
    accel=rng.normal(size=(8, 3)) * 0.5 + np.array([0.0, 0.0, 9.81]),
    dt=DT,
  )

  together = propagate(start, samples)

  apart = start
  for i in range(len(samples)):
    apart = propagate(
      apart,
      ImuSamples.uniform(samples.gyro[i], samples.accel[i], DT),
    )

  np.testing.assert_allclose(apart.position, together.position, atol=1e-12)
  np.testing.assert_allclose(apart.velocity, together.velocity, atol=1e-12)
  assert quat_angle(apart.attitude, together.attitude) < 1e-14


# -- the noise covariance, structurally -------------------------------------


def test_the_covariance_is_symmetric_and_positive_semidefinite():
  start = NavState.at_rest()
  cov = imu_noise_covariance(start, resting_samples(start, k=30))

  np.testing.assert_allclose(cov, cov.T, atol=1e-18)
  assert np.min(np.linalg.eigvalsh(cov)) > -1e-18


def test_a_noiseless_imu_adds_nothing():
  start = NavState.at_rest()
  cov = imu_noise_covariance(
    start,
    resting_samples(start, k=30),
    ImuNoise(gyro=0.0, accel=0.0, gyro_bias=0.0, accel_bias=0.0),
  )
  np.testing.assert_array_equal(cov, np.zeros((DOF, DOF)))


def test_the_covariance_grows_with_every_sample():
  start = NavState.at_rest()
  traces = [
    np.trace(imu_noise_covariance(start, resting_samples(start, k=k)))
    for k in (1, 5, 10, 30)
  ]
  assert all(a < b for a, b in pairwise(traces))


def level_samples(rate, duration=2.0):
  """A level vehicle at rest, sampled at ``rate`` for ``duration`` seconds."""
  start = NavState.at_rest()
  force = -quat_to_rotmat(start.attitude).T @ GRAVITY
  k = round(rate * duration)
  return start, ImuSamples.uniform(
    np.zeros((k, 3)), np.tile(force, (k, 1)), 1 / rate
  )


@pytest.mark.parametrize("rate", [10.0, 30.0, 400.0])
def test_the_bias_blocks_are_the_random_walk_whatever_the_rate(rate):
  """Bias variance is ``density^2 * time``."""
  noise = ImuNoise()
  start, samples = level_samples(rate)
  cov = imu_noise_covariance(start, samples, noise)
  np.testing.assert_allclose(
    cov[GYRO_BIAS, GYRO_BIAS], noise.gyro_bias**2 * 2.0 * np.eye(3), rtol=1e-12
  )
  np.testing.assert_allclose(
    cov[ACCEL_BIAS, ACCEL_BIAS],
    noise.accel_bias**2 * 2.0 * np.eye(3),
    rtol=1e-12,
  )


@pytest.mark.parametrize("rate", [10.0, 30.0, 400.0])
def test_accelerometer_noise_walks_velocity_whatever_the_rate(rate):
  """Velocity variance is ``density^2 * time``."""
  noise = ImuNoise(gyro=0.0, accel=0.05, gyro_bias=0.0, accel_bias=0.0)
  start, samples = level_samples(rate)
  cov = imu_noise_covariance(start, samples, noise)
  np.testing.assert_allclose(
    cov[VELOCITY, VELOCITY], 0.05**2 * 2.0 * np.eye(3), rtol=1e-12
  )


def test_position_variance_approaches_the_continuous_limit():
  """Position variance tends to ``density^2 T^3 / 3``."""
  noise = ImuNoise(gyro=0.0, accel=0.05, gyro_bias=0.0, accel_bias=0.0)
  start, samples = level_samples(1000.0)
  cov = imu_noise_covariance(start, samples, noise)
  np.testing.assert_allclose(
    np.diag(cov[POSITION, POSITION]), 0.05**2 * 2.0**3 / 3, rtol=1e-3
  )


def test_gyro_noise_leaks_into_velocity_through_gravity():
  """Gyro noise alone reaches velocity and position via tilted gravity."""
  start = NavState.at_rest()
  cov = imu_noise_covariance(
    start,
    resting_samples(start, k=30),
    ImuNoise(gyro=0.01, accel=0.0, gyro_bias=0.0, accel_bias=0.0),
  )
  assert np.trace(cov[VELOCITY, VELOCITY]) > 0.0
  assert np.trace(cov[POSITION, POSITION]) > 0.0


# -- the noise covariance, against sampling ---------------------------------
#
# The noisy model is rewritten here with rotation matrices, independent of
# ``propagate`` and its quaternions, so a convention error cannot cancel.


def batch_exp(rotvec):
  """Rodrigues, batched over the leading axis. ``(n, 3) -> (n, 3, 3)``."""
  theta = np.linalg.norm(rotvec, axis=-1)[:, None]
  axis = rotvec / np.where(theta > 0.0, theta, 1.0)

  K = np.zeros((len(rotvec), 3, 3))
  K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
  K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
  K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]

  theta = theta[:, :, None]
  return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def batch_log(rotations):
  """Inverse of :func:`batch_exp` for small rotations. ``(n,3,3) -> (n,3)``."""
  vee = (
    np.stack(
      [
        rotations[:, 2, 1] - rotations[:, 1, 2],
        rotations[:, 0, 2] - rotations[:, 2, 0],
        rotations[:, 1, 0] - rotations[:, 0, 1],
      ],
      axis=-1,
    )
    / 2.0
  )
  # |vee| is sin(theta); valid for rotations well below pi/2.
  norm = np.linalg.norm(vee, axis=-1)
  scale = np.where(norm > 1e-12, np.arcsin(np.clip(norm, 0.0, 1.0)), norm)
  return vee * (scale / np.where(norm > 0.0, norm, 1.0))[:, None]


def sample_errors(state, samples, noise, rng, trials):
  """``(trials, 15)`` error states of noisy against noise-free propagation."""
  position = np.tile(state.position, (trials, 1))
  velocity = np.tile(state.velocity, (trials, 1))
  rotation = np.tile(quat_to_rotmat(state.attitude), (trials, 1, 1))
  gyro_bias = np.tile(state.gyro_bias, (trials, 1))
  accel_bias = np.tile(state.accel_bias, (trials, 1))

  for gyro, accel, dt in zip(samples.gyro, samples.accel, samples.dt):
    sigma = np.sqrt(np.diag(noise.covariance(dt)))
    eta_gyro = rng.normal(scale=sigma[0:3], size=(trials, 3))
    eta_accel = rng.normal(scale=sigma[3:6], size=(trials, 3))

    force = accel - accel_bias - eta_accel
    acceleration = np.einsum("nij,nj->ni", rotation, force) + GRAVITY

    position = position + dt * velocity + 0.5 * dt**2 * acceleration
    velocity = velocity + dt * acceleration
    rotation = rotation @ batch_exp((gyro - gyro_bias - eta_gyro) * dt)

    gyro_bias = gyro_bias + rng.normal(scale=sigma[6:9], size=(trials, 3))
    accel_bias = accel_bias + rng.normal(scale=sigma[9:12], size=(trials, 3))

  nominal = propagate(state, samples)
  nominal_rotation = quat_to_rotmat(nominal.attitude)

  return np.concatenate(
    [
      position - nominal.position,
      batch_log(nominal_rotation.T @ rotation),
      velocity - nominal.velocity,
      gyro_bias - nominal.gyro_bias,
      accel_bias - nominal.accel_bias,
    ],
    axis=-1,
  )


def per_sample(gyro, accel, gyro_bias, accel_bias, dt=DT):
  """The densities whose per-sample values at ``dt`` are the ones given."""
  root = np.sqrt(dt)
  return ImuNoise(
    gyro * root, accel * root, gyro_bias / root, accel_bias / root
  )


def turning_case():
  """A turning, accelerating second, so no Jacobian block is only at zero."""
  start = NavState.at_rest(
    attitude=quat_normalize(np.array([0.9, 0.1, -0.2, 0.3])),
    velocity=np.array([0.6, -0.2, 0.05]),
  )
  k = 30
  samples = ImuSamples.uniform(
    gyro=np.tile([0.15, -0.1, 0.25], (k, 1)),
    accel=np.tile(
      -quat_to_rotmat(start.attitude).T @ GRAVITY + [0.4, -0.3, 0.2],
      (k, 1),
    ),
    dt=DT,
  )
  return start, samples


def normalised(cov, reference):
  """``cov`` scaled to a correlation-like matrix by ``reference``'s diagonal."""
  scale = np.sqrt(np.outer(np.diag(reference), np.diag(reference)))
  return cov / scale


def test_the_covariance_matches_the_noise_it_claims_to_model():
  """Monte Carlo agrees with the analytic recursion over all fifteen states."""
  rng = np.random.default_rng(7)
  noise = per_sample(gyro=0.05, accel=0.2, gyro_bias=2e-3, accel_bias=5e-3)
  start, samples = turning_case()

  errors = sample_errors(start, samples, noise, rng, trials=20000)
  analytic = imu_noise_covariance(start, samples, noise)

  np.testing.assert_allclose(
    normalised(np.cov(errors, rowvar=False), analytic),
    normalised(analytic, analytic),
    atol=0.06,
  )


def test_the_sampled_error_is_centred():
  rng = np.random.default_rng(8)
  noise = per_sample(gyro=0.05, accel=0.2, gyro_bias=2e-3, accel_bias=5e-3)
  start, samples = turning_case()

  errors = sample_errors(start, samples, noise, rng, trials=20000)

  mean = errors.mean(axis=0)
  standard_error = errors.std(axis=0) / np.sqrt(len(errors))
  assert np.all(np.abs(mean) < 4.0 * standard_error + 1e-12)


def test_the_rotation_block_alone_matches_sampling():
  """Catches a right-versus-left perturbation mix-up."""
  rng = np.random.default_rng(9)
  noise = per_sample(gyro=0.05, accel=0.0, gyro_bias=5e-3, accel_bias=0.0)

  start = NavState.at_rest(attitude=quat_exp(np.array([0.2, -0.4, 0.9])))
  k = 30
  samples = ImuSamples.uniform(
    gyro=np.tile([0.3, 0.2, -0.4], (k, 1)), accel=np.zeros((k, 3)), dt=DT
  )

  errors = sample_errors(start, samples, noise, rng, trials=20000)[:, ROTATION]
  analytic = imu_noise_covariance(start, samples, noise)[ROTATION, ROTATION]

  np.testing.assert_allclose(
    normalised(np.cov(errors, rowvar=False), analytic),
    normalised(analytic, analytic),
    atol=0.04,
  )
