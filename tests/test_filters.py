"""Causal recursive estimators."""

import numpy as np
import pytest

from auv_pose.estimation.filters import (
  AttitudeFilter,
  ConstantVelocityEKF,
  gravity_trust,
  position,
  velocity,
)
from auv_pose.estimation.quaternion import (
  G_NED,
  quat_angle,
  quat_from_gyro,
  quat_to_rotmat,
)
from auv_pose.estimation.typing import GaussianState, Measurement

POSITION_H = np.hstack([np.eye(3), np.zeros((3, 3))])
DEPTH_H = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])


def position_of(mean, cov=None):
  return GaussianState(
    mean=np.asarray(mean, dtype=float),
    cov=np.eye(6) if cov is None else cov,
  )


def test_initial_defaults_to_rest_and_unit_covariance():
  state = ConstantVelocityEKF.initial([1.0, 2.0, 3.0])
  np.testing.assert_allclose(position(state), [1.0, 2.0, 3.0])
  np.testing.assert_allclose(velocity(state), np.zeros(3))
  np.testing.assert_allclose(state.cov, np.eye(6))


def test_predict_matches_hand_computed_step():
  """x = x0 + v0 dt + a dt^2 / 2, v = v0 + a dt."""
  ekf = ConstantVelocityEKF()
  state = ekf.predict(
    position_of([1.0, 2.0, 3.0, 0.5, 0.0, -0.5]), [0.0, 2.0, 0.0], dt=2.0
  )

  np.testing.assert_allclose(position(state), [2.0, 6.0, 2.0])
  np.testing.assert_allclose(velocity(state), [0.5, 4.0, -0.5])


def test_predict_does_not_mutate_its_input():
  """State is a value; a filter must not write through it."""
  ekf = ConstantVelocityEKF()
  before = ConstantVelocityEKF.initial([1.0, 2.0, 3.0])

  ekf.predict(before, [1.0, 1.0, 1.0], dt=0.5)

  np.testing.assert_allclose(before.mean, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
  np.testing.assert_allclose(before.cov, np.eye(6))


def test_condition_does_not_mutate_its_input():
  ekf = ConstantVelocityEKF()
  before = ConstantVelocityEKF.initial([0.0, 0.0, 0.0])

  ekf.condition(before, Measurement([9.0], DEPTH_H, np.array([[0.1]])))

  np.testing.assert_allclose(before.mean, np.zeros(6))
  np.testing.assert_allclose(before.cov, np.eye(6))


def test_predict_grows_uncertainty():
  ekf = ConstantVelocityEKF()
  state = ConstantVelocityEKF.initial(np.zeros(3))
  assert np.trace(ekf.predict(state, np.zeros(3), 0.1).cov) > np.trace(
    state.cov
  )


def test_condition_shrinks_uncertainty():
  ekf = ConstantVelocityEKF()
  state = ekf.predict(
    ConstantVelocityEKF.initial(np.zeros(3)), np.zeros(3), 0.1
  )
  updated = ekf.condition(
    state, Measurement(np.zeros(3), POSITION_H, np.eye(3) * 0.5)
  )
  assert np.trace(updated.cov) < np.trace(state.cov)


def test_condition_keeps_covariance_symmetric_positive_definite():
  ekf = ConstantVelocityEKF()
  state = ConstantVelocityEKF.initial(np.zeros(3))
  obs = Measurement([0.0, 0.0, 1.0], POSITION_H, np.eye(3) * 0.25)

  for _ in range(20):
    state = ekf.step(state, [0.1, 0.0, 0.0], 0.05, [obs])

  np.testing.assert_allclose(state.cov, state.cov.T, atol=1e-12)
  assert np.all(np.linalg.eigvalsh(state.cov) > 0)


def test_perfect_measurement_pulls_state_to_it():
  ekf = ConstantVelocityEKF()
  state = ekf.predict(
    ConstantVelocityEKF.initial([10.0, 10.0, 10.0]), np.zeros(3), 0.1
  )
  updated = ekf.condition(
    state, Measurement(np.zeros(3), POSITION_H, np.eye(3) * 1e-9)
  )
  np.testing.assert_allclose(position(updated), np.zeros(3), atol=1e-4)


def test_stacked_measurement_equals_sequential_conditioning():
  """Two independent observations, stacked or applied in turn, agree.

  This is why navigate.py folds its two depth observations into one
  Measurement rather than conditioning twice per step.
  """
  z1, z2, r1, r2 = 4.0, 6.0, 0.5, 2.0
  ekf = ConstantVelocityEKF()
  prior = ekf.predict(
    ConstantVelocityEKF.initial([0.0, 0.0, 5.0]), np.zeros(3), 0.1
  )

  stacked = ekf.condition(
    prior,
    Measurement([z1, z2], np.vstack([DEPTH_H, DEPTH_H]), np.diag([r1, r2])),
  )

  sequential = ekf.condition(prior, Measurement([z1], DEPTH_H, [[r1]]))
  sequential = ekf.condition(sequential, Measurement([z2], DEPTH_H, [[r2]]))

  np.testing.assert_allclose(stacked.mean, sequential.mean, atol=1e-10)
  np.testing.assert_allclose(stacked.cov, sequential.cov, atol=1e-10)


def test_depth_only_leaves_horizontal_position_untouched():
  """With H measuring z alone, x and y are unobservable."""
  ekf = ConstantVelocityEKF()
  state = ekf.predict(
    ConstantVelocityEKF.initial([3.0, -4.0, 0.0]), np.zeros(3), 0.1
  )
  updated = ekf.condition(state, Measurement([100.0], DEPTH_H, [[0.1]]))
  np.testing.assert_allclose(position(updated)[:2], [3.0, -4.0], atol=1e-12)


def test_transition_is_the_constant_velocity_matrix():
  F = ConstantVelocityEKF().transition(0.5)
  expected = np.eye(6)
  expected[:3, 3:] = np.eye(3) * 0.5
  np.testing.assert_allclose(F, expected)


def test_step_records_one_entry_per_call():
  ekf = ConstantVelocityEKF()
  state = ConstantVelocityEKF.initial(np.zeros(3))
  obs = Measurement(np.zeros(3), POSITION_H, np.eye(3))

  for _ in range(5):
    state = ekf.step(state, np.zeros(3), 0.1, [obs])

  assert len(ekf.history) == 5


def test_step_records_prior_and_posterior():
  ekf = ConstantVelocityEKF()
  state = ConstantVelocityEKF.initial([0.0, 0.0, 0.0])
  ekf.step(state, np.zeros(3), 0.1, [Measurement([5.0], DEPTH_H, [[0.01]])])

  recorded = ekf.history[0]
  assert recorded.prior.mean[2] == pytest.approx(0.0)
  assert recorded.posterior.mean[2] > 4.0
  np.testing.assert_allclose(recorded.transition, ekf.transition(0.1))


def test_step_with_no_observations_is_prediction_only():
  ekf = ConstantVelocityEKF()
  state = ConstantVelocityEKF.initial(np.zeros(3))
  stepped = ekf.step(state, [1.0, 0.0, 0.0], 0.1)

  recorded = ekf.history[0]
  np.testing.assert_allclose(recorded.prior.mean, recorded.posterior.mean)
  np.testing.assert_allclose(stepped.mean, recorded.prior.mean)


def test_history_starts_empty():
  assert ConstantVelocityEKF().history == []


LEVEL_ACCEL = -G_NED  # what an accelerometer reads at rest, level


def test_gravity_trust_peaks_at_one_g():
  assert gravity_trust(-G_NED) == pytest.approx(1.0)
  assert gravity_trust([0.0, 0.0, -19.62]) < gravity_trust([0.0, 0.0, -11.0])
  assert 0.0 < gravity_trust([0.0, 0.0, -30.0]) < 1.0


def test_attitude_filter_holds_level_at_rest():
  filt = AttitudeFilter()
  for _ in range(200):
    filt.update(np.zeros(3), LEVEL_ACCEL, dt=0.01)
  np.testing.assert_allclose(filt.rotation, np.eye(3), atol=1e-6)


def test_attitude_filter_corrects_a_tilt_error():
  """Started wrong in roll, the accelerometer should pull it level."""
  tilted = quat_from_gyro([1.0, 0.0, 0.0], 0.3)  # 17 degrees of roll error
  filt = AttitudeFilter(kp=2.0, q0=tilted)

  before = quat_angle(filt.q, IDENTITY_Q)
  for _ in range(2000):
    filt.update(np.zeros(3), LEVEL_ACCEL, dt=0.005)
  after = quat_angle(filt.q, IDENTITY_Q)

  assert after < before / 10


def test_attitude_filter_does_not_touch_heading():
  """The whole reason this is a cross product and not a Wahba solve.

  Gravity carries no heading information. A yaw error must survive untouched --
  solving Wahba's problem with a single gravity vector would instead drag the
  estimate to zero yaw.
  """
  yawed = quat_from_gyro([0.0, 0.0, 1.0], np.pi / 3)  # 60 degrees of yaw
  filt = AttitudeFilter(kp=2.0, q0=yawed)

  # Level and at rest, so the accelerometer reads pure gravity: nothing about
  # this observation says anything about heading.
  body_accel = quat_to_rotmat(yawed).T @ LEVEL_ACCEL
  for _ in range(2000):
    filt.update(np.zeros(3), body_accel, dt=0.005)

  assert quat_angle(filt.q, yawed) == pytest.approx(0.0, abs=1e-3)


def test_attitude_filter_estimates_gyro_bias():
  """A constant gyro bias about a horizontal axis must be learned out."""
  bias = np.array([0.02, -0.01, 0.0])
  filt = AttitudeFilter(kp=1.0, ki=0.3)

  for _ in range(20000):
    filt.update(bias, LEVEL_ACCEL, dt=0.005)

  np.testing.assert_allclose(filt.bias[:2], bias[:2], atol=5e-3)
  np.testing.assert_allclose(filt.rotation, np.eye(3), atol=1e-3)


def test_attitude_filter_distrusts_a_manoeuvring_accelerometer():
  """Under heavy acceleration the reading is not a gravity reference."""
  filt = AttitudeFilter()
  filt.update(np.zeros(3), LEVEL_ACCEL, dt=0.01)
  resting = filt.trust

  filt.update(np.zeros(3), LEVEL_ACCEL + np.array([20.0, 0.0, 0.0]), dt=0.01)
  assert filt.trust < resting / 10


def test_attitude_filter_ignores_a_dead_accelerometer():
  """Zero reading carries no information; fall back to the gyro."""
  filt = AttitudeFilter()
  omega = np.array([0.0, 0.0, np.pi / 2])
  for _ in range(100):
    filt.update(omega, np.zeros(3), dt=0.01)

  assert filt.trust == 0.0
  np.testing.assert_allclose(
    filt.rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-6
  )
