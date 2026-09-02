"""Causal recursive estimators."""

import numpy as np
import pytest

from auv_pose.smoothing.filters import (
  AttitudeFilter,
  ConstantVelocityEKF,
  position,
  velocity,
)
from auv_pose.smoothing.quaternion import quat_to_rotmat, rotmat_to_quat
from auv_pose.smoothing.typing import GaussianState, Measurement

POSITION_H = np.hstack([np.eye(3), np.zeros((3, 3))])
DEPTH_H = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
DOWN = np.array([0.0, 0.0, 1.0])


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


def test_filter_stays_level_when_at_rest():
  """Gravity-only observations with no rotation: attitude must not wander."""
  filt = AttitudeFilter(kp=1.0, ki=0.05)
  for _ in range(200):
    filt.update(np.zeros(3), [DOWN, DOWN], dt=0.01)
  np.testing.assert_allclose(filt.rotation, np.eye(3), atol=1e-6)


def test_filter_tracks_pure_gyro_rotation():
  """With no usable accelerometer, the estimate is the gyro integral."""
  filt = AttitudeFilter()
  omega = np.array([0.0, 0.0, np.pi / 2])
  for _ in range(100):
    filt.update(omega, [np.zeros(3)], dt=0.01)

  np.testing.assert_allclose(
    filt.rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-6
  )


def test_filter_corrects_an_initial_tilt_error():
  """Starting wrong, level accelerometers should pull the estimate back."""
  tilt = rotmat_to_quat(quat_to_rotmat([np.cos(0.1), np.sin(0.1), 0.0, 0.0]))
  filt = AttitudeFilter(kp=1.0, q0=tilt)

  start = abs(filt.rotation[2, 2])
  for _ in range(500):
    filt.update(np.zeros(3), [DOWN], dt=0.01)

  assert abs(filt.rotation[2, 2]) > start
  assert abs(filt.rotation[2, 2]) == pytest.approx(1.0, abs=1e-4)


def test_cross_check_reports_solver_agreement():
  """The diagnostic is opt-in; both solvers must still agree when it is on."""
  filt = AttitudeFilter(cross_check=True)
  filt.update(np.zeros(3), [DOWN, DOWN], dt=0.01)
  assert filt.solver_disagreement < 1e-6
