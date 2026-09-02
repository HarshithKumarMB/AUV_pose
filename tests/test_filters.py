"""Causal recursive estimators."""

import numpy as np
import pytest

from auv_pose.estimation.filters import (
  AttitudeFilter,
  BodyAcceleration,
  ConstantVelocityEKF,
  gravity_trust,
  position,
  velocity,
)
from auv_pose.estimation.quaternion import (
  GRAVITY_NWU,
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


LEVEL_ACCEL = -GRAVITY_NWU  # what an accelerometer reads at rest, level


def test_gravity_trust_peaks_at_one_g():
  assert gravity_trust(-GRAVITY_NWU) == pytest.approx(1.0)
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


def test_gravity_trust_is_blind_to_a_horizontal_manoeuvre():
  """Why distrusting on magnitude was not enough.

  A horizontal component adds to gravity in quadrature, so the acceleration an
  AUV actually flies barely moves ``|f|`` -- while tilting it by degrees. The
  20 m/s^2 case above is the one a magnitude test can see; this is the one that
  matters, and it sails straight through.
  """
  cruise = LEVEL_ACCEL + np.array([0.5, 0.0, 0.0])

  assert gravity_trust(cruise) > 0.99
  assert np.degrees(np.arctan(0.5 / 9.81)) == pytest.approx(2.9, abs=0.1)


def test_attitude_filter_chases_an_unmodelled_manoeuvre():
  """The defect ``kinematic_accel`` exists to fix.

  Level, perfect gyro, accelerating gently forward. The specific force tilts by
  ``atan(a/g)`` and the filter follows it, because from one accelerometer a
  manoeuvre and a tilt are the same observation. What makes this expensive is
  not the attitude error itself but that the strapdown integrating the same
  reading then rotates the acceleration back out from under itself, so dead
  reckoning comes out short rather than merely noisy.

  Detecting the manoeuvre instead of subtracting it is the obvious alternative
  when there is no DVL, and it does not work -- see the note in
  ``visuals/README.md``. Any such test must distinguish a manoeuvre from a
  rotation, and specific force changes under both.
  """
  accel = np.array([0.5, 0.0, 0.0])
  reading = LEVEL_ACCEL + accel

  filt = AttitudeFilter(kp=1.0, ki=0.0)
  for _ in range(300):
    filt.update(np.zeros(3), reading, dt=1 / 30)

  assert np.degrees(quat_angle(filt.q, IDENTITY_Q)) == pytest.approx(
    np.degrees(np.arctan(0.5 / 9.81)), abs=0.2
  )


def test_kinematic_accel_makes_the_reference_survive_a_manoeuvre():
  """The same run, told what the vehicle is doing."""
  accel = np.array([0.5, 0.0, 0.0])
  reading = LEVEL_ACCEL + accel

  filt = AttitudeFilter(kp=1.0, ki=0.0)
  for _ in range(300):
    filt.update(np.zeros(3), reading, dt=1 / 30, kinematic_accel=accel)

  assert np.degrees(quat_angle(filt.q, IDENTITY_Q)) < 0.01
  assert filt.trust > 0.999


def test_body_acceleration_recovers_a_straight_line_acceleration():
  """Differencing the DVL, once the low-pass has caught up."""
  body_accel = BodyAcceleration(tau=0.2)
  dt = 1 / 30

  speed = 0.0
  for _ in range(300):
    speed += 0.5 * dt
    recovered = body_accel.update([speed, 0.0, 0.0], np.zeros(3), dt)

  np.testing.assert_allclose(recovered, [0.5, 0.0, 0.0], atol=0.01)


def test_body_acceleration_includes_the_transport_term():
  """A body frame turning under a constant body velocity still accelerates.

  ``omega x v`` needs no differencing and no smoothing, so it is exact from the
  second sample -- and in a turn at speed it is the larger of the two terms.
  """
  body_accel = BodyAcceleration(tau=0.5)
  velocity_body = np.array([1.0, 0.0, 0.0])
  omega = np.array([0.0, 0.0, 0.5])
  dt = 1 / 30

  body_accel.update(velocity_body, omega, dt)
  recovered = body_accel.update(velocity_body, omega, dt)

  np.testing.assert_allclose(recovered, [0.0, 0.5, 0.0], atol=1e-12)


def test_body_acceleration_starts_at_zero():
  """One sample cannot be differenced; say so rather than inventing a step."""
  body_accel = BodyAcceleration()
  first = body_accel.update([1.0, 2.0, 3.0], np.zeros(3), 1 / 30)
  np.testing.assert_allclose(first, np.zeros(3))
