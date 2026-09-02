"""Open-loop inertial dead reckoning.

These pin the body-to-world conversion, which is otherwise only exercised
against the simulator. An accelerometer reports specific force, so at rest it
reads ``-GRAVITY_NWU`` and the recovered linear acceleration must be zero -- getting
that sign wrong is the classic way to make dead reckoning fall out of the sky.
"""

import numpy as np
import pytest

from auv_pose.estimation.quaternion import (
  GRAVITY_NWU,
  quat_from_gyro,
  quat_to_rotmat,
  rotmat_to_quat,
)
from auv_pose.estimation.strapdown import StrapdownIntegrator

LEVEL = np.array([1.0, 0.0, 0.0, 0.0])
AT_REST = -GRAVITY_NWU  # what an accelerometer reads when stationary and level


def test_at_rest_does_not_move():
  """The whole point of specific force: gravity must cancel exactly."""
  dr = StrapdownIntegrator(np.zeros(3), LEVEL)

  for _ in range(1000):
    accel_world = dr.step(np.zeros(3), AT_REST, dt=0.01)
    np.testing.assert_allclose(accel_world, np.zeros(3), atol=1e-12)

  np.testing.assert_allclose(dr.position, np.zeros(3), atol=1e-12)
  np.testing.assert_allclose(dr.velocity, np.zeros(3), atol=1e-12)


def test_specific_force_is_rotated_before_gravity_is_removed():
  """Level, accelerating north: world acceleration is that acceleration."""
  dr = StrapdownIntegrator(np.zeros(3), LEVEL)
  accel_world = dr.step(np.zeros(3), AT_REST + np.array([2.0, 0.0, 0.0]), 0.01)
  np.testing.assert_allclose(accel_world, [2.0, 0.0, 0.0], atol=1e-12)


def test_rotation_is_applied_to_the_body_reading():
  """Yawed 90 degrees, a body-x push must come out as world y."""
  yawed = quat_from_gyro([0.0, 0.0, 1.0], np.pi / 2)
  dr = StrapdownIntegrator(np.zeros(3), yawed)

  body = quat_to_rotmat(yawed).T @ AT_REST + np.array([2.0, 0.0, 0.0])
  accel_world = dr.step(np.zeros(3), body, 0.01)

  np.testing.assert_allclose(accel_world, [0.0, 2.0, 0.0], atol=1e-9)


def test_constant_acceleration_integrates_quadratically():
  """s = at^2 / 2 after one second at 1 m/s^2."""
  dr = StrapdownIntegrator(np.zeros(3), LEVEL)
  dt, steps = 1e-4, 10_000
  push = AT_REST + np.array([1.0, 0.0, 0.0])

  for _ in range(steps):
    dr.step(np.zeros(3), push, dt)

  assert dr.velocity[0] == pytest.approx(1.0, rel=1e-6)
  assert dr.position[0] == pytest.approx(0.5, rel=1e-3)


def test_accelerometer_bias_produces_quadratic_drift():
  """The error growth that motivates terrain aiding in the first place."""
  bias = np.array([0.01, 0.0, 0.0])
  dr = StrapdownIntegrator(np.zeros(3), LEVEL)
  dt = 0.01

  drift = []
  for step in range(3000):
    dr.step(np.zeros(3), AT_REST + bias, dt)
    if step in (999, 1999, 2999):
      drift.append(dr.position[0])

  # Doubling then tripling the time should quadruple then multiply by nine.
  assert drift[1] / drift[0] == pytest.approx(4.0, rel=0.02)
  assert drift[2] / drift[0] == pytest.approx(9.0, rel=0.02)


def test_gyro_integrates_attitude():
  """A quarter turn about z in one second."""
  dr = StrapdownIntegrator(np.zeros(3), LEVEL)
  omega = np.array([0.0, 0.0, np.pi / 2])

  for _ in range(1000):
    dr.step(omega, AT_REST, dt=1e-3)

  np.testing.assert_allclose(
    dr.rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-6
  )


def test_attitude_stays_a_unit_quaternion():
  """Renormalisation must hold over a long run."""
  rng = np.random.default_rng(0)
  dr = StrapdownIntegrator(np.zeros(3), LEVEL)

  for _ in range(2000):
    dr.step(rng.normal(scale=0.5, size=3), AT_REST, dt=0.01)

  assert np.linalg.norm(dr.attitude) == pytest.approx(1.0, abs=1e-12)
  np.testing.assert_allclose(dr.rotation @ dr.rotation.T, np.eye(3), atol=1e-9)


def test_initial_state_is_copied_not_aliased():
  start = np.array([1.0, 2.0, 3.0])
  dr = StrapdownIntegrator(start, LEVEL)
  dr.step(np.zeros(3), AT_REST + np.array([1.0, 0.0, 0.0]), 0.1)

  np.testing.assert_allclose(start, [1.0, 2.0, 3.0])


def test_rotation_matches_the_attitude_quaternion():
  dr = StrapdownIntegrator(np.zeros(3), LEVEL)
  dr.step([0.1, 0.2, 0.3], AT_REST, 0.05)
  np.testing.assert_allclose(dr.rotation, quat_to_rotmat(dr.attitude))
  np.testing.assert_allclose(
    rotmat_to_quat(dr.rotation), dr.attitude, atol=1e-12
  )


def test_set_attitude_overrides_an_attached_filter():
  """A bare assignment is discarded; ``set_attitude`` must survive a step.

  ``step`` takes its next attitude from the filter's own state, so injecting a
  known attitude by assigning to :attr:`attitude` silently does nothing as soon
  as a filter is attached -- which is exactly how a ground-truth diagnostic
  quietly turns into a second copy of the run it was meant to bound.
  """
  from auv_pose.estimation.filters import AttitudeFilter

  turned = quat_from_gyro([0.0, 0.0, 1.0], np.pi / 2)

  drifted = StrapdownIntegrator(
    np.zeros(3), LEVEL, attitude_filter=AttitudeFilter(q0=LEVEL)
  )
  drifted.attitude = turned
  drifted.step(np.zeros(3), AT_REST, 0.01)
  assert not np.allclose(drifted.attitude, turned, atol=1e-3)

  injected = StrapdownIntegrator(
    np.zeros(3), LEVEL, attitude_filter=AttitudeFilter(q0=LEVEL)
  )
  injected.set_attitude(turned)
  injected.step(np.zeros(3), AT_REST, 0.01)
  assert np.allclose(injected.attitude, turned, atol=1e-3)


def test_initial_velocity_is_carried():
  """Starting from an assumed rest is a bias, not an offset.

  The simulated vehicle is dropped in negatively buoyant and is still sinking
  at 0.4 m/s when the first tick returns. An integrator started at rest there
  does not begin 0.4 m/s wrong and recover -- nothing observes velocity in the
  open loop, so it stays 0.4 m/s wrong and integrates that into 4 m of drift
  over 10 s.
  """
  dr = StrapdownIntegrator(np.zeros(3), LEVEL, velocity=[0.0, 0.0, -0.4])

  for _ in range(1000):
    dr.step(np.zeros(3), AT_REST, dt=0.01)

  np.testing.assert_allclose(dr.velocity, [0.0, 0.0, -0.4], atol=1e-12)
  np.testing.assert_allclose(dr.position, [0.0, 0.0, -4.0], atol=1e-9)


def test_kinematic_accel_reaches_the_attitude_filter():
  """Level and accelerating: the tilt the reading implies must not be taken.

  This is the coupling that made dead reckoning come out short. Without the
  compensation the attitude filter reads the manoeuvre as tilt, and ``step``
  then rotates by that tilt and cancels most of the same acceleration back out.
  """
  from auv_pose.estimation.filters import AttitudeFilter

  accel = np.array([0.5, 0.0, 0.0])
  reading = AT_REST + accel

  chased = StrapdownIntegrator(
    np.zeros(3), LEVEL, attitude_filter=AttitudeFilter(kp=1.0, ki=0.0)
  )
  compensated = StrapdownIntegrator(
    np.zeros(3), LEVEL, attitude_filter=AttitudeFilter(kp=1.0, ki=0.0)
  )

  chased_accel = np.zeros(3)
  compensated_accel = np.zeros(3)
  for _ in range(300):
    chased_accel = chased.step(np.zeros(3), reading, 1 / 30)
    compensated_accel = compensated.step(np.zeros(3), reading, 1 / 30, accel)

  # Most of the acceleration has been rotated away by the time the attitude
  # filter has settled on the manoeuvre.
  assert chased_accel[0] < 0.2
  np.testing.assert_allclose(compensated_accel, accel, atol=1e-4)


def test_kinematic_accel_does_not_enter_the_integration():
  """It is a gravity reference, not a velocity source.

  Subtracting it from the reading being integrated would quietly turn dead
  reckoning into a doubly-integrated DVL, which is not the baseline this
  exists to measure.
  """
  from auv_pose.estimation.filters import AttitudeFilter

  dr = StrapdownIntegrator(
    np.zeros(3), LEVEL, attitude_filter=AttitudeFilter(kp=0.0, ki=0.0)
  )
  accel_world = dr.step(
    np.zeros(3), AT_REST + [1.0, 0.0, 0.0], 0.01, [7.0, 0.0, 0.0]
  )

  np.testing.assert_allclose(accel_world, [1.0, 0.0, 0.0], atol=1e-9)


def test_set_attitude_without_a_filter():
  """Same contract when attitude comes from the gyro alone."""
  integrator = StrapdownIntegrator(np.zeros(3), LEVEL)
  turned = quat_from_gyro([0.0, 0.0, 1.0], np.pi / 2)
  integrator.set_attitude(turned)
  integrator.step(np.zeros(3), AT_REST, 0.01)
  assert np.allclose(integrator.attitude, turned, atol=1e-3)


# The simulator's world is z-up while a sensor in the IMU socket reports in a
# z-down body frame, so a level vehicle's body-to-world rotation is a half turn
# about x, not identity. This is the geometry the frame constants exist for.
Z_DOWN_BODY = rotmat_to_quat(np.diag([1.0, -1.0, -1.0]))
TRUE_ACCEL = np.array([1.0, 2.0, 0.5])


def _specific_force(accel_world, attitude, gravity):
  """What an accelerometer in that body frame would read."""
  R = quat_to_rotmat(attitude)
  return R.T @ (np.asarray(accel_world) - np.asarray(gravity))


def test_z_down_body_in_a_z_up_world_recovers_the_acceleration():
  integrator = StrapdownIntegrator(
    np.zeros(3), Z_DOWN_BODY, gravity=GRAVITY_NWU
  )
  reading = _specific_force(TRUE_ACCEL, Z_DOWN_BODY, GRAVITY_NWU)
  recovered = integrator.step(np.zeros(3), reading, 0.01)
  np.testing.assert_allclose(recovered, TRUE_ACCEL, atol=1e-9)


def test_mismatched_frame_and_gravity_mirror_each_other_at_rest():
  """Why this was invisible for so long.

  Taking orientation from the wrong socket (identity instead of a half turn)
  and gravity from the wrong convention are two errors that cancel exactly when
  the vehicle is at rest. Nothing falls out of the sky, no test of the resting
  case fails, and the pair survives.
  """
  from auv_pose.estimation.quaternion import GRAVITY_NED

  at_rest = _specific_force(np.zeros(3), Z_DOWN_BODY, GRAVITY_NWU)
  wrong = StrapdownIntegrator(np.zeros(3), LEVEL, gravity=GRAVITY_NED)
  np.testing.assert_allclose(
    wrong.step(np.zeros(3), at_rest, 0.01), np.zeros(3), atol=1e-9
  )


def test_mismatched_frame_and_gravity_mirror_y_and_z_once_moving():
  """And what it costs: every recovered acceleration is mirrored in y and z."""
  from auv_pose.estimation.quaternion import GRAVITY_NED

  reading = _specific_force(TRUE_ACCEL, Z_DOWN_BODY, GRAVITY_NWU)
  wrong = StrapdownIntegrator(np.zeros(3), LEVEL, gravity=GRAVITY_NED)
  np.testing.assert_allclose(
    wrong.step(np.zeros(3), reading, 0.01),
    TRUE_ACCEL * np.array([1.0, -1.0, -1.0]),
    atol=1e-9,
  )
