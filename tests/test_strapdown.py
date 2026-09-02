"""Open-loop inertial dead reckoning.

These pin the body-to-world conversion, which is otherwise only exercised
against the simulator. An accelerometer reports specific force, so at rest it
reads ``-G_NED`` and the recovered linear acceleration must be zero -- getting
that sign wrong is the classic way to make dead reckoning fall out of the sky.
"""

import numpy as np
import pytest

from auv_pose.estimation.quaternion import (
  G_NED,
  quat_from_gyro,
  quat_to_rotmat,
  rotmat_to_quat,
)
from auv_pose.estimation.strapdown import StrapdownIntegrator

LEVEL = np.array([1.0, 0.0, 0.0, 0.0])
AT_REST = -G_NED  # what an accelerometer reads when stationary and level


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
