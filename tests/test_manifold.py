"""The state manifold, its ``+``/``-`` chart, and the SO(3) maps under it."""

import numpy as np
import pytest

from auv_pose.estimation.manifold import (
  DOF,
  ROTATION,
  NavState,
  covariance_transport,
  manifold_mean,
)
from auv_pose.estimation.quaternion import (
  quat_angle,
  quat_conjugate,
  quat_exp,
  quat_log,
  quat_multiply,
  quat_normalize,
  quat_to_rotmat,
  skew,
  so3_right_jacobian,
)


def random_state(rng):
  """A state with nothing zero, so a dropped term cannot hide."""
  return NavState(
    position=rng.normal(size=3) * 10.0,
    attitude=quat_normalize(rng.normal(size=4)),
    velocity=rng.normal(size=3),
    gyro_bias=rng.normal(size=3) * 0.01,
    accel_bias=rng.normal(size=3) * 0.1,
  )


# -- exp and log ------------------------------------------------------------


def test_exp_log_round_trip():
  rng = np.random.default_rng(0)
  for _ in range(50):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    theta = rng.uniform(0.0, np.pi - 1e-6)
    rotvec = axis * theta
    np.testing.assert_allclose(quat_log(quat_exp(rotvec)), rotvec, atol=1e-12)


def test_exp_log_round_trip_at_the_awkward_angles():
  """Zero, denormal-small, and just under a half turn."""
  for rotvec in [
    np.zeros(3),
    np.array([1e-12, 0.0, 0.0]),
    np.array([1e-7, -2e-7, 3e-8]),
    np.array([0.0, 0.0, np.pi - 1e-9]),
  ]:
    np.testing.assert_allclose(quat_log(quat_exp(rotvec)), rotvec, atol=1e-14)


def test_log_of_identity_is_exactly_zero():
  """Not NaN."""
  assert np.all(quat_log(np.array([1.0, 0.0, 0.0, 0.0])) == 0.0)


def test_log_folds_a_rotation_past_half_a_turn():
  """The result has norm at most pi and names the same orientation."""
  rotvec = np.array([0.0, 0.0, 1.5 * np.pi])
  folded = quat_log(quat_exp(rotvec))

  assert np.linalg.norm(folded) <= np.pi + 1e-12
  assert quat_angle(quat_exp(folded), quat_exp(rotvec)) < 1e-12


def test_exp_matches_rodrigues():
  rng = np.random.default_rng(2)
  for _ in range(20):
    rotvec = rng.normal(size=3) * 0.7
    theta = np.linalg.norm(rotvec)
    W = skew(rotvec / theta)
    rodrigues = np.eye(3) + np.sin(theta) * W + (1 - np.cos(theta)) * (W @ W)
    np.testing.assert_allclose(
      quat_to_rotmat(quat_exp(rotvec)), rodrigues, atol=1e-12
    )


# -- the right Jacobian -----------------------------------------------------


def test_right_jacobian_is_the_identity_at_zero():
  np.testing.assert_allclose(so3_right_jacobian(np.zeros(3)), np.eye(3))


def test_right_jacobian_matches_barfoots_closed_form():
  """Against Barfoot eq. 8.82a, via ``J_l(-phi) = J_r(phi)``."""
  rng = np.random.default_rng(14)
  for _ in range(20):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    theta = rng.uniform(1e-3, np.pi)

    sinc = np.sin(theta) / theta
    barfoot = (
      sinc * np.eye(3)
      + (1.0 - sinc) * np.outer(axis, axis)
      - ((1.0 - np.cos(theta)) / theta) * skew(axis)
    )

    np.testing.assert_allclose(
      so3_right_jacobian(axis * theta), barfoot, atol=1e-12
    )


def test_right_jacobian_is_continuous_across_its_series_threshold():
  """No step where the implementation switches to the series, at 1e-2."""
  axis = np.array([0.6, -0.8, 0.0])
  delta = 1e-12

  below = so3_right_jacobian(axis * (1e-2 - delta))
  above = so3_right_jacobian(axis * (1e-2 + delta))

  # The residual is the closed form's own rounding at this angle.
  np.testing.assert_allclose(below, above, rtol=1e-9, atol=1e-17)


def test_right_jacobian_is_accurate_where_cancellation_bites():
  """Small angles match the exact series, where the closed form would not."""
  axis = np.array([0.0, 0.0, 1.0])
  for theta in (1e-6, 1e-5, 1e-4, 1e-3):
    coeff_w = 0.5 - theta**2 / 24.0 + theta**4 / 720.0
    coeff_ww = 1.0 / 6.0 - theta**2 / 120.0 + theta**4 / 5040.0
    W = skew(axis * theta)
    expected = np.eye(3) - coeff_w * W + coeff_ww * (W @ W)

    np.testing.assert_allclose(
      so3_right_jacobian(axis * theta), expected, rtol=1e-15, atol=1e-18
    )


def test_right_jacobian_matches_finite_differences():
  """``exp(phi + d) ~= exp(phi) exp(Jr(phi) d)``."""
  phi = np.array([0.3, -0.2, 0.5])
  jacobian = so3_right_jacobian(phi)
  eps = 1e-7

  for axis in range(3):
    step = np.zeros(3)
    step[axis] = eps
    measured = quat_log(
      quat_multiply(quat_conjugate(quat_exp(phi)), quat_exp(phi + step))
    )
    np.testing.assert_allclose(measured / eps, jacobian[:, axis], atol=1e-6)


# -- state + xi and state - other ------------------------------------------


def test_boxminus_inverts_boxplus():
  rng = np.random.default_rng(4)
  for _ in range(50):
    state = random_state(rng)
    xi = rng.normal(size=DOF) * 0.3
    np.testing.assert_allclose(((state + xi) - state), xi, atol=1e-12)


def test_boxplus_inverts_boxminus():
  rng = np.random.default_rng(5)
  for _ in range(50):
    a, b = random_state(rng), random_state(rng)
    recovered = b + (a - b)

    np.testing.assert_allclose(recovered.position, a.position, atol=1e-12)
    np.testing.assert_allclose(recovered.velocity, a.velocity, atol=1e-12)
    assert quat_angle(recovered.attitude, a.attitude) < 1e-10


def test_boxminus_of_a_state_with_itself_vanishes():
  """The vector blocks cancel exactly; the attitude only to rounding."""
  state = random_state(np.random.default_rng(6))
  delta = state - state

  assert np.all(delta[:3] == 0.0)
  assert np.all(delta[6:] == 0.0)
  np.testing.assert_allclose(delta[ROTATION], np.zeros(3), atol=1e-15)


def test_boxplus_perturbs_the_rotation_on_the_right():
  """The rotation increment is applied in the body frame, not the world."""
  # Yaw 90 degrees: body +x points along world +y.
  state = NavState.at_rest(attitude=quat_exp(np.array([0.0, 0.0, np.pi / 2])))
  # A further quarter turn about *body* +x.
  turned = state + np.concatenate([np.zeros(3), [np.pi / 2, 0, 0], np.zeros(9)])

  # A body-x turn leaves body +x on world +y; a world-x turn would not.
  np.testing.assert_allclose(
    turned.rotation @ np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    atol=1e-12,
  )


def test_an_increment_on_the_left_is_refused_not_broadcast():
  with pytest.raises(TypeError):
    _ = np.zeros(DOF) + NavState.at_rest()  # pyright: ignore[reportOperatorIssue]


@pytest.mark.parametrize("shape", [(DOF - 1,), (1, DOF), ()])
def test_an_increment_of_the_wrong_shape_is_refused(shape):
  with pytest.raises(ValueError, match="increment"):
    _ = NavState.at_rest() + np.zeros(shape)


def test_boxplus_only_touches_the_blocks_it_is_given():
  state = random_state(np.random.default_rng(7))
  xi = np.zeros(DOF)
  xi[0] = 1.0

  moved = state + xi
  np.testing.assert_allclose(moved.position, state.position + [1, 0, 0])
  np.testing.assert_allclose(moved.velocity, state.velocity)
  np.testing.assert_allclose(moved.gyro_bias, state.gyro_bias)
  np.testing.assert_allclose(moved.accel_bias, state.accel_bias)
  assert quat_angle(moved.attitude, state.attitude) < 1e-15


# -- covariance transport ---------------------------------------------------


def test_transport_is_the_identity_for_a_zero_increment():
  np.testing.assert_allclose(covariance_transport(np.zeros(DOF)), np.eye(DOF))


def test_transport_leaves_the_vector_blocks_alone():
  xi = np.random.default_rng(8).normal(size=DOF)
  jacobian = covariance_transport(xi)

  flat = np.ones(DOF, dtype=bool)
  flat[ROTATION] = False
  np.testing.assert_allclose(
    jacobian[np.ix_(flat, flat)], np.eye(DOF - 3), atol=1e-15
  )


# -- the intrinsic mean -----------------------------------------------------


def test_mean_of_one_state_is_that_state():
  state = random_state(np.random.default_rng(9))
  mean = manifold_mean([state], [1.0])

  np.testing.assert_allclose(mean.position, state.position, atol=1e-12)
  assert quat_angle(mean.attitude, state.attitude) < 1e-12


def test_mean_recovers_the_centre_of_a_symmetric_cloud():
  rng = np.random.default_rng(10)
  centre = random_state(rng)

  offsets = rng.normal(size=(8, DOF)) * 0.1
  states = [(centre + xi) for xi in offsets] + [
    (centre + -xi) for xi in offsets
  ]
  weights = np.full(len(states), 1.0 / len(states))

  mean = manifold_mean(states, weights, initial=centre)
  np.testing.assert_allclose(mean.position, centre.position, atol=1e-12)
  assert quat_angle(mean.attitude, centre.attitude) < 1e-10


def test_mean_of_two_rotations_is_the_midpoint():
  a = NavState.at_rest(attitude=quat_exp(np.array([0.0, 0.0, -0.4])))
  b = NavState.at_rest(attitude=quat_exp(np.array([0.0, 0.0, 0.4])))

  mean = manifold_mean([a, b], [0.5, 0.5])
  np.testing.assert_allclose(quat_log(mean.attitude), np.zeros(3), atol=1e-10)


def test_mean_weights_are_respected():
  a = NavState.at_rest(position=np.zeros(3))
  b = NavState.at_rest(position=np.array([10.0, 0.0, 0.0]))

  mean = manifold_mean([a, b], [0.25, 0.75])
  np.testing.assert_allclose(mean.position, [7.5, 0.0, 0.0], atol=1e-12)


def test_mean_gets_the_vector_blocks_right_in_one_pass():
  """One pass lands the vector blocks exactly; the attitude still iterates."""
  rng = np.random.default_rng(11)
  centre = random_state(rng)
  states = [(centre + xi) for xi in rng.normal(size=(6, DOF)) * 0.05]
  weights = np.full(6, 1.0 / 6)

  seed = states[3]
  converged = manifold_mean(states, weights)

  one_pass = seed + sum(w * (s - seed) for s, w in zip(states, weights))

  np.testing.assert_allclose(one_pass.position, converged.position, atol=1e-12)
  np.testing.assert_allclose(one_pass.velocity, converged.velocity, atol=1e-12)
  np.testing.assert_allclose(
    one_pass.gyro_bias, converged.gyro_bias, atol=1e-12
  )
  np.testing.assert_allclose(
    one_pass.accel_bias, converged.accel_bias, atol=1e-12
  )
  assert quat_angle(one_pass.attitude, converged.attitude) > 1e-9


def test_mean_raises_rather_than_returning_an_unconverged_attitude():
  spread = [
    NavState.at_rest(attitude=quat_exp(v))
    for v in [
      np.array([0.0, 0.0, 0.0]),
      np.array([0.0, 0.0, 2.6]),
      np.array([0.0, 2.6, 0.0]),
      np.array([2.6, 0.0, 0.0]),
    ]
  ]
  try:
    manifold_mean(spread, np.full(4, 0.25), max_iter=1)
  except ValueError as error:
    assert "did not converge" in str(error)
    return
  raise AssertionError("expected a ValueError for an unconverged mean")


def test_mean_rejects_mismatched_weights():
  state = random_state(np.random.default_rng(12))
  try:
    manifold_mean([state, state], [1.0])
  except ValueError:
    return
  raise AssertionError("expected a ValueError for a weight count mismatch")
