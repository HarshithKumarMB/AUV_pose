"""The state manifold and its chart.

The round-trip tests carry most of the weight: boxplus and boxminus are each
other's inverse by construction, and every moment the estimators compute is a
weighted sum of boxminus results mapped back through boxplus. A sign or a
transpose wrong in either one is silent in a single step and fatal over a run.
"""

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
  quat_from_gyro,
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
  """Not NaN. The quotient it comes from is 0/0 without the series branch."""
  assert np.all(quat_log(np.array([1.0, 0.0, 0.0, 0.0])) == 0.0)


def test_log_folds_a_rotation_past_half_a_turn():
  """``q`` and ``-q`` are one rotation, so the result stays within pi."""
  rotvec = np.array([0.0, 0.0, 1.5 * np.pi])
  folded = quat_log(quat_exp(rotvec))

  assert np.linalg.norm(folded) <= np.pi + 1e-12
  # Same orientation, expressed the short way round.
  assert quat_angle(quat_exp(folded), quat_exp(rotvec)) < 1e-12


def test_exp_agrees_with_the_gyro_increment_already_in_use():
  """``quat_from_gyro`` now delegates; this pins that it still means the same."""
  rng = np.random.default_rng(1)
  for _ in range(20):
    omega = rng.normal(size=3)
    dt = 0.03
    np.testing.assert_allclose(
      quat_from_gyro(omega, dt), quat_exp(omega * dt), atol=1e-15
    )


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
  """Against Barfoot, *State Estimation for Robotics* (2024), eq. 8.82a.

  The implementation uses the series form built from ``skew(phi)``; Barfoot
  writes the same matrix in axis-angle terms. The two are equal after
  substituting ``W = theta a^`` and ``W^2 = theta^2 (a a^T - I)``, so agreeing
  numerically checks the implementation against an independently written
  formula rather than against a rearrangement of itself.

  Note Barfoot's convention is the *left* Jacobian, ``J = J_l``, with
  ``J_l(-phi) = J_r(phi)`` (eq. 8.85). This is the right one.
  """
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
  """No step where the implementation switches to the series, at 1e-2.

  Regression: the threshold was first placed at 1e-6, where the closed form
  has already lost most of its precision to cancellation in ``1 - cos(theta)``
  and ``theta - sin(theta)``. That put a relative step of ~1e-4 into the
  Jacobian -- and so into the attitude block of every propagated covariance --
  at an angle small enough to occur on every quiet step of a run.
  """
  axis = np.array([0.6, -0.8, 0.0])
  delta = 1e-12

  below = so3_right_jacobian(axis * (1e-2 - delta))
  above = so3_right_jacobian(axis * (1e-2 + delta))

  # What is left is the closed form's own rounding at this angle -- 4e-10
  # relative, against the ~1e-4 the old threshold produced. Removing it
  # entirely would mean using the series everywhere, which trades a negligible
  # step for a growing truncation error.
  np.testing.assert_allclose(below, above, rtol=1e-9, atol=1e-17)


def test_right_jacobian_is_accurate_where_cancellation_bites():
  """The series branch must beat the closed form, not merely differ from it.

  At these angles ``(1 - cos t) / t^2`` computed directly is wrong in the
  fourth significant digit. The exact coefficients are known from the series,
  so this checks the implementation against them rather than against the
  expression it is replacing.
  """
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
  """``exp(phi + d) ~= exp(phi) exp(Jr(phi) d)`` is the defining property."""
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


# -- boxplus and boxminus ---------------------------------------------------


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
  """The vector blocks cancel exactly; the attitude to rounding.

  ``conj(q) * q`` is the identity only to floating point, so the rotation block
  lands at ~1e-17 rather than at zero. That is why the prediction step reuses
  the sigma-point offsets it already has instead of recovering them with a
  second ``boxminus``.
  """
  state = random_state(np.random.default_rng(6))
  delta = state - state

  assert np.all(delta[:3] == 0.0)
  assert np.all(delta[6:] == 0.0)
  np.testing.assert_allclose(delta[ROTATION], np.zeros(3), atol=1e-15)


def test_boxplus_perturbs_the_rotation_on_the_right():
  """The increment is a body-frame rotation, not a world-frame one.

  Getting this backwards leaves every round-trip test above passing and the
  attitude block of every cross-covariance transposed.
  """
  # Yaw 90 degrees: body +x points along world +y.
  state = NavState.at_rest(attitude=quat_exp(np.array([0.0, 0.0, np.pi / 2])))
  # A further quarter turn about *body* +x.
  turned = state + np.concatenate([np.zeros(3), [np.pi / 2, 0, 0], np.zeros(9)])

  # Body +x is unmoved by a rotation about body +x, so it still points at world +y.
  np.testing.assert_allclose(
    turned.rotation @ np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    atol=1e-12,
  )
  # Had the increment been applied in the world frame, it would have been a
  # turn about world +x, which moves body +x off world +y.


def test_an_increment_on_the_left_is_refused_not_broadcast():
  """NumPy would otherwise add the state to each element of the increment."""
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
  """The truth is a fixed point: symmetric offsets must cancel exactly."""
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
  """Only the attitude iterates, which is why this mean is cheap.

  A single weighted pass, from a deliberately poor seed, already lands the
  twelve vector components on the converged answer -- their update is linear.
  The attitude, from that same seed, does not.
  """
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
  """A cloud wider than the chart is a broken belief, not a number to return."""
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
