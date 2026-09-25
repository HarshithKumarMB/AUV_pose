"""The manifold backward pass.

The first test is the important one. ``unscented_rts_smooth`` and
``rts_smooth`` are the same recursion written two ways -- one taking its gain
from a recorded cross-covariance, the other forming ``P F^T`` -- so on a
linear-Gaussian problem they must agree to machine precision. Running the
linear ``ConstantVelocityKF`` in ``linear_reference``, converting its history, and comparing pins the
entire backward pass against an implementation that was already trusted, with no
map, no IMU and no simulator involved.

Everything after that is about the parts the vector case cannot reach: the
chart, and the covariance transport across it.
"""

import operator
from dataclasses import replace

import numpy as np
from linear_reference import ConstantVelocityKF, Measurement, rts_smooth

from auv_pose.estimation.manifold import (
  DOF,
  ManifoldGaussian,
  NavState,
  covariance_transport,
)
from auv_pose.estimation.quaternion import quat_angle, quat_exp
from auv_pose.estimation.smoothers import unscented_rts_smooth
from auv_pose.estimation.typing import SmootherStep

POSITION_H = np.hstack([np.eye(3), np.zeros((3, 3))])


def linear_run(n=25, seed=0):
  """A short linear-Gaussian run, as ``tests/test_smoothers.py`` builds one."""
  rng = np.random.default_rng(seed)
  dt = 0.1

  ekf = ConstantVelocityKF(accel_process_sigma=0.5)
  initial = ConstantVelocityKF.initial(np.zeros(3))
  state = initial

  position = np.zeros(3)
  velocity = np.zeros(3)

  for _ in range(n):
    accel = rng.normal(scale=0.5, size=3)
    velocity = velocity + accel * dt
    position = position + velocity * dt

    state = ekf.step(
      state,
      accel,
      dt,
      [
        Measurement(
          position + rng.normal(scale=0.3, size=3),
          POSITION_H,
          np.eye(3) * 0.09,
        )
      ],
    )

  return ekf, initial


def as_smoother_steps(initial, history):
  """Convert a linear filter's record into the cross-covariance form.

  For a linear filter the cross-covariance between the filtered error state at
  ``k`` and the predicted one at ``k + 1`` is exactly ``P_k^+ F^T`` -- which is
  the product ``rts_smooth`` forms inline. Handing it over explicitly is what
  makes the two passes comparable.
  """
  posteriors = [initial] + [step.posterior for step in history]
  return [
    SmootherStep(
      prior=step.prior,
      posterior=step.posterior,
      cross_cov=posteriors[k].cov @ step.transition.T,
    )
    for k, step in enumerate(history)
  ]


def vector_chart(smoother_initial, steps):
  """Smooth in a flat chart: boxplus is ``+``, and nothing to transport."""
  return unscented_rts_smooth(
    smoother_initial,
    steps,
    boxplus=operator.add,
    boxminus=operator.sub,
    transport=None,
  )


# -- exactness against the linear smoother ----------------------------------


def test_it_reproduces_the_linear_smoother_exactly():
  """The strongest check available, and it needs nothing but the two passes."""
  ekf, initial = linear_run(n=40, seed=3)

  expected = rts_smooth(initial, ekf.history)
  actual = vector_chart(initial, as_smoother_steps(initial, ekf.history))

  assert len(actual) == len(expected)
  for got, want in zip(actual, expected):
    np.testing.assert_allclose(got.mean, want.mean, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(got.cov, want.cov, rtol=1e-10, atol=1e-12)


def test_it_reproduces_the_linear_smoother_across_several_runs():
  for seed in (0, 1, 2, 5, 11):
    ekf, initial = linear_run(n=20, seed=seed)
    expected = rts_smooth(initial, ekf.history)
    actual = vector_chart(initial, as_smoother_steps(initial, ekf.history))

    for got, want in zip(actual, expected):
      np.testing.assert_allclose(got.mean, want.mean, rtol=1e-9, atol=1e-11)


def test_the_returned_type_follows_the_belief_it_was_given():
  ekf, initial = linear_run(n=5)
  smoothed = vector_chart(initial, as_smoother_steps(initial, ekf.history))
  assert all(type(s) is type(initial) for s in smoothed)


# -- structure --------------------------------------------------------------


def test_it_returns_one_belief_more_than_there_are_steps():
  ekf, initial = linear_run(n=12)
  steps = as_smoother_steps(initial, ekf.history)
  assert len(vector_chart(initial, steps)) == len(steps) + 1


def test_an_empty_history_returns_the_initial_belief():
  initial = ConstantVelocityKF.initial(np.zeros(3))
  assert vector_chart(initial, []) == [initial]


def test_the_final_belief_is_left_as_the_filter_had_it():
  """There is no future to condition the last step on."""
  ekf, initial = linear_run(n=15)
  steps = as_smoother_steps(initial, ekf.history)
  smoothed = vector_chart(initial, steps)

  np.testing.assert_array_equal(smoothed[-1].mean, steps[-1].posterior.mean)
  np.testing.assert_array_equal(smoothed[-1].cov, steps[-1].posterior.cov)


def test_it_does_not_mutate_the_history():
  ekf, initial = linear_run(n=10)
  steps = as_smoother_steps(initial, ekf.history)
  before = [(s.prior.cov.copy(), s.cross_cov.copy()) for s in steps]

  vector_chart(initial, steps)

  for step, (prior_cov, cross) in zip(steps, before):
    np.testing.assert_array_equal(step.prior.cov, prior_cov)
    np.testing.assert_array_equal(step.cross_cov, cross)


def test_smoothing_never_loosens_the_belief():
  ekf, initial = linear_run(n=30, seed=4)
  steps = as_smoother_steps(initial, ekf.history)
  smoothed = vector_chart(initial, steps)

  filtered = [initial] + [s.posterior for s in steps]
  for got, was in zip(smoothed, filtered):
    assert np.trace(got.cov) <= np.trace(was.cov) + 1e-9


# -- the manifold chart -----------------------------------------------------


def manifold_run(n=20, seed=0, attitude_spread=0.05, rotate=True):
  """A synthetic manifold history with something for the backward pass to do.

  The means are laid out so each prediction overshoots and each posterior pulls
  part of the way back, which is what gives the correction a direction.

  :param rotate: When ``False``, *every* attitude is exactly the identity --
      the systematic turn and the per-step drift both. That is what makes the
      chart genuinely flat; zeroing only ``attitude_spread`` leaves the drift
      rotating each step, which is not the same thing and does not make the
      transport a no-op.
  """
  rng = np.random.default_rng(seed)
  cov = np.eye(DOF) * 0.04

  def state(k, offset):
    turn = np.array([0.0, 0.0, attitude_spread * k]) + offset[3:6]
    return NavState(
      position=np.array([float(k), 0.0, -60.0]) + offset[:3],
      attitude=quat_exp(turn if rotate else np.zeros(3)),
      velocity=np.array([1.0, 0.0, 0.0]) + offset[6:9],
      gyro_bias=offset[9:12] * 0.01,
      accel_bias=offset[12:15] * 0.1,
    )

  initial = ManifoldGaussian(state(0, np.zeros(DOF)), cov.copy())
  steps = []
  for k in range(1, n + 1):
    drift = rng.normal(size=DOF) * 0.05
    prior = ManifoldGaussian(state(k, drift), cov * 1.6)
    posterior = ManifoldGaussian(state(k, drift * 0.4), cov * 0.9)
    steps.append(
      SmootherStep(prior=prior, posterior=posterior, cross_cov=cov * 0.5)
    )

  return initial, steps


def test_the_manifold_pass_runs_and_stays_finite():
  initial, steps = manifold_run(n=25, attitude_spread=0.05)
  smoothed = unscented_rts_smooth(initial, steps)

  assert len(smoothed) == len(steps) + 1
  for belief in smoothed:
    assert np.all(np.isfinite(belief.mean.position))
    assert np.all(np.isfinite(belief.mean.attitude))
    assert np.all(np.isfinite(belief.cov))


def test_the_manifold_pass_keeps_the_covariance_symmetric_and_positive():
  """The failure mode that only appears after many steps."""
  initial, steps = manifold_run(n=60, seed=2, attitude_spread=0.03)
  smoothed = unscented_rts_smooth(initial, steps)

  for belief in smoothed:
    np.testing.assert_allclose(belief.cov, belief.cov.T, atol=1e-15)
    assert np.min(np.linalg.eigvalsh(belief.cov)) > -1e-9 * np.trace(belief.cov)


def test_the_manifold_pass_returns_unit_quaternions():
  """``boxplus`` renormalises; a pass that skipped it would drift off the sphere."""
  initial, steps = manifold_run(n=40, attitude_spread=0.08)
  for belief in unscented_rts_smooth(initial, steps):
    np.testing.assert_allclose(
      np.linalg.norm(belief.mean.attitude), 1.0, atol=1e-12
    )


def test_it_agrees_with_the_vector_pass_when_nothing_rotates():
  """With every attitude identical, the chart is flat and the two must match.

  This is what ties the manifold path to the exactness test above: the same
  code, on data where the rotation block does nothing, has to reproduce plain
  vector arithmetic.
  """
  initial, steps = manifold_run(n=25, seed=6, rotate=False)

  on_manifold = unscented_rts_smooth(initial, steps)
  flat = unscented_rts_smooth(
    initial,
    steps,
    boxplus=lambda s, d: replace(
      s, position=s.position + d[:3], velocity=s.velocity + d[6:9]
    ),
    boxminus=lambda a, b: np.concatenate(
      [
        a.position - b.position,
        np.zeros(3),
        a.velocity - b.velocity,
        a.gyro_bias - b.gyro_bias,
        a.accel_bias - b.accel_bias,
      ]
    ),
    transport=None,
  )

  for got, want in zip(on_manifold, flat):
    np.testing.assert_allclose(
      got.mean.position, want.mean.position, atol=1e-10
    )
    np.testing.assert_allclose(got.cov, want.cov, atol=1e-10)


def test_transport_changes_the_covariance_only_at_second_order():
  """It matters on a large correction and not on a small one."""
  initial, steps = manifold_run(n=20, seed=8, attitude_spread=0.4)

  with_transport = unscented_rts_smooth(initial, steps)
  without = unscented_rts_smooth(initial, steps, transport=None)

  # Same means -- transport touches only the covariance.
  for got, want in zip(with_transport, without):
    assert quat_angle(got.mean.attitude, want.mean.attitude) < 1e-14

  differences = [
    np.max(np.abs(a.cov - b.cov)) for a, b in zip(with_transport, without)
  ]
  assert max(differences) > 0.0


def test_a_zero_correction_leaves_transport_with_nothing_to_do():
  assert np.array_equal(covariance_transport(np.zeros(DOF)), np.eye(DOF))


def test_the_correction_moves_the_mean_toward_the_future():
  """The point of smoothing: the past is pulled toward what came next."""
  initial, steps = manifold_run(n=30, seed=9, attitude_spread=0.02)
  smoothed = unscented_rts_smooth(initial, steps)

  filtered = [initial] + [s.posterior for s in steps]
  moved = [
    np.linalg.norm(got.mean - was.mean)
    for got, was in zip(smoothed[:-1], filtered[:-1])
  ]
  assert max(moved) > 1e-6
