"""Non-causal estimators.

The backward pass could not run at all before ``f5edc90``: the predicted states
it recurses over were never recorded, so it raised IndexError on an empty list.
Several of these pin that.
"""

import numpy as np

from auv_pose.estimation.filters import ConstantVelocityEKF
from auv_pose.estimation.smoothers import rts_smooth
from auv_pose.estimation.typing import GaussianState, Measurement, Step

POSITION_H = np.hstack([np.eye(3), np.zeros((3, 3))])


def run(n=25, seed=0):
  """A short linear-Gaussian run; returns the filter, its start, and truth."""
  rng = np.random.default_rng(seed)
  dt = 0.1

  ekf = ConstantVelocityEKF(accel_process_sigma=0.5)
  initial = ConstantVelocityEKF.initial(np.zeros(3))
  state = initial

  truth = []
  position = np.zeros(3)
  velocity = np.zeros(3)

  for _ in range(n):
    accel = rng.normal(scale=0.5, size=3)
    velocity = velocity + accel * dt
    position = position + velocity * dt
    truth.append(position.copy())

    observed = position + rng.normal(scale=0.3, size=3)
    state = ekf.step(
      state,
      accel,
      dt,
      [Measurement(observed, POSITION_H, np.eye(3) * 0.09)],
    )

  return ekf, initial, np.array(truth)


def test_smoother_runs():
  """Regression: this raised IndexError, the predicted states being absent."""
  ekf, initial, truth = run()
  smoothed = rts_smooth(initial, ekf.history)

  assert len(smoothed) == len(truth) + 1  # + the initial belief
  assert all(np.all(np.isfinite(s.mean)) for s in smoothed)
  assert all(np.all(np.isfinite(s.cov)) for s in smoothed)


def test_smoother_is_at_least_as_accurate_as_the_filter():
  ekf, initial, truth = run(n=40, seed=3)
  smoothed = rts_smooth(initial, ekf.history)

  filtered_track = np.array([s.posterior.mean[:3] for s in ekf.history])
  smoothed_track = np.array([s.mean[:3] for s in smoothed[1:]])

  filter_rmse = np.sqrt(((filtered_track - truth) ** 2).sum(axis=1).mean())
  smoother_rmse = np.sqrt(((smoothed_track - truth) ** 2).sum(axis=1).mean())

  assert smoother_rmse <= filter_rmse


def test_smoother_reduces_uncertainty():
  ekf, initial, _ = run()
  smoothed = rts_smooth(initial, ekf.history)
  filtered = [initial] + [s.posterior for s in ekf.history]

  # The final step is shared; every earlier one gains from future measurements.
  for k in range(len(smoothed) - 1):
    assert np.trace(smoothed[k].cov) <= np.trace(filtered[k].cov) + 1e-9


def test_final_state_is_unchanged():
  """Nothing follows the last step, so smoothing cannot improve it."""
  ekf, initial, _ = run()
  smoothed = rts_smooth(initial, ekf.history)

  np.testing.assert_allclose(smoothed[-1].mean, ekf.history[-1].posterior.mean)
  np.testing.assert_allclose(smoothed[-1].cov, ekf.history[-1].posterior.cov)


def test_empty_history_returns_the_initial_belief():
  initial = ConstantVelocityEKF.initial([1.0, 2.0, 3.0])
  smoothed = rts_smooth(initial, [])

  assert len(smoothed) == 1
  np.testing.assert_allclose(smoothed[0].mean, initial.mean)


def test_smooths_hand_built_steps_with_no_filter():
  """The decoupling: rts_smooth needs a record, not a filter.

  A scalar random walk observed only at the end. The smoothed estimate at
  step 0 must move toward that late observation -- which is precisely the
  information a causal filter cannot use.
  """
  F = np.array([[1.0]])
  initial = GaussianState(mean=np.array([0.0]), cov=np.array([[1.0]]))

  # Two steps of pure prediction, then one that observes 10.
  history = [
    Step(
      prior=GaussianState(np.array([0.0]), np.array([[2.0]])),
      posterior=GaussianState(np.array([0.0]), np.array([[2.0]])),
      transition=F,
    ),
    Step(
      prior=GaussianState(np.array([0.0]), np.array([[3.0]])),
      posterior=GaussianState(np.array([10.0]), np.array([[0.1]])),
      transition=F,
    ),
  ]

  smoothed = rts_smooth(initial, history)

  assert len(smoothed) == 3
  assert smoothed[0].mean[0] > 0.0  # pulled toward the future observation
  assert smoothed[1].mean[0] > 0.0
  np.testing.assert_allclose(smoothed[-1].mean, [10.0])


def test_smoother_does_not_mutate_the_history():
  ekf, initial, _ = run(n=5)
  before = [(s.prior.mean.copy(), s.posterior.mean.copy()) for s in ekf.history]

  rts_smooth(initial, ekf.history)

  for step, (prior_mean, posterior_mean) in zip(ekf.history, before):
    np.testing.assert_allclose(step.prior.mean, prior_mean)
    np.testing.assert_allclose(step.posterior.mean, posterior_mean)
