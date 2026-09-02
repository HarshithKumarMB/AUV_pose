"""Diagnostics emitted during a navigation run.

These run inside the per-tick loop, so a crash here kills a long simulator run
well after the point where restarting is cheap.
"""

import numpy as np

from auv_pose.estimation.typing import GaussianState, Measurement, Step
from experiments.navigate import DEPTH_H, innovation, report

PRIOR = GaussianState(
  mean=np.array([1.0, 2.0, 5.0, 0.0, 0.0, 0.0]), cov=np.eye(6)
)
STEP = Step(prior=PRIOR, posterior=PRIOR, transition=np.eye(6))


def test_innovation_is_measurement_minus_prediction():
  """Prior depth is 5.0; observing 7.0 and 6.5 leaves residuals 2.0 and 1.5."""
  obs = Measurement([7.0, 6.5], DEPTH_H, np.eye(2))
  np.testing.assert_allclose(innovation(STEP, [obs]), [2.0, 1.5])


def test_innovation_is_zero_when_the_prediction_is_right():
  obs = Measurement([5.0], DEPTH_H[:1], np.array([[1.0]]))
  np.testing.assert_allclose(innovation(STEP, [obs]), [0.0], atol=1e-12)


def test_innovation_concatenates_multiple_observations():
  a = Measurement([7.0], DEPTH_H[:1], np.array([[1.0]]))
  b = Measurement([4.0], DEPTH_H[:1], np.array([[1.0]]))
  np.testing.assert_allclose(innovation(STEP, [a, b]), [2.0, -1.0])


def test_innovation_with_no_observations_is_empty():
  """A tick can legitimately have nothing to condition on."""
  assert innovation(STEP, []).shape == (0,)


def test_report_prints_one_line(capsys):
  report(
    step=120,
    step_record=STEP,
    observations=[Measurement([7.0, 6.5], DEPTH_H, np.eye(2))],
    truth=np.array([1.0, 2.0, 5.5]),
    estimate=PRIOR,
    dead_reckoned=np.array([3.0, 2.0, 5.5]),
    sonar_range=68.2,
    map_depth=-64.9,
    waypoint=3,
    attitude_error=14.3,
  )
  out = capsys.readouterr().out.strip()

  assert out.count("\n") == 0
  assert "step    120" in out
  assert "wp 3" in out
  assert "att  14.30 deg" in out
  assert "0.50 m" in out  # ekf error: truth z 5.5 against estimate 5.0
  assert "2.00 m" in out  # dead-reckoning error: 2 m out in x, matching in z


def test_report_survives_a_no_echo_tick(capsys):
  """map_depth is NaN whenever the sonar returns nothing usable."""
  report(
    step=0,
    step_record=STEP,
    observations=[Measurement([5.0], DEPTH_H[:1], np.array([[1.0]]))],
    truth=np.zeros(3),
    estimate=PRIOR,
    dead_reckoned=np.zeros(3),
    sonar_range=float("nan"),
    map_depth=float("nan"),
    waypoint=0,
    attitude_error=0.0,
  )
  assert "nan" in capsys.readouterr().out
