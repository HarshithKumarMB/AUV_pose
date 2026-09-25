"""The convergence diagnostic printed after a fit."""

import pytest

from experiments.train_map import last_tenth


def test_last_tenth_spans_a_tenth_of_the_steps():
  assert last_tenth([float(i) for i in range(101)]) == 10.0


@pytest.mark.parametrize("trace", [[5.0], [5.0, 7.0], [1.0, 2.0, 4.0]])
def test_a_short_run_reports_its_last_step(trace):
  """Under ten steps, still a real change, never a silent zero."""
  assert last_tenth(trace) == trace[-1] - trace[max(0, len(trace) - 2)]
