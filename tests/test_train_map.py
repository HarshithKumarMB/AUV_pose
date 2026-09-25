"""The selections that decide what the map is fitted on and scored against."""

import numpy as np
import pytest

from experiments.train_map import (
  Soundings,
  blocked_split,
  calibration,
  decimate,
  last_tenth,
  ping_split,
  score,
)


def survey(n=500, seed=0, pings=50):
  """Soundings with every optional column, truth a known function of X."""
  rng = np.random.default_rng(seed)
  X = rng.uniform(-10.0, 10.0, size=(n, 2))
  y = rng.normal(-70.0, 1.0, size=n)
  ping = np.sort(rng.integers(0, pings, size=n))
  truth = np.column_stack([X + 0.5, y - 1.0])
  return Soundings(X, y, ping, truth)


# -- keeping the columns together -------------------------------------------


def test_a_selection_keeps_every_column_in_step():
  data = survey()
  mask = data.X[:, 0] > 0
  kept = data.where(mask)

  assert kept.ping is not None and kept.truth is not None
  assert len(kept.X) == len(kept.y) == len(kept.ping) == len(kept.truth)
  np.testing.assert_array_equal(kept.truth[:, :2], kept.X + 0.5)
  np.testing.assert_array_equal(kept.truth[:, 2], kept.y - 1.0)


def test_an_empty_selection_is_empty_not_an_error():
  kept = survey().where(np.zeros(500, dtype=bool))
  assert len(kept) == 0
  assert kept.truth is not None and kept.truth.shape == (0, 3)


def test_joining_puts_the_other_soundings_after():
  data = survey()
  first, second = data.where(data.y < -70), data.where(data.y >= -70)
  joined = first.then(second)

  assert len(joined) == len(data)
  np.testing.assert_array_equal(joined.y[: len(first)], first.y)
  np.testing.assert_array_equal(joined.y[len(first) :], second.y)


def test_a_column_one_side_lacks_is_dropped_not_padded():
  data = survey()
  joined = Soundings(data.X, data.y).then(data)
  assert joined.ping is None and joined.truth is None


# -- decimation -------------------------------------------------------------


def test_decimate_takes_the_median_per_cell():
  """Median, not mean: a mean would drag the cell 2 m toward the bad beam."""
  X = np.array([[0.10, 0.20], [0.12, 0.22], [0.14, 0.24], [0.0, 0.0]])
  y = np.array([-70.0, -70.1, -60.0, -69.9])
  reduced = decimate(Soundings(X, y), cell=1.0)

  assert len(reduced) == 1
  assert reduced.y[0] == pytest.approx(-69.95)
  np.testing.assert_allclose(reduced.X[0], [0.11, 0.21])


def test_decimate_floors_rather_than_truncates_across_zero():
  """Truncation would put -0.1 and +0.1 in one cell and average them."""
  X = np.array([[-0.1, 0.0], [0.1, 0.0], [-0.1, -0.1], [0.1, -0.1]])
  y = np.array([-70.0, -68.0, -66.0, -64.0])
  reduced = decimate(Soundings(X, y), cell=0.25)
  assert sorted(reduced.y) == pytest.approx(sorted(y))


def test_decimate_groups_every_column_by_the_same_cells():
  """Each point's truth is its own cell's truth, not a neighbour's."""
  data = survey(n=2000)
  reduced = decimate(data, cell=2.0)

  assert reduced.truth is not None
  np.testing.assert_allclose(reduced.truth[:, :2], reduced.X + 0.5)
  np.testing.assert_allclose(reduced.truth[:, 2], reduced.y - 1.0)


def test_decimate_does_not_depend_on_the_input_order():
  data = survey(n=2000)
  shuffled = data.where(np.random.default_rng(1).permutation(len(data)))
  a, b = decimate(data, 1.0), decimate(shuffled, 1.0)
  np.testing.assert_array_equal(a.X, b.X)
  np.testing.assert_array_equal(a.y, b.y)


def test_decimate_leaves_one_point_per_occupied_cell():
  data = survey(n=2000)
  reduced = decimate(data, cell=1.0)
  occupied = {tuple(c) for c in np.floor(data.X)}
  assert len(reduced) == len(occupied)
  assert {tuple(c) for c in np.floor(reduced.X)} == occupied


def test_decimate_preserves_dtype_and_drops_the_ping():
  """A cell median mixes pings, so no ping number describes it."""
  data = survey()
  single = Soundings(
    data.X.astype(np.float32), data.y.astype(np.float32), data.ping
  )
  reduced = decimate(single, cell=0.5)
  assert reduced.X.dtype == reduced.y.dtype == np.float32
  assert reduced.ping is None


def test_decimate_leaves_a_lone_sounding_untouched():
  data = survey(n=1)
  reduced = decimate(data, cell=0.25)
  np.testing.assert_array_equal(reduced.X, data.X)
  np.testing.assert_array_equal(reduced.y, data.y)


# -- the holdouts -----------------------------------------------------------


def test_a_ping_is_held_out_whole_or_not_at_all():
  data = survey(n=5000, pings=200)
  assert data.ping is not None
  train, test = ping_split(data.ping, fraction=0.3, seed=0)

  assert np.all(train ^ test)
  assert not set(data.ping[train]) & set(data.ping[test])
  assert len(set(data.ping[test])) == 60


def test_the_ping_split_is_reproducible_and_seeded():
  ping = survey(n=5000, pings=200).ping
  assert ping is not None
  a, _ = ping_split(ping, 0.3, seed=0)
  b, _ = ping_split(ping, 0.3, seed=0)
  c, _ = ping_split(ping, 0.3, seed=1)
  np.testing.assert_array_equal(a, b)
  assert np.any(a != c)


def test_ping_numbers_need_not_be_contiguous():
  ping = np.array([7, 7, 1000, 1000, 1000, -3])
  train, test = ping_split(ping, fraction=1 / 3, seed=0)
  assert len(set(ping[test])) == 1
  assert not set(ping[train]) & set(ping[test])


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_the_ping_split_extremes(fraction):
  ping = np.arange(20) // 4
  _, test = ping_split(ping, fraction, seed=0)
  assert test.all() if fraction else not test.any()


def test_a_held_out_cell_holds_no_training_sounding():
  X = survey(n=5000).X
  train, test = blocked_split(X, fraction=0.2, cell=2.0, seed=0)

  assert np.all(train ^ test)
  cells_train = {tuple(c) for c in np.floor(X[train] / 2.0)}
  cells_test = {tuple(c) for c in np.floor(X[test] / 2.0)}
  assert cells_train and cells_test
  assert not cells_train & cells_test


def test_decimation_finer_than_the_holdout_cell_keeps_the_split_shut():
  """The constraint train_map enforces before it decimates.

  Decimating coarser than the holdout would merge a training and a held-out
  sounding into one point, leaking the answer across the split.
  """
  reduced = decimate(survey(n=5000), cell=0.25)
  train, test = blocked_split(reduced.X, fraction=0.2, cell=1.0, seed=0)
  cells_train = {tuple(c) for c in np.floor(reduced.X[train])}
  cells_test = {tuple(c) for c in np.floor(reduced.X[test])}
  assert not cells_train & cells_test


# -- scoring ----------------------------------------------------------------


class _Calibrated:
  """A stand-in map with a spread we control, to test the coverage measure."""

  def __init__(self, bias, spread):
    self.bias = bias
    self.spread = spread

  def predict(self, points, with_std=False, observation_noise=False):
    elevation = np.full(len(points), self.bias)
    if not with_std:
      return elevation
    return elevation, np.full(len(points), self.spread)


@pytest.mark.parametrize(
  ("spread", "low", "high"),
  [(1.0, 0.93, 0.97), (0.25, 0.0, 0.6), (8.0, 0.99, 1.0)],
  ids=["honest", "overconfident", "underconfident"],
)
def test_calibration_is_the_coverage_of_the_95_percent_interval(
  spread, low, high
):
  truth = np.random.default_rng(0).normal(size=2000)
  test = np.ones(2000, dtype=bool)
  covered = calibration(
    _Calibrated(0.0, spread), np.zeros((2000, 2)), truth, test
  )
  assert low <= covered <= high


def test_calibration_counts_only_the_held_out_soundings():
  y = np.array([0.0, 100.0, 0.0])
  test = np.array([True, False, True])
  assert calibration(_Calibrated(0.0, 1.0), np.zeros((3, 2)), y, test) == 1.0


def test_score_returns_the_rmse_it_printed(capsys):
  """So a caller comparing two maps need not predict all over again."""
  points = np.random.default_rng(3).uniform(-10.0, 10.0, size=(200, 2))
  y = np.full(200, -60.0)
  y[150:] += np.array([3.0, -3.0] * 25)
  train = np.arange(200) < 150

  returned = score(_Calibrated(-60.0, 1.0), points, y, train, ~train)

  assert returned == pytest.approx(3.0)
  assert "held out 50 soundings" in capsys.readouterr().out


# -- convergence diagnostic ---------------------------------------------------


def test_last_tenth_spans_a_tenth_of_the_steps():
  assert last_tenth(list(map(float, range(101)))) == 10.0


@pytest.mark.parametrize("trace", [[5.0], [5.0, 7.0], [1.0] * 9])
def test_last_tenth_of_a_short_run_is_at_most_one_step(trace):
  assert last_tenth(trace) == trace[-1] - trace[max(0, len(trace) - 2)]
