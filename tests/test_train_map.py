"""Decimation and the blocked split that guard the GP's training data."""

import numpy as np
import pytest

from experiments.train_map import (
  blocked_split,
  calibration,
  decimate,
  score,
)


def test_decimate_collapses_a_cell_to_one_point():
  X = np.array([[0.0, 0.0], [0.05, 0.05], [0.06, 0.0]])
  y = np.array([-70.0, -70.2, -69.8])

  reduced_x, reduced_y = decimate(X, y, cell=0.25)
  assert len(reduced_x) == 1
  assert reduced_y[0] == pytest.approx(-70.0)


def test_decimate_keeps_separate_cells_separate():
  X = np.array([[0.0, 0.0], [5.0, 5.0]])
  y = np.array([-70.0, -65.0])

  reduced_x, reduced_y = decimate(X, y, cell=0.25)
  assert len(reduced_x) == 2
  assert sorted(reduced_y) == pytest.approx([-70.0, -65.0])


def test_decimate_rejects_a_minority_of_bad_beams():
  """Why median and not mean.

  A strongest-return picker occasionally lands on the wrong feature, and those
  errors are one-sided. A mean would drag the cell 2.5 m toward the outlier;
  the median ignores it.
  """
  X = np.zeros((5, 2))
  y = np.array([-70.0, -70.1, -69.9, -70.0, -60.0])

  _, reduced_y = decimate(X, y, cell=0.25)
  assert reduced_y[0] == pytest.approx(-70.0)
  assert np.mean(y) == pytest.approx(-68.0)


def test_decimate_places_the_point_at_the_cell_median():
  X = np.array([[0.10, 0.20], [0.12, 0.22], [0.14, 0.24]])
  y = np.array([-70.0, -70.0, -70.0])

  reduced_x, _ = decimate(X, y, cell=1.0)
  assert reduced_x[0] == pytest.approx([0.12, 0.22])


def test_decimate_preserves_dtype():
  """The GP is fitted in single precision; a silent upcast wastes memory."""
  X = np.zeros((3, 2), dtype=np.float32)
  y = np.zeros(3, dtype=np.float32)

  reduced_x, reduced_y = decimate(X, y, cell=0.25)
  assert reduced_x.dtype == np.float32
  assert reduced_y.dtype == np.float32


def test_decimate_does_not_fold_across_zero():
  """The survey box is negative in x and y, so this must floor, not truncate.

  Truncation maps -0.1 and +0.1 to the same cell 0, silently averaging two
  soundings 0.2 m apart across the origin.
  """
  X = np.array([[-0.1, 0.0], [0.1, 0.0]])
  y = np.array([-70.0, -68.0])

  reduced_x, reduced_y = decimate(X, y, cell=0.25)
  assert len(reduced_x) == 2
  assert sorted(reduced_y) == pytest.approx([-70.0, -68.0])


def test_decimate_groups_within_a_negative_cell():
  X = np.array([[-10.1, -5.1], [-10.15, -5.15], [-10.6, -5.1]])
  y = np.array([-70.0, -70.0, -68.0])

  reduced_x, _ = decimate(X, y, cell=0.25)
  assert len(reduced_x) == 2


def test_decimation_finer_than_the_holdout_cell_keeps_the_split_shut():
  """The constraint train_map enforces: cells must not straddle the boundary.

  Decimating at a cell coarser than the holdout would merge a training and a
  held-out sounding into one point, leaking the answer across the split.
  """
  rng = np.random.default_rng(0)
  X = rng.uniform(-10.0, 10.0, size=(2000, 2))
  y = rng.normal(-70.0, 1.0, size=2000)

  reduced_x, _ = decimate(X, y, cell=0.25)
  train, test = blocked_split(reduced_x, fraction=0.2, cell=1.0, seed=0)

  assert train.sum() and test.sum()
  # No decimated point may fall in a holdout cell and a training cell at once,
  # which is guaranteed if every point sits in exactly one 1 m cell.
  cells_train = {tuple(c) for c in np.floor(reduced_x[train] / 1.0)}
  cells_test = {tuple(c) for c in np.floor(reduced_x[test] / 1.0)}
  assert not (cells_train & cells_test)


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


def test_calibration_counts_what_lands_inside_the_interval():
  points = np.zeros((1000, 2))
  truth = np.random.default_rng(0).normal(size=1000)
  test = np.ones(1000, dtype=bool)

  # Spread matching the truth's own: about 95% should land inside 1.96 sigma.
  honest = calibration(_Calibrated(0.0, 1.0), points, truth, test)
  assert 0.93 < honest < 0.97, honest


def test_calibration_exposes_an_overconfident_map():
  """The failure the measure exists to catch.

  A map that understates its spread scores well on rmse and badly here, which
  is the combination that would make a filter trusting it too sure of itself.
  """
  points = np.zeros((1000, 2))
  truth = np.random.default_rng(1).normal(size=1000)
  test = np.ones(1000, dtype=bool)

  assert calibration(_Calibrated(0.0, 0.25), points, truth, test) < 0.6


def test_calibration_exposes_an_underconfident_map():
  points = np.zeros((500, 2))
  truth = np.random.default_rng(2).normal(size=500)
  test = np.ones(500, dtype=bool)

  assert calibration(_Calibrated(0.0, 8.0), points, truth, test) > 0.99


def test_score_returns_the_rmse_it_printed(capsys):
  """So a caller comparing two maps need not predict all over again."""
  rng = np.random.default_rng(3)
  points = rng.uniform(-10.0, 10.0, size=(200, 2))
  truth = np.full(200, -60.0)

  train = np.zeros(200, dtype=bool)
  train[:150] = True
  test = ~train

  returned = score(_Calibrated(-60.0, 1.0), points, truth, train, test)

  assert returned == pytest.approx(0.0, abs=1e-12)
  assert "held out 50 soundings" in capsys.readouterr().out
