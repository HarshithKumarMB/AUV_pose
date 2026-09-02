"""Survey CSV loading and the depth sign convention."""

import numpy as np
import pandas as pd
import pytest

from auv_pose.io.soundings import load_soundings, soundings_to_arrays


@pytest.fixture
def survey(tmp_path):
  def write(name, rows):
    path = tmp_path / name
    pd.DataFrame(rows, columns=["x", "y", "sonar_depth"]).to_csv(
      path, index=False
    )
    return path

  return write


def test_loads_a_single_file(survey):
  path = survey("a.csv", [[0.0, 0.0, 70.0], [1.0, 2.0, 71.5]])
  frame = load_soundings([path])

  assert list(frame.columns) == ["x", "y", "sonar_depth"]
  assert len(frame) == 2


def test_concatenates_in_order(survey):
  a = survey("a.csv", [[0.0, 0.0, 70.0]])
  b = survey("b.csv", [[9.0, 9.0, 60.0]])
  frame = load_soundings([a, b])

  assert len(frame) == 2
  assert frame["x"].tolist() == [0.0, 9.0]


def test_depth_is_negated_for_the_gp(survey):
  """The GP models an upward-increasing surface, so depth flips sign."""
  path = survey("a.csv", [[1.0, 2.0, 70.0]])
  X, y = soundings_to_arrays(load_soundings([path]))

  np.testing.assert_allclose(X, [[1.0, 2.0]])
  np.testing.assert_allclose(y, [-70.0])


def test_arrays_are_float32(survey):
  """gpytorch is fitted in single precision."""
  path = survey("a.csv", [[1.0, 2.0, 70.0]])
  X, y = soundings_to_arrays(load_soundings([path]))
  assert X.dtype == np.float32
  assert y.dtype == np.float32


def test_drops_rows_with_no_echo(survey):
  """survey.py records NaN when the sonar returned nothing usable."""
  path = survey(
    "a.csv", [[0.0, 0.0, 70.0], [1.0, 1.0, np.nan], [2.0, 2.0, 72.0]]
  )
  frame = load_soundings([path])

  assert len(frame) == 2
  assert not frame["sonar_depth"].isna().any()


def test_drops_infinities(survey):
  path = survey("a.csv", [[0.0, 0.0, 70.0], [1.0, 1.0, np.inf]])
  assert len(load_soundings([path])) == 1


def test_keeps_bad_rows_when_asked(survey):
  path = survey("a.csv", [[0.0, 0.0, 70.0], [1.0, 1.0, np.nan]])
  assert len(load_soundings([path], drop_invalid=False)) == 2


def test_index_is_contiguous_after_dropping(survey):
  """Downstream code indexes positionally; a gappy index would misalign it."""
  path = survey("a.csv", [[0.0, 0.0, np.nan], [1.0, 1.0, 70.0]])
  frame = load_soundings([path])
  assert frame.index.tolist() == [0]


def test_rejects_a_file_missing_columns(tmp_path):
  path = tmp_path / "bad.csv"
  pd.DataFrame({"x": [1.0], "y": [2.0]}).to_csv(path, index=False)

  with pytest.raises(ValueError, match="sonar_depth"):
    load_soundings([path])


def test_rejects_an_empty_file_list():
  with pytest.raises(ValueError, match="no sounding files"):
    load_soundings([])


def test_reads_the_real_survey_data():
  """Guards the committed map.csv against silent schema drift."""
  frame = load_soundings(["map.csv", "map1.csv"])
  X, y = soundings_to_arrays(frame)

  assert len(frame) > 80_000
  assert X.shape == (len(frame), 2)
  assert np.all(np.isfinite(y))
  assert np.all(y < 0)  # every sounding is below the vehicle
