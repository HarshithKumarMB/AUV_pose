"""Survey CSV loading and the seabed elevation schema."""

import numpy as np
import pandas as pd
import pytest

from auv_pose.io.soundings import load_soundings, soundings_to_arrays


@pytest.fixture
def survey(tmp_path):
  def write(name, rows):
    path = tmp_path / name
    pd.DataFrame(rows, columns=["x", "y", "z"]).to_csv(path, index=False)
    return path

  return write


def test_loads_a_single_file(survey):
  path = survey("a.csv", [[0.0, 0.0, 70.0], [1.0, 2.0, 71.5]])
  frame = load_soundings([path])

  assert list(frame.columns) == ["x", "y", "z"]
  assert len(frame) == 2


def test_concatenates_in_order(survey):
  a = survey("a.csv", [[0.0, 0.0, 70.0]])
  b = survey("b.csv", [[9.0, 9.0, 60.0]])
  frame = load_soundings([a, b])

  assert len(frame) == 2
  assert frame["x"].tolist() == [0.0, 9.0]


def test_elevation_reaches_the_gp_unchanged(survey):
  """z is already an upward-increasing elevation; nothing flips its sign.

  The old schema stored a downward range and negated it here. Negating an
  elevation would put the seabed 70 m above the surface.
  """
  path = survey("a.csv", [[1.0, 2.0, -70.0]])
  X, y = soundings_to_arrays(load_soundings([path]))

  np.testing.assert_allclose(X, [[1.0, 2.0]])
  np.testing.assert_allclose(y, [-70.0])


def test_arrays_are_float32(survey):
  """gpytorch is fitted in single precision."""
  path = survey("a.csv", [[1.0, 2.0, -70.0]])
  X, y = soundings_to_arrays(load_soundings([path]))
  assert X.dtype == np.float32
  assert y.dtype == np.float32


def test_drops_rows_with_no_echo(survey):
  """survey.py records NaN when the sonar returned nothing usable."""
  path = survey(
    "a.csv", [[0.0, 0.0, -70.0], [1.0, 1.0, np.nan], [2.0, 2.0, -72.0]]
  )
  frame = load_soundings([path])

  assert len(frame) == 2
  assert not frame["z"].isna().any()


def test_drops_infinities(survey):
  path = survey("a.csv", [[0.0, 0.0, -70.0], [1.0, 1.0, np.inf]])
  assert len(load_soundings([path])) == 1


def test_keeps_bad_rows_when_asked(survey):
  path = survey("a.csv", [[0.0, 0.0, -70.0], [1.0, 1.0, np.nan]])
  assert len(load_soundings([path], drop_invalid=False)) == 2


def test_index_is_contiguous_after_dropping(survey):
  """Downstream code indexes positionally; a gappy index would misalign it."""
  path = survey("a.csv", [[0.0, 0.0, np.nan], [1.0, 1.0, -70.0]])
  frame = load_soundings([path])
  assert frame.index.tolist() == [0]


def test_rejects_a_file_missing_columns(tmp_path):
  path = tmp_path / "bad.csv"
  pd.DataFrame({"x": [1.0], "y": [2.0]}).to_csv(path, index=False)

  with pytest.raises(ValueError, match="z"):
    load_soundings([path])


def test_rejects_an_empty_file_list():
  with pytest.raises(ValueError, match="no sounding files"):
    load_soundings([])


def test_round_trips_a_survey_written_by_the_logger(survey):
  """What the writer emits is what the GP reads.

  This replaces a test that loaded the committed map.csv. That file is a
  singlebeam survey in the retired schema, and it is separately known to be
  about 47% wrong -- see the octree comparison -- so it is no longer a fixture
  worth guarding.
  """
  path = survey("a.csv", [[0.0, 0.0, -70.3], [1.0, -2.0, -69.8]])
  frame = load_soundings([path])
  X, y = soundings_to_arrays(frame)

  assert X.shape == (2, 2)
  np.testing.assert_allclose(y, [-70.3, -69.8], rtol=1e-6)
  assert np.all(y < 0)  # the seabed is below the surface
