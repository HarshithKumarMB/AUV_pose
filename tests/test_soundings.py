"""Survey CSV loading and the seabed elevation schema."""

import numpy as np
import pandas as pd
import pytest

from auv_pose.io.soundings import (
  load_soundings,
  placement_covariance,
  soundings_to_arrays,
)


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
  """What the writer emits is what the GP reads."""
  path = survey("a.csv", [[0.0, 0.0, -70.3], [1.0, -2.0, -69.8]])
  frame = load_soundings([path])
  X, y = soundings_to_arrays(frame)

  assert X.shape == (2, 2)
  np.testing.assert_allclose(y, [-70.3, -69.8], rtol=1e-6)
  assert np.all(y < 0)  # the seabed is below the surface


# -- the optional covariance columns ----------------------------------------


def placed(tmp_path, name, cov_xy=0.1):
  """A georeferenced survey: two soundings with covariance and truth."""
  path = tmp_path / name
  pd.DataFrame(
    {
      "x": [0.0, 1.0],
      "y": [0.0, 2.0],
      "z": [-70.0, -69.5],
      "cov_xx": [0.5, 0.6],
      "cov_xy": [cov_xy, cov_xy],
      "cov_xz": [0.02, 0.02],
      "cov_yy": [0.7, 0.8],
      "cov_yz": [0.03, 0.03],
      "cov_zz": [0.01, 0.01],
      "ping": [0, 0],
      "true_x": [0.1, 1.1],
      "true_y": [0.0, 2.0],
      "true_z": [-70.0, -69.5],
    }
  ).to_csv(path, index=False)
  return path


def test_the_covariance_reads_back_as_three_by_three(tmp_path):
  frame = load_soundings([placed(tmp_path, "a.csv")])
  cov = placement_covariance(frame)

  assert cov.shape == (2, 3, 3)
  np.testing.assert_array_equal(
    cov[1], [[0.6, 0.1, 0.02], [0.1, 0.8, 0.03], [0.02, 0.03, 0.01]]
  )


def test_a_column_only_some_files_carry_is_dropped(survey, tmp_path):
  """Filling it with zero would claim the other survey was placed exactly."""
  frame = load_soundings(
    [placed(tmp_path, "a.csv"), survey("b.csv", [[5.0, 5.0, -68.0]])]
  )

  assert list(frame.columns) == ["x", "y", "z"]
  with pytest.raises(ValueError, match="georeference.py --pose smoothed"):
    placement_covariance(frame)


def test_files_that_all_carry_it_keep_it(tmp_path):
  frame = load_soundings([placed(tmp_path, "a.csv"), placed(tmp_path, "b.csv")])
  assert len(placement_covariance(frame)) == 4
  assert "true_x" in frame.columns
