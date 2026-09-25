"""The raw survey log round-trips, and says so when it is not one."""

from typing import Any

import numpy as np
import pytest

from auv_pose.io.raw_survey import RawSurveyWriter, load_raw_survey

TRUTH: dict[str, Any] = {
  "true_position": [1.0, 2.0, -0.4],
  "true_attitude": [1.0, 0.0, 0.0, 0.0],
  "true_gyro_bias": [0.0, 0.0, 1e-5],
  "true_accel_bias": [1e-4, 0.0, 0.0],
}


META: dict[str, Any] = {
  "commit": "abc123-dirty",
  "tick_rate_hz": 30,
  "depth_sigma": 0.05,
  "sonar": {"azimuth": 60.0, "azimuth_bins": 240},
  "initial_mean": {"position": [1.0, 2.0, -0.4], "attitude": [1, 0, 0, 0]},
  "waypoints": [[0.0, 0.0, 0.0], [5.0, 0.0, -1.0]],
}


def assert_same_tree(read, written):
  assert set(read) == set(written)
  for key, value in written.items():
    if isinstance(value, dict):
      assert_same_tree(read[key], value)
    else:
      np.testing.assert_array_equal(read[key], value)
      assert type(read[key]) is type(value) or isinstance(read[key], np.ndarray)


def write(directory, n_beams=4):
  meta = {**META, "bearings": [0.1, 0.2, 0.3, 0.4][:n_beams]}
  with RawSurveyWriter(directory, n_beams, meta) as log:
    log.tick(0, gyro=[0, 0, 0.1], accel=[0, 0, -9.81], **TRUTH)
    log.tick(
      1,
      gyro=[0, 0, 0.2],
      accel=[0, 0, -9.8],
      dvl=[1.0, 0.0, 0.1],
      depth=-0.5,
      magnetometer=[1.0, 0.0, 0.0],
      **TRUTH,
    )
    log.ping(1, [70.0, np.nan, 71.5, 72.0][:n_beams])
  return meta


def test_it_round_trips(tmp_path):
  meta = write(tmp_path / "pass0")
  survey = load_raw_survey(tmp_path / "pass0")

  assert_same_tree(survey.meta, meta)
  np.testing.assert_array_equal(survey.ping_ticks, [1])
  np.testing.assert_array_equal(survey.ranges, [[70.0, np.nan, 71.5, 72.0]])
  np.testing.assert_array_equal(survey.readings("gyro")[:, 2], [0.1, 0.2])
  np.testing.assert_array_equal(survey.ticks["true_gyro_bias_z"], [1e-5] * 2)


def test_a_silent_sensor_reads_back_as_nan(tmp_path):
  write(tmp_path / "pass0")
  survey = load_raw_survey(tmp_path / "pass0")

  assert np.isnan(survey.readings("dvl")[0]).all()
  np.testing.assert_array_equal(survey.readings("dvl")[1], [1.0, 0.0, 0.1])
  assert np.isnan(survey.ticks["depth"][0])


def test_a_ping_must_cover_every_beam(tmp_path):
  with (
    RawSurveyWriter(tmp_path / "pass0", 4, {}) as log,
    pytest.raises(ValueError, match="4 ranges"),
  ):
    log.ping(0, [1.0, 2.0])


def test_a_soundings_file_is_not_mistaken_for_a_log(tmp_path):
  (tmp_path / "pass0.csv").write_text("x,y,z\n0,0,-70\n")
  with pytest.raises(FileNotFoundError, match="not a raw survey log"):
    load_raw_survey(tmp_path / "pass0.csv")


@pytest.mark.parametrize(
  "meta",
  [
    {"a/b": 1.0},
    {"x": None},
    {"x": {}},
    {"x": [1.0, "two"]},
    {"file": 1.0},
    {"allow_pickle": False},
  ],
  ids=[
    "slash in key",
    "None",
    "empty dict",
    "mixed list",
    "file",
    "allow_pickle",
  ],
)
def test_metadata_that_cannot_round_trip_is_refused(tmp_path, meta):
  with pytest.raises(ValueError), RawSurveyWriter(tmp_path / "pass0", 1, meta):
    pass
