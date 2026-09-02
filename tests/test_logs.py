"""CSV run logging."""

import csv

import pandas as pd
import pytest

from auv_pose.io.logs import CsvLogger

COLUMNS = ("step", "x", "y", "z")


def read(path):
  with open(path, newline="") as handle:
    return list(csv.reader(handle))


def test_header_is_written_on_enter(tmp_path):
  path = tmp_path / "log.csv"
  with CsvLogger(path, COLUMNS):
    pass
  assert read(path) == [list(COLUMNS)]


def test_rows_follow_declared_column_order(tmp_path):
  """Keyword order at the call site must not affect the file."""
  path = tmp_path / "log.csv"
  with CsvLogger(path, COLUMNS) as log:
    log.write(z=3.0, step=0, y=2.0, x=1.0)

  assert read(path)[1] == ["0", "1.0", "2.0", "3.0"]


def test_missing_column_is_rejected(tmp_path):
  with (
    CsvLogger(tmp_path / "log.csv", COLUMNS) as log,
    pytest.raises(ValueError, match=r"missing=\[.y., .z.\]"),
  ):
    log.write(step=0, x=1.0)


def test_unexpected_column_is_rejected(tmp_path):
  with (
    CsvLogger(tmp_path / "log.csv", COLUMNS) as log,
    pytest.raises(ValueError, match=r"unexpected=\['w'\]"),
  ):
    log.write(step=0, x=1.0, y=2.0, z=3.0, w=4.0)


def test_write_outside_the_context_manager_is_rejected(tmp_path):
  log = CsvLogger(tmp_path / "log.csv", COLUMNS)
  with pytest.raises(RuntimeError, match="context manager"):
    log.write(step=0, x=1.0, y=2.0, z=3.0)


def test_write_after_exit_is_rejected(tmp_path):
  path = tmp_path / "log.csv"
  with CsvLogger(path, COLUMNS) as log:
    log.write(step=0, x=1.0, y=2.0, z=3.0)

  with pytest.raises(RuntimeError, match="context manager"):
    log.write(step=1, x=1.0, y=2.0, z=3.0)


def test_round_trips_through_pandas(tmp_path):
  path = tmp_path / "log.csv"
  with CsvLogger(path, COLUMNS) as log:
    for step in range(5):
      log.write(step=step, x=float(step), y=-float(step), z=0.5)

  frame = pd.read_csv(path)
  assert list(frame.columns) == list(COLUMNS)
  assert len(frame) == 5
  assert frame["x"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
  assert (frame["z"] == 0.5).all()


def test_reentering_truncates(tmp_path):
  """The file is opened "w", so a second run does not append to the first."""
  path = tmp_path / "log.csv"
  logger = CsvLogger(path, COLUMNS)

  with logger as log:
    log.write(step=0, x=1.0, y=2.0, z=3.0)
  with logger as log:
    log.write(step=9, x=9.0, y=9.0, z=9.0)

  rows = read(path)
  assert len(rows) == 2
  assert rows[1][0] == "9"


def test_accepts_a_string_path(tmp_path):
  path = tmp_path / "log.csv"
  with CsvLogger(str(path), COLUMNS) as log:
    log.write(step=0, x=1.0, y=2.0, z=3.0)
  assert path.exists()


def test_nan_is_written_as_a_readable_value(tmp_path):
  """navigate.py logs NaN for map_depth when the sonar returns no echo."""
  path = tmp_path / "log.csv"
  with CsvLogger(path, COLUMNS) as log:
    log.write(step=0, x=float("nan"), y=2.0, z=3.0)

  frame = pd.read_csv(path)
  assert frame["x"].isna().all()
