"""The raw survey log: every sensor reading, before anything is placed.

A survey flown on the vehicle's own navigation cannot write soundings as it
goes, because where a sounding lies depends on a pose that the smoother only
settles once the whole run is in. So the survey writes what the sensors said,
and :mod:`experiments.georeference` places the soundings afterwards -- which is
also how a real survey is processed.

A log is a directory holding three files:

``ticks.csv``
    One row per simulator tick: the IMU, whichever aiding sensors reported
    that tick (``nan`` otherwise), and **ground truth**, logged for scoring
    only. Nothing in the navigation reads a ``true_`` column.
``pings.csv``
    One row per sonar ping: the tick it was taken at, and the picked range of
    every beam (``nan`` for a beam with no echo).
``meta.json``
    What is needed to interpret the other two: tick rate, beam bearings and
    axes, sensor noise, and the initial belief navigation started from.

Both CSVs are streamed row by row, so a run that dies partway keeps everything
up to that point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from auv_pose.io.logs import CsvLogger

__all__ = [
  "TICK_COLUMNS",
  "RawSurvey",
  "RawSurveyWriter",
  "load_raw_survey",
  "ping_columns",
]

#: One row per tick. ``true_`` columns are ground truth and scoring-only.
TICK_COLUMNS = (
  "tick",
  "gyro_x",
  "gyro_y",
  "gyro_z",
  "accel_x",
  "accel_y",
  "accel_z",
  "dvl_x",
  "dvl_y",
  "dvl_z",
  "depth",
  "mag_x",
  "mag_y",
  "mag_z",
  "true_x",
  "true_y",
  "true_z",
  "true_qw",
  "true_qx",
  "true_qy",
  "true_qz",
  "true_gyro_bias_x",
  "true_gyro_bias_y",
  "true_gyro_bias_z",
  "true_accel_bias_x",
  "true_accel_bias_y",
  "true_accel_bias_z",
)

_FILES = ("ticks.csv", "pings.csv", "meta.json")


def ping_columns(n_beams: int) -> tuple[str, ...]:
  """Columns of ``pings.csv`` for a fan of ``n_beams``."""
  return ("tick", *(f"range_{i}" for i in range(n_beams)))


def _triple(prefix: str, values: ArrayLike | None) -> dict[str, float]:
  if values is None:
    return {f"{prefix}_{axis}": np.nan for axis in "xyz"}
  x, y, z = np.asarray(values, dtype=float)
  return {f"{prefix}_x": x, f"{prefix}_y": y, f"{prefix}_z": z}


class RawSurveyWriter:
  """Stream a raw survey log into a directory.

  Use as a context manager. ``meta.json`` is written on entry, so a log that
  dies partway is still interpretable.

  :param directory: Where to write; created if absent.
  :param n_beams: Beams per ping.
  :param meta: Everything needed to interpret the log; must be JSON-able.
  """

  def __init__(
    self, directory: str | Path, n_beams: int, meta: dict[str, Any]
  ) -> None:
    self.directory = Path(directory)
    self.n_beams = n_beams
    self.meta = meta
    self._ticks = CsvLogger(self.directory / "ticks.csv", TICK_COLUMNS)
    self._pings = CsvLogger(self.directory / "pings.csv", ping_columns(n_beams))

  def __enter__(self) -> Self:
    self.directory.mkdir(parents=True, exist_ok=True)
    (self.directory / "meta.json").write_text(json.dumps(self.meta, indent=2))
    self._ticks.__enter__()
    self._pings.__enter__()
    return self

  def __exit__(self, *exc_info) -> None:
    self._ticks.__exit__(*exc_info)
    self._pings.__exit__(*exc_info)

  def tick(
    self,
    index: int,
    gyro: ArrayLike,
    accel: ArrayLike,
    true_position: ArrayLike,
    true_attitude: ArrayLike,
    true_gyro_bias: ArrayLike,
    true_accel_bias: ArrayLike,
    dvl: ArrayLike | None = None,
    depth: float | None = None,
    magnetometer: ArrayLike | None = None,
  ) -> None:
    """Record one tick. Aiding left as ``None`` did not report this tick."""
    qw, qx, qy, qz = np.asarray(true_attitude, dtype=float)
    true_x, true_y, true_z = np.asarray(true_position, dtype=float)
    self._ticks.write(
      tick=index,
      **_triple("gyro", gyro),
      **_triple("accel", accel),
      **_triple("dvl", dvl),
      depth=np.nan if depth is None else float(depth),
      **_triple("mag", magnetometer),
      true_x=true_x,
      true_y=true_y,
      true_z=true_z,
      true_qw=qw,
      true_qx=qx,
      true_qy=qy,
      true_qz=qz,
      **_triple("true_gyro_bias", true_gyro_bias),
      **_triple("true_accel_bias", true_accel_bias),
    )

  def ping(self, index: int, ranges: ArrayLike) -> None:
    """Record the picked range of every beam of one ping."""
    ranges = np.asarray(ranges, dtype=float)
    if ranges.shape != (self.n_beams,):
      raise ValueError(f"expected {self.n_beams} ranges, got {ranges.shape}")
    self._pings.write(
      tick=index, **{f"range_{i}": r for i, r in enumerate(ranges)}
    )


@dataclass(frozen=True)
class RawSurvey:
  """A raw survey log, read back.

  :param ticks: ``ticks.csv`` as a frame, one row per tick.
  :param ping_ticks: Tick of each ping, ``(n_pings,)``.
  :param ranges: Picked range per beam, ``(n_pings, n_beams)``.
  :param meta: ``meta.json``.
  """

  ticks: pd.DataFrame
  ping_ticks: NDArray[np.int64]
  ranges: NDArray[np.float64]
  meta: dict[str, Any]

  def readings(self, prefix: str) -> NDArray[np.float64]:
    """An ``(n_ticks, 3)`` block of one sensor, ``nan`` where it was silent."""
    return self.ticks[[f"{prefix}_{axis}" for axis in "xyz"]].to_numpy(float)


def load_raw_survey(directory: str | Path) -> RawSurvey:
  """Read a raw survey log written by :class:`RawSurveyWriter`.

  :raises FileNotFoundError: If the directory lacks one of its three files --
      most likely a soundings CSV passed where a raw log was expected.
  """
  directory = Path(directory)
  missing = [name for name in _FILES if not (directory / name).exists()]
  if missing:
    raise FileNotFoundError(
      f"{directory} is not a raw survey log: missing {', '.join(missing)}. "
      "Raw logs are directories written by experiments/survey.py"
    )

  ticks = pd.read_csv(directory / "ticks.csv")
  absent = set(TICK_COLUMNS) - set(ticks.columns)
  if absent:
    raise ValueError(f"{directory}/ticks.csv lacks {sorted(absent)}")

  pings = pd.read_csv(directory / "pings.csv")
  return RawSurvey(
    ticks=ticks,
    ping_ticks=pings["tick"].to_numpy(np.int64),
    ranges=pings.drop(columns="tick").to_numpy(float),
    meta=json.loads((directory / "meta.json").read_text()),
  )
