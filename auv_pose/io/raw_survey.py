"""The raw survey log: every sensor reading, before soundings are placed.

A log directory holds ``ticks.csv`` (sensors, ``nan`` when silent, plus
scoring-only ``true_*`` columns), ``pings.csv`` (tick and range per beam) and
``meta.npz``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from auv_pose.io.logs import CsvWriter

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

TICKS, PINGS, META = "ticks.csv", "pings.csv", "meta.npz"


def ping_columns(n_beams: int) -> tuple[str, ...]:
  """Columns of ``pings.csv`` for a fan of ``n_beams``."""
  return ("tick", *(f"range_{i}" for i in range(n_beams)))


def _flatten(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
  """Nested metadata as flat arrays keyed ``outer/inner``."""
  flat: dict[str, Any] = {}
  for key, value in tree.items():
    if "/" in key:
      raise ValueError(f"metadata key {key!r} contains '/'")
    name = prefix + key
    if name in ("file", "allow_pickle"):
      raise ValueError(f"metadata key {name!r} is reserved by np.savez")
    if isinstance(value, dict):
      if not value:
        raise ValueError(f"metadata {name!r} is an empty dict")
      flat |= _flatten(value, name + "/")
      continue
    array = np.asarray(value)
    # NumPy turns a list mixing numbers and strings into strings, silently.
    mixed = array.dtype.kind == "U" and not all(
      isinstance(item, str) for item in np.asarray(value, dtype=object).ravel()
    )
    if array.dtype == object or mixed:
      raise ValueError(f"metadata {name!r} is not a plain array: {value!r}")
    flat[name] = array
  return flat


def _unflatten(stored: Any) -> dict[str, Any]:
  """Inverse of :func:`_flatten`; 0-d arrays come back as plain values."""
  tree: dict[str, Any] = {}
  for key in stored.files:
    *outer, leaf = key.split("/")
    node = tree
    for part in outer:
      node = node.setdefault(part, {})
    value = stored[key]
    node[leaf] = value.item() if value.ndim == 0 else value
  return tree


def _triple(prefix: str, values: ArrayLike | None) -> dict[str, float]:
  if values is None:
    return {f"{prefix}_{axis}": np.nan for axis in "xyz"}
  x, y, z = np.asarray(values, dtype=float)
  return {f"{prefix}_x": x, f"{prefix}_y": y, f"{prefix}_z": z}


class RawSurveyWriter:
  """Stream a raw survey log into a directory.

  Use as a context manager. ``meta.npz`` is written on entry, so a log that
  dies partway is still interpretable.

  :param meta: Nested dicts of numbers, strings and arrays.
  """

  def __init__(
    self, directory: str | Path, n_beams: int, meta: dict[str, Any]
  ) -> None:
    self.directory = Path(directory)
    self.n_beams = n_beams
    self.meta = meta
    self._ticks = CsvWriter(self.directory / TICKS, TICK_COLUMNS)
    self._pings = CsvWriter(self.directory / PINGS, ping_columns(n_beams))

  def __enter__(self) -> Self:
    self.directory.mkdir(parents=True, exist_ok=True)
    np.savez(self.directory / META, **_flatten(self.meta))
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

  :param ranges: Picked range per beam, ``(n_pings, n_beams)``.
  :param meta: ``meta.npz``, as nested dicts.
  """

  ticks: pd.DataFrame
  ping_ticks: NDArray[np.int64]
  ranges: NDArray[np.float64]
  meta: dict[str, Any]

  @property
  def start_offset(self) -> NDArray[np.float64]:
    """Believed start minus true start, horizontal: the map frame's offset.

    Falls back to the first tick's truth when ``initial_truth`` is absent.
    """
    believed = np.asarray(self.meta["initial_mean"]["position"][:2], float)
    recorded = self.meta.get("initial_truth")
    if recorded is not None:
      return believed - np.asarray(recorded["position"][:2], float)
    return believed - self.ticks[["true_x", "true_y"]].to_numpy(float)[0]

  def readings(self, prefix: str) -> NDArray[np.float64]:
    """An ``(n_ticks, 3)`` block of one sensor, ``nan`` where it was silent."""
    return self.ticks[[f"{prefix}_{axis}" for axis in "xyz"]].to_numpy(float)


def _meta(path: Path) -> dict[str, Any]:
  with np.load(path, allow_pickle=False) as stored:
    return _unflatten(stored)


def load_raw_survey(directory: str | Path) -> RawSurvey:
  """Read a raw survey log written by :class:`RawSurveyWriter`.

  :raises FileNotFoundError: If the directory lacks one of its three files.
  """
  directory = Path(directory)
  missing = [
    name for name in (TICKS, PINGS, META) if not (directory / name).exists()
  ]
  if missing:
    raise FileNotFoundError(
      f"{directory} is not a raw survey log: missing {', '.join(missing)}. "
      "Raw logs are directories written by experiments/survey.py"
    )

  ticks = pd.read_csv(directory / TICKS)
  absent = set(TICK_COLUMNS) - set(ticks.columns)
  if absent:
    raise ValueError(f"{directory / TICKS} lacks {sorted(absent)}")

  pings = pd.read_csv(directory / PINGS)
  return RawSurvey(
    ticks=ticks,
    ping_ticks=pings["tick"].to_numpy(np.int64),
    ranges=pings.drop(columns="tick").to_numpy(float),
    meta=_meta(directory / META),
  )
