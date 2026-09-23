"""Loading sonar survey data.

Survey CSVs have columns ``x, y, z``: the **world-frame position of the seabed**
where a beam struck it. ``z`` increases upward, so the seabed is negative and the
GP models it directly with no sign flip.

.. note::

   This replaced an older ``x, y, sonar_depth`` schema in which ``sonar_depth``
   was a positive *range* from the vehicle, recorded at the vehicle's own
   ``(x, y)`` and with the vehicle's ``z`` never written down at all. That works
   only for a single downward beam over a survey that happens to fly at
   ``z = 0``, and it cannot express a multibeam at all: every beam but nadir
   lands ``range * sin(bearing)`` away from the vehicle. Surveys written under
   the old schema cannot be converted, because the depth they were taken from
   was not recorded.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
  "COVARIANCE_COLUMNS",
  "OPTIONAL_COLUMNS",
  "SOUNDING_COLUMNS",
  "load_soundings",
  "position_covariance",
  "soundings_to_arrays",
]

SOUNDING_COLUMNS = ("x", "y", "z")

#: The sounding's own position uncertainty, as ``georeference.py`` derives it
#: from the smoothed pose: the horizontal block's upper triangle, and vertical.
COVARIANCE_COLUMNS = ("cov_xx", "cov_xy", "cov_yy", "cov_zz")

#: Written by ``georeference.py`` and optional on read. ``ping`` groups the
#: beams of one ping, which share the vehicle's error; ``true_`` is where the
#: same range would have landed from the true pose, **in the map's frame** --
#: shifted by the survey's shared start offset, so it can be compared with the
#: map directly. Scoring only.
OPTIONAL_COLUMNS = (*COVARIANCE_COLUMNS, "ping", "true_x", "true_y", "true_z")


def load_soundings(
  paths: Iterable[str | Path], drop_invalid: bool = True
) -> pd.DataFrame:
  """Read one or more survey CSVs into a single frame.

  Args:
      paths: Survey CSV paths, concatenated in order.
      drop_invalid: Drop rows with a missing or non-finite value in any of the
          sounding columns. Survey runs record NaN whenever the sonar returned no
          usable echo, and those rows must not reach the GP.

  Returns:
      A frame with columns ``x, y, z``, plus whichever of
      :data:`OPTIONAL_COLUMNS` every file carries. A column only some files
      have is dropped, not filled: a covariance of zero would claim a
      sounding was placed exactly.

  Raises:
      ValueError: If no paths are given, or a file lacks the expected columns.
  """
  paths = [Path(p) for p in paths]
  if not paths:
    raise ValueError("no sounding files given")

  frames = []
  for path in paths:
    frame = pd.read_csv(path)
    missing = set(SOUNDING_COLUMNS) - set(frame.columns)
    if missing:
      raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
    frames.append(frame)

  shared = [
    column
    for column in OPTIONAL_COLUMNS
    if all(column in frame.columns for frame in frames)
  ]
  columns = [*SOUNDING_COLUMNS, *shared]
  combined = pd.concat([frame[columns] for frame in frames], ignore_index=True)

  if drop_invalid:
    for column in SOUNDING_COLUMNS:
      combined[column] = pd.to_numeric(combined[column], errors="coerce")
    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined = combined.dropna(subset=list(SOUNDING_COLUMNS))
    combined = combined.reset_index(drop=True)

  return combined


def soundings_to_arrays(
  frame: pd.DataFrame,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
  """Split a sounding frame into GP training arrays.

  Returns:
      ``(X, y)`` where ``X`` is ``(n, 2)`` of horizontal position and ``y`` is
      ``(n,)`` of seabed elevation, taken as recorded. The old schema needed a
      sign flip here because it stored a downward range; an elevation already
      points the way the GP models it.
  """
  X = frame[["x", "y"]].to_numpy(dtype=np.float32)
  y = frame["z"].to_numpy(dtype=np.float32)
  return X, y


def position_covariance(frame: pd.DataFrame) -> NDArray[np.float64]:
  """Each sounding's horizontal position covariance, ``(n, 2, 2)``.

  Raises:
      ValueError: If the frame lacks the covariance columns -- a survey placed
          from ground truth, or one written before they existed. Regenerate it
          with ``experiments/georeference.py --pose smoothed``.
  """
  missing = [c for c in COVARIANCE_COLUMNS if c not in frame.columns]
  if missing:
    raise ValueError(
      f"soundings carry no position covariance (missing {', '.join(missing)}); "
      "place them from a raw log with "
      "`experiments/georeference.py --pose smoothed`"
    )
  xx, xy, yy = (
    frame[c].to_numpy(np.float64) for c in ("cov_xx", "cov_xy", "cov_yy")
  )
  return np.stack([np.stack([xx, xy], -1), np.stack([xy, yy], -1)], -2)
