"""Loading sonar survey CSVs.

Columns ``x, y, z`` are the world-frame seabed point a beam struck; ``z`` is
up, so the seabed is negative.
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

SOUNDING_COLUMNS = ("x", "y", "z")

#: Optional on read. ``ping`` groups the beams of one ping; ``true_*`` is where
#: the range lands from the true pose, in the map's frame. Scoring only.
OPTIONAL_COLUMNS = ("ping", "true_x", "true_y", "true_z")


def load_soundings(
  paths: Iterable[str | Path], drop_invalid: bool = True
) -> pd.DataFrame:
  """Read one or more survey CSVs into a single frame.

  Args:
      drop_invalid: Drop rows with a missing or non-finite ``x``, ``y`` or
          ``z`` (beams with no echo).

  Returns:
      Columns ``x, y, z`` plus whichever :data:`OPTIONAL_COLUMNS` every file
      carries.
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
      ``(X, y)``: ``(n, 2)`` horizontal positions and ``(n,)`` elevations.
  """
  X = frame[["x", "y"]].to_numpy(dtype=np.float32)
  y = frame["z"].to_numpy(dtype=np.float32)
  return X, y
