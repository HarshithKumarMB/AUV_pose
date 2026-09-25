"""Separating seabed from objects standing on it, using the soundings alone.

A morphological opening of the lowest return per cell gives the ground; a
hysteresis on height above it (core, then flanks) tells pipes from mounds.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage


def _grid(points: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
  """Integer cell of each point, shifted so the lowest index is zero."""
  index = np.floor(points / cell).astype(np.int64)
  return index - index.min(axis=0), index.max(axis=0) - index.min(axis=0) + 1


def ground_surface(
  points: ArrayLike, z: ArrayLike, cell: float = 1.0, window: float = 15.0
) -> NDArray[np.float64]:
  """Opened seabed height under each sounding.

  :param points: Horizontal positions, ``(n, 2)``, metres.
  :param z: Elevation, ``(n,)``, z-up.
  :param cell: Grid cell side, metres.
  :param window: Side of the square structuring element, metres. Must exceed
      the widest object to be removed.
  :return: ``(n,)`` ground height beneath each sounding.

  Spuriously deep returns drag the ground down; gate them first.
  """
  points = np.asarray(points, dtype=float)
  z = np.asarray(z, dtype=float)
  index, shape = _grid(points, cell)

  lowest = np.full(tuple(shape), np.inf)
  np.minimum.at(lowest, (index[:, 0], index[:, 1]), z)

  size = max(1, round(window / cell))
  size += 1 - size % 2  # odd, so the element is centred on its cell

  eroded = ndimage.grey_erosion(lowest, size=(size, size), mode="nearest")
  # An empty cell must not win the dilation.
  eroded[~np.isfinite(eroded)] = -np.inf
  opened = ndimage.grey_dilation(eroded, size=(size, size), mode="nearest")

  ground = opened[index[:, 0], index[:, 1]]
  return np.where(np.isfinite(ground), ground, np.nan)


def object_soundings(
  points: ArrayLike,
  z: ArrayLike,
  cell: float = 1.0,
  window: float = 15.0,
  core: float = 3.5,
  flank: float = 0.5,
  reach: float = 4.0,
) -> NDArray[np.bool_]:
  """Which soundings stand on an object rather than the seabed.

  Arguments up to ``window`` as for :func:`ground_surface`; heights are
  above the opened ground, in metres.

  :param core: Height that makes a sounding an object on its own.
  :param flank: Height that makes a sounding an object within ``reach`` of a
      core.
  """
  points = np.asarray(points, dtype=float)
  above = np.nan_to_num(
    np.asarray(z, dtype=float) - ground_surface(points, z, cell, window),
    nan=0.0,
  )

  index, shape = _grid(points, cell)
  cores = np.zeros(tuple(shape), dtype=bool)
  cores[index[above > core, 0], index[above > core, 1]] = True

  radius = max(0, round(reach / cell))
  offsets = np.arange(-radius, radius + 1)
  disc = offsets[:, None] ** 2 + offsets[None, :] ** 2 <= radius**2
  near = np.asarray(ndimage.binary_dilation(cores, structure=disc), dtype=bool)

  return (above > core) | ((above > flank) & near[index[:, 0], index[:, 1]])
