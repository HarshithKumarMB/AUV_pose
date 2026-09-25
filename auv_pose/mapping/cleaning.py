"""Separating seabed from objects standing on it, using the soundings alone.

The Dam seabed carries pipelines and a valve manifold: 4-5 m tall, about 10 m
across, and sharp-edged. A bathymetric map should not be asked to fit them.
Its smooth mean spreads each edge over a wide ramp, so a misplaced sounding
costs almost nothing over most of the ramp and far more than the slope predicts
at the edge -- errors no smooth map can explain, and which dominate its rmse.

So objects are a separate class, found the way ground is separated from
buildings in airborne LiDAR: a morphological **opening** (erosion, then
dilation) of the lowest return per cell, with a window wider than any object.
An opening leaves any surface that is flat or sloping at the window's scale
exactly as it is, and cuts off anything narrower than the window that stands
above it.

**Width alone does not tell a pipe from a mound.** An opening also clips the
rounded cap of any mound whose top is narrower than the window, and the Dam has
broad 1-3 m mounds: a single 1 m threshold flagged 37% of a pass, a third of it
seabed. What separates them is height -- pipes and the manifold stand 4-6 m
proud, mounds about 3 -- so the threshold is a hysteresis, as in edge detection:
a **core** must stand well above the opened ground, and an object then takes in
its **flanks**, anything modestly proud within a few metres of a core. That
reaches a pipe's sides without swallowing a mound beside it.

Nothing here reads a reference surface or the simulator, so it runs on
hardware data as it does on simulated.
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
  :param cell: Grid cell side, metres. The lowest return per cell is taken as
      that cell's candidate ground.
  :param window: Side of the square structuring element, metres. Must exceed
      the widest object to be removed: a pipe here is about 10 m across.
  :return: ``(n,)`` ground height beneath each sounding; ``nan`` where the
      window around it held no soundings to erode against.

  Note:
      An opening removes what stands *above* the surface, not what dips below
      it. A single spuriously deep return drags the ground down around it. The
      multibeam here returns every beam and has no such spikes; a real sonar's
      would be rejected by a range gate before this runs.
  """
  points = np.asarray(points, dtype=float)
  z = np.asarray(z, dtype=float)
  index, shape = _grid(points, cell)

  lowest = np.full(tuple(shape), np.inf)
  np.minimum.at(lowest, (index[:, 0], index[:, 1]), z)

  size = max(1, round(window / cell))
  size += 1 - size % 2  # odd, so the element is centred on its cell

  eroded = ndimage.grey_erosion(lowest, size=(size, size), mode="nearest")
  # An empty cell must not win the dilation: erosion leaves it infinite only
  # where no sounding lies within the window at all.
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

  :param points: Horizontal positions, ``(n, 2)``, metres.
  :param z: Elevation, ``(n,)``, z-up.
  :param cell: See :func:`ground_surface`.
  :param window: See :func:`ground_surface`.
  :param core: Height above the opened ground that makes a sounding an
      object on its own, metres. Above the Dam's mounds, below its pipes.
  :param flank: Height that makes a sounding an object when it lies within
      ``reach`` of a core, metres. Well above the sonar's 0.1 m quantisation.
  :param reach: How far an object's flanks extend from its core, metres:
      about the part of a pipe's width below the core height.
  :return: ``(n,)`` boolean, True for an object sounding.
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
