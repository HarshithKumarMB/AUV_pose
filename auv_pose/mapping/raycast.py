"""Tracing sonar beams through the seabed surface.

:mod:`~auv_pose.mapping.octree` reads the simulator's cached octree into a
surface; this asks where a beam pointed along a given direction first meets it.

**Ray-casting rather than a vertical lookup is what makes an off-nadir beam
scorable.** :func:`~auv_pose.mapping.octree.surface_residual` compares a
sounding to the surface directly beneath it, which answers "is this point on
the seabed" but not "should the beam have come back at this range". Over ground
that is flat those agree. Over real ground they do not: at the Dam test site the
seabed under the vehicle varies by 0.02 m while the swath spans 4.04 m of
relief, so a beam pointed at a mound legitimately returns shorter than the
vehicle's altitude and a vertical test calls it a defect.

That distinction is not academic. A whole investigation concluded the sonar
fabricated returns across half its fan, on the strength of a flat-ground test
applied to ground that was not flat.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["Heightfield", "raycast"]


class Heightfield:
  """A surface point cloud as a raster, so a ray can be marched by indexing.

  Marching a ray with a KD-tree query per step costs tens of millions of
  queries for one capture. Rasterising once and indexing gives the same answer
  far faster -- and costs nothing in fidelity, because
  :func:`~auv_pose.mapping.octree.load_surface` has already reduced the leaves
  to one elevation per horizontal cell. The surface *is* a raster; this only
  stores it as one.

  Cells with no surface point hold ``-inf``, so a ray passes through a gap
  rather than stopping at it. That is the safe direction to fail: a missing
  cell yields no hit, which is visible, instead of a confident wrong range.

  Args:
      surface: ``(n, 3)`` world-frame surface points, as
          :func:`~auv_pose.mapping.octree.load_surface` returns.
      cell: Raster cell side in metres. Should match the ``cell`` the surface
          was reduced at; finer only adds empty cells for rays to fall through.

  Raises:
      ValueError: If ``surface`` is not ``(n, 3)`` and non-empty, or ``cell``
          is not positive.
  """

  def __init__(self, surface: ArrayLike, cell: float = 0.10) -> None:
    surface = np.asarray(surface, dtype=float)
    if surface.ndim != 2 or surface.shape[1] != 3:
      raise ValueError(f"expected (n, 3) surface points, got {surface.shape}")
    if not len(surface):
      raise ValueError("cannot build a heightfield from an empty surface")
    if not cell > 0:
      raise ValueError(f"cell must be positive, got {cell}")

    self.cell = float(cell)
    self.origin = surface[:, :2].min(axis=0)

    # Size from the indices the points actually round to, not from
    # ceil(extent / cell). The latter is a float division of a float extent,
    # so a grid that exactly fills its bounds gains a phantom row whenever the
    # extent lands a hair above an integer multiple -- an all-empty edge that
    # rays then fall through.
    index = self._index(surface)
    self.grid = np.full(tuple(index.max(axis=0) + 1), -np.inf)

    # Maximum rather than last-wins: two points can land in one raster cell,
    # and the surface is the top of the geometry.
    np.maximum.at(self.grid, (index[..., 0], index[..., 1]), surface[:, 2])

  @property
  def coverage(self) -> float:
    """Fraction of raster cells holding a surface point.

    Worth checking before trusting a miss: a sparse raster produces NaN ranges
    that look like "the beam saw nothing" but mean "the surface has holes".
    """
    return float(np.isfinite(self.grid).mean())

  def _index(self, points: np.ndarray) -> NDArray[np.int_]:
    return np.rint((points[..., :2] - self.origin) / self.cell).astype(int)

  def elevation(self, points: ArrayLike) -> NDArray[np.float64]:
    """Surface height under each point; ``-inf`` outside the raster.

    Args:
        points: ``(..., 3)`` or ``(..., 2)``; only the horizontal part is used.

    Returns:
        Elevations, shaped like ``points`` without its last axis.
    """
    index = self._index(np.asarray(points, dtype=float))
    inside = np.all((index >= 0) & (index < np.array(self.grid.shape)), axis=-1)

    out = np.full(index.shape[:-1], -np.inf)
    rows, columns = index[..., 0][inside], index[..., 1][inside]
    out[inside] = self.grid[rows, columns]
    return out


def raycast(
  field: Heightfield,
  origin: ArrayLike,
  directions: ArrayLike,
  t_max: float,
  step: float = 0.02,
) -> NDArray[np.float64]:
  """Range at which each ray first meets the surface.

  Fixed-step marching rather than a DDA traversal: the step is a fraction of a
  range bin, so the quantisation it adds is below what the sonar reports
  anyway, and the whole fan marches as one array operation.

  Args:
      field: The surface to trace against.
      origin: Ray origin in world coordinates, ``(3,)``.
      directions: Unit directions, ``(n, 3)``, in the world frame. Beam
          directions in the body frame must be rotated first -- a fan is
          defined about the vehicle, not the world.
      t_max: Stop marching beyond this range, metres. Use the sonar's
          ``RangeMax``; a ray that would hit beyond it does not return either.
      step: March step in metres.

  Returns:
      ``(n,)`` ranges, NaN where the ray reached ``t_max`` without meeting the
      surface -- off the edge of the extracted region, or over a gap in it.

  Raises:
      ValueError: If ``directions`` is not ``(n, 3)``, or ``step`` is not
          positive and smaller than ``t_max``.
  """
  origin = np.asarray(origin, dtype=float).reshape(3)
  directions = np.asarray(directions, dtype=float)
  if directions.ndim != 2 or directions.shape[1] != 3:
    raise ValueError(f"expected (n, 3) directions, got {directions.shape}")
  if not 0 < step < t_max:
    raise ValueError(f"need 0 < step < t_max, got step={step}, t_max={t_max}")

  # From `step`, not 0: a ray starting exactly on the surface would otherwise
  # report a range of zero for every beam.
  t = np.arange(step, t_max, step)
  points = origin[None, None, :] + t[:, None, None] * directions[None, :, :]

  below = points[..., 2] <= field.elevation(points)
  return np.where(below.any(axis=0), t[np.argmax(below, axis=0)], np.nan)
