"""Reading the seabed out of HoloOcean's cached octree.

The simulator builds an octree of the world for its sonar to raycast against and
caches it on disk as JSON, under::

    <HOLODECKPATH>/<version>/worlds/Ocean/<platform>/Holodeck/Octrees/<world>/
        min<a>_max<b>/<x>_<y>_<z>.json

``min2_max512`` is the cache for ``octree_min: 0.02`` and ``octree_max: 5.12``;
tile filenames are the tile centre in Unreal **centimetres**. Each tile is a
recursive ``{"p": [x, y, z], "l": [children]}``; a node with no ``"l"`` is a leaf
carrying its centre ``"p"``, a surface normal ``"n"`` and a material ``"m"``.

This is the same geometry the sonar sees, so it is ground truth for the survey
rather than an independent model of it -- which is exactly what makes it useful:
a sounding that disagrees with it is a sonar defect, not terrain.

Pure parsing and numpy, no simulator dependency, so this is testable offline.

Note:
    The full Dam cache is 2289 tiles and 45 GB. Always pass a bounds box --
    :func:`tile_paths` filters by filename so only the tiles you need are opened.
"""

import hashlib
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
  "TILE_NAME",
  "cached_surface",
  "default_cache_dir",
  "leaves",
  "load_surface",
  "robust_spread",
  "surface_residual",
  "tile_paths",
  "top_surface",
]

#: Tile filenames are ``<x>_<y>_<z>.json`` in centimetres. ``roots.json`` sits
#: alongside them as an index and does not match.
TILE_NAME = re.compile(r"(-?\d+)_(-?\d+)_(-?\d+)\.json\Z")

#: Unreal stores the octree in centimetres; everything else here is metres.
CENTIMETRES_PER_METRE = 100.0

#: Tiles are addressed by centre, but their leaves extend beyond it. Measured on
#: the Dam cache, leaves reach +-2.55 m from the centre of a ``max512`` tile
#: (half of 5.12, less one leaf), so a bounds query has to over-select by at
#: least that or it clips its own edges.
TILE_HALF_SPAN_M = 2.56


def tile_paths(
  directory: str | Path,
  bounds: tuple[float, float, float, float] | None = None,
  margin: float = TILE_HALF_SPAN_M,
) -> list[Path]:
  """Tiles whose leaves may fall inside a horizontal bounds box.

  Selection is by filename alone, so this opens nothing. That matters: the Dam
  cache is 45 GB and a survey box needs about 4% of it.

  Args:
      directory: A ``min<a>_max<b>`` directory of tile JSON files.
      bounds: ``(x_min, x_max, y_min, y_max)`` in metres, or None for every tile.
      margin: How far beyond its centre a tile's leaves may reach, metres. The
          default over-selects rather than clipping the edges of the box.

  Returns:
      Tile paths, sorted, so a run is reproducible.

  Raises:
      FileNotFoundError: If the directory does not exist.
  """
  directory = Path(directory)
  if not directory.is_dir():
    raise FileNotFoundError(f"no octree directory at {directory}")

  paths = []
  for path in sorted(directory.iterdir()):
    match = TILE_NAME.search(path.name)
    if match is None:
      continue
    if bounds is None:
      paths.append(path)
      continue

    x, y, _ = (int(v) / CENTIMETRES_PER_METRE for v in match.groups())
    x_min, x_max, y_min, y_max = bounds
    if (
      x_min - margin <= x <= x_max + margin
      and y_min - margin <= y <= y_max + margin
    ):
      paths.append(path)

  return paths


def leaves(tile: dict) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Leaf centres and normals of one parsed tile.

  Walks iteratively. The tree reaches the 2 cm leaf level from a 5.12 m root,
  which is deep enough that a recursive walk is a real risk on the default
  recursion limit.

  Args:
      tile: A parsed tile, ``{"p": [...], "l": [...]}``.

  Returns:
      ``(points, normals)``, each ``(n, 3)``. Points are in **metres**; a leaf
      with no recorded normal gets zeros.
  """
  points: list[list[float]] = []
  normals: list[list[float]] = []

  stack = [tile]
  while stack:
    node = stack.pop()
    # A leaf is a node with no ``"l"`` at all. Testing truthiness instead would
    # turn a childless interior node into a surface point at its own centre,
    # which is metres above the geometry it was meant to subdivide.
    if "l" in node:
      stack.extend(node["l"])
    else:
      points.append(node["p"])
      normals.append(node.get("n", (0.0, 0.0, 0.0)))

  if not points:
    return np.empty((0, 3)), np.empty((0, 3))

  return (
    np.asarray(points, dtype=float) / CENTIMETRES_PER_METRE,
    np.asarray(normals, dtype=float),
  )


def top_surface(
  points: ArrayLike,
  normals: ArrayLike | None = None,
  cell: float = 0.10,
  min_normal_z: float = 0.0,
) -> NDArray[np.float64]:
  """Reduce a leaf cloud to the highest surface per horizontal cell.

  The octree stores a one-voxel-thick shell rather than a filled volume -- leaf
  density on the Dam seabed is ~2000 within a 0.5 m disc against 1963 for a
  single 2 cm layer -- so the highest leaf in a cell is the surface, not the top
  of a solid.

  Args:
      points: Leaf centres, ``(n, 3)`` in metres.
      normals: Leaf normals, ``(n, 3)``. Required if ``min_normal_z`` is set.
      cell: Horizontal cell side in metres.
      min_normal_z: Drop leaves whose normal's z component is below this. The Dam
          seabed is 99.8% upward-facing so the default keeps everything, but on a
          world with walls or overhangs this is what separates seabed from
          structure.

  Returns:
      ``(m, 3)`` of the highest leaf in each occupied cell, sorted by cell.

  Raises:
      ValueError: If ``min_normal_z`` is set without normals, or the shapes
          disagree.
  """
  points = np.asarray(points, dtype=float)
  if points.ndim != 2 or points.shape[1] != 3:
    raise ValueError(f"expected (n, 3) points, got {points.shape}")

  if min_normal_z > 0.0:
    if normals is None:
      raise ValueError("min_normal_z needs normals")
    normals = np.asarray(normals, dtype=float)
    if normals.shape != points.shape:
      raise ValueError(
        f"points and normals must match: {points.shape} vs {normals.shape}"
      )
    points = points[normals[:, 2] >= min_normal_z]

  if len(points) == 0:
    return np.empty((0, 3))

  key = np.floor(points[:, :2] / cell).astype(np.int64)
  # Sort by cell, then by descending z, so the first row of each cell is its
  # highest leaf.
  order = np.lexsort((-points[:, 2], key[:, 1], key[:, 0]))
  points, key = points[order], key[order]

  first = np.ones(len(points), dtype=bool)
  first[1:] = (key[1:] != key[:-1]).any(axis=1)
  return points[first]


def load_surface(
  directory: str | Path,
  bounds: tuple[float, float, float, float] | None = None,
  cell: float = 0.10,
  min_normal_z: float = 0.0,
  paths: Iterable[str | Path] | None = None,
) -> NDArray[np.float64]:
  """Seabed surface over a bounds box, as world-frame ``(x, y, z)`` in metres.

  Reduces each tile as it is read rather than accumulating every leaf: the Dam
  survey box is 1.4 GB of JSON and tens of millions of leaves, but only a few
  hundred thousand surface cells.

  Args:
      directory: A ``min<a>_max<b>`` directory of tile JSON files.
      bounds: ``(x_min, x_max, y_min, y_max)`` in metres, or None for everything.
      cell: Horizontal cell side in metres.
      min_normal_z: See :func:`top_surface`.
      paths: Explicit tile paths, bypassing ``directory``/``bounds`` selection.

  Returns:
      ``(m, 3)`` surface points, one per occupied cell.
  """
  if paths is None:
    paths = tile_paths(directory, bounds)

  reduced = []
  for path in paths:
    with open(path) as handle:
      points, normals = leaves(json.load(handle))
    if len(points):
      reduced.append(top_surface(points, normals, cell, min_normal_z))

  if not reduced:
    return np.empty((0, 3))

  # Tiles overlap at their edges, so cells can appear twice and the reduction
  # has to run once more across the join.
  surface = np.concatenate(reduced)
  if bounds is not None:
    x_min, x_max, y_min, y_max = bounds
    inside = (
      (surface[:, 0] >= x_min)
      & (surface[:, 0] <= x_max)
      & (surface[:, 1] >= y_min)
      & (surface[:, 1] <= y_max)
    )
    surface = surface[inside]

  return top_surface(surface, cell=cell)


def surface_residual(
  points: ArrayLike, surface: ArrayLike, max_distance: float | None = 1.0
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
  """How far each sounding sits above the surface, vertically.

  The one place the "nearest horizontal cell, then subtract" comparison lives.
  It is the measurement every claim in this project rests on -- whether a sonar
  is working, whether a beam geometry is right, whether a map is any good -- so
  it should not be reimplemented per caller.

  Args:
      points: Soundings, ``(n, 3)``. Non-finite rows are dropped rather than
          poisoning the result; a beam with no echo is normal, not an error.
      surface: Reference surface, ``(m, 3)``, from :func:`load_surface`.
      max_distance: Drop soundings with no surface cell within this horizontal
          distance, metres. **The default is not a tuning knob, it is a trap
          guard.** A nearest-neighbour lookup always returns something, so a
          sounding outside the extracted region silently snaps to the nearest
          edge cell and reports a residual of whatever the terrain does there.
          Measured, that turned a survey whose beams reach 40 m either side of
          the track into a 1.99 m "error" against a surface covering only the
          20 m-wide box. Pass None to disable.

  Returns:
      ``(residual, kept)`` where ``residual`` is the signed vertical offset of
      each *kept* sounding from the surface beneath it, and ``kept`` is the
      boolean mask selecting those soundings from ``points``. The mask is
      returned because callers routinely need to line residuals back up with
      per-beam or per-ping metadata -- and because a low ``kept.mean()`` is the
      signal that the surface does not cover the data.

  Raises:
      ValueError: If either array is not ``(n, 3)``.
  """
  points = np.asarray(points, dtype=float)
  surface = np.asarray(surface, dtype=float)
  for name, array in (("points", points), ("surface", surface)):
    if array.ndim != 2 or array.shape[1] != 3:
      raise ValueError(f"expected (n, 3) {name}, got {array.shape}")

  kept = np.isfinite(points).all(axis=1)
  if not kept.any() or len(surface) == 0:
    return np.empty(0), np.zeros(len(points), dtype=bool)

  # Imported here: scipy is a heavy dependency and the parsing half of this
  # module is useful without it.
  from scipy.spatial import KDTree

  distance, nearest = KDTree(surface[:, :2]).query(points[kept, :2])
  if max_distance is not None:
    covered = distance <= max_distance
    nearest = nearest[covered]
    # Fold the coverage test back into the caller-facing mask, so residuals and
    # metadata stay aligned.
    kept[np.flatnonzero(kept)[~covered]] = False

  return points[kept, 2] - surface[nearest, 2], kept


def robust_spread(residual: ArrayLike) -> tuple[float, float]:
  """Median and MAD-derived std of a residual.

  **Always report these rather than mean and standard deviation.** The errors
  here are routinely bimodal -- the singlebeam's soundings split 53/47 between
  the true seabed echo and one 4.5 m short -- and the mean of that mixture
  describes neither population. Reporting a mean is how the defect stayed
  hidden through several rounds of analysis.

  Returns:
      ``(median, mad_std)``, the latter scaled by 1.4826 so it matches a
      standard deviation for normally distributed residuals.
  """
  residual = np.asarray(residual, dtype=float)
  if residual.size == 0:
    return float("nan"), float("nan")
  median = float(np.median(residual))
  return median, float(np.median(np.abs(residual - median)) * 1.4826)


def default_cache_dir() -> Path:
  """Where :func:`cached_surface` keeps reductions, honouring ``XDG_CACHE_HOME``.

  Deliberately not inside the repository or the simulator's install: the
  reduction is a derived artefact of a 45 GB cache that is itself derived.
  """
  root = os.environ.get("XDG_CACHE_HOME")
  base = Path(root) if root else Path.home() / ".cache"
  return base / "auv_pose" / "octree"


def cached_surface(
  directory: str | Path,
  bounds: tuple[float, float, float, float],
  cell: float = 0.10,
  min_normal_z: float = 0.0,
  cache_dir: str | Path | None = None,
  refresh: bool = False,
) -> NDArray[np.float64]:
  """:func:`load_surface`, memoised to a ``.npy`` beside a key of its arguments.

  Reducing a survey-sized box takes tens of seconds and a wide one takes
  minutes, which is too slow for anything you have to run more than once --
  fitting beam geometry means evaluating against the same surface repeatedly.

  The key covers the resolved directory, the bounds, the cell and the normal
  filter. It does **not** cover the contents of the tiles: the octree cache is
  written once by the simulator and thereafter only appended to, so a stale
  entry means the box was extended, not that its geometry changed. Pass
  ``refresh`` if that assumption ever breaks.

  Args:
      directory: A ``min<a>_max<b>`` directory of tile JSON files.
      bounds: ``(x_min, x_max, y_min, y_max)`` in metres.
      cell: Horizontal cell side in metres.
      min_normal_z: See :func:`top_surface`.
      cache_dir: Where to keep reductions; defaults to
          :func:`default_cache_dir`.
      refresh: Recompute and overwrite an existing entry.

  Returns:
      ``(m, 3)`` surface points, as :func:`load_surface` would.
  """
  directory = Path(directory)
  cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()

  key = hashlib.sha256(
    repr(
      (
        str(directory.resolve()),
        tuple(round(float(b), 4) for b in bounds),
        round(float(cell), 6),
        round(float(min_normal_z), 6),
      )
    ).encode()
  ).hexdigest()[:16]
  path = cache / f"surface-{key}.npy"

  if path.exists() and not refresh:
    return np.load(path)

  surface = load_surface(directory, bounds, cell, min_normal_z)
  cache.mkdir(parents=True, exist_ok=True)
  # Write via a temporary file so an interrupted run cannot leave a truncated
  # .npy that later loads as a silently short surface.
  temporary = path.with_name(path.name + ".partial")
  # Write through an open handle: np.save appends ".npy" to a path that does
  # not already end in it, which would silently put the data somewhere else.
  with open(temporary, "wb") as handle:
    np.save(handle, surface)
  temporary.replace(path)
  return surface
