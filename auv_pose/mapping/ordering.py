"""Maximin ordering and conditioning sets for the Vecchia approximation.

A bad ordering fails quietly: still a valid Gaussian, just a worse one.
"""

import heapq

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import KDTree


def maximin_order(
  points: ArrayLike, first: int | None = None
) -> NDArray[np.int64]:
  """Order points so each is as far as possible from all its predecessors.

  :param first: Index to start from; defaults to the point nearest the
      centroid.
  :return: A permutation of ``0..n-1``.

  Ties break by index, so the ordering is reproducible on grid-like data.
  """
  points = np.asarray(points, dtype=float)
  if points.ndim != 2:
    raise ValueError(f"expected (n, d) points, got {points.shape}")

  n = len(points)
  if n == 0:
    return np.empty(0, dtype=np.int64)

  if first is None:
    centroid = points.mean(axis=0)
    first = int(np.argmin(np.linalg.norm(points - centroid, axis=1)))

  order = np.empty(n, dtype=np.int64)
  order[0] = first

  tree = KDTree(points)
  distance = np.linalg.norm(points - points[first], axis=1)
  distance[first] = -np.inf

  # Negated for a max-heap; the index breaks ties.
  heap = [(-distance[i], i) for i in range(n) if i != first]
  heapq.heapify(heap)

  for position in range(1, n):
    while True:
      key, candidate = heapq.heappop(heap)
      # Lazy deletion: a stale key overstates the distance.
      if -key == distance[candidate]:
        break

    order[position] = candidate
    radius = distance[candidate]
    distance[candidate] = -np.inf

    # Only points within `radius` of the new one can have moved closer.
    affected = np.asarray(
      tree.query_ball_point(points[candidate], radius), dtype=np.int64
    )
    if affected.size:
      moved = np.linalg.norm(points[affected] - points[candidate], axis=1)
      closer = moved < distance[affected]
      for index, value in zip(affected[closer], moved[closer]):
        distance[index] = value
        heapq.heappush(heap, (-value, int(index)))

  return order


def _spread_predecessors(
  row: int, taken: NDArray[np.int64], count: int
) -> list[int]:
  """Up to ``count`` distinct predecessors of ``row``, evenly spaced in order.

  :param taken: Indices already chosen as nearest neighbours, to avoid.
  """
  if count <= 0 or row <= len(taken):
    return []

  used = {int(index) for index in taken}
  chosen: list[int] = []

  for step in range(1, count + 1):
    target = (step * row) // (count + 1)

    probe = target
    while probe < row and probe in used:
      probe += 1
    if probe >= row:
      probe = target
      while probe >= 0 and probe in used:
        probe -= 1
    if 0 <= probe < row:
      used.add(probe)
      chosen.append(probe)

  return chosen


def ordered_neighbours(
  points: ArrayLike, m: int, block: int = 2048, near: int | None = None
) -> NDArray[np.int64]:
  """Each point's conditioning set, drawn from its predecessors.

  :param points: Positions **already in the intended order**, shape ``(n, d)``.
  :param block: Points per tree build; performance only.
  :param near: How many of the ``m`` are nearest predecessors; the other
      ``m - near`` are spread across the ordering, which helps estimate the
      range (Stein, Chi and Welty 2004). Defaults to ``m``, which suits
      prediction.
  :return: Indices into the same ordering, shape ``(n, m)``, padded with
      ``-1``.
  """
  points = np.asarray(points, dtype=float)
  if points.ndim != 2:
    raise ValueError(f"expected (n, d) points, got {points.shape}")
  if m < 1:
    raise ValueError(f"expected m >= 1, got {m}")

  near = m if near is None else near
  if not 1 <= near <= m:
    raise ValueError(f"expected 1 <= near <= m = {m}, got {near}")
  far = m - near

  n = len(points)
  neighbours = np.full((n, m), -1, dtype=np.int64)

  for start in range(1, n, block):
    stop = min(start + block, n)
    rows = np.arange(start, stop)

    # Candidates from before the block, via one tree.
    take = min(near, start)
    tree = KDTree(points[:start])
    outside_distance, outside_index = tree.query(points[rows], k=take)
    outside_distance = np.atleast_2d(outside_distance.reshape(len(rows), take))
    outside_index = np.atleast_2d(outside_index.reshape(len(rows), take))

    # Candidates inside the block, masked to strictly earlier.
    inside = points[start:stop]
    within = np.linalg.norm(inside[:, None] - inside[None], axis=-1)
    earlier = np.tril(np.ones_like(within, dtype=bool), -1)
    within = np.where(earlier, within, np.inf)
    inside_index = np.arange(start, stop)

    for local, row in enumerate(rows):
      candidates = np.concatenate([outside_index[local], inside_index])
      distances = np.concatenate([outside_distance[local], within[local]])

      finite = np.isfinite(distances)
      candidates, distances = candidates[finite], distances[finite]

      keep = np.argsort(distances, kind="stable")[:near]
      chosen = candidates[keep]
      neighbours[row, : len(chosen)] = chosen

      spread = _spread_predecessors(int(row), chosen, far)
      neighbours[row, len(chosen) : len(chosen) + len(spread)] = spread

  return neighbours
