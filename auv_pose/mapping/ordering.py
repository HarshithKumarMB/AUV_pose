"""Ordering the soundings, and finding each one's predecessors.

A Vecchia approximation factorises a joint Gaussian as a chain,
``p(x) = prod_i p(x_i | x_{g(i)})``, where ``g(i)`` holds at most ``m`` indices
drawn from earlier in some ordering. Both halves of that sentence matter: the
approximation's accuracy depends on the **ordering** quite as much as on ``m``,
and a badly ordered map fails quietly -- it is still a valid, positive-definite
Gaussian, merely a worse approximation of the one intended.

This module is deliberately free of torch. Ordering is geometry, it runs once
per map rather than once per optimiser step, and keeping it in numpy makes the
brute-force reference implementations in the tests trivial to write.

**Maximin ordering**, following the recommendation of Katzfuss, Guinness, Gong
and Zilber (2020): take the point furthest from everything chosen so far, so
early points sketch the whole survey coarsely and later ones fill it in. A
coordinate or arrival ordering instead conditions each point on a set lying
entirely to one side of it, which is a poor summary of its surroundings.
"""

from __future__ import annotations

import heapq

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["maximin_order", "ordered_neighbours"]


def maximin_order(
  points: ArrayLike, first: int | None = None
) -> NDArray[np.int64]:
  """Order points so each is as far as possible from all its predecessors.

  :param points: Positions, shape ``(n, d)``.
  :param first: Index to start from. Defaults to the point nearest the
      centroid, which is reproducible and puts the coarse sketch in the middle
      of the survey rather than at a corner.
  :return: A permutation of ``0..n-1``, shape ``(n,)``.

  Note:
      Exact, and near ``O(n log n)`` rather than the naive ``O(n^2)``. Two
      observations do the work.

      First, selecting a point can only *lower* other points' distance to the
      selected set, and only for points **within that distance of the new
      one**. Since every unselected point's distance is at most the one just
      selected, a ball query of that radius finds every point that can have
      changed. The radii shrink as the ordering proceeds -- that is what
      maximin means -- so the balls get cheap, and one tree over *all* the
      points, built once, serves every query.

      Second, the running maximum comes off a heap with lazy deletion. A stale
      entry always records a distance no smaller than the truth, so it surfaces
      early and is discarded on sight by comparing its key against the live
      value.

      The earlier version of this rebuilt a tree over the *selected* set at
      geometric intervals and brute-forced the remainder. It was also exact,
      but the brute-forced remainder grew to ``O(n)``, giving a measured
      ``n^1.6`` -- about twenty minutes at the size of a decimated survey.

  Note:
      Ties break by index. After decimation the soundings sit near a regular
      grid and exact ties are common; a stored ordering that does not reproduce
      is a thoroughly unpleasant bug to chase.
  """
  from scipy.spatial import KDTree

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

  # Negated because heapq is a min-heap and we want the largest; the index
  # breaks ties, which is what makes the ordering reproducible.
  heap = [(-distance[i], i) for i in range(n) if i != first]
  heapq.heapify(heap)

  for position in range(1, n):
    while True:
      key, candidate = heapq.heappop(heap)
      # A stale key always overstates the distance, so it surfaces first and
      # is recognised by disagreeing with the live value.
      if -key == distance[candidate]:
        break

    order[position] = candidate
    radius = distance[candidate]
    distance[candidate] = -np.inf

    # Only points within `radius` of the new one can have moved closer to the
    # selected set, because none of them was further than `radius` to begin
    # with.
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
  """``count`` predecessors of ``row`` spread across the whole ordering.

  :param row: The point being conditioned, so its predecessors are
      ``0..row-1``.
  :param taken: Indices already chosen as nearest neighbours, to avoid.
  :param count: How many to return.
  :return: Up to ``count`` distinct indices below ``row``.

  Note:
      Targets are placed at evenly spaced *ordering positions*. Collisions with
      the nearest set are resolved by walking forward, then backward, which
      keeps the result deterministic -- the same requirement that makes
      :func:`maximin_order` break ties by index.
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
  :param m: Conditioning-set size.
  :param block: Points handled per tree build. An optimisation only -- the
      result does not depend on it, and there is a test saying so.
  :param near: How many of the ``m`` are nearest predecessors. The remaining
      ``m - near`` are spread across the ordering. Defaults to ``m``, i.e. all
      nearest, which is the right choice for *prediction*.
  :return: Indices into the same ordering, shape ``(n, m)``, padded with ``-1``
      where a point has fewer than ``m`` predecessors.

  Note:
      **Why anything but nearest neighbours.** Stein, Chi and Welty (2004)
      show that conditioning only on nearest neighbours is a poor design for
      *estimating* covariance parameters, even though it is the best design for
      predicting. Their table 2 measures the damage where it applies to this
      map exactly -- a linear mean, profiled out by a restricted likelihood.
      Relative efficiency for the range parameter at ``m = 32``:

      ==========  =============  ===========
      ``near/m``  constant mean  linear mean
      ==========  =============  ===========
      1.0         96.3%          **77.1%**
      0.75        96.6%          **93.8%**
      0.5         95.3%          92.8%
      ==========  =============  ===========

      With a constant mean the choice barely matters; with a linear mean,
      replacing a quarter of the nearest neighbours by distant points recovers
      most of what was lost. Their section 4.2 gives the mechanism: when the
      correlation range is an appreciable fraction of the survey, the
      information about it "is contained in the dependences at longer
      distances", which a purely local conditioning set never sees.

  Note:
      **This is not their construction, and the difference is worth stating.**
      Stein et al. select the distant members by *distance rank* among all
      ``j - 1`` predecessors, which needs every pairwise distance -- ``O(n^2)``,
      affordable at their ``n = 1000`` and not at a survey's several hundred
      thousand. Here the distant members are taken at evenly spaced *ordering
      positions* instead.

      Under maximin ordering that is a reasonable substitute rather than a
      coincidence: the first ``k`` points of a maximin ordering are spread over
      the whole survey by construction, so an index drawn from early in the
      ordering is a spatially spread sample, and one drawn from anywhere in it
      is at a typical survey-scale distance from the point being conditioned.
      What it supplies is long-range information, which is what the range
      parameter is starved of. It does **not** reproduce their graded ladder of
      intermediate distances, so the gains above are an upper bound on what to
      expect, and :func:`~auv_pose.mapping.vecchia.fit_vecchia` is the place
      that has to earn them on real soundings.

  Note:
      The obvious implementation -- a fresh tree over each prefix -- is ``n``
      tree builds, ``O(n^2 log n)``. Instead each block of points gets **one**
      tree over everything before the block, plus a brute-force scan within the
      block masked to strictly-earlier entries. Every candidate is covered by
      exactly one of those two, so the nearest part is exact.

      Cost is ``(n/c) O(n log n) + n c``, minimised near ``c = sqrt(n log n)``,
      which at the ~144k soundings of a decimated survey is about 1560. Hence
      the default of 2048.
  """
  from scipy.spatial import KDTree

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
