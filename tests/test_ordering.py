"""Maximin ordering and predecessor neighbour sets.

Both functions are optimised versions of things with obvious slow definitions,
so both are checked against those definitions written out longhand here. That is
the whole point: the lazy-greedy heap and the blocked tree builds are supposed
to produce *identical* answers to the naive loops, not merely similar ones, and
nothing else in the suite would notice if they did not.
"""

from itertools import pairwise

import numpy as np

from auv_pose.mapping.ordering import maximin_order, ordered_neighbours


def scatter(n, seed, spread=50.0):
  return np.random.default_rng(seed).uniform(-spread, spread, size=(n, 2))


def brute_force_maximin(points, first):
  """The definition, written out: repeatedly take the furthest point."""
  n = len(points)
  order = [first]
  remaining = [i for i in range(n) if i != first]

  while remaining:
    distance = np.linalg.norm(
      points[remaining][:, None] - points[order][None], axis=-1
    ).min(axis=1)
    # Ties by index, matching the implementation.
    best = int(np.lexsort((remaining, -distance))[0])
    order.append(remaining.pop(best))

  return np.array(order)


def brute_force_neighbours(points, m):
  """The definition: for each point, the m nearest among its predecessors."""
  n = len(points)
  out = np.full((n, m), -1, dtype=np.int64)
  for i in range(1, n):
    distance = np.linalg.norm(points[:i] - points[i], axis=1)
    keep = np.argsort(distance, kind="stable")[:m]
    out[i, : len(keep)] = keep
  return out


# -- maximin ----------------------------------------------------------------


def test_it_matches_a_brute_force_greedy():
  """The pin on the lazy-greedy heap. Identical, not similar."""
  for seed in (0, 1, 2):
    points = scatter(200, seed)
    np.testing.assert_array_equal(
      maximin_order(points, first=0), brute_force_maximin(points, first=0)
    )


def test_the_selection_distances_never_increase():
  """The defining property: each point is further out than the next.

  Catches an ordering that is a valid permutation but not actually maximin --
  which is the failure the KL test downstream would see only as "slightly worse
  accuracy", if at all.
  """
  points = scatter(400, 3)
  order = maximin_order(points)

  distances = [
    np.linalg.norm(points[order[:k]] - points[order[k]], axis=1).min()
    for k in range(1, len(order))
  ]
  assert all(a >= b - 1e-12 for a, b in pairwise(distances))


def test_it_is_a_permutation():
  order = maximin_order(scatter(300, 4))
  np.testing.assert_array_equal(np.sort(order), np.arange(300))


def test_it_starts_where_it_is_told():
  points = scatter(50, 5)
  for first in (0, 17, 49):
    assert maximin_order(points, first=first)[0] == first


def test_the_default_start_is_nearest_the_centroid():
  points = scatter(120, 6)
  expected = int(
    np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1))
  )
  assert maximin_order(points)[0] == expected


def test_it_is_deterministic_on_tied_distances():
  """A decimated survey is nearly a grid, so exact ties are the common case."""
  grid = np.stack(
    np.meshgrid(np.arange(12) * 0.25, np.arange(12) * 0.25), axis=-1
  ).reshape(-1, 2)

  first = maximin_order(grid, first=0)
  second = maximin_order(grid.copy(), first=0)
  np.testing.assert_array_equal(first, second)


def test_it_handles_degenerate_inputs():
  np.testing.assert_array_equal(maximin_order(np.empty((0, 2))), np.empty(0))
  np.testing.assert_array_equal(maximin_order(np.zeros((1, 2))), [0])
  assert len(maximin_order(np.zeros((5, 2)))) == 5


def test_it_spreads_before_it_fills_in():
  """The property the approximation actually relies on.

  The first handful of points should span the survey, not huddle. Compare the
  spread of the first eight against the spread of eight consecutive arrivals.
  """
  points = scatter(500, 7)
  order = maximin_order(points)

  early = points[order[:8]]
  arbitrary = points[:8]

  def spread(block):
    return np.linalg.norm(block[:, None] - block[None], axis=-1).mean()

  assert spread(early) > spread(arbitrary)


# -- neighbour sets ---------------------------------------------------------


def test_they_match_a_brute_force_search():
  for seed in (8, 9):
    points = scatter(500, seed)
    np.testing.assert_array_equal(
      ordered_neighbours(points, m=10, block=64),
      brute_force_neighbours(points, m=10),
    )


def test_the_block_size_does_not_change_the_answer():
  """Blocking is an optimisation, not a model choice."""
  points = scatter(400, 10)
  reference = ordered_neighbours(points, m=8, block=4096)

  for block in (1, 7, 64, 333):
    np.testing.assert_array_equal(
      ordered_neighbours(points, m=8, block=block), reference
    )


def test_every_neighbour_is_earlier_in_the_order():
  """The Vecchia condition. Violating it would not be an approximation at all."""
  points = scatter(600, 11)
  neighbours = ordered_neighbours(points, m=12, block=128)

  for i, row in enumerate(neighbours):
    present = row[row >= 0]
    assert np.all(present < i)
    assert len(np.unique(present)) == len(present)


def test_the_head_is_padded_where_there_are_too_few_predecessors():
  points = scatter(40, 12)
  neighbours = ordered_neighbours(points, m=10)

  assert np.all(neighbours[0] == -1)
  for i in range(1, 10):
    assert (neighbours[i] >= 0).sum() == i
  assert np.all(neighbours[10:] >= 0)


def test_a_neighbour_set_larger_than_the_data_still_works():
  points = scatter(6, 13)
  neighbours = ordered_neighbours(points, m=20)

  assert neighbours.shape == (6, 20)
  for i, row in enumerate(neighbours):
    assert (row >= 0).sum() == i


def test_the_nearest_predecessor_really_is_nearest():
  """Independent of the brute-force helper, in case that is wrong too."""
  points = scatter(300, 14)
  neighbours = ordered_neighbours(points, m=5, block=32)

  for i in range(1, len(points)):
    present = neighbours[i][neighbours[i] >= 0]
    chosen = np.linalg.norm(points[present] - points[i], axis=1).max()
    others = np.setdiff1d(np.arange(i), present)
    if len(others):
      assert (
        chosen
        <= np.linalg.norm(points[others] - points[i], axis=1).min() + 1e-12
      )


def test_it_rejects_a_nonsense_conditioning_size():
  points = scatter(10, 15)
  for m in (0, -1):
    try:
      ordered_neighbours(points, m=m)
    except ValueError:
      continue
    raise AssertionError(f"expected a ValueError for m={m}")


# -- mixed near/far conditioning --------------------------------------------


def test_all_nearest_is_the_default():
  """`near=m` must reproduce the plain nearest-neighbour sets exactly."""
  points = scatter(400, 20)
  np.testing.assert_array_equal(
    ordered_neighbours(points, m=10),
    ordered_neighbours(points, m=10, near=10),
  )


def test_the_nearest_part_is_still_exactly_nearest():
  """Splitting the set must not disturb the neighbours it does keep."""
  points = scatter(500, 21)
  reference = brute_force_neighbours(points, m=6)
  mixed = ordered_neighbours(points, m=10, near=6)

  # The first `near` columns are the nearest, in the same order.
  np.testing.assert_array_equal(mixed[:, :6], reference)


def test_the_far_points_are_predecessors_and_distinct():
  """The Vecchia condition still has to hold for the spread members."""
  points = scatter(600, 22)
  neighbours = ordered_neighbours(points, m=12, near=8)

  for i, row in enumerate(neighbours):
    present = row[row >= 0]
    assert np.all(present < i)
    assert len(np.unique(present)) == len(present)


def test_the_far_points_really_are_further_away():
  """The property the design exists for.

  Averaged over rows with a full conditioning set, the spread members must sit
  substantially further from the point than the nearest members do. Without
  this the change is cosmetic.
  """
  points = scatter(800, 23)
  near, m = 8, 12
  neighbours = ordered_neighbours(points, m=m, near=near)

  close, distant = [], []
  for i in range(200, len(points)):
    row = neighbours[i]
    close.append(np.linalg.norm(points[row[:near]] - points[i], axis=1).mean())
    spread = row[near:]
    spread = spread[spread >= 0]
    distant.append(np.linalg.norm(points[spread] - points[i], axis=1).mean())

  assert np.mean(distant) > 5.0 * np.mean(close)


def test_the_conditioning_set_stays_full():
  """A collision with the nearest set must not silently shrink the row."""
  points = scatter(700, 24)
  neighbours = ordered_neighbours(points, m=15, near=10)

  for i in range(15, len(points)):
    assert (neighbours[i] >= 0).sum() == 15


def test_it_is_deterministic():
  points = scatter(300, 25)
  first = ordered_neighbours(points, m=10, near=7)
  second = ordered_neighbours(points.copy(), m=10, near=7)
  np.testing.assert_array_equal(first, second)


def test_the_block_size_still_does_not_change_the_answer():
  points = scatter(400, 26)
  reference = ordered_neighbours(points, m=10, near=7, block=4096)

  for block in (1, 7, 64, 333):
    np.testing.assert_array_equal(
      ordered_neighbours(points, m=10, near=7, block=block), reference
    )


def test_it_rejects_a_nonsense_split():
  points = scatter(50, 27)
  for near in (0, -1, 11):
    try:
      ordered_neighbours(points, m=10, near=near)
    except ValueError:
      continue
    raise AssertionError(f"expected a ValueError for near={near}")


def test_the_head_still_takes_everything_it_can():
  """Early rows have too few predecessors to split; they take all of them."""
  points = scatter(60, 28)
  neighbours = ordered_neighbours(points, m=12, near=8)

  for i in range(1, 12):
    present = neighbours[i][neighbours[i] >= 0]
    assert len(present) == i
    np.testing.assert_array_equal(np.sort(present), np.arange(i))
