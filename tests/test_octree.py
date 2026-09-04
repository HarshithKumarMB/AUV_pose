"""Reading the seabed out of HoloOcean's cached octree."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from auv_pose.mapping.octree import (
  cached_surface,
  leaves,
  load_surface,
  robust_spread,
  surface_residual,
  tile_paths,
  top_surface,
)

CACHE = (
  Path(os.environ.get("HOLODECKPATH", Path.home() / "data" / "holoocean"))
  / "2.3.0/worlds/Ocean/Linux/Holodeck/Octrees/Dam/min2_max512"
)


def leaf(x, y, z, normal=(0.0, 0.0, 1.0)):
  """A leaf node, positioned in centimetres as the cache stores them."""
  return {"p": [x, y, z], "n": list(normal), "m": "MaterialNotFound"}


def write_tile(directory, name, node):
  path = directory / name
  path.write_text(json.dumps(node))
  return path


def test_leaves_are_returned_in_metres():
  points, normals = leaves(leaf(-1029, -1055, -6927))
  assert points == pytest.approx(np.array([[-10.29, -10.55, -69.27]]))
  assert normals == pytest.approx(np.array([[0.0, 0.0, 1.0]]))


def test_walks_nested_nodes():
  tile = {
    "p": [0, 0, 0],
    "l": [
      {"p": [0, 0, 0], "l": [leaf(100, 0, -100), leaf(200, 0, -100)]},
      leaf(300, 0, -100),
    ],
  }
  points, _ = leaves(tile)
  assert sorted(points[:, 0]) == pytest.approx([1.0, 2.0, 3.0])


def test_walks_deeply_without_hitting_the_recursion_limit():
  """2 cm leaves under a 5.12 m root is deep enough for this to matter."""
  node = leaf(0, 0, 0)
  for _ in range(5000):
    node = {"p": [0, 0, 0], "l": [node]}

  points, _ = leaves(node)
  assert len(points) == 1


def test_leaf_without_a_normal_gets_zeros():
  _, normals = leaves({"p": [0, 0, 0]})
  assert normals == pytest.approx(np.zeros((1, 3)))


def test_empty_tile_returns_empty_arrays():
  points, normals = leaves({"p": [0, 0, 0], "l": []})
  assert points.shape == (0, 3)
  assert normals.shape == (0, 3)


def test_top_surface_keeps_the_highest_leaf_per_cell():
  points = [[0.0, 0.0, -70.0], [0.02, 0.0, -68.0], [0.04, 0.0, -69.0]]
  surface = top_surface(points, cell=0.10)
  assert len(surface) == 1
  assert surface[0, 2] == pytest.approx(-68.0)


def test_top_surface_separates_cells():
  points = [[0.0, 0.0, -70.0], [0.5, 0.0, -60.0]]
  surface = top_surface(points, cell=0.10)
  assert len(surface) == 2
  assert sorted(surface[:, 2]) == pytest.approx([-70.0, -60.0])


def test_normal_filter_drops_a_vertical_wall():
  """A wall stands above the seabed and would otherwise win the cell."""
  points = np.array([[0.0, 0.0, -70.0], [0.02, 0.0, -60.0]])
  normals = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

  kept = top_surface(points, normals, cell=0.10, min_normal_z=0.7)
  assert len(kept) == 1
  assert kept[0, 2] == pytest.approx(-70.0)

  # Without the filter the wall is the top of the cell.
  assert top_surface(points, normals, cell=0.10)[0, 2] == pytest.approx(-60.0)


def test_normal_filter_needs_normals():
  with pytest.raises(ValueError, match="normals"):
    top_surface([[0.0, 0.0, 0.0]], min_normal_z=0.7)


def test_rejects_points_that_are_not_three_dimensional():
  with pytest.raises(ValueError, match=r"\(n, 3\)"):
    top_surface([[0.0, 0.0]])


def test_top_surface_of_nothing_is_empty():
  assert top_surface(np.empty((0, 3))).shape == (0, 3)


def test_tile_selection_is_by_filename(tmp_path):
  for name in ["0_0_0.json", "10000_0_0.json", "roots.json"]:
    write_tile(tmp_path, name, leaf(0, 0, 0))

  selected = tile_paths(tmp_path, bounds=(-1.0, 1.0, -1.0, 1.0))
  assert [p.name for p in selected] == ["0_0_0.json"]


def test_tile_selection_ignores_the_roots_index(tmp_path):
  write_tile(tmp_path, "roots.json", leaf(0, 0, 0))
  assert tile_paths(tmp_path) == []


def test_tile_selection_over_selects_by_the_tile_span(tmp_path):
  """Leaves reach +-2.55 m past the centre, so an exact filter clips the edges."""
  write_tile(tmp_path, "500_0_0.json", leaf(0, 0, 0))

  assert tile_paths(tmp_path, bounds=(-1.0, 3.0, -1.0, 1.0)) != []
  assert tile_paths(tmp_path, bounds=(-1.0, 3.0, -1.0, 1.0), margin=0.0) == []


def test_missing_directory_is_an_error(tmp_path):
  with pytest.raises(FileNotFoundError):
    tile_paths(tmp_path / "absent")


def test_load_surface_merges_tiles_across_their_join(tmp_path):
  """Tiles overlap at the edges; the same cell must not appear twice."""
  write_tile(tmp_path, "0_0_0.json", {"p": [0, 0, 0], "l": [leaf(1, 0, -7000)]})
  write_tile(
    tmp_path, "512_0_0.json", {"p": [0, 0, 0], "l": [leaf(1, 0, -6900)]}
  )

  surface = load_surface(tmp_path, bounds=(-10.0, 10.0, -10.0, 10.0))
  assert len(surface) == 1
  assert surface[0, 2] == pytest.approx(-69.0)


def test_load_surface_clips_to_the_bounds(tmp_path):
  write_tile(
    tmp_path,
    "0_0_0.json",
    {"p": [0, 0, 0], "l": [leaf(0, 0, -7000), leaf(200, 0, -7000)]},
  )

  surface = load_surface(tmp_path, bounds=(-0.5, 0.5, -0.5, 0.5))
  assert len(surface) == 1
  assert surface[0, 0] == pytest.approx(0.0)


def test_load_surface_of_an_empty_box_is_empty(tmp_path):
  write_tile(tmp_path, "0_0_0.json", leaf(0, 0, 0))
  assert load_surface(tmp_path, bounds=(1000.0, 1001.0, 0.0, 1.0)).shape == (
    0,
    3,
  )


def test_surface_residual_measures_height_above_the_surface():
  surface = np.array([[0.0, 0.0, -70.0], [1.0, 0.0, -68.0]])
  points = np.array([[0.01, 0.0, -69.5], [1.02, 0.0, -68.5]])

  residual, kept = surface_residual(points, surface)
  assert kept.all()
  assert residual == pytest.approx([0.5, -0.5])


def test_surface_residual_drops_beams_with_no_echo():
  """NaN rows are the normal case at the edge of a swath, not an error."""
  surface = np.array([[0.0, 0.0, -70.0]])
  points = np.array([[0.0, 0.0, -69.0], [np.nan, np.nan, np.nan]])

  residual, kept = surface_residual(points, surface)
  assert kept.tolist() == [True, False]
  assert residual == pytest.approx([1.0])


def test_surface_residual_mask_lines_up_with_the_input():
  surface = np.array([[0.0, 0.0, -70.0]])
  points = np.array([[0.0, 0.0, np.inf], [0.0, 0.0, -69.0]])

  residual, kept = surface_residual(points, surface)
  assert kept.tolist() == [False, True]
  assert len(residual) == int(kept.sum())


def test_surface_residual_drops_soundings_the_surface_does_not_cover():
  """The trap: a nearest-neighbour lookup always returns *something*.

  A sounding 40 m outside the extracted region snaps to the nearest edge cell
  and reports a confident residual of whatever the terrain does there.
  """
  surface = np.array([[0.0, 0.0, -70.0]])
  points = np.array([[0.0, 0.0, -69.0], [40.0, 0.0, -69.0]])

  residual, kept = surface_residual(points, surface)
  assert kept.tolist() == [True, False]
  assert residual == pytest.approx([1.0])

  # Disabling the guard reproduces the silent snap.
  unguarded, kept = surface_residual(points, surface, max_distance=None)
  assert kept.all()
  assert unguarded == pytest.approx([1.0, 1.0])


def test_surface_residual_coverage_mask_stays_aligned():
  """kept must index the original array, not the finite subset."""
  surface = np.array([[0.0, 0.0, -70.0]])
  points = np.array(
    [
      [40.0, 0.0, -69.0],  # uncovered
      [np.nan, np.nan, np.nan],  # no echo
      [0.0, 0.0, -69.5],  # good
    ]
  )

  residual, kept = surface_residual(points, surface)
  assert kept.tolist() == [False, False, True]
  assert residual == pytest.approx([0.5])


def test_surface_residual_of_nothing_finite_is_empty():
  residual, kept = surface_residual(
    np.full((2, 3), np.nan), np.array([[0.0, 0.0, 0.0]])
  )
  assert len(residual) == 0
  assert not kept.any()


def test_surface_residual_rejects_bad_shapes():
  with pytest.raises(ValueError, match="points"):
    surface_residual(np.zeros((2, 2)), np.zeros((1, 3)))
  with pytest.raises(ValueError, match="surface"):
    surface_residual(np.zeros((2, 3)), np.zeros((1, 2)))


def test_robust_spread_ignores_a_second_population():
  """The property the mean lacks, and why this function exists."""
  bulk = np.zeros(60)
  outliers = np.full(40, -4.5)
  median, spread = robust_spread(np.concatenate([bulk, outliers]))

  assert median == pytest.approx(0.0)
  assert spread == pytest.approx(0.0)
  # A mean would land at -1.8, describing neither population.
  assert np.mean(np.concatenate([bulk, outliers])) == pytest.approx(-1.8)


def test_robust_spread_matches_std_on_clean_data():
  sample = np.random.default_rng(0).normal(3.0, 2.0, 20000)
  median, spread = robust_spread(sample)
  assert median == pytest.approx(3.0, abs=0.05)
  assert spread == pytest.approx(2.0, abs=0.05)


def test_robust_spread_of_nothing_is_nan():
  median, spread = robust_spread(np.empty(0))
  assert np.isnan(median) and np.isnan(spread)


def test_cached_surface_returns_the_same_answer(tmp_path):
  tiles = tmp_path / "tiles"
  tiles.mkdir()
  write_tile(tiles, "0_0_0.json", {"p": [0, 0, 0], "l": [leaf(0, 0, -7000)]})
  bounds = (-1.0, 1.0, -1.0, 1.0)

  direct = load_surface(tiles, bounds=bounds)
  cached = cached_surface(tiles, bounds, cache_dir=tmp_path / "cache")
  assert cached == pytest.approx(direct)


def test_cached_surface_does_not_reread_the_tiles(tmp_path):
  """The point of the cache: a second call must not touch the 45 GB source."""
  tiles = tmp_path / "tiles"
  tiles.mkdir()
  write_tile(tiles, "0_0_0.json", {"p": [0, 0, 0], "l": [leaf(0, 0, -7000)]})
  bounds = (-1.0, 1.0, -1.0, 1.0)
  cache = tmp_path / "cache"

  first = cached_surface(tiles, bounds, cache_dir=cache)
  (tiles / "0_0_0.json").unlink()

  assert cached_surface(tiles, bounds, cache_dir=cache) == pytest.approx(first)


def test_cached_surface_keys_on_the_bounds(tmp_path):
  tiles = tmp_path / "tiles"
  tiles.mkdir()
  write_tile(
    tiles,
    "0_0_0.json",
    {"p": [0, 0, 0], "l": [leaf(0, 0, -7000), leaf(500, 0, -6900)]},
  )
  cache = tmp_path / "cache"

  narrow = cached_surface(tiles, (-1.0, 1.0, -1.0, 1.0), cache_dir=cache)
  wide = cached_surface(tiles, (-1.0, 10.0, -1.0, 1.0), cache_dir=cache)
  assert len(narrow) == 1
  assert len(wide) == 2


def test_refresh_recomputes(tmp_path):
  tiles = tmp_path / "tiles"
  tiles.mkdir()
  write_tile(tiles, "0_0_0.json", {"p": [0, 0, 0], "l": [leaf(0, 0, -7000)]})
  bounds = (-1.0, 1.0, -1.0, 1.0)
  cache = tmp_path / "cache"

  cached_surface(tiles, bounds, cache_dir=cache)
  write_tile(tiles, "0_0_0.json", {"p": [0, 0, 0], "l": [leaf(0, 0, -6800)]})

  assert cached_surface(tiles, bounds, cache_dir=cache)[0, 2] == pytest.approx(
    -70.0
  )
  refreshed = cached_surface(tiles, bounds, cache_dir=cache, refresh=True)
  assert refreshed[0, 2] == pytest.approx(-68.0)


def test_cache_leaves_no_partial_file(tmp_path):
  tiles = tmp_path / "tiles"
  tiles.mkdir()
  write_tile(tiles, "0_0_0.json", {"p": [0, 0, 0], "l": [leaf(0, 0, -7000)]})
  cache = tmp_path / "cache"

  cached_surface(tiles, (-1.0, 1.0, -1.0, 1.0), cache_dir=cache)
  assert list(cache.glob("*.partial")) == []


@pytest.mark.skipif(
  not CACHE.is_dir(), reason="no local octree cache; needs a simulator run"
)
def test_the_real_cache_reduces_to_a_plausible_seabed():
  """The reader works on the actual 107 GB cache, not just synthetic tiles.

  Deliberately weak: it checks the survey box reduces to a dense surface with
  physically sensible relief, and nothing about accuracy.

  Accuracy is no longer checkable here. It used to be, against ``map.csv``'s
  far-return population -- but those surveys were the pre-``a1fd5b1``
  ``x, y, sonar_depth`` schema and could not be migrated, since the vehicle's
  own z was never recorded. The equivalent check now lives in
  ``experiments/check_beam_validity.py``, which scores a real capture against a
  ray-cast through this cache and gets -0.049 m with a 0.035 m MAD-std. It
  needs a capture and so cannot be a unit test.
  """
  surface = load_surface(CACHE, bounds=(-40.0, 0.0, -20.0, 0.0))

  assert len(surface) > 10_000
  relief = float(np.ptp(surface[:, 2]))
  assert 0.5 < relief < 50.0, f"implausible relief {relief:.2f} m"
