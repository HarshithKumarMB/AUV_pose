"""Reading the seabed out of HoloOcean's cached octree."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from auv_pose.mapping.octree import (
  leaves,
  load_surface,
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


@pytest.mark.skipif(
  not CACHE.is_dir(), reason="no local octree cache; needs a simulator run"
)
def test_agrees_with_the_survey_soundings_that_picked_the_right_peak():
  """The claim the whole diagnosis rests on, checked against the real cache.

  ``map.csv``'s soundings are bimodal: the far population is the true seabed
  echo and the near one is 12 range bins short. The far population must agree
  with this surface to better than the sonar's own 0.113 m quantisation, or the
  reader is not ground truth and nothing may be concluded from it.
  """
  from scipy.spatial import KDTree

  surface = load_surface(CACHE, bounds=(-40.0, 0.0, -20.0, 0.0))
  assert len(surface) > 10_000

  frame = pd.concat([pd.read_csv("map.csv"), pd.read_csv("map1.csv")])
  _, nearest = KDTree(surface[:, :2]).query(frame[["x", "y"]].to_numpy())
  residual = surface[nearest, 2] + frame["sonar_depth"].to_numpy()

  far = residual[frame["sonar_depth"].to_numpy() >= 67.5]
  median = np.median(far)
  spread = np.median(np.abs(far - median)) * 1.4826

  assert spread < 0.113, (
    f"MAD-std {spread:.4f} m exceeds the quantisation floor"
  )
  # The offset is the survey vehicle's actual depth, which was never recorded.
  assert abs(median) < 0.5
