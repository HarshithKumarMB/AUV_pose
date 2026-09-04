"""Tracing beams through the seabed surface."""

import numpy as np
import pytest

from auv_pose.mapping.raycast import Heightfield, raycast


def plane(z=-70.0, extent=40.0, cell=0.1):
  """A flat surface, as a point cloud on a regular grid."""
  axis = np.arange(-extent, extent + cell / 2, cell)
  x, y = np.meshgrid(axis, axis)
  return np.column_stack([x.ravel(), y.ravel(), np.full(x.size, z)])


def down_and_out(degrees):
  """Unit directions in the x-z plane, tilted from straight down by `degrees`."""
  angles = np.radians(np.atleast_1d(degrees))
  return np.column_stack(
    [np.sin(angles), np.zeros_like(angles), -np.cos(angles)]
  )


def test_a_tilted_beam_returns_the_slant_range():
  """The whole reason this exists: altitude / cos(bearing), not altitude."""
  field = Heightfield(plane(z=-70.0), cell=0.1)
  origin = np.array([0.0, 0.0, -50.0])
  altitude = 20.0

  ranges = raycast(field, origin, down_and_out([0.0, 20.0, 40.0]), t_max=60.0)

  expected = altitude / np.cos(np.radians([0.0, 20.0, 40.0]))
  assert ranges == pytest.approx(expected, abs=0.05)


def test_a_beam_over_a_gap_returns_nan_rather_than_a_range():
  """A missing cell must fail visibly, not confidently.

  The alternative -- treating an empty cell as ground -- is the mistake
  `surface_residual`'s nearest-neighbour lookup made, where every sounding got
  an answer whether or not the surface covered it.
  """
  field = Heightfield(plane(z=-70.0, extent=5.0), cell=0.1)
  origin = np.array([0.0, 0.0, -50.0])

  # Straight down hits; 60 degrees off nadir leaves the 5 m patch entirely.
  ranges = raycast(field, origin, down_and_out([0.0, 60.0]), t_max=60.0)

  assert ranges[0] == pytest.approx(20.0, abs=0.05)
  assert np.isnan(ranges[1])


def test_a_ray_that_never_reaches_the_surface_is_nan():
  field = Heightfield(plane(z=-70.0), cell=0.1)
  origin = np.array([0.0, 0.0, -50.0])

  # Upward, and downward but stopped short by t_max.
  assert np.isnan(raycast(field, origin, [[0.0, 0.0, 1.0]], t_max=60.0)[0])
  assert np.isnan(raycast(field, origin, [[0.0, 0.0, -1.0]], t_max=10.0)[0])


def test_a_step_is_hit_on_its_near_face():
  """A beam must stop at the first thing it meets, not the ground behind it.

  This is what separates a range from a vertical lookup: directly beneath the
  ray's endpoint the seabed is 20 m down, but the beam crosses a 5 m riser on
  the way and should come back short.
  """
  low, high = plane(z=-70.0, extent=40.0), plane(z=-65.0, extent=40.0)
  # High ground for x > 5 only, so a beam tilted into +x meets its flank.
  surface = np.vstack([low[low[:, 0] <= 5.0], high[high[:, 0] > 5.0]])
  field = Heightfield(surface, cell=0.1)
  origin = np.array([0.0, 0.0, -50.0])

  straight_down, tilted = raycast(
    field, origin, down_and_out([0.0, 45.0]), t_max=60.0
  )

  assert straight_down == pytest.approx(20.0, abs=0.05)
  # At 45 degrees the ray reaches x = 5 having dropped 5 m, to z = -55, still
  # above the -65 step; it keeps going until z = -65 at x = 15.
  assert tilted == pytest.approx(15.0 * np.sqrt(2.0), abs=0.05)
  assert tilted < 20.0 / np.cos(np.radians(45.0))


def test_the_heightfield_takes_the_top_of_stacked_points():
  """Two points in one cell: the surface is the higher one."""
  surface = np.array([[0.0, 0.0, -70.0], [0.02, 0.0, -65.0], [1.0, 0.0, -70.0]])
  field = Heightfield(surface, cell=0.1)

  assert field.elevation(np.array([[0.0, 0.0, 0.0]])) == pytest.approx([-65.0])


def test_elevation_outside_the_raster_is_minus_infinity():
  field = Heightfield(plane(z=-70.0, extent=1.0), cell=0.1)

  elevation = field.elevation(np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]))
  assert elevation[0] == pytest.approx(-70.0)
  assert elevation[1] == -np.inf


def test_coverage_reports_holes():
  full = Heightfield(plane(z=-70.0, extent=2.0), cell=0.1)
  assert full.coverage == pytest.approx(1.0)

  sparse = Heightfield(
    np.array([[0.0, 0.0, -70.0], [2.0, 2.0, -70.0]]), cell=0.1
  )
  assert sparse.coverage < 0.01


def test_bad_input_is_rejected():
  with pytest.raises(ValueError, match="empty surface"):
    Heightfield(np.empty((0, 3)))
  with pytest.raises(ValueError, match=r"\(n, 3\)"):
    Heightfield(np.zeros((4, 2)))
  with pytest.raises(ValueError, match="cell must be positive"):
    Heightfield(plane(extent=1.0), cell=0.0)

  field = Heightfield(plane(extent=1.0))
  with pytest.raises(ValueError, match=r"\(n, 3\)"):
    raycast(field, [0.0, 0.0, 0.0], np.zeros((4, 2)), t_max=10.0)
  with pytest.raises(ValueError, match="step < t_max"):
    raycast(field, [0.0, 0.0, 0.0], [[0.0, 0.0, -1.0]], t_max=10.0, step=20.0)
