"""Separating seabed from objects on it, by morphological opening."""

import numpy as np

from auv_pose.mapping.cleaning import ground_surface, object_soundings


def seabed(rng, slope=(0.0, 0.0), n=40_000, extent=100.0):
  points = rng.uniform(0.0, extent, size=(n, 2))
  z = -68.0 + points @ np.asarray(slope) + rng.normal(scale=0.05, size=n)
  return points, z


def pipe(points, centre_x=50.0, width=8.0, height=4.5):
  """A pipe along y: a cylindrical cap ``width`` across and ``height`` tall."""
  offset = np.abs(points[:, 0] - centre_x) / (width / 2)
  return height * np.sqrt(np.clip(1.0 - offset**2, 0.0, None))


def test_a_flat_seabed_has_no_objects():
  points, z = seabed(np.random.default_rng(0))
  assert object_soundings(points, z).mean() == 0.0


def test_a_sloping_seabed_has_no_objects():
  """An opening passes a plane through unchanged, whatever its slope."""
  points, z = seabed(np.random.default_rng(1), slope=(0.08, -0.05))
  assert object_soundings(points, z).mean() < 0.001


def test_a_pipe_is_found_and_the_seabed_beside_it_is_not():
  points, z = seabed(np.random.default_rng(2), slope=(0.03, 0.0))
  bump = pipe(points)
  flagged = object_soundings(points, z + bump)

  assert flagged[bump > 1.5].all()
  assert not flagged[bump == 0.0].any()


def test_the_ground_under_a_pipe_is_the_seabed():
  points, z = seabed(np.random.default_rng(3))
  ground = ground_surface(points, z + pipe(points))
  under = pipe(points) > 0.0
  np.testing.assert_allclose(ground[under], -68.0, atol=0.3)


def test_an_object_wider_than_the_window_is_ground():
  """The window is the definition of an object: wider than it is terrain."""
  points, z = seabed(np.random.default_rng(4))
  mound = pipe(points, width=40.0, height=3.0)
  assert not object_soundings(points, z + mound, window=15.0)[mound > 0].any()


def test_isolated_soundings_are_left_alone():
  """Nothing within the window to compare against is not evidence of an object."""
  points = np.array([[0.0, 0.0], [500.0, 500.0]])
  z = np.array([-68.0, -60.0])
  assert not object_soundings(points, z).any()
