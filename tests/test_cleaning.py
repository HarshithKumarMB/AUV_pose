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
  points, z = seabed(np.random.default_rng(4))
  mound = pipe(points, width=40.0, height=3.0)
  assert not object_soundings(points, z + mound, window=15.0)[mound > 0].any()


def test_isolated_soundings_are_left_alone():
  points = np.array([[0.0, 0.0], [500.0, 500.0]])
  z = np.array([-68.0, -60.0])
  assert not object_soundings(points, z).any()


def mound(points, centre=(30.0, 30.0), radius=12.0, height=2.5):
  """A broad, gentle mound: a Gaussian cap, too low to be an object."""
  r2 = ((points - np.asarray(centre)) ** 2).sum(axis=1)
  return height * np.exp(-r2 / (2 * (radius / 2) ** 2))


def test_a_mounds_cap_is_seabed_even_where_the_opening_clips_it():
  points, z = seabed(np.random.default_rng(5))
  bump = mound(points)
  assert (z + bump - ground_surface(points, z + bump)).max() > 1.0
  assert not object_soundings(points, z + bump).any()


def test_a_pipes_flanks_go_with_it():
  """Below core height but beside a core: part of the pipe."""
  points, z = seabed(np.random.default_rng(6))
  bump = pipe(points)
  flanks = (bump > 0.6) & (bump < 3.0)
  assert object_soundings(points, z + bump)[flanks].all()


def test_a_mound_beside_a_pipe_is_not_swallowed():
  points, z = seabed(np.random.default_rng(7))
  bump = pipe(points, centre_x=50.0) + mound(points, centre=(30.0, 50.0))
  flagged = object_soundings(points, z + bump)
  far_side = mound(points, centre=(30.0, 50.0)) > 0.6
  far_side &= np.abs(points[:, 0] - 50.0) > 10.0
  assert not flagged[far_side].any()
