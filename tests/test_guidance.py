"""Waypoint following and thruster mixing."""

import numpy as np
import pytest

from experiments.guidance import (
  THRUST_LIMIT,
  WaypointFollower,
  thruster_command,
)

SQUARE = np.array(
  [
    [0.0, 0.0, 0.0],
    [10.0, 0.0, 0.0],
    [10.0, 10.0, 0.0],
  ]
)


def test_zero_error_gives_zero_thrust():
  np.testing.assert_allclose(thruster_command(np.zeros(3)), np.zeros(8))


def test_vertical_thrusters_all_carry_depth_error():
  """The first four are the vertical bank; nothing else should move on pure z."""
  command = thruster_command([0.0, 0.0, 3.0])
  np.testing.assert_allclose(command[:4], [3.0, 3.0, 3.0, 3.0])
  np.testing.assert_allclose(command[4:], np.zeros(4))


def test_horizontal_mixing_is_the_vectored_layout():
  """[e_x + e_y, e_x - e_y, e_y, -e_y] -- a transposed pair would steer wrong."""
  command = thruster_command([2.0, 5.0, 0.0])
  np.testing.assert_allclose(command[4:], [7.0, -3.0, 5.0, -5.0])


def test_surge_alone_drives_only_the_diagonal_pair():
  command = thruster_command([4.0, 0.0, 0.0])
  np.testing.assert_allclose(command[4:], [4.0, 4.0, 0.0, 0.0])


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_saturation_is_symmetric(sign):
  command = thruster_command(sign * np.array([1e6, 1e6, 1e6]))
  assert np.all(np.abs(command) <= THRUST_LIMIT)
  assert np.any(np.abs(command) == THRUST_LIMIT)


def test_rejects_an_error_that_is_not_three_elements():
  with pytest.raises(ValueError):
    thruster_command([1.0, 2.0])


def test_rejects_waypoints_that_are_not_n_by_3():
  with pytest.raises(ValueError, match=r"\(n, 3\)"):
    WaypointFollower(np.zeros((4, 2)))
  with pytest.raises(ValueError, match=r"\(n, 3\)"):
    WaypointFollower(np.zeros(3))


def test_far_from_target_returns_the_thruster_command():
  follower = WaypointFollower(SQUARE, arrival_radius=0.5)
  position = np.array([-5.0, -5.0, -5.0])

  np.testing.assert_allclose(
    follower.command(position), thruster_command(SQUARE[0] - position)
  )
  assert follower.index == 0  # no advance while still travelling


def test_arrival_returns_none_and_advances():
  """The contract every driver branches on: None means 'try again next tick'."""
  follower = WaypointFollower(SQUARE, arrival_radius=1.0)

  assert follower.command([0.0, 0.0, 0.0]) is None
  assert follower.index == 1
  assert not follower.finished


def test_finished_distinguishes_arrival_from_completion():
  """Both return None, so `finished` is the only way to tell them apart."""
  follower = WaypointFollower(SQUARE[:1], arrival_radius=1.0)

  assert follower.command([0.0, 0.0, 0.0]) is None
  assert follower.finished

  assert follower.command([0.0, 0.0, 0.0]) is None
  assert follower.finished  # and stays there, not running off the end


def test_arrival_radius_is_strict():
  """Exactly at the radius is not yet arrival."""
  follower = WaypointFollower(SQUARE, arrival_radius=2.0)
  assert follower.command([2.0, 0.0, 0.0]) is not None
  assert follower.index == 0

  assert follower.command([1.999, 0.0, 0.0]) is None
  assert follower.index == 1


def test_walking_the_course_reaches_finished():
  follower = WaypointFollower(SQUARE, arrival_radius=0.5)

  for waypoint in SQUARE:
    assert follower.command(waypoint) is None

  assert follower.finished
  assert follower.index == len(SQUARE)


def test_target_is_the_current_waypoint():
  follower = WaypointFollower(SQUARE, arrival_radius=1.0)
  np.testing.assert_allclose(follower.target, SQUARE[0])

  follower.command(SQUARE[0])
  np.testing.assert_allclose(follower.target, SQUARE[1])


def test_target_raises_once_finished():
  """command() guards the access, but a direct caller is not protected."""
  follower = WaypointFollower(SQUARE[:1], arrival_radius=1.0)
  follower.command(SQUARE[0])

  with pytest.raises(IndexError):
    _ = follower.target


def test_waypoints_are_copied_from_a_list():
  follower = WaypointFollower([[1.0, 2.0, 3.0]])
  assert follower.waypoints.shape == (1, 3)
  assert follower.waypoints.dtype == float
