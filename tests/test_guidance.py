"""Waypoint following and thruster mixing."""

import numpy as np
import pytest

from experiments.guidance import (
  THRUST_LIMIT,
  WaypointFollower,
  thruster_command,
)

SQUARE = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]])

# HoloOcean's level attitude at zero yaw: a z-down body frame.
LEVEL = np.diag([1.0, -1.0, -1.0])


def yawed(degrees):
  angle = np.radians(degrees)
  c, s = np.cos(angle), np.sin(angle)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ LEVEL


def to_body(error, degrees):
  angle = np.radians(degrees)
  c, s = np.cos(angle), np.sin(angle)
  return np.array(
    [c * error[0] + s * error[1], -s * error[0] + c * error[1], error[2]]
  )


def steer(follower, position, rotation):
  command = follower.command(position, rotation)
  assert command is not None, "follower unexpectedly finished"
  return command


def turning_part(follower, position, degrees):
  """What turning adds to the command, beyond steering toward the target."""
  error = follower.target - position
  return steer(follower, position, yawed(degrees)) - thruster_command(
    to_body(error, degrees)
  )


# -- mixing -----------------------------------------------------------------


def test_the_mix_is_the_vectored_layout():
  """Vertical bank carries z; ``[x+y, x-y, y-yaw, -y+yaw]`` horizontally."""
  np.testing.assert_allclose(
    thruster_command([2.0, 5.0, 3.0], yaw=1.0),
    [3.0, 3.0, 3.0, 3.0, 7.0, -3.0, 4.0, -4.0],
  )


def test_rejects_an_error_that_is_not_three_elements():
  with pytest.raises(ValueError):
    thruster_command([1.0, 2.0])


# BlueROV2 thruster directions and positions, body frame, from holoocean.
_S = 1.0 / np.sqrt(2.0)
THRUSTER_DIR = np.array(
  [[0, 0, 1]] * 4 + [[_S, _S, 0], [_S, -_S, 0], [_S, _S, 0], [_S, -_S, 0]],
  dtype=float,
)
THRUSTER_POS = np.array(
  [
    [0.25, -0.22, -0.04],
    [0.25, 0.22, -0.04],
    [-0.25, 0.22, -0.04],
    [-0.25, -0.22, -0.04],
    [0.14, -0.18, 0.0],
    [0.14, 0.18, 0.0],
    [-0.14, 0.18, 0.0],
    [-0.14, -0.18, 0.0],
  ]
)


@pytest.mark.parametrize(
  "error",
  [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [40.0, -15.0, -5.0],
    [1e4, -1e4, 1e3],
  ],
)
def test_a_translation_command_is_torque_free_even_when_saturated(error):
  command = thruster_command(error)
  assert np.all(np.abs(command) <= THRUST_LIMIT + 1e-9)
  moment = np.cross(THRUSTER_POS, THRUSTER_DIR * command[:, None]).sum(axis=0)
  np.testing.assert_allclose(moment, 0.0, atol=1e-9)


@pytest.mark.parametrize("yaw", [0.0, 15.0, -40.0])
def test_saturation_scales_the_whole_command(yaw):
  """A capped command is the uncapped one scaled down."""
  error = np.array([40.0, -15.0, -5.0])
  capped = thruster_command(error, yaw)
  small = thruster_command(error / 1000.0, yaw / 1000.0)
  np.testing.assert_allclose(
    capped / np.abs(capped).max(), small / np.abs(small).max()
  )
  assert np.abs(capped).max() == pytest.approx(THRUST_LIMIT)


# -- following the course -----------------------------------------------------


def test_rejects_waypoints_that_are_not_n_by_3():
  for bad in (np.zeros((4, 2)), np.zeros(3)):
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
      WaypointFollower(bad)


def test_a_target_dead_ahead_is_approached_without_turning():
  follower = WaypointFollower(SQUARE, arrival_radius=0.5)
  position = np.array([-5.0, 0.0, -5.0])
  np.testing.assert_allclose(
    steer(follower, position, LEVEL), thruster_command(SQUARE[0] - position)
  )
  assert follower.index == 0


def test_arrival_is_strictly_inside_the_radius_and_advances():
  follower = WaypointFollower(SQUARE, arrival_radius=2.0)
  assert follower.command([2.0, 0.0, 0.0], LEVEL) is not None
  assert follower.index == 0
  assert follower.command([1.999, 0.0, 0.0], LEVEL) is None
  assert follower.index == 1
  np.testing.assert_allclose(follower.target, SQUARE[1])


def test_the_course_finishes_and_stays_finished():
  follower = WaypointFollower(SQUARE, arrival_radius=0.5)
  for waypoint in SQUARE:
    assert not follower.finished
    assert follower.command(waypoint, LEVEL) is None
  assert follower.finished
  assert follower.command(SQUARE[-1], LEVEL) is None
  with pytest.raises(IndexError):
    _ = follower.target


# -- frames -----------------------------------------------------------------


@pytest.mark.parametrize("degrees", [0.0, 45.0, 90.0, 180.0, -90.0])
def test_depth_control_is_never_inverted(degrees):
  follower = WaypointFollower([[0.0, 0.0, 10.0]], arrival_radius=0.5)
  command = steer(follower, np.zeros(3), yawed(degrees))
  np.testing.assert_allclose(command[:4], 10.0)


def test_a_yawed_vehicle_steers_in_its_own_frame():
  """At yaw 90 a world +x target is to starboard: body -y, no surge."""
  follower = WaypointFollower([[10.0, 0.0, 0.0]], arrival_radius=0.5)
  command = steer(follower, np.zeros(3), yawed(90.0))
  np.testing.assert_allclose(command[4:6], [-10.0, 10.0])


def test_roll_and_pitch_do_not_reach_the_mix():
  """LEVEL and the identity differ by a 180 degree roll, same heading."""
  a = WaypointFollower([[10.0, 3.0, 0.0]])
  b = WaypointFollower([[10.0, 3.0, 0.0]])
  np.testing.assert_allclose(
    steer(a, np.zeros(3), LEVEL), steer(b, np.zeros(3), np.eye(3))
  )


# -- turning ----------------------------------------------------------------


@pytest.mark.parametrize(
  ("target", "degrees"),
  [
    ([0.0, 3.0, 0.0], 0.0),
    ([0.0, -3.0, 1.0], 30.0),
    ([-3.0, -0.5, 0.0], 170.0),
  ],
)
def test_turning_only_drives_the_yaw_pair(target, degrees):
  added = turning_part(WaypointFollower([target]), np.zeros(3), degrees)
  np.testing.assert_allclose(added[:6], 0.0, atol=1e-12)
  assert added[6] == pytest.approx(-added[7])


def test_a_target_to_port_turns_anticlockwise_and_starboard_clockwise():
  port = turning_part(WaypointFollower([[0.0, 3.0, 0.0]]), np.zeros(3), 0.0)
  starboard = turning_part(
    WaypointFollower([[0.0, -3.0, 0.0]]), np.zeros(3), 0.0
  )
  assert port[7] > 0 > starboard[7]


def test_the_heading_error_takes_the_short_way_round():
  """Heading 170 degrees, target at -170: 20 degrees left, not 340 right."""
  added = turning_part(
    WaypointFollower([[-3.0, -0.53, 0.0]]), np.zeros(3), 170.0
  )
  assert added[7] > 0


def test_turning_is_damped_by_the_heading_rate():
  still = WaypointFollower([[0.0, 3.0, 0.0]])
  steer(still, np.zeros(3), yawed(10.0))
  swinging = WaypointFollower([[0.0, 3.0, 0.0]])
  steer(swinging, np.zeros(3), yawed(0.0))
  calm = steer(still, np.zeros(3), yawed(10.0))[7]
  damped = steer(swinging, np.zeros(3), yawed(10.0))[7]
  assert damped < calm


def test_the_heading_target_is_held_close_to_a_waypoint():
  follower = WaypointFollower([[10.0, 0.0, 0.0]], hold_within=1.5)
  steer(follower, np.zeros(3), LEVEL)
  added = turning_part(follower, np.array([9.5, 0.9, 0.0]), 0.0)
  np.testing.assert_allclose(added, 0.0, atol=1e-12)
