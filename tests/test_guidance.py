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


def steer(follower, *args):
  """``command`` for a follower still on its route, where None would be a bug."""
  command = follower.command(*args)
  assert command is not None, "follower unexpectedly finished"
  return command


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
    steer(follower, position), thruster_command(SQUARE[0] - position)
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


# Documented BlueROV2 geometry from holoocean.agents.HoveringAUV: unit thrust
# directions and their positions in the body frame.
_S = 1.0 / np.sqrt(2.0)
THRUSTER_DIR = np.array(
  [
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [_S, _S, 0],
    [_S, -_S, 0],
    [_S, _S, 0],
    [_S, -_S, 0],
  ],
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
  ],
  dtype=float,
)


def _moment(command):
  return np.cross(THRUSTER_POS, THRUSTER_DIR * command[:, None]).sum(axis=0)


@pytest.mark.parametrize(
  "error",
  [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [3.0, -2.0, 1.5],
    # Saturating, and asymmetric in y so that e_x + e_y and e_x - e_y clip by
    # different amounts -- the case that used to leave a yaw moment behind.
    [40.0, -15.0, -5.0],
    [1e4, -1e4, 1e3],
  ],
)
def test_command_is_torque_free_even_when_saturated(error):
  """Thrust may be capped, but never at the cost of spinning the vehicle."""
  command = thruster_command(error)
  assert np.all(np.abs(command) <= THRUST_LIMIT + 1e-9)
  np.testing.assert_allclose(_moment(command), np.zeros(3), atol=1e-9)


def test_saturation_preserves_the_commanded_direction():
  """Scaling, not clipping: a capped command still points where it should."""
  error = [40.0, -15.0, -5.0]
  capped = thruster_command(error)
  unlimited = thruster_command(np.asarray(error) / 1000.0)
  cos = (
    capped @ unlimited / (np.linalg.norm(capped) * np.linalg.norm(unlimited))
  )
  np.testing.assert_allclose(cos, 1.0, atol=1e-12)


# The attitude HoloOcean reports for this vehicle at zero yaw. It is a z-down
# body frame, not a rolled vehicle -- and it is emphatically not the identity,
# which is what made the first attempt at body-frame steering pass its tests
# and stall a survey.
LEVEL = np.diag([1.0, -1.0, -1.0])


def yawed(degrees):
  """The same frame turned by `degrees` about world z."""
  angle = np.radians(degrees)
  turn = np.array(
    [
      [np.cos(angle), -np.sin(angle), 0.0],
      [np.sin(angle), np.cos(angle), 0.0],
      [0.0, 0.0, 1.0],
    ]
  )
  return turn @ LEVEL


def test_the_real_level_attitude_steers_exactly_as_before():
  """The regression this guards: diag(1, -1, -1) must be a no-op.

  Applying the whole rotation inverts e_y and e_z, which inverts depth control.
  Measured, that drove a survey along the one axis it left alone and stalled it
  16 m short of its first waypoint.
  """
  follower = WaypointFollower(SQUARE.copy(), arrival_radius=0.5)
  reference = WaypointFollower(SQUARE.copy(), arrival_radius=0.5)

  position = np.array([2.0, -3.0, 1.0])
  np.testing.assert_allclose(
    steer(follower, position, LEVEL), steer(reference, position)
  )


def test_depth_control_is_never_inverted():
  """Whatever the heading, a target above the vehicle thrusts the same way."""
  for degrees in (0.0, 45.0, 90.0, 180.0, -90.0):
    follower = WaypointFollower([[0.0, 0.0, 10.0]], arrival_radius=0.5)
    command = steer(follower, np.zeros(3), yawed(degrees))
    np.testing.assert_allclose(command[:4], [10.0, 10.0, 10.0, 10.0])


def test_a_yawed_vehicle_steers_in_its_own_frame():
  """At yaw 90 the body x axis points along world +y."""
  follower = WaypointFollower([[10.0, 0.0, 0.0]], arrival_radius=0.5)
  command = steer(follower, np.zeros(3), yawed(90.0))

  # A world +x target is 10 m to starboard: pure body -y, no surge.
  np.testing.assert_allclose(command[4:], [-10.0, 10.0, -10.0, 10.0])


def test_a_reversed_heading_does_not_drive_away():
  """At 180 degrees an unrotated error would be pure positive feedback."""
  follower = WaypointFollower([[10.0, 0.0, 0.0]], arrival_radius=0.5)
  command = steer(follower, np.zeros(3), yawed(180.0))

  # Body-frame surge is negative: the target is behind the vehicle.
  assert command[4] < 0 and command[5] < 0


def test_heading_is_taken_from_the_body_x_axis():
  """Roll and pitch must not reach the mixing; only the heading may."""
  follower = WaypointFollower([[10.0, 0.0, 0.0]], arrival_radius=0.5)
  reference = WaypointFollower([[10.0, 0.0, 0.0]], arrival_radius=0.5)

  # LEVEL and plain identity differ by a 180 degree roll, same heading.
  np.testing.assert_allclose(
    steer(follower, np.zeros(3), LEVEL),
    steer(reference, np.zeros(3), np.eye(3)),
  )


# -- turning ----------------------------------------------------------------


def test_a_pure_yaw_command_drives_thrusters_six_and_seven_against_each_other():
  """The measured pattern: no vertical thrust, no surge or sway pair."""
  command = thruster_command(np.zeros(3), yaw=5.0)
  np.testing.assert_allclose(command, [0, 0, 0, 0, 0, 0, -5.0, 5.0])


def test_turning_saturates_with_the_rest_of_the_command():
  """One scale factor for everything, so a clipped turn leaves no stray moment."""
  command = thruster_command(np.array([30.0, 10.0, 0.0]), yaw=15.0)
  unscaled = np.array([0, 0, 0, 0, 40.0, 20.0, -5.0, 5.0])
  np.testing.assert_allclose(command, unscaled * (20.0 / 40.0))


def turning_part(follower, position, rotation):
  """What turning adds to the command, against the same follower not turning."""
  plain = WaypointFollower(follower.waypoints, follower.arrival_radius)
  plain.index = follower.index
  return steer(follower, position, rotation) - steer(plain, position, rotation)


def test_a_target_to_port_turns_the_vehicle_anticlockwise():
  """Heading 0, waypoint due north: turn left, which is positive yaw."""
  follower = WaypointFollower([[0.0, 3.0, 0.0]], turn=True)
  added = turning_part(follower, np.zeros(3), LEVEL)
  assert added[6] < 0 < added[7]
  np.testing.assert_allclose(added[:6], 0.0)


def test_a_target_to_starboard_turns_it_clockwise():
  follower = WaypointFollower([[0.0, -3.0, 0.0]], turn=True)
  added = turning_part(follower, np.zeros(3), LEVEL)
  assert added[7] < 0 < added[6]


def test_the_heading_error_takes_the_short_way_round():
  """Heading 170 degrees, target -170: a 20 degree turn left, not 340 right."""
  follower = WaypointFollower([[-3.0, -0.53, 0.0]], turn=True)
  added = turning_part(follower, np.zeros(3), yawed(170.0))
  assert added[7] > 0


def test_turning_is_damped_by_the_heading_rate():
  """A heading already swinging toward the target is turned less hard."""
  still = WaypointFollower([[0.0, 3.0, 0.0]], turn=True)
  steer(still, np.zeros(3), yawed(10.0))
  swinging = WaypointFollower([[0.0, 3.0, 0.0]], turn=True)
  steer(swinging, np.zeros(3), yawed(0.0))

  calm = steer(still, np.zeros(3), yawed(10.0))[7]
  damped = steer(swinging, np.zeros(3), yawed(10.0))[7]
  assert damped < calm


def test_the_heading_target_is_held_close_to_a_waypoint():
  """Arrival must not spin the vehicle toward a point it is sitting on."""
  follower = WaypointFollower([[10.0, 0.0, 0.0]], turn=True, hold_within=1.5)
  steer(follower, np.zeros(3), LEVEL)
  steer(follower, np.array([9.5, 0.9, 0.0]), LEVEL)
  assert follower._heading_target == pytest.approx(0.0)


def test_without_turning_nothing_changes():
  """Every run flown before turning existed is reproduced exactly."""
  follower = WaypointFollower([[4.0, -2.0, 1.0]])
  np.testing.assert_allclose(
    steer(follower, np.zeros(3), yawed(30.0))[4:],
    thruster_command(
      np.array(
        [
          4.0 * np.cos(np.radians(30)) - 2.0 * np.sin(np.radians(30)),
          -4.0 * np.sin(np.radians(30)) - 2.0 * np.cos(np.radians(30)),
          1.0,
        ]
      )
    )[4:],
  )
