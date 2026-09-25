"""Waypoint following for the BlueROV2.

Vehicle-specific control, shared by the drivers. Not in ``auv_pose`` because the
thruster mixing is a fact about this hull and HoloOcean's control scheme 0, not an
algorithm.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["WaypointFollower", "thruster_command", "wrap"]

THRUST_LIMIT = 20.0


def thruster_command(error: ArrayLike, yaw: float = 0.0) -> NDArray[np.float64]:
  """Proportional thruster command driving ``error`` to zero, and turning.

  Control scheme 0 takes eight thruster values: four vertical, then four in the
  horizontal plane mixed for the BlueROV2's 45-degree vectored layout.

  The mix is torque-free: against the documented thruster geometry, pure surge,
  sway and heave each produce zero net moment. **Saturating it element-wise is
  not.** Clipping ``e_x + e_y`` and ``e_x - e_y`` by different amounts leaves
  the two angled front thrusters unbalanced, and the residual is a yaw moment
  that appears exactly when the position error is largest. Nothing in the loop
  observes yaw, so it integrates freely -- measured against the simulator, the
  vehicle began rotating within sixteen samples of the first horizontal clip and
  went on to tumble through 135 degrees.

  So scale rather than clip: divide the whole vector down by a single factor
  when it exceeds the limit, which caps the thrust while preserving both the
  commanded direction and the torque balance.

  **Yaw is thrusters 6 and 7 driven against each other.** Measured in the
  simulator (``experiments/check_thrusters.py``) rather than taken from the
  vendored geometry, whose comment would have this vehicle's forward mix push
  it backwards: fired alone from rest, each horizontal thruster turns the
  vehicle about 28 degrees in half a second, and the pattern that turns it
  with no net force solves to ``[0.045, -0.043, -1, 1]`` over thrusters 4-7.
  ``[0, 0, -1, 1]`` turned it 59 degrees anticlockwise in half a second with
  3 cm of stray translation. The same single scale factor caps it, so a turn
  never leaves an unbalanced moment behind either.

  :param error: Body-frame position error, ``(x, y, z)``.
  :param yaw: Turning command, positive anticlockwise seen from above.
  """
  error = np.asarray(error, dtype=float)
  e_x, e_y, e_z = error
  command = np.array(
    [e_z, e_z, e_z, e_z, e_x + e_y, e_x - e_y, e_y - yaw, -e_y + yaw]
  )

  peak = float(np.abs(command).max())
  if peak > THRUST_LIMIT:
    command = command * (THRUST_LIMIT / peak)
  return command


def wrap(angle: float) -> float:
  """An angle in ``(-pi, pi]``."""
  return float(np.angle(np.exp(1j * angle)))


class WaypointFollower:
  """Steers through a waypoint list, advancing on arrival.

  With ``turn`` the vehicle also points its bow along the direction of travel,
  as a survey AUV does, instead of crabbing at its starting heading. That turns
  a body-fixed sonar fan with the track: across-track on every leg, and
  sweeping round on a loop.

  Args:
      waypoints: Sequence of ``(x, y, z)`` targets.
      arrival_radius: Distance at which a waypoint counts as reached, metres.
      turn: Steer the heading toward the direction of travel.
      yaw_gain: Turning command per radian of heading error.
      yaw_damping: Turning command per radian per second of heading rate.
          The vehicle yaws readily -- one thruster turns it 28 degrees in half
          a second -- so without damping it overshoots.
      dt: Interval between calls, seconds, for the heading rate.
      hold_within: Keep the last heading target inside this distance of a
          waypoint, metres, so arrival does not spin the vehicle toward it.
  """

  def __init__(
    self,
    waypoints: ArrayLike,
    arrival_radius: float = 0.5,
    turn: bool = False,
    yaw_gain: float = 4.0,
    yaw_damping: float = 3.0,
    dt: float = 1.0 / 30.0,
    hold_within: float = 1.5,
  ) -> None:
    self.waypoints = np.asarray(waypoints, dtype=float)
    if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 3:
      raise ValueError(f"expected (n, 3) waypoints, got {self.waypoints.shape}")
    self.arrival_radius = arrival_radius
    self.index = 0
    self.turn = turn
    self.yaw_gain = yaw_gain
    self.yaw_damping = yaw_damping
    self.dt = dt
    self.hold_within = hold_within
    self._heading_target: float | None = None
    self._last_heading: float | None = None

  @property
  def finished(self) -> bool:
    return self.index >= len(self.waypoints)

  @property
  def target(self) -> NDArray[np.float64]:
    return self.waypoints[self.index]

  def command(
    self, position: ArrayLike, rotation: ArrayLike | None = None
  ) -> NDArray[np.float64] | None:
    """Thruster command steering from ``position`` toward the current waypoint.

    Args:
        position: Current world position, ``(3,)``.
        rotation: Body-to-world rotation, ``(3, 3)``. **Required for any run
            that is not aligned with the world axes.** The waypoint error is a
            world vector and :func:`thruster_command` mixes body thrusters, so
            without this the two frames are silently assumed identical --
            measured at yaw 90 degrees a commanded world ``+x`` produces world
            ``+y``, and at 180 degrees ``-x``, which is positive feedback.

            Only the **heading** is taken from it. HoloOcean reports this
            vehicle's attitude as ``diag(1, -1, -1)`` at zero yaw -- a
            z-down body frame, not a rolled vehicle -- so applying the whole
            matrix inverts ``e_y`` and ``e_z`` and inverts depth control with
            them. That is not hypothetical: it stalled a survey 16 m short of
            its first waypoint, having driven the one axis it left alone.

    Returns:
        None when the waypoint has just been reached -- advance and try again
        on the next tick -- or when the course is complete. Check
        :attr:`finished` to tell the two apart.
    """
    if self.finished:
      return None

    error = self.target - np.asarray(position, dtype=float)
    if np.linalg.norm(error) < self.arrival_radius:
      self.index += 1
      return None

    yaw = 0.0
    if rotation is not None:
      rotation = np.asarray(rotation, dtype=float)
      # Heading of the body x axis in the world. At zero yaw this is a no-op,
      # so every run flown before headings existed is reproduced exactly.
      heading = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
      if self.turn:
        yaw = self._turning(heading, error)
      cos, sin = np.cos(heading), np.sin(heading)
      error = np.array(
        [
          cos * error[0] + sin * error[1],
          -sin * error[0] + cos * error[1],
          error[2],
        ]
      )

    return thruster_command(error, yaw)

  def _turning(self, heading: float, error: NDArray[np.float64]) -> float:
    """PD turning command toward the direction of travel."""
    if self._heading_target is None or (
      np.hypot(error[0], error[1]) > self.hold_within
    ):
      self._heading_target = float(np.arctan2(error[1], error[0]))

    rate = (
      0.0
      if self._last_heading is None
      else wrap(heading - self._last_heading) / self.dt
    )
    self._last_heading = heading
    return self.yaw_gain * wrap(self._heading_target - heading) - (
      self.yaw_damping * rate
    )
