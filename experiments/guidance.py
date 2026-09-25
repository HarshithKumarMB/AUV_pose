"""Waypoint following for the BlueROV2.

Vehicle-specific control, shared by the drivers. Not in ``auv_pose`` because the
thruster mixing is a fact about this hull and HoloOcean's control scheme 0, not an
algorithm.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

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
  simulator rather than taken from the vendored geometry, whose comment would have this vehicle's forward mix push
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

  The bow is turned along the direction of travel, as a survey AUV does, so a
  body-fixed sonar fan stays across-track on every leg.

  Args:
      waypoints: Sequence of ``(x, y, z)`` targets.
      arrival_radius: Distance at which a waypoint counts as reached, metres.
      yaw_gain: Turning command per radian of heading error.
      yaw_damping: Turning command per radian per second of heading rate;
          the vehicle yaws readily and overshoots without it.
      dt: Interval between calls, seconds, for the heading rate.
      hold_within: Keep the last heading target inside this distance of a
          waypoint, metres, so arrival does not spin the vehicle toward it.
  """

  def __init__(
    self,
    waypoints: ArrayLike,
    arrival_radius: float = 0.5,
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
    self, position: ArrayLike, rotation: ArrayLike
  ) -> NDArray[np.float64] | None:
    """Thruster command steering from ``position`` toward the current waypoint.

    Args:
        position: Current world position, ``(3,)``.
        rotation: Body-to-world rotation, ``(3, 3)``. Only its heading is used:
            HoloOcean's level attitude is ``diag(1, -1, -1)``, and applying the
            whole matrix would invert depth control.

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

    rotation = np.asarray(rotation, dtype=float)
    heading = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    yaw = self._turning(heading, error)
    cos, sin = np.cos(heading), np.sin(heading)
    body = np.array(
      [
        cos * error[0] + sin * error[1],
        -sin * error[0] + cos * error[1],
        error[2],
      ]
    )
    return thruster_command(body, yaw)

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
