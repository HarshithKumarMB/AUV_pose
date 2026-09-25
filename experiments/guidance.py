"""Waypoint following for the BlueROV2 under HoloOcean's control scheme 0."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

THRUST_LIMIT = 20.0


def thruster_command(error: ArrayLike, yaw: float = 0.0) -> NDArray[np.float64]:
  """Proportional eight-thruster command driving ``error`` to zero, and turning.

  Saturate by scaling the whole vector, never by clipping element-wise:
  clipping unbalances the angled thrusters and leaves a yaw moment. Yaw drives
  thrusters 6 and 7 against each other.

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
  """Steers through ``(x, y, z)`` waypoints, bow along the direction of travel.

  Args:
      arrival_radius: Distance at which a waypoint counts as reached, metres.
      yaw_gain: Turning command per radian of heading error.
      yaw_damping: Turning command per rad/s of heading rate.
      dt: Interval between calls, seconds.
      hold_within: Freeze the heading target inside this distance of a
          waypoint, metres, so arrival does not spin the vehicle.
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
    """Thruster command toward the current waypoint.

    Args:
        rotation: Body-to-world rotation; only its heading is used, since
            applying HoloOcean's level ``diag(1, -1, -1)`` would invert depth.

    Returns:
        None on arrival at a waypoint or when :attr:`finished`.
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
