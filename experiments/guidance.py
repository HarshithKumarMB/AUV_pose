"""Waypoint following for the BlueROV2.

Vehicle-specific control, shared by the drivers. Not in ``auv_pose`` because the
thruster mixing is a fact about this hull and HoloOcean's control scheme 0, not an
algorithm.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["WaypointFollower", "thruster_command"]

THRUST_LIMIT = 20.0


def thruster_command(error: ArrayLike) -> NDArray[np.float64]:
  """Proportional thruster command driving ``error`` to zero.

  Control scheme 0 takes eight thruster values: four vertical, then four in the
  horizontal plane mixed for the BlueROV2's 45-degree vectored layout.
  """
  error = np.asarray(error, dtype=float)
  e_x, e_y, e_z = error
  return np.clip(
    np.array([e_z, e_z, e_z, e_z, e_x + e_y, e_x - e_y, e_y, -e_y]),
    -THRUST_LIMIT,
    THRUST_LIMIT,
  )


class WaypointFollower:
  """Steers through a waypoint list, advancing on arrival.

  Args:
      waypoints: Sequence of ``(x, y, z)`` targets.
      arrival_radius: Distance at which a waypoint counts as reached, metres.
  """

  def __init__(self, waypoints: ArrayLike, arrival_radius: float = 0.5) -> None:
    self.waypoints = np.asarray(waypoints, dtype=float)
    if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 3:
      raise ValueError(f"expected (n, 3) waypoints, got {self.waypoints.shape}")
    self.arrival_radius = arrival_radius
    self.index = 0

  @property
  def finished(self) -> bool:
    return self.index >= len(self.waypoints)

  @property
  def target(self) -> NDArray[np.float64]:
    return self.waypoints[self.index]

  def command(self, position: ArrayLike) -> NDArray[np.float64] | None:
    """Thruster command steering from ``position`` toward the current waypoint.

    Returns None when the waypoint has just been reached -- advance and try
    again on the next tick -- or when the course is complete. Check
    :attr:`finished` to tell the two apart.
    """
    if self.finished:
      return None

    error = self.target - np.asarray(position, dtype=float)
    if np.linalg.norm(error) < self.arrival_radius:
      self.index += 1
      return None

    return thruster_command(error)
