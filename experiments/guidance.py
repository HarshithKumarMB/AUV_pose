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
  """
  error = np.asarray(error, dtype=float)
  e_x, e_y, e_z = error
  command = np.array([e_z, e_z, e_z, e_z, e_x + e_y, e_x - e_y, e_y, -e_y])

  peak = float(np.abs(command).max())
  if peak > THRUST_LIMIT:
    command = command * (THRUST_LIMIT / peak)
  return command


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
