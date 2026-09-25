"""HoloOcean sensor and scenario configuration."""

from typing import Any

import numpy as np

from auv_pose.estimation.inertial import ImuNoise


def imu_sensor(
  noise: ImuNoise, hz: int, name: str = "imu", return_bias: bool = False
) -> dict[str, Any]:
  """An IMU with ``noise``, returning ``[accel; ang_vel]``.

  With ``return_bias`` it also returns the true ``[accel_bias; ang_vel_bias]``.
  HoloOcean takes per-sample sigmas, so the densities are converted at ``hz``.
  """
  dt = 1.0 / hz
  return {
    "sensor_name": name,
    "sensor_type": "IMUSensor",
    "socket": "IMUSocket",
    "Hz": hz,
    "configuration": {
      "AccelSigma": noise.accel / np.sqrt(dt),
      "AngVelSigma": noise.gyro / np.sqrt(dt),
      "AccelBiasSigma": noise.accel_bias * np.sqrt(dt),
      "AngVelBiasSigma": noise.gyro_bias * np.sqrt(dt),
      "ReturnBias": return_bias,
    },
  }


#: The multibeam fan's axes in the body frame, given :func:`profiling_sonar`'s
#: mount ``rotation``. Update them together: a wrong swath sign mirrors the map
#: across the track yet still looks plausible.
PROFILER_NADIR_AXIS = (0.0, 0.0, 1.0)
PROFILER_SWATH_AXIS = (0.0, 1.0, 0.0)


def profiling_sonar(
  name: str = "multibeam",
  hz: int = 5,
  range_min: float = 0.5,
  range_max: float = 100.0,
  range_bins: int = 1000,
  azimuth: float = 60.0,
  azimuth_bins: int = 240,
  elevation: float = 1.0,
  use_approx: bool = False,
  init_octree_range: float | None = 100.0,
) -> dict[str, Any]:
  """Downward-facing multibeam, for seabed mapping.

  Noise is deliberately absent:
  :func:`~auv_pose.mapping.sonar.bottom_return_ranges` detects a beam with no
  return by its flat profile, which noise would break.

  :param range_max: Maximum range, metres; must exceed the slant range at the
      swath edge or the outer beams return nothing.
  :param azimuth: Total swath width, degrees.
  :param elevation: Along-track beam width, degrees.
  :param use_approx: Let the simulator approximate ``atan2`` when binning.
  :param init_octree_range: Radius of octree built at startup, metres. ``None``
      lets the simulator build the whole world, which costs tens of GB of cache.
  """
  configuration: dict[str, Any] = {
    "RangeMin": range_min,
    "RangeMax": range_max,
    "RangeBins": range_bins,
    "Azimuth": azimuth,
    "AzimuthBins": azimuth_bins,
    "Elevation": elevation,
    "UseApprox": use_approx,
  }
  if init_octree_range is not None:
    configuration["InitOctreeRange"] = init_octree_range

  return {
    "sensor_name": name,
    "sensor_type": "ProfilingSonar",
    "socket": "IMUSocket",
    "rotation": [0, -90, 0],
    "Hz": hz,
    "configuration": configuration,
  }


def blue_rov_agent(
  location: list[float],
  sensors: list[dict[str, Any]],
  name: str = "rov",
  rotation: list[float] | None = None,
) -> dict[str, Any]:
  """A BlueROV2 under control scheme 0 (direct thruster commands).

  :param rotation: Starting ``[roll, pitch, yaw]``, degrees; held for the whole
      run, since guidance never commands yaw.
  """
  agent: dict[str, Any] = {
    "agent_name": name,
    "agent_type": "BlueROV2",
    "location": location,
    "control_scheme": 0,
    "sensors": sensors,
  }
  if rotation is not None:
    agent["rotation"] = rotation
  return agent


def pose_sensor(
  name: str = "pose", socket: str | None = "IMUSocket"
) -> dict[str, Any]:
  """Ground-truth pose ``[[R, p], [0, 1]]``, for initialisation and scoring."""
  block: dict[str, Any] = {"sensor_name": name, "sensor_type": "PoseSensor"}
  if socket:
    block["socket"] = socket
  return block


def orientation_sensor(
  name: str = "orient", socket: str | None = "IMUSocket"
) -> dict[str, Any]:
  """Ground-truth orientation, body to world, as a 3x3 matrix.

  Keep it in ``IMUSocket``: holoocean reports it in its socket's frame, and
  the default socket silently mirrors every rotated reading in y and z.
  """
  block: dict[str, Any] = {
    "sensor_name": name,
    "sensor_type": "OrientationSensor",
  }
  if socket:
    block["socket"] = socket
  return block


def depth_sensor(
  name: str = "depthsensor", hz: int = 30, sigma: float = 0.0
) -> dict[str, Any]:
  """Pressure depth: world ``z``, increasing upward.

  :param sigma: Noise standard deviation, metres.
  """
  block: dict[str, Any] = {
    "sensor_name": name,
    "sensor_type": "DepthSensor",
    "socket": "IMUSocket",
    "Hz": hz,
  }
  if sigma > 0.0:
    block["configuration"] = {"Sigma": sigma}
  return block


def magnetometer_sensor(
  name: str = "magnetometer", hz: int = 5, sigma: float = 0.03
) -> dict[str, Any]:
  """Magnetometer: ``R^T [1, 0, 0]`` plus noise, in the body frame.

  :param sigma: Per-axis noise on the unit reading; roughly heading error, rad.
  """
  return {
    "sensor_name": name,
    "sensor_type": "MagnetometerSensor",
    "socket": "IMUSocket",
    "Hz": hz,
    "configuration": {"Sigma": sigma},
  }


def ocean_scenario(
  name: str,
  start: list[float],
  sensors: list[dict[str, Any]],
  world: str = "Dam",
  octree_min: float = 0.02,
  octree_max: float = 5.0,
  rotation: list[float] | None = None,
) -> dict[str, Any]:
  """A single-BlueROV2 scenario in a world from the Ocean package.

  :param start: Initial world position.
  :param octree_min: Finest octree voxel, metres. Changing it changes sonar
      returns, so surveys at different values are inconsistent.
  :param rotation: Starting attitude in degrees; see :func:`blue_rov_agent`.
  """
  return {
    "name": name,
    "world": world,
    "package_name": "Ocean",
    "octree_min": octree_min,
    "octree_max": octree_max,
    "agents": [
      blue_rov_agent(location=start, sensors=sensors, rotation=rotation)
    ],
  }


def dvl_sensor(
  name: str = "dvl",
  hz: int = 30,
  vel_sigma: float = 0.02,
  elevation: float = 22.5,
  return_range: bool = False,
) -> dict[str, Any]:
  """Doppler velocity log: velocity over ground, in the body frame.

  :param vel_sigma: Std applied to each of the four beam velocities, m/s.
  :param elevation: Beam angle off the downward z axis, degrees.
  :param return_range: Also return the four beam ranges (a 7-vector).
  """
  return {
    "sensor_name": name,
    "sensor_type": "DVLSensor",
    "socket": "IMUSocket",
    "Hz": hz,
    "configuration": {
      "Elevation": elevation,
      "VelSigma": vel_sigma,
      "ReturnRange": return_range,
    },
  }
