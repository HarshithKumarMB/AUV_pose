"""Shared HoloOcean scenario fragments.

Sensor configuration, not algorithms -- hence here rather than in ``auv_pose``.

Note:
    The original scripts configured the IMU with ``"AddNoise": True`` and
    ``"GyroBiasSigma"``. Neither is a recognised ``IMUSensor`` option: ``AddNoise``
    appears nowhere in ``holoocean/sensors.py``, and the angular-rate bias key is
    ``AngVelBiasSigma``. Only ``AccelBiasSigma`` was taking effect, so those runs had
    no gyro noise and no accelerometer measurement noise at all. The documented keys
    are used below, which makes dead reckoning drift considerably more than before.
"""

from __future__ import annotations

from typing import Any

__all__ = [
  "DVL_AXES",
  "blue_rov_agent",
  "depth_sensor",
  "dvl_sensor",
  "imaging_sonar",
  "imu_sensor",
  "ocean_scenario",
  "orientation_sensor",
  "pose_sensor",
  "sidescan_sonar",
  "singlebeam_sonar",
]


def imu_sensor(
  name: str = "imu",
  hz: int = 30,
  accel_sigma: float = 0.05,
  ang_vel_sigma: float = 0.01,
  accel_bias_sigma: float = 0.01,
  ang_vel_bias_sigma: float = 0.01,
) -> dict[str, Any]:
  """An IMU with noise that actually takes effect. Returns ``[accel; ang_vel]``."""
  return {
    "sensor_name": name,
    "sensor_type": "IMUSensor",
    "socket": "IMUSocket",
    "Hz": hz,
    "configuration": {
      "AccelSigma": accel_sigma,
      "AngVelSigma": ang_vel_sigma,
      "AccelBiasSigma": accel_bias_sigma,
      "AngVelBiasSigma": ang_vel_bias_sigma,
      "ReturnBias": False,
    },
  }


def singlebeam_sonar(
  name: str = "singlebeam",
  hz: int = 30,
  range_min: float = 0.5,
  range_max: float = 100.0,
  range_bins: int = 256,
  opening_angle: float = 10.0,
) -> dict[str, Any]:
  """Downward-facing echosounder, for seabed ranging."""
  return {
    "sensor_name": name,
    "sensor_type": "SinglebeamSonar",
    "rotation": [0, -90, 0],
    "socket": "IMUSocket",
    "Hz": hz,
    "configuration": {
      "OpeningAngle": opening_angle,
      "RangeMin": range_min,
      "RangeMax": range_max,
      "RangeBins": range_bins,
    },
  }


def sidescan_sonar(
  name: str = "sidescan",
  hz: int = 10,
  range_min: float = 0.5,
  range_max: float = 70.0,
  range_bins: int = 256,
  azimuth_bins: int = 256,
) -> dict[str, Any]:
  """Side-looking sonar, for imaging the seabed either side of the track."""
  return {
    "sensor_name": name,
    "sensor_type": "SidescanSonar",
    "socket": "IMUSocket",
    "rotation": [0, -90, 0],
    "Hz": hz,
    "configuration": {
      "RangeMin": range_min,
      "RangeMax": range_max,
      "RangeBins": range_bins,
      "AzimuthBins": azimuth_bins,
      "AddNoise": True,
    },
  }


def blue_rov_agent(
  location: list[float],
  sensors: list[dict[str, Any]],
  name: str = "rov",
) -> dict[str, Any]:
  """A BlueROV2 under control scheme 0 (direct thruster commands)."""
  return {
    "agent_name": name,
    "agent_type": "BlueROV2",
    "location": location,
    "control_scheme": 0,
    "sensors": sensors,
  }


def pose_sensor(name: str = "pose") -> dict[str, Any]:
  """Ground-truth pose as ``[[R, p], [0, 1]]``. For initialisation and error only."""
  return {"sensor_name": name, "sensor_type": "PoseSensor"}


def orientation_sensor(name: str = "orient") -> dict[str, Any]:
  """Ground-truth orientation as a 3x3 matrix, NED in the IMU socket."""
  return {"sensor_name": name, "sensor_type": "OrientationSensor"}


def depth_sensor(name: str = "depthsensor", hz: int = 30) -> dict[str, Any]:
  """Pressure depth."""
  return {
    "sensor_name": name,
    "sensor_type": "DepthSensor",
    "socket": "IMUSocket",
    "Hz": hz,
  }


def imaging_sonar(
  name: str = "sonar",
  hz: int = 10,
  range_min: float = 0.5,
  range_max: float = 50.0,
  range_bins: int = 256,
  azimuth_bins: int = 256,
  azimuth: float = 90.0,
) -> dict[str, Any]:
  """Forward-looking imaging sonar."""
  return {
    "sensor_name": name,
    "sensor_type": "ImagingSonar",
    "socket": "IMUSocket",
    "rotation": [0, -90, 0],
    "Hz": hz,
    "configuration": {
      "RangeMin": range_min,
      "RangeMax": range_max,
      "RangeBins": range_bins,
      "AzimuthBins": azimuth_bins,
      "Azimuth": azimuth,
      "AddNoise": True,
    },
  }


def ocean_scenario(
  name: str,
  start: list[float],
  sensors: list[dict[str, Any]],
  world: str = "Dam",
  octree_min: float = 0.02,
  octree_max: float = 5.0,
) -> dict[str, Any]:
  """A single-BlueROV2 scenario in a world from the Ocean package.

  :param name: Scenario name.
  :param start: Initial world position.
  :param sensors: Sensor configuration blocks.
  :param world: World from the Ocean package.
  :param octree_min: Finest octree voxel, metres. Sonar raycasting needs an
      octree, which the simulator builds on first use and caches. The default
      matches holoocean's own (``environments.py:150``) and is **expensive** --
      the Dam world spans 664 x 664 x 400 m, and at 0.02 m this generates tens
      of gigabytes at several GB per minute. Raise it to make runs practical,
      but note that sonar ``AzimuthBins`` and ``ShadowEpsilon`` are derived from
      it, so changing it changes sonar returns and makes new surveys
      inconsistent with maps built at a different value.
  :param octree_max: Coarsest octree voxel, metres.
  :return: A scenario dict for :func:`holoocean.make`.
  """
  return {
    "name": name,
    "world": world,
    "package_name": "Ocean",
    "octree_min": octree_min,
    "octree_max": octree_max,
    "agents": [blue_rov_agent(location=start, sensors=sensors)],
  }


# The DVL reports velocity with y and z inverted relative to the NED body frame
# that the OrientationSensor uses in the IMU socket. Measured against ground
# truth over 199 samples spanning all three body axes: per-axis correlation is
# x +1.0000, y -1.0000, z -1.0000, and applying this gives exact agreement
# (rms error 0.0000 m/s). Multiply a raw reading by this to get NED body.
DVL_AXES = (1.0, -1.0, -1.0)


def dvl_sensor(
  name: str = "dvl",
  hz: int = 30,
  vel_sigma: float = 0.02,
  elevation: float = 22.5,
  return_range: bool = False,
) -> dict[str, Any]:
  """Doppler velocity log: velocity over ground, in the body frame.

  The sensor that makes velocity observable. Without one, position comes from
  doubly integrating acceleration and drifts quadratically; with one, velocity
  error is bounded and position drifts only linearly.

  :param name: Sensor name in the state dict.
  :param hz: Update rate.
  :param vel_sigma: Std applied to each of the four beam velocities, m/s.
  :param elevation: Beam angle off the downward z axis, degrees.
  :param return_range: Also return the four beam ranges, making the reading a
      7-vector instead of a 3-vector. Off here -- only velocity is used.
  :return: A sensor configuration block.
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
