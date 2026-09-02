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
    "blue_rov_agent",
    "depth_sensor",
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
) -> dict[str, Any]:
    """A single-BlueROV2 scenario in a world from the Ocean package."""
    return {
        "name": name,
        "world": world,
        "package_name": "Ocean",
        "agents": [blue_rov_agent(location=start, sensors=sensors)],
    }
