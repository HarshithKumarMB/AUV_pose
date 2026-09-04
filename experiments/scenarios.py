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
  "PROFILER_NADIR_AXIS",
  "PROFILER_SWATH_AXIS",
  "blue_rov_agent",
  "depth_sensor",
  "dvl_sensor",
  "imaging_sonar",
  "imu_sensor",
  "ocean_scenario",
  "orientation_sensor",
  "pose_sensor",
  "profiling_sonar",
  "sidescan_sonar",
  "singlebeam_sonar",
  "viewport_capture",
]


def imu_sensor(
  name: str = "imu",
  hz: int = 30,
  accel_sigma: float = 0.05,
  ang_vel_sigma: float = 0.01,
  accel_bias_sigma: float = 6e-5,
  ang_vel_bias_sigma: float = 5e-5,
) -> dict[str, Any]:
  """An IMU with noise that actually takes effect. Returns ``[accel; ang_vel]``.

  .. warning::

     The two ``*_bias_sigma`` values are **per-sample random-walk increments**,
     not standard deviations of a fixed bias. The bias grows as
     ``sigma * sqrt(n)``, so a value that reads like a plausible IMU spec is
     off by the square root of the run length. Measured with ``ReturnBias``
     against a motionless vehicle, ``0.01`` reaches 14 deg/s of gyro bias and
     0.35 m/s^2 of accelerometer bias after 300 samples at 30 Hz -- roughly
     200x a real MEMS unit, and enough on its own to swing heading through 70
     degrees in ten seconds.

     Size them backwards from the bias you want at the end of a run:
     ``sigma = bias_at_n / sqrt(n)``. The defaults here target about 0.05 deg/s
     of gyro bias and 0.001 m/s^2 of accelerometer bias over a 300-sample run.
  """
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


#: Where the multibeam's fan points, in the **body** frame, given the
#: ``rotation`` in :func:`profiling_sonar`'s block below.
#:
#: These belong next to the sensor block because they are two halves of one
#: fact: the block aims the sensor, and these say where it ended up.
#: :func:`~auv_pose.mapping.sonar.seabed_points` does not apply the mount
#: rotation itself, so if the block's ``rotation`` changes and these do not, the
#: reconstruction silently keeps using the old aim -- and a wrong swath sign
#: mirrors the whole map across the track while still looking plausible.
#:
#: Measured against the octree: enumeration picked this pair, and a
#: five-parameter fit moved it by 0.005 degrees.
PROFILER_NADIR_AXIS = (0.0, 0.0, 1.0)
PROFILER_SWATH_AXIS = (0.0, -1.0, 0.0)


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

  Replaces :func:`singlebeam_sonar` for surveying, because a fan of narrow beams
  makes a defect in any one of them visible against its neighbours, which one
  beam cannot do.

  **The singlebeam's soundings are bimodal, not smeared.** Against the octree
  they fall in two *tight* populations 4.87 m apart, the far one agreeing to a
  0.059 m MAD-std -- below the sensor's own quantisation. The "a constant beats
  every bin-selection rule" figure that used to justify this switch was measured
  over a four-metre strip where the seabed barely varies, so a constant won by
  construction; it does not generalise.

  Where the two disagree, do not assume the sonar is at fault. Measured with
  ``experiments/check_beam_validity.py``, the multibeam agrees with a ray-cast
  through the octree to a 0.035 m MAD-std except over compact patches where it
  reports 4-5 m shorter -- and those patches hold **fixed world positions and a
  fixed ~10 m size across altitudes of 17, 40 and 69 m**, which is an object on
  the seabed, not a property of the fan. The octree contains no geometry there.

  This fan is ~0.31 m across-track by ~1.2 m along-track at 70 m altitude, small
  enough that treating a beam as a point sounding is a fair approximation.

  Note:
      Noise is deliberately absent. :func:`~auv_pose.mapping.sonar.bottom_return_ranges`
      recognises a beam that saw nothing by its profile being flat, and that test
      never fires once additive noise is on -- every off-swath beam would then
      report a confident sounding at whatever bin the noise peaked in.

  :param name: Sensor name in the state dict.
  :param hz: Update rate. Sonar raycasting dominates the tick, and a 240-beam
      fan is far more work per ping than one beam, so this is well below the
      simulation rate.
  :param range_min: Minimum range, metres.
  :param range_max: Maximum range, metres. Must exceed the slant range at the
      edge of the swath -- at 70 m altitude and a 60 degree fan that is 80.8 m --
      or the outer beams return nothing.
  :param range_bins: Range bins. 1000 over 99.5 m is 0.0995 m per bin, against
      the singlebeam's 0.39 m.
  :param azimuth: Total swath width, degrees. holoocean defaults to 120, which
      at survey altitude is a 242 m swath against a 100 m range -- most of that
      fan never reaches the seabed.
  :param azimuth_bins: Beams across the swath.
  :param elevation: Along-track beam width, degrees.
  :param use_approx: Let the simulator bin returns into azimuth columns with a
      fast approximation of ``atan2``. **Off here, against holoocean's default
      of on**, on the reasoning that an approximation whose error varies with
      angle misplaces returns by an angle-dependent amount.

      **It changes nothing measurable.** Same track, same 60 pings, the flag the
      only difference: 1.879 vs 1.896 m MAD-std against the octree, with the
      disagreement pattern identical. An earlier capture suggested otherwise and
      was a confound -- two flights over different ground. The flag stays off
      because exactness is cheap at 5 Hz, not because it buys anything measured.
  :param init_octree_range: How far around the vehicle to build octree at
      startup, metres. **Leaving this unset is expensive and the cost is
      invisible.** The octree is generated once per (world, octree_min,
      octree_max) and cached to disk; with no range given the Dam cache reached
      **107 GB** at ``octree_min = 0.02``, and a second one at a different
      ``octree_max`` added 80 GB more. holoocean's own scenarios set 70. The
      default here covers a survey box with margin. Pass None to leave it to the
      simulator, which is what produced those numbers.

      Note this bounds what is built *at startup*, not the sonar's reach, and
      the cache is shared between runs -- so a later run over new ground extends
      it rather than replacing it.
  :return: A sensor configuration block.
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


def viewport_capture(
  name: str = "ViewportCapture", width: int = 1280, height: int = 720
) -> dict[str, Any]:
  """The viewport's own frame, for looking at the world rather than measuring it.

  Faster than an ``RGBCamera`` and, more usefully here, it renders whatever
  :meth:`~holoocean.environments.HoloOceanEnvironment.move_viewport` is pointed
  at -- so the camera is not tied to the vehicle and a patch of seabed can be
  viewed from any side.

  :param name: Sensor name in the state dict. The default is what holoocean
      calls it, and the key the frame arrives under.
  :param width: Capture width in pixels. **Must equal the viewport width** or
      the returned buffer does not match the frame.
  :param height: Capture height in pixels; same constraint.
  :return: A sensor configuration block.
  """
  return {
    "sensor_name": name,
    "sensor_type": "ViewportCapture",
    "socket": "IMUSocket",
    "configuration": {"CaptureWidth": width, "CaptureHeight": height},
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
  rotation: list[float] | None = None,
) -> dict[str, Any]:
  """A BlueROV2 under control scheme 0 (direct thruster commands).

  :param rotation: Starting ``[roll, pitch, yaw]`` in degrees. Nothing in the
      guidance commands yaw, so whatever is set here is held for the whole run
      -- which makes it the way to ask whether a sensor defect is fixed in the
      sensor's own frame or in the world's.
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
  """Ground-truth pose as ``[[R, p], [0, 1]]``. For initialisation and error only.

  In the IMU socket so its rotation block shares a frame with the inertial
  sensors, and so its translation is taken at the same point on the hull.
  """
  block: dict[str, Any] = {"sensor_name": name, "sensor_type": "PoseSensor"}
  if socket:
    block["socket"] = socket
  return block


def orientation_sensor(
  name: str = "orient", socket: str | None = "IMUSocket"
) -> dict[str, Any]:
  """Ground-truth orientation, body to world, as a 3x3 matrix.

  **The socket is load-bearing.** holoocean reports this sensor in the frame of
  the socket it sits in -- NED in ``IMUSocket``, NWU in the default COM socket
  (``sensors.py:166``). The IMU, DVL and depth sensor all sit in ``IMUSocket``,
  so the orientation used to rotate their readings into the world must come
  from there too. Omitting the socket returns identity at rest instead of
  ``diag(1, -1, -1)``, which silently mirrors every rotated reading in y and z.
  """
  block: dict[str, Any] = {
    "sensor_name": name,
    "sensor_type": "OrientationSensor",
  }
  if socket:
    block["socket"] = socket
  return block


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
  rotation: list[float] | None = None,
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
  :param rotation: Starting attitude in degrees; see :func:`blue_rov_agent`.
  :return: A scenario dict for :func:`holoocean.make`.
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
