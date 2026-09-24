"""Survey the seabed on a lawnmower track, navigating on the vehicle's own filter.

    nix run .#sim -- -c "python -u experiments/survey.py --out pass0 --yaw 0"

Flies a boustrophedon pattern with a downward **multibeam**, steering on the
estimate of an unscented inertial filter aided by a DVL, a pressure sensor and
a magnetometer -- no absolute position fix, as on the hardware vehicle. It
writes a **raw log** (:mod:`auv_pose.io.raw_survey`), not soundings: where a
sounding lies depends on a pose that is only settled once the whole run can be
smoothed, so ``georeference.py`` places them afterwards. That is how a real
survey is processed, and it means one flight yields both the navigated map and
a ground-truth-placed control, from the same pings.

Ground truth is logged beside every reading for scoring and never read by the
filter or the controller. The one exception is the initial belief, which is
drawn *around* the true start pose with the stated covariance -- standing in for
the surface fix a real vehicle takes before it dives.

The multibeam's per-beam return is 0.40 m wide, and 100% of beams answer at
survey altitude. Sonar and aiding sensors all run at 5 Hz, so every filter
cycle closes on a ping tick and the smoothed pose lands exactly on it.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import holoocean
import numpy as np

from auv_pose.estimation.inertial import ImuNoise
from auv_pose.estimation.manifold import (
  DOF,
  ManifoldGaussian,
  NavState,
  boxplus,
)
from auv_pose.estimation.navigation import (
  MAGNETIC_NORTH,
  AidingNoise,
  InertialNavigator,
  dvl_noise_covariance,
)
from auv_pose.estimation.quaternion import quat_to_rotmat, rotmat_to_quat
from auv_pose.io.raw_survey import RawSurveyWriter
from auv_pose.mapping.sonar import (
  azimuth_angles,
  bottom_return_ranges,
  range_bins,
)
from experiments.captures import write_capture
from experiments.cli import configure_sdl, refuse_overwrite
from experiments.guidance import WaypointFollower
from experiments.scenarios import (
  PROFILER_NADIR_AXIS,
  PROFILER_SWATH_AXIS,
  depth_sensor,
  dvl_sensor,
  imu_sensor,
  magnetometer_sensor,
  ocean_scenario,
  orientation_sensor,
  pose_sensor,
  profiling_sonar,
)

TICK_RATE_HZ = 30

# 5 Hz, not the tick rate: raycasting a 240-beam fan is the expensive part of a
# tick, and at survey speed 5 Hz still oversamples the along-track footprint.
SONAR_HZ = 5
SONAR = {
  "range_min": 0.5,
  "range_max": 100.0,
  "range_bins": 1000,
  "azimuth": 60.0,
  "azimuth_bins": 240,
  "elevation": 1.0,
}

#: The aiding sensors run at the sonar's rate, as a survey DVL typically does,
#: so a filter cycle closes on every ping. Running depth at the 30 Hz tick rate
#: instead is what made the older filter 3-10 sigma overconfident in z: each
#: reading was treated as independent when the noise was not.
AIDING_HZ = SONAR_HZ
DVL_BEAM_SIGMA = 0.02
DVL_ELEVATION = 22.5
DEPTH_SIGMA = 0.05
COMPASS_SIGMA = 0.03

#: IMU noise for a survey. The white-noise terms are the scenarios' defaults;
#: the bias random walks are sized for a *pass*, not for the 300-sample runs
#: :func:`~experiments.scenarios.imu_sensor`'s defaults target. A diagonal pass
#: is about 25,000 ticks, over which those defaults would grow the gyro bias to
#: 0.45 deg/s. These give about 0.01 deg/s and 1e-3 m/s^2 by the end, which is
#: a tactical-grade MEMS unit.
SURVEY_IMU = ImuNoise(gyro=0.01, accel=0.05, gyro_bias=1e-6, accel_bias=6e-6)

#: Spread of the belief navigation starts from, around the true start pose:
#: position as from a surface GNSS fix, attitude as from a levelled AHRS, and
#: the biases at what a pass accumulates.
INITIAL_SIGMA = {
  "position": [1.0, 1.0, 0.1],
  "attitude_deg": [1.0, 1.0, 3.0],
  "velocity": [0.05, 0.05, 0.05],
  "gyro_bias": [2e-4] * 3,
  "accel_bias": [1e-3] * 3,
}


#: The box to cover, ``(x_min, x_max, y_min, y_max)`` in metres.
#:
#: Sized from what navigation will *look at*, not where it will go.
#: ``navigate.py`` flies x -30..-10, y -15..-5, and at 70 m altitude a 60 degree
#: fan reaches 40.2 m either side of the vehicle -- so every beam of every ping
#: it takes lands somewhere in this box. Surveying only the track would map the
#: nadir beam's footprint and leave the other 239 over unsurveyed ground.
SURVEY_BOX = (-70.0, 30.0, -55.0, 35.0)


def lawnmower(
  heading: float = 0.0,
  box: tuple[float, float, float, float] = SURVEY_BOX,
  spacing: float = 5.0,
  z: float = 0.0,
) -> list[list[float]]:
  """Boustrophedon track aligned to ``heading``, covering ``box``.

  **The vehicle travels along its own x, so the fan is across-track.** That is
  the whole point of the alignment: the fan opens along body -y and the vehicle
  never rotates, so a track running parallel to the fan sweeps the same strip of
  seabed 240 times and leaves the across-track sampling at the line spacing.
  Measured that way, a 240-beam fan bought no more coverage than one beam.

  Flying the same box at several headings is then how a patch gets seen from
  several directions, which fills the shadow a pipeline casts from any one of
  them. Each heading is a separate run -- the spawn attitude is held for the
  whole flight and nothing commands yaw.

  Args:
      heading: Track direction in degrees, counter-clockwise from world ``+x``.
          Must match the vehicle's spawn yaw or the fan is not across-track.
      box: Region to cover.
      spacing: Line spacing in metres. Generous compared to the old 2 m,
          because at 70 m altitude a 60 degree fan reaches 40 m either side and
          the redundancy is now real rather than the same strip re-measured.
      z: Depth to hold.

  Returns:
      Waypoints, ``(n, 3)`` as a list.
  """
  angle = np.radians(heading)
  forward = np.array([np.cos(angle), np.sin(angle)])
  across = np.array([-np.sin(angle), np.cos(angle)])

  x_min, x_max, y_min, y_max = box
  centre = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
  length, width = x_max - x_min, y_max - y_min

  # Extent of an axis-aligned box measured along a rotated axis: the projection
  # of both sides onto it. Covers the box at any heading without flying a square
  # big enough for the worst one.
  along = abs(length * forward[0]) + abs(width * forward[1])
  side = abs(length * across[0]) + abs(width * across[1])

  lines = np.arange(-side / 2, side / 2 + spacing / 2, spacing)
  ends = np.array([-along / 2, along / 2])

  waypoints: list[list[float]] = []
  for index, offset in enumerate(lines):
    legs = ends if index % 2 == 0 else ends[::-1]
    for reach in legs:
      point = centre + reach * forward + offset * across
      waypoints.append([float(point[0]), float(point[1]), z])

  return waypoints


def figure_eight(
  centre: tuple[float, float] = (-20.0, -10.0),
  radius: float = 15.0,
  depths: tuple[float, float] = (0.0, -30.0),
  loop_points: int = 36,
) -> list[list[float]]:
  """A figure-eight test track: two tangent circles, each at its own depth.

  Made to be flown *after* a survey, as a test set the map never saw: its
  soundings land between the survey's, on every heading, from a navigation
  run with its own drift. The two depths put the fan at two altitudes, so the
  map is tested at two sounding densities.

  Loop one is flown anticlockwise from the crossing at ``depths[0]``; the
  vehicle then changes depth at the crossing and flies loop two clockwise at
  ``depths[1]``, ending back where it started -- the crossing is where the
  track overlaps itself at both altitudes.

  Args:
      centre: Where the loops touch, world ``(x, y)``.
      radius: Radius of each loop, metres.
      depths: World ``z`` of loop one and loop two.
      loop_points: Waypoints per loop.

  Returns:
      Waypoints, ``(n, 3)`` as a list, starting at the crossing.
  """
  cx, cy = centre
  first, second = depths
  steps = 2.0 * np.pi * np.arange(1, loop_points + 1) / loop_points

  # Loop one's centre is west of the crossing, so the crossing is at angle 0.
  one = [
    [cx - radius + radius * np.cos(a), cy + radius * np.sin(a), first]
    for a in steps
  ]
  # Loop two's centre is east, so the crossing is at angle pi; clockwise.
  two = [
    [
      cx + radius + radius * np.cos(np.pi - a),
      cy + radius * np.sin(np.pi - a),
      second,
    ]
    for a in steps
  ]
  start = [[cx, cy, first]]
  descend = [[cx, cy, second]]
  return [[float(v) for v in point] for point in start + one + descend + two]


def initial_covariance() -> np.ndarray:
  """The initial belief's covariance, from :data:`INITIAL_SIGMA`."""
  sigma = np.concatenate(
    [
      INITIAL_SIGMA["position"],
      np.radians(INITIAL_SIGMA["attitude_deg"]),
      INITIAL_SIGMA["velocity"],
      INITIAL_SIGMA["gyro_bias"],
      INITIAL_SIGMA["accel_bias"],
    ]
  )
  return np.diag(sigma**2)


def build_scenario(
  start: list[float], octree_min: float, yaw: float = 0.0
) -> dict:
  return ocean_scenario(
    "bathymetry_survey",
    start=start,
    octree_min=octree_min,
    # Held for the whole run; nothing commands yaw. This is what turns the fan
    # across-track, so it must match the heading lawnmower() was built with.
    rotation=[0.0, 0.0, yaw],
    sensors=[
      # Truth, logged for scoring only. The socket is load-bearing -- see
      # orientation_sensor's docstring.
      pose_sensor(),
      orientation_sensor(),
      imu_sensor(
        "imu",
        hz=TICK_RATE_HZ,
        accel_sigma=SURVEY_IMU.accel,
        ang_vel_sigma=SURVEY_IMU.gyro,
        accel_bias_sigma=SURVEY_IMU.accel_bias,
        ang_vel_bias_sigma=SURVEY_IMU.gyro_bias,
        return_bias=True,
      ),
      dvl_sensor(
        hz=AIDING_HZ, vel_sigma=DVL_BEAM_SIGMA, elevation=DVL_ELEVATION
      ),
      depth_sensor(hz=AIDING_HZ, sigma=DEPTH_SIGMA),
      magnetometer_sensor(hz=AIDING_HZ, sigma=COMPASS_SIGMA),
      profiling_sonar("multibeam", hz=SONAR_HZ, **SONAR),
    ],
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Bathymetry survey")
  parser.add_argument(
    "--out",
    type=Path,
    required=True,
    help="raw log directory to write, e.g. ~/data/auv_pose/surveys_v3/pass0",
  )
  parser.add_argument(
    "--force", action="store_true", help="overwrite --out if it exists"
  )
  parser.add_argument(
    "--yaw",
    type=float,
    default=0.0,
    help=(
      "track heading in degrees from world +x, held for the whole run. The "
      "fan opens along body +y, so this rotates the swath with the track and "
      "keeps it across-track. Fly the same box at several headings -- 0 and 90 "
      "at least -- and fit the map on all of them: a pipeline shadows the "
      "seabed behind it from one direction and not from another"
    ),
  )
  parser.add_argument(
    "--route",
    choices=("lawnmower", "figure8"),
    default="lawnmower",
    help=(
      "lawnmower surveys --box; figure8 flies a test track the map is scored "
      "on, two loops at --depths about --centre. Guidance holds the heading, "
      "so the vehicle crabs around the loops rather than turning into them"
    ),
  )
  parser.add_argument(
    "--centre",
    type=float,
    nargs=2,
    default=[-20.0, -10.0],
    metavar=("X", "Y"),
    help="figure8: where the loops touch",
  )
  parser.add_argument(
    "--radius", type=float, default=15.0, help="figure8: loop radius, m"
  )
  parser.add_argument(
    "--depths",
    type=float,
    nargs=2,
    default=[0.0, -30.0],
    metavar=("Z1", "Z2"),
    help="figure8: world z of each loop",
  )
  parser.add_argument(
    "--loop-points", type=int, default=36, help="figure8: waypoints per loop"
  )
  parser.add_argument(
    "--box",
    type=float,
    nargs=4,
    metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
    default=list(SURVEY_BOX),
    help="region to cover, metres",
  )
  parser.add_argument(
    "--spacing",
    type=float,
    default=20.0,
    help=(
      "line spacing, metres. Sets redundancy, not coverage: the swath is 80 m "
      "wide at survey altitude, so 20 m already gives four looks at every "
      "patch from one heading"
    ),
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="seeds the initial belief's draw around the true start pose",
  )
  parser.add_argument(
    "--settle-steps",
    type=int,
    default=60,
    help=(
      "ticks under zero thrust before navigation starts. The vehicle is "
      "dropped in negatively buoyant and is still sinking on the first tick"
    ),
  )
  parser.add_argument("--max-steps", type=int, default=100_000)
  parser.add_argument("--arrival-radius", type=float, default=0.5)
  parser.add_argument(
    "--profiles",
    type=Path,
    default=None,
    help=(
      "also write the raw sonar images, with true poses, to this .npz for "
      "check_beam_validity.py -- scoring the sonar against the octree over "
      "the whole box rather than one hover"
    ),
  )
  parser.add_argument(
    "--profile-steps",
    type=int,
    default=400,
    help="stop recording images after this many pings; diagnosis, not a survey",
  )
  parser.add_argument(
    "--octree-min",
    type=float,
    default=0.02,
    help=(
      "finest octree voxel in metres. The multibeam's range bins are 0.0995 m, "
      "so holoocean's default of 0.02 is 5x finer -- about the right ratio. "
      "Budget ~30 GB and three minutes of octree generation at startup, after "
      "which it stops. Raising this is not a free speedup: ShadowEpsilon "
      "defaults to 4*OctreeMin, so it changes sonar returns and makes surveys "
      "inconsistent with maps built at another value"
    ),
  )
  parser.add_argument(
    "--headless",
    action="store_true",
    help="run with -RenderOffScreen, for machines without a usable display",
  )
  return parser.parse_args()


def commit() -> str:
  """The code version that flew the survey, recorded in the log's metadata."""
  try:
    return subprocess.run(
      ["git", "describe", "--always", "--dirty"],
      capture_output=True,
      text=True,
      check=True,
    ).stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return "unknown"


def reading(state: dict, name: str, size: int | None = None):
  """A sensor's reading this tick, or ``None`` if it did not report."""
  if name not in state:
    return None
  value = np.asarray(state[name], dtype=float).ravel()
  return value if size is None else value[:size]


def main() -> None:
  args = parse_args()
  args.out = args.out.expanduser()

  refuse_overwrite(args.out, args.force)
  configure_sdl(args.headless)

  if args.route == "figure8":
    waypoints = figure_eight(
      centre=tuple(args.centre),
      radius=args.radius,
      depths=tuple(args.depths),
      loop_points=args.loop_points,
    )
    print(
      f"figure eight about {tuple(args.centre)}: two {args.radius:g} m loops "
      f"at z = {args.depths[0]:g} and {args.depths[1]:g} m, heading held at "
      f"{args.yaw:.0f} deg"
    )
  else:
    waypoints = lawnmower(
      heading=args.yaw, box=tuple(args.box), spacing=args.spacing
    )
    print(
      f"lawnmower, heading {args.yaw:.0f} deg, {args.spacing:.1f} m spacing"
    )
  track = np.asarray(waypoints)
  print(
    f"{len(waypoints)} waypoints, "
    f"{np.linalg.norm(np.diff(track, axis=0), axis=1).sum():.0f} m of track"
  )

  env = holoocean.make(
    scenario_cfg=build_scenario(waypoints[0], args.octree_min, args.yaw),
    show_viewport=not args.headless,
  )
  ranges = range_bins(
    SONAR["range_min"], SONAR["range_max"], SONAR["range_bins"]
  )
  bearings = azimuth_angles(SONAR["azimuth"], SONAR["azimuth_bins"])

  command = np.zeros(8)
  state = env.tick()
  dvl = None
  for _ in range(args.settle_steps):
    state = env.step(command)
    dvl = reading(state, "dvl", 3) if "dvl" in state else dvl

  # The belief starts *around* the truth, not on it: the stand-in for a
  # surface fix. Velocity comes from the last DVL reading, rotated by the
  # believed attitude, as a vehicle would have it.
  rng = np.random.default_rng(args.seed)
  true_rotation = np.array(state["orient"], dtype=float)
  truth = NavState.at_rest(
    position=np.array(state["pose"])[:3, 3],
    attitude=rotmat_to_quat(true_rotation),
  )
  cov = initial_covariance()
  mean = boxplus(truth, np.sqrt(np.diag(cov)) * rng.normal(size=DOF))
  velocity = np.zeros(3) if dvl is None else mean.rotation @ dvl
  mean = mean._replace(velocity=velocity)
  initial = ManifoldGaussian(mean, cov)

  aiding_noise = AidingNoise(
    dvl=dvl_noise_covariance(DVL_BEAM_SIGMA, DVL_ELEVATION),
    depth=DEPTH_SIGMA,
    magnetometer=COMPASS_SIGMA,
  )
  navigator = InertialNavigator(
    initial, 1.0 / TICK_RATE_HZ, aiding_noise, imu_noise=SURVEY_IMU
  )

  meta = {
    "commit": commit(),
    "tick_rate_hz": TICK_RATE_HZ,
    "sonar_hz": SONAR_HZ,
    "aiding_hz": AIDING_HZ,
    "sonar": SONAR,
    "bearings": bearings.tolist(),
    "swath_axis": list(PROFILER_SWATH_AXIS),
    "nadir_axis": list(PROFILER_NADIR_AXIS),
    "imu_noise": SURVEY_IMU._asdict(),
    "dvl_beam_sigma": DVL_BEAM_SIGMA,
    "dvl_elevation_deg": DVL_ELEVATION,
    "depth_sigma": DEPTH_SIGMA,
    "compass_sigma": COMPASS_SIGMA,
    "magnetic_field": MAGNETIC_NORTH.tolist(),
    "initial_sigma": INITIAL_SIGMA,
    # Truth at the moment navigation started, so the pose's error can be
    # scored relative to the start's own -- which is what an anchored map's
    # covariance describes. Scoring only.
    "initial_truth": {
      "position": truth.position.tolist(),
      "attitude": truth.attitude.tolist(),
    },
    "initial_mean": {
      k: np.asarray(v).tolist() for k, v in mean._asdict().items()
    },
    "seed": args.seed,
    "yaw_deg": args.yaw,
    "route": args.route,
    "box": list(args.box),
    "spacing": args.spacing,
    "waypoints": waypoints,
  }

  follower = WaypointFollower(waypoints, args.arrival_radius)
  ticks = 0
  pings = 0
  live_beams = 0
  images: list[np.ndarray] = []
  image_poses: list[np.ndarray] = []
  image_rotations: list[np.ndarray] = []

  with RawSurveyWriter(args.out, len(bearings), meta) as log:
    for step in range(args.max_steps):
      state = env.step(command)
      ticks += 1

      imu = np.asarray(state["imu"], dtype=float)
      accel, gyro, accel_bias, gyro_bias = imu[:4]
      true_position = np.array(state["pose"])[:3, 3]
      true_rotation = np.array(state["orient"], dtype=float)

      dvl = reading(state, "dvl", 3)
      depth_reading = reading(state, "depthsensor", 1)
      depth = None if depth_reading is None else float(depth_reading[0])
      compass = reading(state, "magnetometer", 3)

      beam_ranges = None
      if "multibeam" in state:
        image = np.asarray(state["multibeam"], dtype=float)
        beam_ranges = bottom_return_ranges(image, ranges)
        if args.profiles and len(images) < args.profile_steps:
          images.append(image.copy())
          image_poses.append(true_position.copy())
          image_rotations.append(true_rotation.copy())

      log.tick(
        step,
        gyro=gyro,
        accel=accel,
        true_position=true_position,
        true_attitude=rotmat_to_quat(true_rotation),
        true_gyro_bias=gyro_bias,
        true_accel_bias=accel_bias,
        dvl=dvl,
        depth=depth,
        magnetometer=compass,
      )
      navigator.tick(
        step,
        gyro,
        accel,
        dvl=dvl,
        depth=depth,
        magnetometer=compass,
        close=beam_ranges is not None,
      )

      if beam_ranges is not None:
        log.ping(step, beam_ranges)
        pings += 1
        live_beams += int(np.isfinite(beam_ranges).sum())

        # Count pings, not steps: the sonar runs at 5 Hz against a 30 Hz tick.
        if pings % 200 == 0:
          estimate = navigator.belief
          error = np.linalg.norm(estimate.mean.position[:2] - true_position[:2])
          sigma = np.sqrt(np.trace(estimate.cov[:2, :2]))
          print(
            f"step {step} | waypoint {follower.index}/{len(waypoints)} "
            f"| {pings} pings | horizontal error {error:.2f} m "
            f"against {sigma:.2f} m (1 sigma)"
          )

      # Steer on the estimate, in the body frame: at a non-zero yaw a
      # world-frame error drives the vehicle sideways.
      estimate = navigator.belief.mean
      next_command = follower.command(
        estimate.position, quat_to_rotmat(estimate.attitude)
      )
      if next_command is None:
        if follower.finished:
          print("survey complete")
          break
        continue
      command = next_command
    else:
      print(f"stopped after {args.max_steps} steps without finishing")

  print(f"Wrote {pings} pings and {ticks} ticks to {args.out}")
  if pings:
    fraction = live_beams / (pings * len(bearings))
    print(f"{100 * fraction:.1f}% of beams returned an echo")
  for name, values in navigator.nis.items():
    if values:
      dof = {"dvl": 3, "depth": 1, "compass": 3}[name]
      print(
        f"{name:8s} mean NIS {np.mean(values):.2f} against {dof} "
        f"over {len(values)} updates"
      )

  if args.profiles and images:
    write_capture(args.profiles, images, image_poses, image_rotations, SONAR)
    print(f"Wrote {len(images)} pings to {args.profiles}")


if __name__ == "__main__":
  main()
