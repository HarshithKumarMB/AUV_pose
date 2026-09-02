"""Terrain-aided navigation: fly a waypoint course using the bathymetry map.

    nix run .#sim -- -c "python experiments/navigate.py"

Dead reckons the IMU, ranges the seabed with a singlebeam sonar, looks the seabed
depth up in the fitted GP map, and corrects the depth estimate in an EKF.

Three properties worth knowing before comparing runs:

* **Depth and velocity are measured; position is not.** The sonar-plus-map and
  the depth sensor observe ``z``; the DVL observes velocity. Nothing observes
  ``x`` or ``y`` directly, so horizontal position still drifts -- but at a rate
  bounded by the DVL, linearly rather than quadratically. Pinning it down needs
  terrain correlation over varying relief, or a bathymetric particle filter.
* **The DVL reading is rotated by the attitude estimate**, so heading error
  turns it in the horizontal plane. Its speed is accurate to under 1%, but the
  direction error tracks the attitude error almost exactly, so the observation
  noise is built as ``sigma_dvl^2 + (speed * sigma_heading)^2``. Nothing here
  observes heading, so that second term is an assumption.
* **The DVL also feeds the attitude filter**, which is a separate job from
  aiding velocity. An accelerometer is a gravity reference only at rest;
  under a manoeuvre the specific force tilts by ``atan(a_h / g)`` and a
  complementary filter chases that tilt, which then rotates the same
  acceleration back out of the strapdown. Differentiating the DVL gives the
  vehicle's own acceleration, so it can be subtracted first. Without a DVL
  there is nothing to subtract and this is unfixed -- see ``--no-dvl``.
* **Guidance runs on the estimate**, not on ground truth, so tracking error
  reflects estimator quality honestly.

Ground truth is read only to initialise attitude and to log error. It never enters
the filter or the controller -- except under ``--truth-attitude``, a diagnostic
flag that deliberately breaks that rule to isolate attitude error, and which
announces itself loudly when used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import holoocean
import numpy as np

from auv_pose.estimation.filters import (
  AttitudeFilter,
  BodyAcceleration,
  ConstantVelocityEKF,
  position,
  velocity,
)
from auv_pose.estimation.quaternion import (
  GRAVITY_NED,
  GRAVITY_NWU,
  quat_angle,
  quat_to_rotmat,
  rotmat_to_quat,
)
from auv_pose.estimation.strapdown import StrapdownIntegrator
from auv_pose.estimation.typing import Measurement
from auv_pose.io.checkpoints import load_map
from auv_pose.io.logs import CsvLogger
from auv_pose.mapping.sonar import bottom_return_range, range_bins
from experiments.cli import configure_sdl
from experiments.guidance import WaypointFollower
from experiments.scenarios import (
  depth_sensor,
  dvl_sensor,
  imu_sensor,
  ocean_scenario,
  orientation_sensor,
  pose_sensor,
  singlebeam_sonar,
)

TICK_RATE_HZ = 30
START = [-20.0, -10.0, 0.0]

WAYPOINTS = np.array(
  [
    [-20.0, -10.0, 0.0],
    [-10.0, -5.0, 0.0],
    [-20.0, -5.0, 0.0],
    [-30.0, -5.0, 0.0],
    [-30.0, -10.0, 0.0],
    [-30.0, -15.0, 0.0],
    [-20.0, -15.0, 0.0],
    [-10.0, -15.0, 0.0],
    [-20.0, -10.0, 0.0],
  ]
)

# Depth is the only observed quantity. Both rows measure z: the sonar range plus
# the mapped seabed depth, and the pressure depth sensor.
DEPTH_H = np.array(
  [
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
  ]
)

# The DVL observes velocity directly. This is what makes velocity observable
# at all: without it, position comes from doubly integrating acceleration.
VELOCITY_H = np.hstack([np.zeros((3, 3)), np.eye(3)])

SONAR = {"range_min": 0.5, "range_max": 100.0, "range_bins": 256}

COLUMNS = (
  "step",
  "att_err_deg",
  "ax",
  "ay",
  "az",
  "sonar_range",
  "map_depth",
  "x",
  "y",
  "z",
  "ekf_x",
  "ekf_y",
  "ekf_z",
  "ekf_vx",
  "ekf_vy",
  "ekf_vz",
  "dr_x",
  "dr_y",
  "dr_z",
  "dr_vx",
  "dr_vy",
  "dr_vz",
  "dvl_vx",
  "dvl_vy",
  "dvl_vz",
  # Gyro as read, alongside the truth orientation it is meant to propagate.
  # Together they recover the truth body rate as 2 vec(q_k* (x) q_k+1) / dt,
  # which is what identifies a frame error in the gyro -- the DVL's axes
  # turned out to be flipped, and nothing rules the gyro's out.
  "wx",
  "wy",
  "wz",
  "true_qw",
  "true_qx",
  "true_qy",
  "true_qz",
)


def build_scenario(
  octree_min: float,
  accel_bias_sigma: float,
  ang_vel_bias_sigma: float,
  legacy_frames: bool = False,
) -> dict:
  # See --legacy-frames: the COM socket is the misconfiguration, not a choice.
  socket = None if legacy_frames else "IMUSocket"
  return ocean_scenario(
    "terrain_aided_navigation",
    start=START,
    octree_min=octree_min,
    sensors=[
      pose_sensor(socket=socket),
      orientation_sensor(socket=socket),
      depth_sensor(hz=TICK_RATE_HZ),
      imu_sensor(
        "imu_1",
        hz=TICK_RATE_HZ,
        accel_bias_sigma=accel_bias_sigma,
        ang_vel_bias_sigma=ang_vel_bias_sigma,
      ),
      dvl_sensor("dvl", hz=TICK_RATE_HZ),
      singlebeam_sonar("singlebeam", hz=TICK_RATE_HZ, **SONAR),
    ],
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Terrain-aided waypoint navigation"
  )
  parser.add_argument("--map", type=Path, default=Path("svgp_bathymetry.pkl"))
  parser.add_argument("--out", type=Path, default=Path("wp_c.csv"))
  parser.add_argument("--max-steps", type=int, default=20_000)
  parser.add_argument("--arrival-radius", type=float, default=0.5)
  parser.add_argument(
    "--report-every",
    type=int,
    default=30,
    help="print a diagnostic line every N steps; 0 to silence",
  )
  parser.add_argument(
    "--sigma-map",
    type=float,
    default=1.0,
    help="std of the sonar+map depth observation, m",
  )
  parser.add_argument(
    "--sigma-depth",
    type=float,
    default=0.5,
    help="std of the depth sensor observation, m",
  )
  parser.add_argument(
    "--octree-min",
    type=float,
    default=0.02,
    help=(
      "finest octree voxel in metres. holoocean's default of 0.02 is 19x "
      "finer than the singlebeam's 0.39 m range bins and generates octrees "
      "at several GB per minute; 0.1 is still 4x finer than a bin"
    ),
  )
  parser.add_argument(
    "--accel-bias-sigma",
    type=float,
    default=6e-5,
    help=(
      "per-sample random-walk increment for the accelerometer bias. Grows as "
      "sigma*sqrt(n), so this is 0.001 m/s^2 after 300 samples. Pass 0.01 to "
      "reproduce the runs before the sensor was measured"
    ),
  )
  parser.add_argument(
    "--gyro-bias-sigma",
    type=float,
    default=5e-5,
    help=(
      "per-sample random-walk increment for the gyro bias, reaching "
      "0.05 deg/s after 300 samples. At 0.01 it reaches 14 deg/s, which is "
      "what drove heading through 70 degrees in 10 s"
    ),
  )
  parser.add_argument(
    "--no-dvl",
    action="store_true",
    help=(
      "drop the DVL entirely. The unaided baseline: velocity is then "
      "unobservable and position drifts quadratically, the run starts from an "
      "assumed rest, and the attitude filter loses its manoeuvre compensation"
    ),
  )
  parser.add_argument(
    "--settle-steps",
    type=int,
    default=60,
    help=(
      "ticks with a zero command before the estimators are initialised. The "
      "vehicle is dropped in negatively buoyant and is still sinking at "
      "0.4 m/s when the first tick returns; starting an integrator from an "
      "assumed rest there costs 4 m of vertical drift over 10 s"
    ),
  )
  parser.add_argument(
    "--dvl-accel-tau",
    type=float,
    default=0.1,
    help=(
      "time constant of the low-pass on DVL velocity before it is differenced "
      "for the attitude filter's manoeuvre compensation, s. Lag and noise pull "
      "opposite ways and the minimum-error choice is the wrong one: 0.3-0.5 "
      "minimises rms error on the acceleration but lags it by 30-40%%, and that "
      "shortfall is systematic where the noise at 0.1 is not. Measured mean "
      "attitude error over a run: 0.56 deg at 0.5, 0.28 at 0.1, 0.38 at 0.05"
    ),
  )
  parser.add_argument(
    "--legacy-frames",
    action="store_true",
    help=(
      "reproduce the frame mismatch fixed in this branch: orientation read "
      "from the default COM socket while the IMU and DVL report in the IMU "
      "socket, gravity taken as z-down in a z-up world, and the DVL patched "
      "with a sign flip. Cancels exactly at rest and mirrors every recovered "
      "acceleration in y and z once moving"
    ),
  )
  parser.add_argument(
    "--sigma-dvl",
    type=float,
    default=0.02,
    help="the DVL's own velocity noise, m/s; matches the sensor VelSigma",
  )
  parser.add_argument(
    "--sigma-heading",
    type=float,
    default=30.0,
    help=(
      "assumed heading uncertainty in degrees. The DVL measures speed well "
      "but its direction comes from the attitude estimate, so heading error "
      "dominates: measured direction error tracks attitude error almost "
      "exactly. Nothing observes heading here, hence a fixed assumption"
    ),
  )
  parser.add_argument(
    "--gyro-only",
    action="store_true",
    help=(
      "propagate attitude from the gyro with no accelerometer correction. "
      "The uncorrected baseline: attitude then drifts without bound and "
      "gravity leaks into the acceleration as g*sin(theta)"
    ),
  )
  parser.add_argument(
    "--attitude-kp",
    type=float,
    default=1.0,
    help="proportional gain on the tilt correction, 1/s",
  )
  parser.add_argument(
    "--truth-attitude",
    action="store_true",
    help=(
      "DIAGNOSTIC ONLY: take attitude from the OrientationSensor each tick "
      "instead of propagating the gyro. This is ground truth inside the "
      "estimator, so results are not valid -- it exists to bisect attitude "
      "divergence from position integration"
    ),
  )
  parser.add_argument(
    "--headless",
    action="store_true",
    help="run with -RenderOffScreen, for machines without a usable display",
  )
  return parser.parse_args()


def innovation(step_record, observations) -> np.ndarray:
  """Measurement minus prediction, before conditioning.

  The sharpest early signal that something is wrong. A frame convention error,
  a sign slip in the strapdown, or a bad map lookup shows up here as a large
  innovation on the very first ticks, where the position error would take
  thousands of steps to become obviously wrong.

  :param step_record: The :class:`Step` the filter just recorded.
  :param observations: Measurements conditioned on this step.
  :return: Concatenated residuals, empty if there were no observations.
  """
  residuals = [
    np.asarray(obs.z, dtype=float)
    - np.atleast_2d(obs.H) @ step_record.prior.mean
    for obs in observations
  ]
  return np.concatenate(residuals) if residuals else np.empty(0)


def report(
  step,
  step_record,
  observations,
  truth,
  estimate,
  dead_reckoned,
  sonar_range,
  map_depth,
  waypoint,
  attitude_error,
) -> None:
  """One diagnostic line.

  Ground truth appears here and in the log only. It never enters the filter or
  the controller.
  """
  ekf_error = np.linalg.norm(truth - position(estimate))
  dr_error = np.linalg.norm(truth - dead_reckoned)
  resid = innovation(step_record, observations)
  resid_text = np.array2string(resid, precision=2, suppress_small=True)

  print(
    f"step {step:6d} | wp {waypoint} | att {attitude_error:6.2f} deg | "
    f"ekf {ekf_error:7.2f} m | dr {dr_error:8.2f} m | "
    f"sonar {sonar_range:6.2f} | map {map_depth:8.2f} | innov {resid_text}"
  )


def main() -> None:
  args = parse_args()

  configure_sdl(args.headless)

  if args.truth_attitude:
    print(
      "*** --truth-attitude: attitude comes from ground truth. This run is a "
      "diagnostic bisection, not a valid result. ***"
    )

  bathymetry = load_map(args.map)
  ranges = range_bins(
    SONAR["range_min"], SONAR["range_max"], SONAR["range_bins"]
  )
  dt = 1.0 / TICK_RATE_HZ

  env = holoocean.make(
    scenario_cfg=build_scenario(
      args.octree_min,
      args.accel_bias_sigma,
      args.gyro_bias_sigma,
      legacy_frames=args.legacy_frames,
    ),
    show_viewport=not args.headless,
  )
  state = env.tick()

  # The vehicle is dropped in negatively buoyant and has not settled by the
  # first tick -- it is sinking at 0.4 m/s. Let it ride that out under no
  # thrust, then start from what the DVL says it is actually doing. Both
  # matter: an integrator started from an assumed rest carries that 0.4 m/s as
  # a bias for the whole run, which is 4 m of vertical drift over 10 s and was
  # the entire dead-reckoning error in the vertical channel.
  command = np.zeros(8)
  for _ in range(args.settle_steps):
    state = env.step(command)

  start_position = np.array(state["pose"])[:3, 3]
  attitude = rotmat_to_quat(np.array(state["orient"], dtype=float))

  if args.no_dvl:
    start_velocity = np.zeros(3)
  else:
    start_velocity = (
      quat_to_rotmat(attitude) @ np.asarray(state["dvl"], dtype=float)[:3]
    )

  gravity = GRAVITY_NED if args.legacy_frames else GRAVITY_NWU

  ekf = ConstantVelocityEKF(accel_process_sigma=0.5)
  estimate = ekf.initial(start_position, start_velocity)
  attitude_filter = (
    None
    if args.gyro_only
    else AttitudeFilter(kp=args.attitude_kp, q0=attitude, gravity=gravity)
  )
  dead_reckoning = StrapdownIntegrator(
    start_position,
    attitude,
    velocity=start_velocity,
    attitude_filter=attitude_filter,
    gravity=gravity,
  )
  # Feeds the attitude filter the vehicle's own acceleration so it is not
  # mistaken for tilt. Unused under --no-dvl, where there is nothing to
  # measure it with.
  body_acceleration = BodyAcceleration(tau=args.dvl_accel_tau)

  measurement_noise = np.diag([args.sigma_map**2, args.sigma_depth**2])
  depth_only_noise = measurement_noise[1:, 1:]

  follower = WaypointFollower(WAYPOINTS, args.arrival_radius)

  with CsvLogger(args.out, COLUMNS) as log:
    for step in range(args.max_steps):
      state = env.step(command)

      truth = np.array(state["pose"])[:3, 3]
      accel_body, gyro_body = np.array(state["imu_1"], dtype=float)[:2]

      truth_attitude = rotmat_to_quat(np.array(state["orient"], dtype=float))

      if args.truth_attitude:
        # Diagnostic bisection only -- see --truth-attitude. step() still
        # propagates one dt of gyro from here, so the rotation it uses carries
        # a single sample of error rather than none.
        dead_reckoning.set_attitude(truth_attitude)

      # Read the DVL before the strapdown, not after: the attitude filter
      # inside it needs the vehicle's own acceleration to tell a manoeuvre
      # apart from a tilt. Rotating the reading into the world stays below, so
      # the velocity measurement still uses the post-update attitude.
      if args.no_dvl:
        dvl_body = None
        kinematic_accel = None
      else:
        dvl_body = np.asarray(state["dvl"], dtype=float)[:3]
        if args.legacy_frames:
          dvl_body = dvl_body * np.array([1.0, -1.0, -1.0])
        kinematic_accel = body_acceleration.update(dvl_body, gyro_body, dt)

      # Strapdown gives both the honest dead-reckoning baseline and the
      # world-frame acceleration the filter needs as its control input.
      accel_world = dead_reckoning.step(
        gyro_body, accel_body, dt, kinematic_accel
      )
      attitude_error = np.degrees(
        quat_angle(dead_reckoning.attitude, truth_attitude)
      )

      sonar_range = bottom_return_range(np.asarray(state["singlebeam"]), ranges)
      depth = float(np.asarray(state["depthsensor"]).ravel()[0])

      if np.isnan(sonar_range):
        # No echo: the depth sensor alone. Skipping the GP here matters --
        # it is the most expensive operation in the tick.
        map_depth = float("nan")
        observations = [Measurement([depth], DEPTH_H[:1], depth_only_noise)]
      else:
        map_depth = float(bathymetry.predict(position(estimate)[None, :2])[0])
        observations = [
          Measurement(
            [sonar_range + map_depth, depth], DEPTH_H, measurement_noise
          )
        ]

      if dvl_body is not None:
        # Body-frame velocity over ground, rotated into the world by the
        # current attitude estimate.
        dvl_world = dead_reckoning.rotation @ dvl_body

        # The reading itself is good -- measured speed matches truth to under
        # 1% -- but rotating it by an uncertain heading is not. A heading error
        # of theta turns the vector, costing roughly speed * theta, which
        # dominates the sensor's own noise by an order of magnitude. Modelled
        # isotropically here, though the induced error is really perpendicular
        # to the velocity; tightening that needs a heading uncertainty to draw
        # on, and nothing currently observes heading.
        heading_term = np.linalg.norm(dvl_world) * np.radians(
          args.sigma_heading
        )
        dvl_var = args.sigma_dvl**2 + heading_term**2

        observations.append(
          Measurement(dvl_world, VELOCITY_H, np.eye(3) * dvl_var)
        )
      else:
        dvl_world = np.full(3, np.nan)

      estimate = ekf.step(estimate, accel_world, dt, observations)
      estimated_position = position(estimate)
      estimated_velocity = velocity(estimate)

      if args.report_every and step % args.report_every == 0:
        report(
          step,
          ekf.history[-1],
          observations,
          truth,
          estimate,
          dead_reckoning.position,
          sonar_range,
          map_depth,
          follower.index,
          attitude_error,
        )

      command_or_none = follower.command(estimated_position)
      if command_or_none is None:
        if follower.finished:
          print(f"step {step}: course complete")
          break
        print(f"step {step}: reached waypoint {follower.index - 1}")
        continue
      command = command_or_none

      log.write(
        step=step,
        att_err_deg=attitude_error,
        ax=accel_world[0],
        ay=accel_world[1],
        az=accel_world[2],
        sonar_range=sonar_range,
        map_depth=map_depth,
        x=truth[0],
        y=truth[1],
        z=truth[2],
        ekf_x=estimated_position[0],
        ekf_y=estimated_position[1],
        ekf_z=estimated_position[2],
        ekf_vx=estimated_velocity[0],
        ekf_vy=estimated_velocity[1],
        ekf_vz=estimated_velocity[2],
        dr_x=dead_reckoning.position[0],
        dr_y=dead_reckoning.position[1],
        dr_z=dead_reckoning.position[2],
        dr_vx=dead_reckoning.velocity[0],
        dr_vy=dead_reckoning.velocity[1],
        dr_vz=dead_reckoning.velocity[2],
        dvl_vx=dvl_world[0],
        dvl_vy=dvl_world[1],
        dvl_vz=dvl_world[2],
        wx=gyro_body[0],
        wy=gyro_body[1],
        wz=gyro_body[2],
        true_qw=truth_attitude[0],
        true_qx=truth_attitude[1],
        true_qy=truth_attitude[2],
        true_qz=truth_attitude[3],
      )
    else:
      # Distinguish "too slow" from "diverged": a crawling but healthy run is
      # making progress toward its waypoint, a broken one is not.
      remaining = np.linalg.norm(follower.target - position(estimate))
      print(
        f"stopped after {args.max_steps} steps "
        f"({args.max_steps / TICK_RATE_HZ:.0f} s simulated) "
        f"having reached {follower.index} of {len(WAYPOINTS)} waypoints; "
        f"{remaining:.1f} m from the next. Raise --max-steps if it was still "
        f"closing, otherwise the estimate has diverged."
      )

  print(f"Wrote {args.out}")


if __name__ == "__main__":
  main()
