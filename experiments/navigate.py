"""Terrain-aided navigation: fly a waypoint course using the bathymetry map.

    nix run .#sim -- -c "python experiments/navigate.py"

Dead reckons the IMU, ranges the seabed with a singlebeam sonar, looks the seabed
depth up in the fitted GP map, and corrects the depth estimate in an EKF.

Two properties worth knowing before comparing runs:

* **Depth is the only measurement**, so ``x`` and ``y`` are unobservable and drift
  with the IMU. That is a true property of this sensor suite; constraining them
  needs terrain correlation over varying relief or a bathymetric particle filter.
* **Guidance runs on the estimate**, not on ground truth, so tracking error
  reflects estimator quality honestly.

Ground truth is read only to initialise attitude and to log error. It never enters
the filter or the controller.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import holoocean
import numpy as np

from auv_pose.estimation.filters import (
  ConstantVelocityEKF,
  position,
  velocity,
)
from auv_pose.estimation.quaternion import rotmat_to_quat
from auv_pose.estimation.strapdown import StrapdownIntegrator
from auv_pose.estimation.typing import Measurement
from auv_pose.io.checkpoints import load_map
from auv_pose.io.logs import CsvLogger
from auv_pose.mapping.sonar import bottom_return_range, range_bins
from experiments.cli import configure_sdl
from experiments.guidance import WaypointFollower
from experiments.scenarios import (
  depth_sensor,
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

SONAR = {"range_min": 0.5, "range_max": 100.0, "range_bins": 256}

COLUMNS = (
  "step",
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
)


def build_scenario() -> dict:
  return ocean_scenario(
    "terrain_aided_navigation",
    start=START,
    sensors=[
      pose_sensor(),
      orientation_sensor(),
      depth_sensor(hz=TICK_RATE_HZ),
      imu_sensor("imu_1", hz=TICK_RATE_HZ),
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
    "--headless",
    action="store_true",
    help="run with -RenderOffScreen, for machines without a usable display",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  configure_sdl(args.headless)

  bathymetry = load_map(args.map)
  ranges = range_bins(
    SONAR["range_min"], SONAR["range_max"], SONAR["range_bins"]
  )
  dt = 1.0 / TICK_RATE_HZ

  env = holoocean.make(
    scenario_cfg=build_scenario(), show_viewport=not args.headless
  )
  state = env.tick()

  start_position = np.array(state["pose"])[:3, 3]
  attitude = rotmat_to_quat(np.array(state["orient"], dtype=float))

  ekf = ConstantVelocityEKF(accel_process_sigma=0.5)
  estimate = ekf.initial(start_position)
  dead_reckoning = StrapdownIntegrator(start_position, attitude)

  measurement_noise = np.diag([args.sigma_map**2, args.sigma_depth**2])
  depth_only_noise = measurement_noise[1:, 1:]

  follower = WaypointFollower(WAYPOINTS, args.arrival_radius)
  command = np.zeros(8)

  with CsvLogger(args.out, COLUMNS) as log:
    for step in range(args.max_steps):
      state = env.step(command)

      truth = np.array(state["pose"])[:3, 3]
      accel_body, gyro_body = np.array(state["imu_1"], dtype=float)[:2]

      # Strapdown gives both the honest dead-reckoning baseline and the
      # world-frame acceleration the filter needs as its control input.
      accel_world = dead_reckoning.step(gyro_body, accel_body, dt)

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

      estimate = ekf.step(estimate, accel_world, dt, observations)
      estimated_position = position(estimate)
      estimated_velocity = velocity(estimate)

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
      )
    else:
      print(
        f"stopped after {args.max_steps} steps without finishing the course"
      )

  print(f"Wrote {args.out}")


if __name__ == "__main__":
  main()
