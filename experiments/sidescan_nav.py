"""Navigation corrected by matching sidescan returns against a labelled obstacle map.

    nix run .#sim -- -c "python experiments/sidescan_nav.py --obstacles <dir>"

Projects bright sidescan returns into the world, classifies them, snaps each to the
nearest known obstacle of that class, and feeds the result to the EKF.

Requires the labelled obstacle CSVs that ``obstacle_train.py`` was fitted on. Those
are **not in this repository**; supply them with ``--obstacles``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import holoocean
import numpy as np

from auv_pose.mapping.sonar import range_bins
from auv_pose.smoothing.filters import ConstantVelocityEKF, position
from auv_pose.smoothing.quaternion import rotmat_to_quat
from auv_pose.smoothing.strapdown import StrapdownIntegrator
from auv_pose.smoothing.typing import Measurement
from experiments.guidance import WaypointFollower
from experiments.obstacle_map import ObstacleMap
from experiments.scenarios import (
  imu_sensor,
  ocean_scenario,
  orientation_sensor,
  pose_sensor,
  sidescan_sonar,
)

TICK_RATE_HZ = 20
START = [1.0, 2.0, -6.0]
INTENSITY_THRESHOLD = 250
MATCH_RADIUS = 50.0

WAYPOINTS = np.array(
  [
    [-6.0, -4.0, -20.0],
    [-14.0, -10.0, -40.0],
    [-22.0, -18.0, -60.0],
    [-30.0, -28.0, -60.0],
    [-35.0, -35.0, -60.0],
  ]
)

POSITION_H = np.hstack([np.eye(3), np.zeros((3, 3))])
SIDESCAN = {
  "range_min": 0.5,
  "range_max": 70.0,
  "range_bins": 256,
  "azimuth_bins": 256,
}


def build_scenario() -> dict:
  return ocean_scenario(
    "sidescan_navigation",
    start=START,
    sensors=[
      pose_sensor(),
      orientation_sensor(),
      imu_sensor("imu_1", hz=TICK_RATE_HZ),
      sidescan_sonar("sidescan", hz=10, **SIDESCAN),
    ],
  )


def project_returns(
  profile: np.ndarray,
  position: np.ndarray,
  ranges: np.ndarray,
  angles: np.ndarray,
) -> np.ndarray:
  """World points for every bin brighter than the threshold.

  Returns an ``(n, 3)`` array, empty when nothing is bright enough.
  """
  spread = np.ptp(profile)
  if spread == 0:
    return np.empty((0, 3))

  normalised = (profile - profile.min()) / spread * 255
  bright = np.flatnonzero(normalised > INTENSITY_THRESHOLD)
  if bright.size == 0:
    return np.empty((0, 3))

  r, theta = ranges[bright], angles[bright]
  local = np.column_stack(
    [np.zeros_like(r), r * np.sin(theta), -r * np.cos(theta)]
  )
  return position + local


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--obstacles",
    type=Path,
    nargs="+",
    required=True,
    help="labelled obstacle CSVs (x, y, z, obstacle_type)",
  )
  parser.add_argument(
    "--classifier", type=Path, default=Path("obstacle_classifier.pkl")
  )
  parser.add_argument("--encoder", type=Path, default=Path("label_encoder.pkl"))
  parser.add_argument("--max-steps", type=int, default=6000)
  parser.add_argument("--arrival-radius", type=float, default=0.5)
  parser.add_argument("--sigma-match", type=float, default=2.0)
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  obstacles = ObstacleMap.load(args.obstacles, args.classifier, args.encoder)

  ranges = range_bins(
    SIDESCAN["range_min"], SIDESCAN["range_max"], SIDESCAN["range_bins"]
  )
  angles = np.linspace(-np.pi / 4, np.pi / 4, SIDESCAN["range_bins"])
  dt = 1.0 / TICK_RATE_HZ

  env = holoocean.make(scenario_cfg=build_scenario())
  state = env.tick()

  start_position = np.array(state["pose"])[:3, 3]
  attitude = rotmat_to_quat(np.array(state["orient"], dtype=float))

  ekf = ConstantVelocityEKF()
  estimate = ekf.initial(start_position)
  dead_reckoning = StrapdownIntegrator(start_position, attitude)
  measurement_noise = np.eye(3) * args.sigma_match**2

  follower = WaypointFollower(WAYPOINTS, args.arrival_radius)
  command = np.zeros(8)

  for step in range(args.max_steps):
    state = env.step(command)
    accel_body, gyro_body = np.array(state["imu_1"], dtype=float)[:2]

    accel_world = dead_reckoning.step(gyro_body, accel_body, dt)

    # Returns are projected from where we think we are, so the prior is needed
    # before conditioning. predict is pure, so step() recomputing it below is
    # two matmuls, not a correctness concern.
    prior = ekf.predict(estimate, accel_world, dt)

    points = project_returns(
      np.asarray(state["sidescan"], dtype=float),
      position(prior),
      ranges,
      angles,
    )

    observations = []
    if points.size:
      match = obstacles.nearest(points.mean(axis=0))
      if match is not None and match.distance < MATCH_RADIUS:
        observations.append(
          Measurement(match.position, POSITION_H, measurement_noise)
        )

    estimate = ekf.step(estimate, accel_world, dt, observations)
    estimated_position = position(estimate)

    next_command = follower.command(estimated_position)
    if next_command is None:
      if follower.finished:
        print(f"step {step}: course complete")
        break
      print(f"step {step}: reached waypoint {follower.index - 1}")
      continue
    command = next_command

    if step % 100 == 0:
      truth = np.array(state["pose"])[:3, 3]
      print(
        f"step {step} | error "
        f"{np.linalg.norm(truth - estimated_position):.2f} m"
      )


if __name__ == "__main__":
  main()
