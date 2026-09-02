"""Survey the seabed on a lawnmower track, logging soundings.

    nix run .#sim -- -c "python experiments/survey.py --out map1.csv"

Flies a boustrophedon pattern with a downward singlebeam sonar and writes
``x, y, sonar_depth`` for every step. The output feeds ``train_map.py``.

Uses the ground-truth pose to position each sounding: this builds the reference map
that navigation is later corrected against, so it must not itself be drifting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import holoocean
import numpy as np

from auv_pose.io.logs import CsvLogger
from auv_pose.io.soundings import SOUNDING_COLUMNS
from auv_pose.mapping.sonar import bottom_return_range, range_bins
from experiments.cli import refuse_overwrite
from experiments.guidance import WaypointFollower
from experiments.scenarios import ocean_scenario, pose_sensor, singlebeam_sonar

TICK_RATE_HZ = 30
SONAR = {"range_min": 0.5, "range_max": 100.0, "range_bins": 256}


def lawnmower(
  x_start: float = 0.0,
  x_end: float = -40.0,
  x_step: float = -2.0,
  y_near: float = 0.0,
  y_far: float = -20.0,
  y_step: float = -5.0,
) -> list[list[float]]:
  """Boustrophedon track: sweep in y, step across in x, reverse each pass."""
  waypoints: list[list[float]] = []
  columns = np.arange(x_start, x_end + x_step / 2, x_step)
  sweep = list(np.arange(y_near, y_far + y_step / 2, y_step))

  for index, x in enumerate(columns):
    legs = sweep if index % 2 == 0 else sweep[::-1]
    waypoints.extend([float(x), float(y), 0.0] for y in legs)

  return waypoints


def build_scenario(start: list[float]) -> dict:
  return ocean_scenario(
    "bathymetry_survey",
    start=start,
    sensors=[
      pose_sensor(),
      singlebeam_sonar("singlebeam", hz=TICK_RATE_HZ, **SONAR),
    ],
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Bathymetry survey")
  parser.add_argument("--out", type=Path, default=Path("map1.csv"))
  parser.add_argument(
    "--force", action="store_true", help="overwrite --out if it exists"
  )
  parser.add_argument("--max-steps", type=int, default=100_000)
  parser.add_argument("--arrival-radius", type=float, default=0.5)
  parser.add_argument(
    "--headless",
    action="store_true",
    help="run with -RenderOffScreen, for machines without a usable display",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  refuse_overwrite(args.out, args.force)

  waypoints = lawnmower()
  print(f"{len(waypoints)} waypoints")

  env = holoocean.make(
    scenario_cfg=build_scenario(waypoints[0]),
    show_viewport=not args.headless,
  )
  ranges = range_bins(
    SONAR["range_min"], SONAR["range_max"], SONAR["range_bins"]
  )

  follower = WaypointFollower(waypoints, args.arrival_radius)
  command = np.zeros(8)
  soundings = 0

  with CsvLogger(args.out, SOUNDING_COLUMNS) as log:
    for step in range(args.max_steps):
      state = env.step(command)
      position = np.array(state["pose"])[:3, 3]

      next_command = follower.command(position)
      if next_command is None:
        if follower.finished:
          print("survey complete")
          break
        continue
      command = next_command

      sonar_depth = bottom_return_range(np.asarray(state["singlebeam"]), ranges)
      if not np.isnan(sonar_depth):
        log.write(x=position[0], y=position[1], sonar_depth=sonar_depth)
        soundings += 1

      if step % 1000 == 0:
        print(
          f"step {step} | waypoint {follower.index}/{len(waypoints)} "
          f"| {soundings} soundings"
        )
    else:
      print(f"stopped after {args.max_steps} steps without finishing")

  print(f"Wrote {soundings} soundings to {args.out}")


if __name__ == "__main__":
  main()
