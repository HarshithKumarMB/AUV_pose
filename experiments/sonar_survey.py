"""Live sonar viewer: watch imaging or sidescan returns while the vehicle moves.

    nix run .#sim -- -c "python experiments/sonar_survey.py --sonar imaging"
    nix run .#sim -- -c "python experiments/sonar_survey.py --sonar sidescan --save"

An exploratory tool for seeing what a sonar returns, not part of the pipeline.
With ``--dead-reckon`` it also integrates the IMU and reports drift against ground
truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import holoocean
import matplotlib.pyplot as plt
import numpy as np

from auv_pose.estimation.quaternion import rotmat_to_quat
from auv_pose.estimation.strapdown import StrapdownIntegrator
from experiments.scenarios import (
  imaging_sonar,
  imu_sensor,
  ocean_scenario,
  orientation_sensor,
  pose_sensor,
  sidescan_sonar,
)

TICK_RATE_HZ = 20


def build_scenario(sonar: str, world: str, start: list[float]) -> dict:
  return ocean_scenario(
    f"{sonar}_survey",
    start=start,
    world=world,
    sensors=[
      pose_sensor(),
      orientation_sensor(),
      imu_sensor("imu_1", hz=TICK_RATE_HZ),
      imaging_sonar() if sonar == "imaging" else sidescan_sonar(),
    ],
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--sonar", choices=["imaging", "sidescan"], default="sidescan"
  )
  parser.add_argument("--world", default="Dam")
  parser.add_argument("--start", type=float, nargs=3, default=[1.0, 2.0, -6.0])
  parser.add_argument("--steps", type=int, default=2000)
  parser.add_argument(
    "--command",
    type=float,
    nargs=8,
    default=[0, 0, 0, 0, 10, 10, 0, 0],
    help="constant thruster command to fly under",
  )
  parser.add_argument(
    "--dead-reckon",
    action="store_true",
    help="integrate the IMU and report drift against ground truth",
  )
  parser.add_argument(
    "--save",
    type=Path,
    default=None,
    help="directory to write the waterfall image into",
  )
  parser.add_argument(
    "--headless",
    action="store_true",
    help="run with -RenderOffScreen, for machines without a usable display",
  )
  return parser.parse_args()


def draw(axes, waterfall, sonar: str, step: int) -> None:
  """Render the accumulated pings as a waterfall."""
  axes.clear()
  axes.imshow(np.array(waterfall), aspect="auto", cmap="viridis")
  axes.set_xlabel("Range bin")
  axes.set_ylabel("Ping")
  axes.set_title(f"{sonar} waterfall, step {step}")


def main() -> None:
  args = parse_args()

  env = holoocean.make(
    scenario_cfg=build_scenario(args.sonar, args.world, args.start),
    show_viewport=not args.headless,
  )
  state = env.tick()

  dead_reckoning = StrapdownIntegrator(
    np.array(state["pose"])[:3, 3],
    rotmat_to_quat(np.array(state["orient"], dtype=float)),
  )
  dt = 1.0 / TICK_RATE_HZ

  command = np.array(args.command, dtype=float)
  sensor_name = "sonar" if args.sonar == "imaging" else "sidescan"
  waterfall: list[np.ndarray] = []

  # Headless means no display at all, so skip the live view and only render at
  # the end for --save.
  live = not args.headless
  if live:
    plt.ion()
  figure, axes = plt.subplots(figsize=(10, 6))

  for step in range(args.steps):
    state = env.step(command)

    if args.dead_reckon:
      accel_body, gyro_body = np.array(state["imu_1"], dtype=float)[:2]
      dead_reckoning.step(gyro_body, accel_body, dt)

      if step % 100 == 0:
        truth = np.array(state["pose"])[:3, 3]
        drift = np.linalg.norm(truth - dead_reckoning.position)
        print(f"step {step} | drift {drift:6.2f} m")

    if sensor_name not in state:
      continue

    scan = np.asarray(state[sensor_name], dtype=float)
    waterfall.append(scan.ravel() if scan.ndim == 1 else scan.max(axis=0))
    del waterfall[:-500]

    if live and step % 10 == 0:
      draw(axes, waterfall, args.sonar, step)
      plt.pause(0.001)

  if live:
    plt.ioff()
  elif waterfall:
    draw(axes, waterfall, args.sonar, args.steps)

  if args.save and waterfall:
    args.save.mkdir(parents=True, exist_ok=True)
    out = args.save / f"{args.sonar}_waterfall.png"
    figure.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")

  plt.close(figure)


if __name__ == "__main__":
  main()
