"""Check that the simulator launches and the sensors report.

    nix run .#sim -- -c "python experiments/smoke.py"
    nix run .#sim -- -c "python experiments/smoke.py --headless"

The smallest thing that exercises the whole simulator path: build a scenario,
start the Unreal binary, tick it, and report what each sensor returned. Use this
before debugging any of the pipeline scripts -- it separates "the binary will not
start" from "the estimator is wrong", which are otherwise easy to confuse.

Exits non-zero on failure, so it works as a gate in a shell script.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from experiments.cli import configure_sdl
from experiments.scenarios import (
  depth_sensor,
  imu_sensor,
  ocean_scenario,
  orientation_sensor,
  pose_sensor,
  singlebeam_sonar,
)

TICK_RATE_HZ = 30

# The sensors the pipeline actually uses, and the shape each should report.
EXPECTED_SHAPES = {
  "pose": (4, 4),
  "orient": (3, 3),
  "depthsensor": None,  # scalar-ish; length varies by build
  "imu_1": (2, 3),
}

SONAR_SHAPE = {"singlebeam": (256,)}


def build_scenario(
  world: str, start: list[float], sonar: bool, octree_min: float
) -> dict:
  sensors = [
    pose_sensor(),
    orientation_sensor(),
    depth_sensor(hz=TICK_RATE_HZ),
    imu_sensor("imu_1", hz=TICK_RATE_HZ),
  ]
  if sonar:
    sensors.append(singlebeam_sonar("singlebeam", hz=TICK_RATE_HZ))

  return ocean_scenario(
    "smoke",
    start=start,
    world=world,
    sensors=sensors,
    octree_min=octree_min,
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--world", default="Dam")
  parser.add_argument(
    "--start", type=float, nargs=3, default=[-20.0, -10.0, 0.0]
  )
  parser.add_argument("--steps", type=int, default=10)
  parser.add_argument(
    "--headless",
    action="store_true",
    help="run with -RenderOffScreen, for machines without a usable display",
  )
  parser.add_argument(
    "--sonar",
    action="store_true",
    help=(
      "also check the singlebeam. Off by default: sonar needs an octree, "
      "which the simulator builds on first use at several GB per minute"
    ),
  )
  parser.add_argument(
    "--octree-min",
    type=float,
    default=0.02,
    help="finest octree voxel in metres; only matters with --sonar",
  )
  return parser.parse_args()


def check_worlds_installed() -> None:
  """Fail early and usefully when the 5.2 GB package is missing."""
  path = (
    Path(os.environ.get("HOLODECKPATH", ""))
    if os.environ.get("HOLODECKPATH")
    else Path.home() / ".local/share/holoocean"
  )

  if not path.exists():
    install = (
      'nix run .#sim -- -c "python -c '
      '\\"import holoocean; holoocean.install(\'Ocean\')\\""'
    )
    raise SystemExit(
      f"No world packages under {path}.\n"
      f"Install them once (5.2 GB) with:\n  {install}"
    )


def main() -> int:
  args = parse_args()
  check_worlds_installed()
  configure_sdl(args.headless)

  print(f"HOLODECKPATH  {os.environ.get('HOLODECKPATH', '<unset>')}")
  print(f"world         {args.world}")
  print(f"viewport      {'off' if args.headless else 'on'}")
  print(f"SDL driver    {os.environ['SDL_VIDEODRIVER']}")
  print(f"sonar         {'on (builds octree)' if args.sonar else 'off'}")

  import holoocean

  print(f"holoocean     {holoocean.__version__}")
  print("\nlaunching...")

  env = holoocean.make(
    scenario_cfg=build_scenario(
      args.world, args.start, args.sonar, args.octree_min
    ),
    show_viewport=not args.headless,
  )
  print("binary started")

  state = env.tick()
  print(f"\nsensors after one tick ({len(state)} reported):")

  expected_shapes = dict(EXPECTED_SHAPES)
  if args.sonar:
    expected_shapes.update(SONAR_SHAPE)

  failures = []
  for name, expected in expected_shapes.items():
    if name not in state:
      print(f"  {name:14s} MISSING")
      failures.append(name)
      continue

    value = np.asarray(state[name])
    ok = expected is None or value.shape == expected
    note = "" if ok else f"  (expected {expected})"
    print(
      f"  {name:14s} shape {value.shape!s:10s} "
      f"finite={np.all(np.isfinite(value))}{note}"
    )
    if not ok:
      failures.append(name)

  # Tick a few more times under zero thrust to confirm it keeps running.
  command = np.zeros(8)
  for _ in range(args.steps):
    state = env.step(command)

  position = np.array(state["pose"])[:3, 3]
  print(f"\nafter {args.steps} steps at zero thrust:")
  print(f"  position      {np.array2string(position, precision=3)}")
  if args.sonar:
    peak = int(np.argmax(np.asarray(state["singlebeam"])))
    print(f"  sonar peak    bin {peak}")

  if failures:
    print(f"\nFAILED: {', '.join(failures)}")
    return 1

  print("\nOK -- simulator, sensors and stepping all work.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
