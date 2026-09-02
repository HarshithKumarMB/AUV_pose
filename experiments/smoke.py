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
  "singlebeam": (256,),
}


def build_scenario(world: str, start: list[float]) -> dict:
  return ocean_scenario(
    "smoke",
    start=start,
    world=world,
    sensors=[
      pose_sensor(),
      orientation_sensor(),
      depth_sensor(hz=TICK_RATE_HZ),
      imu_sensor("imu_1", hz=TICK_RATE_HZ),
      singlebeam_sonar("singlebeam", hz=TICK_RATE_HZ),
    ],
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

  print(f"HOLODECKPATH  {os.environ.get('HOLODECKPATH', '<unset>')}")
  print(f"world         {args.world}")
  print(f"viewport      {'off' if args.headless else 'on'}")

  import holoocean

  print(f"holoocean     {holoocean.__version__}")
  print("\nlaunching...")

  env = holoocean.make(
    scenario_cfg=build_scenario(args.world, args.start),
    show_viewport=not args.headless,
  )
  print("binary started")

  state = env.tick()
  print(f"\nsensors after one tick ({len(state)} reported):")

  failures = []
  for name, expected in EXPECTED_SHAPES.items():
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
  print(
    f"  sonar range   {np.argmax(np.asarray(state['singlebeam']))} (peak bin)"
  )

  if failures:
    print(f"\nFAILED: {', '.join(failures)}")
    return 1

  print("\nOK -- simulator, sensors and stepping all work.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
