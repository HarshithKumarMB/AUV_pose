"""Check the multibeam's geometry against the simulator's own octree.

    nix run .#sim -- -c "python -u experiments/check_multibeam.py --out mb.npz"

Two questions have to be answered before a survey is flown with this sensor, and
both are settled offline against :mod:`auv_pose.mapping.octree`:

1. **Is a beam's return narrow enough for ``argmax`` to mean anything?** The
   singlebeam's is not -- it spans ~14 range bins, and no bin-selection rule
   recovers the depth beneath the vehicle from it. If the profiler smears the
   same way, picking a bin is no more valid here than it was there.
2. **Is the beam geometry right?** :func:`~auv_pose.mapping.sonar.seabed_points`
   takes its ``nadir_axis`` and ``swath_axis`` in the body frame, but the sensor
   carries a mount ``rotation`` that nothing applies. A wrong ``swath_axis``
   sign mirrors the whole swath across the track and still looks entirely
   plausible in isolation -- against a known surface it does not.

Writes the raw images and poses; ``analyse_multibeam.py`` scores them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import holoocean
import numpy as np

from experiments.cli import configure_sdl, refuse_overwrite
from experiments.scenarios import (
  ocean_scenario,
  orientation_sensor,
  pose_sensor,
  profiling_sonar,
)

TICK_RATE_HZ = 30
SONAR_HZ = 5
SONAR = {
  "range_min": 0.5,
  "range_max": 100.0,
  "range_bins": 1000,
  "azimuth": 60.0,
  "azimuth_bins": 240,
  "elevation": 1.0,
}


def build_scenario(start: list[float], octree_min: float) -> dict:
  return ocean_scenario(
    "multibeam_check",
    start=start,
    octree_min=octree_min,
    sensors=[
      pose_sensor(),
      orientation_sensor(),
      profiling_sonar("multibeam", hz=SONAR_HZ, **SONAR),
    ],
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, default=Path("multibeam_check.npz"))
  parser.add_argument(
    "--start", type=float, nargs=3, default=[-15.0, -10.0, 0.0]
  )
  parser.add_argument(
    "--pings", type=int, default=40, help="images to capture before stopping"
  )
  parser.add_argument("--max-steps", type=int, default=3000)
  parser.add_argument(
    "--command",
    type=float,
    nargs=8,
    default=[0, 0, 0, 0, 8, 8, 0, 0],
    help=(
      "constant thruster command. Some translation is wanted: a stationary "
      "check cannot tell a mirrored swath axis from a correct one"
    ),
  )
  parser.add_argument("--octree-min", type=float, default=0.02)
  parser.add_argument("--force", action="store_true")
  parser.add_argument("--headless", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  refuse_overwrite(args.out, args.force)
  configure_sdl(args.headless)

  env = holoocean.make(
    scenario_cfg=build_scenario(args.start, args.octree_min),
    show_viewport=not args.headless,
  )

  images: list[np.ndarray] = []
  positions: list[np.ndarray] = []
  rotations: list[np.ndarray] = []

  command = np.array(args.command, dtype=float)
  for step in range(args.max_steps):
    state = env.step(command)
    if "multibeam" not in state:
      continue

    images.append(np.asarray(state["multibeam"], dtype=float).copy())
    positions.append(np.array(state["pose"])[:3, 3].copy())
    rotations.append(np.array(state["orient"], dtype=float).copy())

    if len(images) % 10 == 0:
      print(f"step {step} | {len(images)}/{args.pings} pings")
    if len(images) >= args.pings:
      break

  if not images:
    raise SystemExit("no sonar returns; check the sensor name and Hz")

  np.savez_compressed(
    args.out,
    images=np.array(images),
    positions=np.array(positions),
    rotations=np.array(rotations),
    **SONAR,
  )
  print(f"Wrote {len(images)} pings of {images[0].shape} to {args.out}")


if __name__ == "__main__":
  main()
