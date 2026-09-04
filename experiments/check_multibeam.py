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
from experiments.guidance import WaypointFollower
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


def track(start: list[float], sweep: float) -> list[list[float]]:
  """A short track that moves across the swath, not just along it.

  Two legs offset in y, joined by a leg along x. Flying only along x cannot
  decorrelate beam index from terrain however long it runs, because the
  across-track offset a beam sees never changes.
  """
  x, y, z = start
  return [
    [x, y, z],
    [x, y + sweep, z],
    [x + 8.0, y + sweep, z],
    [x + 8.0, y, z],
  ]


def build_scenario(
  start: list[float],
  octree_min: float,
  use_approx: bool = False,
  sonar: dict | None = None,
  yaw: float = 0.0,
  octree_max: float = 5.0,
) -> dict:
  return ocean_scenario(
    "multibeam_check",
    start=start,
    octree_min=octree_min,
    octree_max=octree_max,
    rotation=[0.0, 0.0, yaw],
    sensors=[
      pose_sensor(),
      orientation_sensor(),
      profiling_sonar(
        "multibeam",
        hz=SONAR_HZ,
        use_approx=use_approx,
        **(sonar if sonar is not None else SONAR),
      ),
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
  parser.add_argument("--max-steps", type=int, default=20_000)
  parser.add_argument("--arrival-radius", type=float, default=1.0)
  parser.add_argument(
    "--sweep",
    type=float,
    default=18.0,
    help=(
      "how far the track moves ACROSS the swath, metres. This is the whole "
      "point of the flight: the fan opens along body y, so flying along x "
      "leaves every beam staring at the same strip of seabed and beam index "
      "cannot be told apart from terrain. Moving in y makes a given patch be "
      "seen by different beams, which is what separates a misaimed beam from "
      "a feature on the ground"
    ),
  )
  parser.add_argument("--octree-min", type=float, default=0.02)
  parser.add_argument(
    "--octree-max",
    type=float,
    default=5.0,
    help=(
      "coarsest octree voxel, metres. The phantom returns sit 0-5 m above the "
      "true surface against a default of 5.0, which is why this is a knob "
      "worth turning: if the sonar is resolving against coarse nodes rather "
      "than the leaves inside them, the error should scale with this"
    ),
  )
  parser.add_argument(
    "--yaw",
    type=float,
    default=0.0,
    help=(
      "starting heading in degrees, held for the run. Flying the same patch at "
      "0 and 180 asks whether a sensor defect is fixed in the sensor's frame "
      "or the world's: sensor-fixed keeps the same beam indices bad while they "
      "point the opposite way, world-fixed moves which indices are bad"
    ),
  )
  # Overrides for isolating the phantom-return boundary. Beams past ~106 return
  # ranges shorter than the vehicle's altitude over flat ground, which no
  # geometry can produce; varying the fan tells index-based from angle-based.
  for key, value in SONAR.items():
    parser.add_argument(
      f"--{key.replace('_', '-')}",
      type=type(value),
      default=value,
      help=f"sonar {key}, default {value}",
    )
  parser.add_argument(
    "--use-approx",
    action="store_true",
    help=(
      "restore holoocean's approximate atan2 for azimuth binning. Only for "
      "measuring what it costs: flying the same track with and without is the "
      "controlled comparison, since two captures over different ground are not"
    ),
  )
  parser.add_argument("--force", action="store_true")
  parser.add_argument("--headless", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  refuse_overwrite(args.out, args.force)
  configure_sdl(args.headless)

  sonar = {k: getattr(args, k) for k in SONAR}
  if sonar != SONAR:
    print(
      f"sonar overrides: { {k: v for k, v in sonar.items() if v != SONAR[k]} }"
    )

  env = holoocean.make(
    scenario_cfg=build_scenario(
      args.start,
      args.octree_min,
      args.use_approx,
      sonar,
      args.yaw,
      args.octree_max,
    ),
    show_viewport=not args.headless,
  )

  images: list[np.ndarray] = []
  positions: list[np.ndarray] = []
  rotations: list[np.ndarray] = []

  follower = WaypointFollower(
    track(args.start, args.sweep), args.arrival_radius
  )
  command = np.zeros(8)
  for step in range(args.max_steps):
    state = env.step(command)
    position = np.array(state["pose"])[:3, 3]

    next_command = follower.command(position)
    if next_command is None:
      if follower.finished:
        print("track complete")
        break
      continue
    command = next_command

    if "multibeam" not in state:
      continue

    images.append(np.asarray(state["multibeam"], dtype=float).copy())
    positions.append(position.copy())
    rotations.append(np.array(state["orient"], dtype=float).copy())

    if len(images) % 20 == 0:
      print(
        f"step {step} | waypoint {follower.index}/4 | "
        f"{len(images)}/{args.pings} pings"
      )
    if len(images) >= args.pings:
      break

  if not images:
    raise SystemExit("no sonar returns; check the sensor name and Hz")

  np.savez_compressed(
    args.out,
    images=np.array(images),
    positions=np.array(positions),
    rotations=np.array(rotations),
    **sonar,
  )
  print(f"Wrote {len(images)} pings of {images[0].shape} to {args.out}")


if __name__ == "__main__":
  main()
