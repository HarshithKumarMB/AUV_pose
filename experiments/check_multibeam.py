"""Capture raw multibeam pings over a chosen patch of seabed.

    nix run .#sim -- -c "python -u experiments/check_multibeam.py --out mb.npz"

Flies a short track and writes every ping's intensity image with the pose and
attitude it was taken at; ``check_beam_validity.py`` scores them offline against
a ray-cast through the octree.

The two questions this was originally built for are both settled. A beam's
return is **1 bin wide, 0.10 m** -- point-like, so ``argmax`` on it is
meaningful in a way it never was for the singlebeam's ~14 bins. And the body
frame is ``nadir +z, swath +y``, recorded on
:data:`~experiments.scenarios.PROFILER_NADIR_AXIS`.

That swath sign used to read ``-y`` here, and the octree fit that produced it
could not have found the error: a mirror barely moves the residual over terrain
that is symmetric across the track. It was caught by sweeping one patch on two
perpendicular headings -- ``check_beam_geometry.py`` -- which is the test to
reach for whenever the fan's geometry is in question.

What the knobs below are for now is the question those answers raised. The sonar
agrees with the octree to 0.035 m almost everywhere and reports 4-5 m short over
compact patches, and telling a sensor defect from a gap in the reference means
re-flying the *same ground* with one thing changed at a time. Doing that with
two captures over different ground produced three wrong conclusions in a row,
which is the whole reason these are arguments rather than edits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import holoocean
import numpy as np

from experiments.captures import write_capture
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
      "coarsest octree voxel, metres. **Measured, this changes nothing**: "
      "5.12 m against 10.24 m gave bit-identical returns, not one beam moved. "
      "That is worth knowing rather than forgetting -- it says the sonar does "
      "not raycast the octree at all, which is why the octree can be missing "
      "geometry the sonar sees. Changing it strands the existing cache and "
      "costs tens of GB to rebuild"
    ),
  )
  parser.add_argument(
    "--yaw",
    type=float,
    default=0.0,
    help=(
      "starting heading in degrees, held for the run -- nothing commands yaw. "
      "Flying the same patch at 0 and 180 asks whether an effect is fixed in "
      "the sensor's frame or the world's: sensor-fixed keeps the same bearings "
      "affected while they point the opposite way, world-fixed swaps them"
    ),
  )
  # Vary the fan to separate an effect that is a property of the sensor from
  # one that is a property of the ground. Changing Azimuth while holding
  # AzimuthBins moves every beam's index without moving its bearing, so
  # whichever of the two holds still is the one the effect is indexed by.
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
      "restore holoocean's approximate atan2 for azimuth binning. Measured on "
      "the same track it costs nothing -- 1.879 vs 1.896 m MAD-std against the "
      "octree, same pattern. An earlier capture said otherwise and was two "
      "flights over different ground"
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

  write_capture(args.out, images, positions, rotations, sonar)
  print(f"Wrote {len(images)} pings of {images[0].shape} to {args.out}")


if __name__ == "__main__":
  main()
