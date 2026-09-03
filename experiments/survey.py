"""Survey the seabed on a lawnmower track, logging soundings.

    nix run .#sim -- -c "python -u experiments/survey.py --out map1.csv"

Flies a boustrophedon pattern with a downward **multibeam** and writes one
``x, y, z`` row per beam that returned an echo -- the world-frame point where
that beam struck the seabed. The output feeds ``train_map.py``.

The sensor changed because the singlebeam could not measure depth. Scored
against the simulator's own octree, its strongest-return range is biased 4.17 m
against nadir truth and a constant beats every bin-selection rule (0.888 m rms
against 4.184 m): a 10 degree cone at survey altitude is a 12 m footprint, so a
range bin is evidence about seabed *area* at that slant range, not about the
depth under the vehicle. The multibeam's per-beam return is 0.40 m wide.

Uses the ground-truth pose *and attitude* to place each sounding: every beam but
nadir lands ``range * sin(bearing)`` from the vehicle, so a survey that records
soundings at the vehicle's own ``(x, y)`` misplaces all of them. This builds the
reference map that navigation is later corrected against, so it must not itself
be drifting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import holoocean
import numpy as np

from auv_pose.io.logs import CsvLogger
from auv_pose.io.soundings import SOUNDING_COLUMNS
from auv_pose.mapping.sonar import (
  azimuth_angles,
  bottom_return_ranges,
  range_bins,
  seabed_points,
)
from experiments.cli import configure_sdl, refuse_overwrite
from experiments.guidance import WaypointFollower
from experiments.scenarios import (
  PROFILER_NADIR_AXIS,
  PROFILER_SWATH_AXIS,
  ocean_scenario,
  orientation_sensor,
  pose_sensor,
  profiling_sonar,
)

TICK_RATE_HZ = 30

# 5 Hz, not the tick rate: raycasting a 240-beam fan is the expensive part of a
# tick, and at survey speed 5 Hz still oversamples the along-track footprint.
SONAR_HZ = 5
SONAR = {
  "range_min": 0.5,
  "range_max": 100.0,
  "range_bins": 1000,
  "azimuth": 60.0,
  "azimuth_bins": 240,
  "elevation": 1.0,
}


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


def build_scenario(start: list[float], octree_min: float) -> dict:
  return ocean_scenario(
    "bathymetry_survey",
    start=start,
    octree_min=octree_min,
    sensors=[
      pose_sensor(),
      # Seabed points need attitude, not just position. The socket is
      # load-bearing -- see orientation_sensor's docstring.
      orientation_sensor(),
      profiling_sonar("multibeam", hz=SONAR_HZ, **SONAR),
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
    "--profiles",
    type=Path,
    default=None,
    help=(
      "also write the raw sonar images, poses and attitudes to this .npz. "
      "analyse_multibeam.py and fit_beam_geometry.py read it directly, so beam "
      "geometry can be checked on survey data -- where the lawnmower's y sweep "
      "means a patch of seabed is seen by different beams -- rather than on a "
      "dedicated straight-line flight, where beam index and terrain are "
      "confounded and no fit can separate them"
    ),
  )
  parser.add_argument(
    "--profile-steps",
    type=int,
    default=400,
    help="stop recording pings after this many; diagnosis, not a survey",
  )
  parser.add_argument(
    "--octree-min",
    type=float,
    default=0.02,
    help=(
      "finest octree voxel in metres. The multibeam's range bins are 0.0995 m, "
      "so holoocean's default of 0.02 is 5x finer -- about the right ratio. "
      "Budget ~30 GB and three minutes of octree generation at startup, after "
      "which it stops: InitOctreeRange builds the neighbourhood once and the "
      "run writes nothing more. Raising this is not a free speedup, because "
      "ShadowEpsilon defaults to 4*OctreeMin, so it changes sonar returns and "
      "makes surveys inconsistent with maps built at another value"
    ),
  )
  parser.add_argument(
    "--headless",
    action="store_true",
    help="run with -RenderOffScreen, for machines without a usable display",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  refuse_overwrite(args.out, args.force)
  configure_sdl(args.headless)

  waypoints = lawnmower()
  print(f"{len(waypoints)} waypoints")

  env = holoocean.make(
    scenario_cfg=build_scenario(waypoints[0], args.octree_min),
    show_viewport=not args.headless,
  )
  ranges = range_bins(
    SONAR["range_min"], SONAR["range_max"], SONAR["range_bins"]
  )
  bearings = azimuth_angles(SONAR["azimuth"], SONAR["azimuth_bins"])

  follower = WaypointFollower(waypoints, args.arrival_radius)
  command = np.zeros(8)
  soundings = 0
  pings = 0
  live_beams = 0
  images: list[np.ndarray] = []
  image_poses: list[np.ndarray] = []
  image_rotations: list[np.ndarray] = []

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

      # The sonar runs slower than the tick, so most steps carry no image.
      if "multibeam" not in state:
        continue

      image = np.asarray(state["multibeam"], dtype=float)
      rotation = np.array(state["orient"], dtype=float)
      if args.profiles and len(images) < args.profile_steps:
        images.append(image.copy())
        image_poses.append(position.copy())
        image_rotations.append(rotation.copy())

      beam_ranges = bottom_return_ranges(image, ranges)
      points = seabed_points(
        position,
        rotation,
        beam_ranges,
        bearings,
        swath_axis=PROFILER_SWATH_AXIS,
        nadir_axis=PROFILER_NADIR_AXIS,
      )

      pings += 1
      finite = np.isfinite(points).all(axis=1)
      live_beams += int(finite.sum())
      for x, y, z in points[finite]:
        log.write(x=x, y=y, z=z)
      soundings += int(finite.sum())

      if step % 1000 == 0:
        print(
          f"step {step} | waypoint {follower.index}/{len(waypoints)} "
          f"| {soundings} soundings"
        )
    else:
      print(f"stopped after {args.max_steps} steps without finishing")

  print(f"Wrote {soundings} soundings to {args.out}")
  if pings:
    # A geometry or range regression shows up here first: beams past the edge
    # of the swath legitimately return nothing, but a sharp drop means the
    # sensor stopped reaching the seabed.
    fraction = live_beams / (pings * len(bearings))
    print(f"{pings} pings, {100 * fraction:.1f}% of beams returned an echo")

  if args.profiles and images:
    # Same layout check_multibeam.py writes, so analyse_multibeam.py and
    # fit_beam_geometry.py read survey data without a separate flight -- which
    # matters because the lawnmower sweeps across the swath and a dedicated
    # straight-line check does not.
    np.savez_compressed(
      args.profiles,
      images=np.array(images),
      positions=np.array(image_poses),
      rotations=np.array(image_rotations),
      **SONAR,
    )
    print(f"Wrote {len(images)} pings to {args.profiles}")


if __name__ == "__main__":
  main()
