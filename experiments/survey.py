"""Survey the seabed on a lawnmower track, logging soundings.

    nix run .#sim -- -c "python -u experiments/survey.py --out map1.csv"

Flies a boustrophedon pattern with a downward **multibeam** and writes one
``x, y, z`` row per beam that returned an echo -- the world-frame point where
that beam struck the seabed. The output feeds ``train_map.py``.

The sensor changed because the singlebeam's soundings split into two tight
populations 4.87 m apart and one beam gives no way to tell which is the seabed.
A fan does, by letting each beam be checked against its neighbours. (The "a
constant beats every bin-selection rule" figure that used to appear here was
measured over a four-metre strip where the seabed barely varies, so a constant
won by construction. It does not generalise.) The multibeam's per-beam return is
0.40 m wide.

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
from experiments.captures import write_capture
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


#: The box to cover, ``(x_min, x_max, y_min, y_max)`` in metres.
#:
#: Sized from what navigation will *look at*, not where it will go.
#: ``navigate.py`` flies x -30..-10, y -15..-5, and at 70 m altitude a 60 degree
#: fan reaches 40.2 m either side of the vehicle -- so every beam of every ping
#: it takes lands somewhere in this box. Surveying only the track would map the
#: nadir beam's footprint and leave the other 239 over unsurveyed ground.
SURVEY_BOX = (-70.0, 30.0, -55.0, 35.0)


def lawnmower(
  heading: float = 0.0,
  box: tuple[float, float, float, float] = SURVEY_BOX,
  spacing: float = 5.0,
  z: float = 0.0,
) -> list[list[float]]:
  """Boustrophedon track aligned to ``heading``, covering ``box``.

  **The vehicle travels along its own x, so the fan is across-track.** That is
  the whole point of the alignment: the fan opens along body -y and the vehicle
  never rotates, so a track running parallel to the fan sweeps the same strip of
  seabed 240 times and leaves the across-track sampling at the line spacing.
  Measured that way, a 240-beam fan bought no more coverage than one beam.

  Flying the same box at several headings is then how a patch gets seen from
  several directions, which fills the shadow a pipeline casts from any one of
  them. Each heading is a separate run -- the spawn attitude is held for the
  whole flight and nothing commands yaw.

  Args:
      heading: Track direction in degrees, counter-clockwise from world ``+x``.
          Must match the vehicle's spawn yaw or the fan is not across-track.
      box: Region to cover.
      spacing: Line spacing in metres. Generous compared to the old 2 m,
          because at 70 m altitude a 60 degree fan reaches 40 m either side and
          the redundancy is now real rather than the same strip re-measured.
      z: Depth to hold.

  Returns:
      Waypoints, ``(n, 3)`` as a list.
  """
  angle = np.radians(heading)
  forward = np.array([np.cos(angle), np.sin(angle)])
  across = np.array([-np.sin(angle), np.cos(angle)])

  x_min, x_max, y_min, y_max = box
  centre = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
  length, width = x_max - x_min, y_max - y_min

  # Extent of an axis-aligned box measured along a rotated axis: the projection
  # of both sides onto it. Covers the box at any heading without flying a square
  # big enough for the worst one.
  along = abs(length * forward[0]) + abs(width * forward[1])
  side = abs(length * across[0]) + abs(width * across[1])

  lines = np.arange(-side / 2, side / 2 + spacing / 2, spacing)
  ends = np.array([-along / 2, along / 2])

  waypoints: list[list[float]] = []
  for index, offset in enumerate(lines):
    legs = ends if index % 2 == 0 else ends[::-1]
    for reach in legs:
      point = centre + reach * forward + offset * across
      waypoints.append([float(point[0]), float(point[1]), z])

  return waypoints


def build_scenario(
  start: list[float], octree_min: float, yaw: float = 0.0
) -> dict:
  return ocean_scenario(
    "bathymetry_survey",
    start=start,
    octree_min=octree_min,
    # Held for the whole run; nothing commands yaw. This is what turns the fan
    # across-track, so it must match the heading lawnmower() was built with.
    rotation=[0.0, 0.0, yaw],
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
  parser.add_argument(
    "--yaw",
    type=float,
    default=0.0,
    help=(
      "track heading in degrees from world +x, held for the whole run. The "
      "fan opens along body +y, so this rotates the swath with the track and "
      "keeps it across-track. Fly the same box at several headings -- 0 and 90 "
      "at least -- and fit the map on all of them: a pipeline shadows the "
      "seabed behind it from one direction and not from another"
    ),
  )
  parser.add_argument(
    "--box",
    type=float,
    nargs=4,
    metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
    default=list(SURVEY_BOX),
    help="region to cover, metres",
  )
  parser.add_argument(
    "--spacing",
    type=float,
    default=20.0,
    help=(
      "line spacing, metres. Sets redundancy, not coverage: the swath is 80 m "
      "wide at survey altitude, so 20 m already gives four looks at every "
      "patch from one heading. The old 2 m came from a survey whose fan lay "
      "along the track and so covered nothing the line spacing did not"
    ),
  )
  parser.add_argument("--max-steps", type=int, default=100_000)
  parser.add_argument("--arrival-radius", type=float, default=0.5)
  parser.add_argument(
    "--profiles",
    type=Path,
    default=None,
    help=(
      "also write the raw sonar images, poses and attitudes to this .npz. "
      "analyse_multibeam.py and check_beam_validity.py read it directly, so the "
      "sonar can be scored against the octree on survey data -- over the whole "
      "box rather than one hover, which matters because the two disagree only "
      "over particular patches of seabed and a single site cannot tell a sensor "
      "defect from a feature of the ground"
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

  waypoints = lawnmower(
    heading=args.yaw, box=tuple(args.box), spacing=args.spacing
  )
  track = np.asarray(waypoints)[:, :2]
  print(
    f"{len(waypoints)} waypoints, heading {args.yaw:.0f} deg, "
    f"{args.spacing:.1f} m line spacing, "
    f"{np.linalg.norm(np.diff(track, axis=0), axis=1).sum():.0f} m of track"
  )

  env = holoocean.make(
    scenario_cfg=build_scenario(waypoints[0], args.octree_min, args.yaw),
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

      # Steer in the body frame: at a non-zero yaw a world-frame error drives
      # the vehicle sideways, and at 180 degrees it drives it away.
      next_command = follower.command(
        position, np.array(state["orient"], dtype=float)
      )
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

      # Count pings, not steps. The sonar runs at 5 Hz against a 30 Hz tick and
      # this sits below the `continue` for a tick without one, so a step-based
      # test almost never coincides with a ping and a long run prints nothing.
      if pings % 200 == 0:
        print(
          f"step {step} | waypoint {follower.index}/{len(waypoints)} "
          f"| {pings} pings | {soundings} soundings"
        )
    else:
      print(f"stopped after {args.max_steps} steps without finishing")

  print(f"Wrote {soundings} soundings to {args.out}")
  if pings:
    # Measured, this is 100% and stays there -- every beam answers, at every
    # altitude flown. So it is a regression check and not a quality one: a drop
    # means the fan stopped reaching the seabed, while 100% says nothing at all
    # about whether the ranges are right. For that, pass --profiles and score
    # the capture with check_beam_validity.py.
    fraction = live_beams / (pings * len(bearings))
    print(f"{pings} pings, {100 * fraction:.1f}% of beams returned an echo")

  if args.profiles and images:
    # Same schema check_multibeam.py writes, so check_beam_validity.py scores
    # survey data without a separate flight -- which matters because the
    # lawnmower crosses the whole box, and the sonar's disagreements with the
    # octree turned out to be local to particular patches of seabed.
    write_capture(args.profiles, images, image_poses, image_rotations, SONAR)
    print(f"Wrote {len(images)} pings to {args.profiles}")


if __name__ == "__main__":
  main()
