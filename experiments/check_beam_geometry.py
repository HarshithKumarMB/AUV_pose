"""Hold one patch of seabed still and ask the fan to describe it twice.

    nix run .#sim -- -c "python -u experiments/check_beam_geometry.py --out geom.npz"

The four-heading survey disagrees with itself. Soundings that land in the same
0.25 m cell from *different survey lines* differ by a median of 3.4 m, while
soundings from the **same** line agree to 0.000 m. Nadir beams match the octree
(median -0.05 m); off-nadir beams do not (-3.5 m). So the error is angular, and
something about how an off-nadir beam is turned into a sounding is wrong.

What this captures is the one experiment that can say *which* thing, without
appealing to the octree at all.

**The crossing test.** Sweep a short track over a patch, then sweep the *same*
patch again on a perpendicular heading. The seabed has not changed, so the two
reconstructions must agree wherever they cross. They only do if the fan's
body-frame geometry is right: a swath axis with the wrong sign puts every
sounding on the wrong side of the track, which a second heading exposes.

**Not a 180 degree yaw, and this is the whole subtlety.** The obvious version of
this test -- hover, record, yaw 180, record again -- cannot work, and an earlier
version of this file did exactly that and reported a confident PASS. Mirroring
the swath maps the heading-0 fan *onto* the heading-180 fan: for a beam at +20
degrees the true geometry gives world direction ``[0, 0.342, -0.94]`` at heading
0 and ``[0, -0.342, -0.94]`` at heading 180, and the mirrored geometry gives
exactly those two the other way round. A mirror is therefore self-consistent
under a 180 degree yaw and invisible to it. Both geometries scored a median
disagreement of 0.000 m, which is what tipped it off.

Perpendicular headings are not self-consistent that way. They also need a
*track* rather than a hover: one ping's fan is a line on the seabed, because the
along-track beamwidth is 1 degree, and two lines at 90 degrees meet at a single
point. Two swept patches overlap properly.

This needs no reference surface, which matters -- ``check_multibeam.py``'s
docstring records that comparing captures over *different ground* "produced
three wrong conclusions in a row". Here the ground is identical by construction
and the only thing that changes is the vehicle's heading.

**The altitude test.** Repeat at a second altitude. A defect in the fan holds
its *angular* width; an object the octree omits holds its *physical* width. That
is ``check_beam_validity.py``'s own advice, and it separates "the geometry is
wrong" from "the reference is incomplete".

The analysis, and the assertions that make it a test rather than a plot, live in
:mod:`experiments.check_beam_geometry` below ``analyse``. Run it on the capture
with ``--analyse geom.npz`` and no simulator.
"""

from __future__ import annotations

import argparse
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

from auv_pose.mapping.sonar import (
  azimuth_angles,
  bottom_return_ranges,
  range_bins,
  seabed_points,
)
from experiments.scenarios import PROFILER_NADIR_AXIS, PROFILER_SWATH_AXIS

#: Where to hover. Chosen inside the surveyed box and away from its edges, so
#: the whole fan lands on seabed the survey also covered.
DEFAULT_PATCH = (-20.0, -10.0)

#: Headings to sweep at, degrees. They must **not** differ by 180: see the
#: module docstring for why a half turn cannot see a mirrored swath.
DEFAULT_HEADINGS = (0.0, 90.0)

#: Vehicle depths to record at, metres, z-up. Two altitudes over one patch is
#: what separates a fan defect from a missing object.
DEFAULT_DEPTHS = (-0.5, -25.0)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, default=Path("beam_geometry.npz"))
  parser.add_argument(
    "--analyse",
    type=Path,
    default=None,
    help="score an existing capture and exit; needs no simulator",
  )
  parser.add_argument(
    "--patch",
    type=float,
    nargs=2,
    default=DEFAULT_PATCH,
    metavar=("X", "Y"),
    help="where to hover",
  )
  parser.add_argument(
    "--headings",
    type=float,
    nargs="+",
    default=list(DEFAULT_HEADINGS),
    help=(
      "yaw angles to sweep at, degrees. Perpendicular is the test; a pair "
      "differing by 180 cannot detect a mirrored swath"
    ),
  )
  parser.add_argument(
    "--depths",
    type=float,
    nargs="+",
    default=list(DEFAULT_DEPTHS),
    help="vehicle depths to record at, metres, z-up",
  )
  parser.add_argument(
    "--pings", type=int, default=60, help="pings, and so steps, per track"
  )
  parser.add_argument(
    "--spacing",
    type=float,
    default=0.5,
    help="along-track step between pings, metres",
  )
  parser.add_argument(
    "--hold",
    type=int,
    default=20,
    help="ticks to settle after each step within a track",
  )
  parser.add_argument(
    "--settle",
    type=int,
    default=400,
    help=(
      "ticks to hold each pose before recording. The fan must be still: a "
      "vehicle drifting through the capture reintroduces exactly the "
      "different-ground confound this test exists to remove"
    ),
  )
  parser.add_argument(
    "--tolerance",
    type=float,
    default=0.5,
    help="how far the two headings may disagree before the assertion fails, m",
  )
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--force", action="store_true")
  return parser.parse_args()


def soundings_for(capture: dict, index: int, swath, nadir) -> np.ndarray:
  """Reconstruct one pose's fan under a given body-frame geometry."""
  return seabed_points(
    capture["positions"][index],
    capture["rotations"][index],
    capture["beam_ranges"][index],
    capture["bearings"],
    swath_axis=swath,
    nadir_axis=nadir,
  )


def agreement(
  left: np.ndarray, right: np.ndarray, radius: float = 0.6
) -> tuple[float, int]:
  """How far apart two reconstructions of the same ground are.

  Each sounding on the left is matched to its nearest neighbour on the right
  in the horizontal plane, and the elevations compared.

  Nearest-neighbour matching rather than a shared grid, because a fan is a
  *line* on the seabed, not a patch: the along-track beamwidth is 1 degree. The
  two headings put their lines a couple of decimetres apart -- yawing the
  vehicle swings the sonar to the other side of its own origin -- and on a grid
  those two lines fall in adjacent cells and share nothing at all. That is not
  a disagreement about the seabed, it is a disagreement about bookkeeping, and
  an earlier version of this function reported it as zero overlap.

  :param left: Soundings, ``(n, 3)``.
  :param right: Soundings, ``(m, 3)``.
  :param radius: How near a match must be, horizontally, in metres. Wide
      enough to bridge the along-track offset, narrow enough that terrain does
      not dominate the elevation difference.
  :return: ``(median absolute z difference, number of matched soundings)``.
  """
  from scipy.spatial import KDTree

  left = left[np.isfinite(left).all(axis=1)]
  right = right[np.isfinite(right).all(axis=1)]
  if not len(left) or not len(right):
    return float("nan"), 0

  distance, nearest = KDTree(right[:, :2]).query(left[:, :2])
  matched = distance <= radius
  if not matched.any():
    return float("nan"), 0

  difference = np.abs(left[matched, 2] - right[nearest[matched], 2])
  return float(np.median(difference)), int(matched.sum())


def _groups(capture: dict) -> dict:
  """Pings grouped by the pose they were taken at.

  Every pose contributes many pings. Comparing two *pings* rather than two
  *poses* is a mistake that cannot fail: it scores identical data against
  itself and reports perfect agreement. This function exists so the yaw test
  cannot accidentally do that, and :func:`analyse` asserts that the two sides
  of every comparison really are different headings.
  """
  grouped: dict[tuple[float, float], list[int]] = {}
  for index, (depth, heading) in enumerate(
    zip(capture["depths"], capture["headings"])
  ):
    grouped.setdefault((float(depth), float(heading)), []).append(index)
  return grouped


def _fan(capture: dict, indices, swath, nadir) -> np.ndarray:
  """Every sounding from a pose, under a given body-frame geometry."""
  return np.concatenate(
    [soundings_for(capture, int(i), swath, nadir) for i in indices]
  )


def analyse(path: Path, tolerance: float) -> int:
  """Score a capture and assert what it implies. Returns a process exit code."""
  raw = np.load(path, allow_pickle=False)
  capture = {key: raw[key] for key in raw.files}

  grouped = _groups(capture)
  failures: list[str] = []

  print(f"{len(capture['headings'])} pings in {len(grouped)} poses, {path}")
  for (depth, heading), indices in sorted(grouped.items()):
    position = capture["positions"][indices[0]]
    live = int(np.isfinite(capture["beam_ranges"][indices]).sum()) / len(
      indices
    )
    print(
      f"  depth {depth:7.2f} m  heading {heading:6.1f} deg  "
      f"at ({position[0]:7.2f}, {position[1]:7.2f}, {position[2]:7.2f})  "
      f"{len(indices):3d} pings, {live:5.1f} live beams each"
    )

  candidates = {
    "as configured": (PROFILER_SWATH_AXIS, PROFILER_NADIR_AXIS),
    "swath sign flipped": (
      tuple(-np.asarray(PROFILER_SWATH_AXIS)),
      PROFILER_NADIR_AXIS,
    ),
  }

  # -- the yaw test -------------------------------------------------------
  print("\n-- crossing test: the same ground, swept on two headings --")
  print("   A body-frame geometry error shows here and nowhere else.")

  compared = 0
  for depth in sorted({d for d, _ in grouped}):
    headings = sorted(h for d, h in grouped if d == depth)
    if len(headings) < 2:
      failures.append(
        f"depth {depth:.2f}: only heading {headings} recorded, so the yaw "
        "test cannot run. Pass at least two --headings"
      )
      continue

    for left, right in pairwise(headings):
      # The whole point of the test. Without this the comparison can silently
      # become a pose against itself, which always agrees perfectly.
      assert left != right, (
        f"the crossing test needs two headings, got {left} twice"
      )
      assert abs((left - right) % 360.0 - 180.0) > 1.0, (
        f"headings {left} and {right} differ by 180 degrees, which cannot "
        "detect a mirrored swath -- see the module docstring"
      )
      compared += 1
      print(f"\n  depth {depth:.2f} m, heading {left:.0f} vs {right:.0f}")

      for name, (swath, nadir) in candidates.items():
        spread, cells = agreement(
          _fan(capture, grouped[(depth, left)], swath, nadir),
          _fan(capture, grouped[(depth, right)], swath, nadir),
        )
        verdict = "" if spread <= tolerance else "   <-- disagrees"
        print(
          f"    {name:22} median |dz| {spread:7.3f} m over {cells:5d} "
          f"shared cells{verdict}"
        )

      spread, cells = agreement(
        _fan(capture, grouped[(depth, left)], *candidates["as configured"]),
        _fan(capture, grouped[(depth, right)], *candidates["as configured"]),
      )
      if cells < 20:
        failures.append(
          f"depth {depth:.2f}, {left:.0f} vs {right:.0f}: only {cells} shared "
          "cells, too few to conclude anything. The two headings barely "
          "overlap -- check the vehicle held station"
        )
      elif not np.isfinite(spread):
        failures.append(
          f"depth {depth:.2f}, {left:.0f} vs {right:.0f}: no shared ground"
        )
      elif spread > tolerance:
        failures.append(
          f"depth {depth:.2f}, heading {left:.0f} vs {right:.0f}: the "
          f"configured geometry reconstructs the same seabed {spread:.3f} m "
          f"apart, over tolerance {tolerance} m. The fan's body-frame axes "
          "are wrong, not the sonar"
        )

  if compared == 0:
    failures.append(
      "no pair of headings was comparable, so nothing here was tested"
    )

  # -- the altitude test --------------------------------------------------
  print("\n-- altitude test: does a disagreement hold angle, or size? --")
  print("   Angular width fixed across altitude means the fan. Physical width")
  print("   fixed means an object the reference omits.")

  bearings = np.degrees(capture["bearings"])
  swath, nadir = candidates["as configured"]
  for heading in sorted({h for _, h in grouped}):
    depths = sorted(d for d, h in grouped if h == heading)
    if len(depths) < 2:
      continue
    print(f"\n  heading {heading:.0f} deg")
    for depth in depths:
      indices = grouped[(depth, heading)]
      fan = _fan(capture, indices, swath, nadir)
      ranges = capture["beam_ranges"][indices[0]]
      finite = np.isfinite(ranges)
      altitude = float(capture["positions"][indices[0]][2]) - float(
        np.nanmedian(fan[:, 2])
      )
      # Over flat ground a beam's range is altitude / cos(bearing); the excess
      # is what an angular defect would inflate. The seabed is not flat, so
      # read the *difference between altitudes*, not either number alone.
      excess = ranges - altitude / np.cos(capture["bearings"])
      outer = finite & (np.abs(bearings) > 15.0)
      inner = finite & (np.abs(bearings) <= 5.0)
      print(
        f"    depth {depth:7.2f}  altitude {altitude:6.2f} m   "
        f"excess range: nadir {np.median(excess[inner]):+6.2f} m, "
        f"outer {np.median(excess[outer]):+6.2f} m"
      )

  print()
  if failures:
    for failure in failures:
      print(f"FAIL  {failure}")
    return 1

  print("PASS  the configured geometry reconstructs the same seabed from")
  print(f"      every heading recorded ({compared} comparisons), within")
  print(f"      {tolerance} m.")
  return 0


def capture_poses(args) -> dict:
  """Sweep the patch at each heading and depth. Needs the simulator.

  The vehicle is teleported along each track rather than flown. Station-keeping
  error is the one confound this test cannot tolerate -- ``check_multibeam.py``
  records that comparing captures over *different ground* "produced three wrong
  conclusions in a row" -- and the vehicle's controller cannot hold a line to
  the centimetre.
  """
  import holoocean

  from experiments.scenarios import (
    imu_sensor,
    ocean_scenario,
    orientation_sensor,
    pose_sensor,
    profiling_sonar,
  )

  sonar = profiling_sonar("multibeam")
  settings = sonar["configuration"]
  scenario = ocean_scenario(
    name="beam-geometry",
    start=[args.patch[0], args.patch[1], args.depths[0]],
    sensors=[pose_sensor(), orientation_sensor(), imu_sensor(), sonar],
  )

  ranges = range_bins(
    settings["RangeMin"], settings["RangeMax"], settings["RangeBins"]
  )
  bearings = azimuth_angles(settings["Azimuth"], settings["AzimuthBins"])

  positions, rotations, beam_ranges, headings, depths = [], [], [], [], []

  with holoocean.make(
    scenario_cfg=scenario, show_viewport=not args.headless
  ) as env:
    for depth in args.depths:
      for heading in args.headings:
        radians = np.radians(heading)
        along = np.array([np.cos(radians), np.sin(radians)])
        offsets = (
          np.arange(args.pings) - (args.pings - 1) / 2.0
        ) * args.spacing

        print(
          f"sweeping {args.pings * args.spacing:.1f} m at {heading} deg, "
          f"depth {depth}"
        )
        for step, offset in enumerate(offsets):
          target = np.array(
            [
              args.patch[0] + along[0] * offset,
              args.patch[1] + along[1] * offset,
              depth,
            ]
          )
          env.agents["rov"].teleport(target, [0.0, 0.0, float(heading)])
          for _ in range(args.settle if step == 0 else args.hold):
            env.tick()

          state = env.tick()
          while "multibeam" not in state:
            state = env.tick()

          image = np.asarray(state["multibeam"], dtype=float)
          positions.append(np.array(state["pose"])[:3, 3])
          rotations.append(np.array(state["orient"], dtype=float))
          beam_ranges.append(bottom_return_ranges(image, ranges))
          headings.append(heading)
          depths.append(depth)

  return {
    "positions": np.asarray(positions),
    "rotations": np.asarray(rotations),
    "beam_ranges": np.asarray(beam_ranges),
    "bearings": bearings,
    "ranges": ranges,
    "headings": np.asarray(headings),
    "depths": np.asarray(depths),
  }


def main() -> None:
  args = parse_args()

  if args.analyse is not None:
    raise SystemExit(analyse(args.analyse, args.tolerance))

  if args.out.exists() and not args.force:
    raise SystemExit(f"{args.out} exists; pass --force to overwrite")

  capture = capture_poses(args)
  np.savez_compressed(args.out, **capture)
  print(f"Wrote {args.out}")

  raise SystemExit(analyse(args.out, args.tolerance))


if __name__ == "__main__":
  sys.exit(main())
