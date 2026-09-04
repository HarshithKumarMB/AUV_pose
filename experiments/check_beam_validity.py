"""Score every multibeam beam against a ray-cast through the simulator's octree.

    python experiments/check_beam_validity.py multibeam_check.npz

Each beam's reported range is compared to where its ray actually first meets
the seabed, traced with :mod:`auv_pose.mapping.raycast`. Ray-casting is what
makes an off-nadir beam scorable at all: a beam pointed at a mound legitimately
comes back shorter than the vehicle's altitude, so the cheaper "a range cannot
be shorter than the altitude" test calls honest returns fabricated the moment
the swath is not flat -- and at the Dam site the ground under the track varies
by 0.02 m while the swath spans 4.04 m.

**What this found, and what it is for now.** The sonar agrees with the octree to
a 0.035 m MAD-std, inside its own 0.0996 m quantisation. Where it disagrees it
reports 4-5 m short, and that turned out to be **pipelines lying on the Dam
seabed that octree generation omits** -- photographed with ``capture_scene.py``.
So a disagreement here is a question about the *reference*, not a verdict on the
sensor. The sector's angular bounds, and how they move with altitude and
position, are what distinguish the two: a sensor defect holds its angular width
and follows the vehicle, an object holds its physical width and stays put.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from auv_pose.mapping.octree import cached_surface, robust_spread
from auv_pose.mapping.raycast import Heightfield, raycast
from experiments.captures import Capture
from experiments.cli import add_octree_args, octree_directory
from experiments.scenarios import PROFILER_NADIR_AXIS, PROFILER_SWATH_AXIS


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("capture", type=Path)
  add_octree_args(parser)
  parser.add_argument(
    "--pad",
    type=float,
    default=30.0,
    help=(
      "margin around the track to extract, metres. Must cover the swath, not "
      "just the track: at 70 m altitude a 60 degree fan reaches 40 m either "
      "side, and a ray leaving the extracted surface reads as no return"
    ),
  )
  parser.add_argument("--cell", type=float, default=0.10)
  parser.add_argument(
    "--step",
    type=float,
    default=0.02,
    help="ray-march step, metres; a fifth of a range bin",
  )
  parser.add_argument(
    "--tolerance",
    type=float,
    default=0.5,
    help=(
      "how far short of the ray-cast a range may fall before the beam counts "
      "as disagreeing, metres. Well above the 0.0996 m quantisation and well "
      "below the 4-5 m the affected beams are out by, so the sector's bounds "
      "do not depend on it"
    ),
  )
  parser.add_argument(
    "--pings",
    type=int,
    default=12,
    help="pings to ray-cast; the sector is stable, so a subset suffices",
  )
  return parser.parse_args()


def report_beam_width(capture: Capture) -> None:
  """How many range bins carry a single beam's return.

  A point-like return is one or two bins. The singlebeam's spans ~14, which is
  why ``argmax`` on it means little; if the profiler smeared the same way,
  treating a beam as a point sounding would be no more valid here.
  """
  live = capture.images.max(axis=1) > 0
  peak = capture.images.max(axis=1, keepdims=True)
  with np.errstate(invalid="ignore"):
    widths = (capture.images >= 0.5 * peak).sum(axis=1)[live]

  median = float(np.median(widths))
  print(
    f"beam return width: median {median:.1f} bins "
    f"({median * capture.bin_width:.2f} m), "
    f"p90 {np.percentile(widths, 90):.1f}, max {widths.max()}"
  )
  if median > 4:
    print(
      "  *** the return is not point-like. argmax on it is the same mistake "
      "the singlebeam made -- a range bin is evidence about area at that "
      "slant range, not about depth under the beam ***"
    )


def main() -> None:
  args = parse_args()
  capture = Capture.load(args.capture)
  print(capture.describe())

  surface = cached_surface(
    octree_directory(args),
    (
      capture.positions[:, 0].min() - args.pad,
      capture.positions[:, 0].max() + args.pad,
      capture.positions[:, 1].min() - args.pad,
      capture.positions[:, 1].max() + args.pad,
    ),
    cell=args.cell,
  )
  if not len(surface):
    raise SystemExit("no octree geometry under the track")
  field = Heightfield(surface, args.cell)
  print(
    f"octree surface: {len(surface)} cells, "
    f"raster {100 * field.coverage:.1f}% covered"
  )
  print()

  report_beam_width(capture)
  print()

  chosen = np.linspace(
    0, capture.n_pings - 1, min(args.pings, capture.n_pings)
  ).astype(int)
  truth = np.full((len(chosen), capture.n_beams), np.nan)
  for row, ping in enumerate(chosen):
    truth[row] = raycast(
      field,
      capture.positions[ping],
      capture.beam_directions(ping, PROFILER_NADIR_AXIS, PROFILER_SWATH_AXIS),
      capture.settings["range_max"],
      args.step,
    )

  shortfall = truth - capture.beam_ranges[chosen]  # positive: reported too near
  valid = np.isfinite(shortfall)
  print(
    f"ray-cast {len(chosen)} pings; {100 * valid.mean():.1f}% of beams have "
    "both a return and a surface to hit"
  )

  # Relief across the swath, which is why this ray-casts rather than assuming
  # flat ground. Reported because it is the assumption the cheap test made.
  under = field.elevation(capture.positions)
  hit_z = (
    capture.positions[chosen][:, None, 2]
    - truth * np.cos(capture.bearings)[None, :]
  )
  print(
    f"seabed under the track {np.ptp(under):.2f} m of relief; "
    f"across the swath {np.nanmax(hit_z) - np.nanmin(hit_z):.2f} m"
  )
  print()

  disagrees = (shortfall > args.tolerance) & valid
  per_beam = disagrees.sum(axis=0) > 0.5 * np.maximum(valid.sum(axis=0), 1)

  if not per_beam.any():
    print("every beam agrees with the octree. Nothing unmodelled in the swath.")
  else:
    index = np.flatnonzero(per_beam)
    low, high = np.degrees(capture.bearings[index[[0, -1]]])
    print("*** where the sonar and the octree disagree ***")
    print(
      f"  bearings          : {low:+.2f} to {high:+.2f} deg ({high - low:.2f} wide)"
    )
    print(
      f"  beam indices      : {index[0]} to {index[-1]} of {capture.n_beams}"
    )
    print(
      f"  contiguous        : {bool(per_beam[index[0] : index[-1] + 1].all())}"
    )
    print(
      f"  reported short by : median {np.median(shortfall[disagrees]):.2f} m, "
      f"max {shortfall[disagrees].max():.2f} m"
    )

    altitude = float(np.median(capture.positions[chosen][:, 2] - under[chosen]))
    print(
      f"  physical width    : {np.radians(high - low) * altitude:.2f} m at "
      f"{altitude:.1f} m altitude"
    )
    print(
      "\n  Re-fly at another altitude before concluding anything. Constant "
      "angular width means the fan; constant physical width and world "
      "position means an object the octree does not contain, which at Dam is "
      "a pipeline -- see capture_scene.py."
    )

  print()
  print("reported minus ray-cast, metres:")
  print(f"  {'beams':>12s} {'median':>9s} {'MAD-std':>9s} {'n':>8s}")
  for label, keep in (("agreeing", ~per_beam), ("short", per_beam)):
    selected = valid & keep[None, :]
    if not selected.any():
      continue
    median, spread = robust_spread(-shortfall[selected])
    print(
      f"  {label:>12s} {median:+9.4f} {spread:9.4f} {int(selected.sum()):8d}"
    )
  print(f"  range quantisation {capture.bin_width:.4f} m")


if __name__ == "__main__":
  main()
