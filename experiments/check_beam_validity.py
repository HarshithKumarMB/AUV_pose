"""Compare every multibeam beam against a ray-cast through the simulator's octree.

    python experiments/check_beam_validity.py multibeam_check.npz

Each beam's reported range is checked against **where its ray actually first
meets the seabed**, traced through the octree surface. That is the only test
that works over real ground: a beam pointing at a mound legitimately comes back
shorter than the vehicle's altitude, so the cheaper "a range cannot be shorter
than the altitude" check calls honest returns fabricated as soon as the swath
is not flat. Measured here, the swath at the Dam test site has 4.04 m of relief
across it while the ground *under the track* varies by 0.02 m -- so the cheap
test was being applied exactly where it does not hold.

What survives the stricter test: over a contiguous **angular** sector the sonar
reports ranges 3-5 m shorter than the ray-cast, while every beam outside it
agrees to 0.01-0.07 m. Beams do not merely mis-range there -- the true echo is
absent from the profile entirely, so no bin-selection rule recovers it.

The readout is the sector's angular bounds. Re-flying with a different fan, a
different altitude or a different patch of seabed and watching what moves is
what separates a sensor defect from a real object the octree does not contain.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from auv_pose.mapping.octree import cached_surface, robust_spread
from auv_pose.mapping.sonar import (
  azimuth_angles,
  bottom_return_ranges,
  range_bins,
)
from experiments.scenarios import PROFILER_NADIR_AXIS, PROFILER_SWATH_AXIS

DEFAULT_ROOT = Path(
  os.environ.get("HOLODECKPATH", Path.home() / "data" / "holoocean")
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("capture", type=Path)
  parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
  parser.add_argument("--version", default="2.3.0")
  parser.add_argument("--world", default="Dam")
  parser.add_argument("--cache", default="min2_max512")
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
      "as bad, metres. Well above the 0.0996 m range quantisation and well "
      "below the 3-5 m the affected beams are out by, so the sector's bounds "
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


class Heightfield:
  """The octree surface as a raster, so a ray can be marched by indexing.

  A KD-tree lookup per marched point costs tens of millions of queries for one
  capture. Rasterising once and indexing is the same answer far faster, and the
  surface is already a per-cell reduction -- it *is* a raster that happens to be
  stored as points.
  """

  def __init__(self, surface: np.ndarray, cell: float) -> None:
    self.cell = cell
    self.x0, self.y0 = surface[:, 0].min(), surface[:, 1].min()
    nx = int(np.ceil((surface[:, 0].max() - self.x0) / cell)) + 1
    ny = int(np.ceil((surface[:, 1].max() - self.y0) / cell)) + 1
    self.grid = np.full((nx, ny), -np.inf)
    ix = np.rint((surface[:, 0] - self.x0) / cell).astype(int)
    iy = np.rint((surface[:, 1] - self.y0) / cell).astype(int)
    # Maximum rather than last-wins: two points can land in one raster cell and
    # the surface is the top of the geometry.
    np.maximum.at(self.grid, (ix, iy), surface[:, 2])
    self.covered = np.isfinite(self.grid)

  def elevation(self, points: np.ndarray) -> np.ndarray:
    """Surface height under each point; ``-inf`` outside the raster."""
    ix = np.rint((points[..., 0] - self.x0) / self.cell).astype(int)
    iy = np.rint((points[..., 1] - self.y0) / self.cell).astype(int)
    inside = (
      (ix >= 0)
      & (ix < self.grid.shape[0])
      & (iy >= 0)
      & (iy < self.grid.shape[1])
    )
    out = np.full(ix.shape, -np.inf)
    out[inside] = self.grid[ix[inside], iy[inside]]
    return out


def cast(
  field: Heightfield,
  origin: np.ndarray,
  directions: np.ndarray,
  t_max: float,
  step: float,
) -> np.ndarray:
  """First range at which each ray drops to or below the surface."""
  t = np.arange(step, t_max, step)
  points = origin[None, None, :] + t[:, None, None] * directions[None, :, :]
  below = points[..., 2] <= field.elevation(points)
  hit = below.any(axis=0)
  first = np.where(hit, t[np.argmax(below, axis=0)], np.nan)
  return first


def main() -> None:
  args = parse_args()
  data = np.load(args.capture)
  images, positions = data["images"], data["positions"]
  rotations = data["rotations"]

  ranges = range_bins(
    float(data["range_min"]), float(data["range_max"]), int(data["range_bins"])
  )
  bearings = azimuth_angles(float(data["azimuth"]), int(data["azimuth_bins"]))
  reported = np.array([bottom_return_ranges(im, ranges) for im in images])
  n_pings, n_beams = reported.shape

  print(
    f"{n_pings} pings, {n_beams} beams, "
    f"azimuth {float(data['azimuth']):.1f} deg, "
    f"{float(data['range_max']):.1f} m in {int(data['range_bins'])} bins "
    f"({ranges[1] - ranges[0]:.4f} m)"
  )

  directory = (
    args.root
    / args.version
    / "worlds/Ocean/Linux/Holodeck/Octrees"
    / args.world
    / args.cache
  )
  surface = cached_surface(
    directory,
    (
      positions[:, 0].min() - args.pad,
      positions[:, 0].max() + args.pad,
      positions[:, 1].min() - args.pad,
      positions[:, 1].max() + args.pad,
    ),
    cell=args.cell,
  )
  if not len(surface):
    raise SystemExit(f"no octree geometry under the track in {directory}")
  field = Heightfield(surface, args.cell)
  print(f"octree surface: {len(surface)} cells")

  # Beam directions in the body frame, the convention measured in
  # analyse_multibeam.py and stored beside the sensor block.
  nadir = np.asarray(PROFILER_NADIR_AXIS, dtype=float)
  swath = np.asarray(PROFILER_SWATH_AXIS, dtype=float)
  body = np.cos(bearings)[:, None] * nadir + np.sin(bearings)[:, None] * swath

  chosen = np.linspace(0, n_pings - 1, min(args.pings, n_pings)).astype(int)
  truth = np.full((len(chosen), n_beams), np.nan)
  for row, ping in enumerate(chosen):
    directions = body @ rotations[ping].T
    truth[row] = cast(
      field,
      positions[ping],
      directions,
      float(data["range_max"]),
      args.step,
    )

  seen = reported[chosen]
  shortfall = truth - seen  # positive: the sonar reports too near
  valid = np.isfinite(shortfall)
  print(
    f"ray-cast {len(chosen)} pings; {100 * valid.mean():.1f}% of beams have "
    "both a return and a surface to hit"
  )

  # Relief across the swath, which is why this ray-casts rather than assuming
  # a flat seabed. Reported because it is the assumption the cheaper test made.
  under = field.elevation(positions[:, None, :])[:, 0]
  hit_z = positions[chosen][:, None, 2] - truth * np.cos(bearings)[None, :]
  print(
    f"seabed under the track {np.ptp(under):.2f} m of relief; "
    f"across the swath {np.nanmax(hit_z) - np.nanmin(hit_z):.2f} m"
  )
  print()

  bad = (shortfall > args.tolerance) & valid
  per_beam = bad.sum(axis=0) > 0.5 * np.maximum(valid.sum(axis=0), 1)

  if not per_beam.any():
    print("every beam agrees with the ray-cast. The whole fan is usable.")
  else:
    idx = np.flatnonzero(per_beam)
    lo, hi = np.degrees(bearings[idx[0]]), np.degrees(bearings[idx[-1]])
    print("*** the sector, which is the reason to run this ***")
    print(
      f"  bearings          : {lo:+.2f} to {hi:+.2f} deg  ({hi - lo:.2f} wide)"
    )
    print(f"  beam indices      : {idx[0]} to {idx[-1]} of {n_beams}")
    print(f"  contiguous        : {bool(per_beam[idx[0] : idx[-1] + 1].all())}")
    print(
      f"  shortfall         : median {np.median(shortfall[bad]):.2f} m, "
      f"max {shortfall[bad].max():.2f} m"
    )
    print(
      "\n  The index moves with the fan and the bearings do not, so quote the "
      "bearings. Watch what the bounds do against altitude and against a "
      "different patch of seabed: a sensor defect follows the vehicle, a real "
      "object the octree lacks stays where it is."
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

  quantisation = float(ranges[1] - ranges[0])
  print(f"  range quantisation {quantisation:.4f} m")


if __name__ == "__main__":
  main()
