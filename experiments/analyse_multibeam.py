"""Score captured multibeam pings against the simulator's octree.

    python experiments/analyse_multibeam.py multibeam_check.npz

Answers the two questions ``check_multibeam.py`` was flown for:

**Beam width.** How many range bins does a single beam's return span? The
singlebeam's spans ~14, which is why ``argmax`` on it is meaningless. If the
profiler's spans one or two, treating a beam as a point sounding is fair.

**Geometry.** Reconstructs soundings with
:func:`~auv_pose.mapping.sonar.seabed_points` under each candidate
``(nadir_axis, swath_axis)`` convention and scores them against the octree
surface. The right convention puts the swath on the seabed; a mirrored one
leaves residual that grows with beam index, which is reported separately because
a mirrored swath can still have a plausible-looking median.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from auv_pose.mapping.octree import load_surface
from auv_pose.mapping.sonar import (
  azimuth_angles,
  bottom_return_ranges,
  range_bins,
  seabed_points,
)

DEFAULT_ROOT = Path(
  os.environ.get("HOLODECKPATH", Path.home() / "data" / "holoocean")
)

#: Body-frame conventions to try. The sensor's mount ``rotation`` of [0, -90, 0]
#: is not applied by ``seabed_points``, so which body axis the fan opens along
#: -- and its sign -- is a guess until measured.
CONVENTIONS = {
  "nadir +z, swath +y": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
  "nadir +z, swath -y": ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
  "nadir +z, swath +x": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
  "nadir +z, swath -x": ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)),
  "nadir +x, swath +y": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
  "nadir +x, swath -y": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
  "nadir -z, swath +y": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("capture", type=Path)
  parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
  parser.add_argument("--version", default="2.3.0")
  parser.add_argument("--world", default="Dam")
  parser.add_argument("--cache", default="min2_max512")
  return parser.parse_args()


def report_beam_width(images: np.ndarray, ranges: np.ndarray) -> None:
  """How many range bins carry a single beam's return."""
  bin_width = float(ranges[1] - ranges[0])
  live = images.max(axis=1) > 0
  peak = images.max(axis=1, keepdims=True)

  # Bins within half the peak, per beam. A point-like return is one or two.
  with np.errstate(invalid="ignore"):
    wide = (images >= 0.5 * peak).sum(axis=1)
  widths = wide[live]

  print("beam return width, bins above half the beam's peak:")
  print(
    f"  median {np.median(widths):.1f} bins "
    f"({np.median(widths) * bin_width:.2f} m), "
    f"p90 {np.percentile(widths, 90):.1f}, max {widths.max()}"
  )
  print(f"  bin width {bin_width:.4f} m")
  if np.median(widths) > 4:
    print(
      "  *** the return is not point-like. argmax on it is the same mistake "
      "the singlebeam made -- a range bin is evidence about area at that "
      "slant range, not about depth under the beam ***"
    )


def score(points: np.ndarray, tree: KDTree, surface: np.ndarray) -> tuple:
  """Median and MAD-std of reconstructed soundings against the octree."""
  finite = np.isfinite(points).all(axis=1)
  if not finite.any():
    return float("nan"), float("nan"), 0
  good = points[finite]
  _, nearest = tree.query(good[:, :2])
  residual = good[:, 2] - surface[nearest, 2]
  median = float(np.median(residual))
  spread = float(np.median(np.abs(residual - median)) * 1.4826)
  return median, spread, int(finite.sum())


def main() -> None:
  args = parse_args()
  data = np.load(args.capture)
  images = data["images"]
  positions, rotations = data["positions"], data["rotations"]

  ranges = range_bins(
    float(data["range_min"]),
    float(data["range_max"]),
    int(data["range_bins"]),
  )
  bearings = azimuth_angles(float(data["azimuth"]), int(data["azimuth_bins"]))
  print(f"{len(images)} pings of {images[0].shape} (range, azimuth)")
  print()

  report_beam_width(images, ranges)
  print()

  beam_ranges = np.array([bottom_return_ranges(im, ranges) for im in images])
  valid = np.isfinite(beam_ranges)
  print(
    f"beams with an echo: {100 * valid.mean():.1f}% "
    f"({valid.sum()} of {valid.size})"
  )
  print()

  directory = (
    args.root
    / args.version
    / "worlds/Ocean/Linux/Holodeck/Octrees"
    / args.world
    / args.cache
  )
  pad = 60.0
  surface = load_surface(
    directory,
    bounds=(
      positions[:, 0].min() - pad,
      positions[:, 0].max() + pad,
      positions[:, 1].min() - pad,
      positions[:, 1].max() + pad,
    ),
  )
  tree = KDTree(surface[:, :2])
  print(f"octree surface: {len(surface)} cells")
  print()

  print("reconstruction vs the octree, per axis convention:")
  print(
    f"  {'convention':22s} {'median':>9s} {'MAD-std':>9s} "
    f"{'edge-vs-nadir':>14s}  n"
  )
  for label, (nadir, swath) in CONVENTIONS.items():
    points = np.concatenate(
      [
        seabed_points(p, r, br, bearings, swath_axis=swath, nadir_axis=nadir)
        for p, r, br in zip(positions, rotations, beam_ranges)
      ]
    )
    median, spread, n = score(points, tree, surface)

    # A mirrored swath still puts the nadir beams in the right place; it is the
    # outermost beams that land on the wrong side. Score them separately.
    edge = np.zeros(len(bearings), dtype=bool)
    edge[: len(bearings) // 6] = True
    edge[-len(bearings) // 6 :] = True
    edge_points = np.concatenate(
      [
        seabed_points(
          p, r, br[edge], bearings[edge], swath_axis=swath, nadir_axis=nadir
        )
        for p, r, br in zip(positions, rotations, beam_ranges)
      ]
    )
    edge_median, _, _ = score(edge_points, tree, surface)

    print(
      f"  {label:22s} {median:+9.3f} {spread:9.3f} "
      f"{edge_median - median:+14.3f}  {n}"
    )

  print()
  print(
    "The correct convention has a small MAD-std and no edge-vs-nadir drift. A "
    "mirrored swath axis reconstructs the nadir beams correctly and puts the "
    "outer ones on the wrong side of the track, which shows up in the last "
    "column rather than the median."
  )


if __name__ == "__main__":
  main()
