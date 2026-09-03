"""Fit the multibeam's mount rotation against the simulator's octree.

    python experiments/fit_beam_geometry.py multibeam_check.npz

``analyse_multibeam.py`` enumerates candidate body-frame axis conventions and
gets no further than 1.6 m of residual -- 27x the octree's own agreement with
good soundings. Enumeration cannot do better, because the remaining error is
continuous: the sensor's mount ``rotation`` is a rotation, not a choice among
six axes, and roughly one degree of misalignment across a 40 m half-swath is
exactly this size.

So fit it. Five parameters:

* a rotation vector for the sensor-to-body rotation that
  :func:`~auv_pose.mapping.sonar.seabed_points` does not apply, and
* a gain and offset on bearing, which absorb a bin-centre-versus-edge
  convention error and any sign or scale mistake in the swath.

The objective is the median absolute vertical residual against the octree
surface. Median, not mean: beams that miss the swath or land on a step produce
outliers that a least-squares fit would chase.

Reports residual against beam index before and after, because *how* it varies
identifies what was wrong -- a linear trend is an angular scale error, a
symmetric V is a nadir offset, a step is a mirrored swath.
"""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from auv_pose.mapping.octree import (
  cached_surface,
  robust_spread,
  surface_residual,
)
from auv_pose.mapping.sonar import (
  azimuth_angles,
  bottom_return_ranges,
  range_bins,
)

DEFAULT_ROOT = Path(
  os.environ.get("HOLODECKPATH", Path.home() / "data" / "holoocean")
)

#: Best convention found by enumeration: the fan points along body +z (down in
#: the IMU socket) and opens along body -y. The fit starts here.
INITIAL_MOUNT = np.array(
  [
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
  ]
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
    default=50.0,
    help=(
      "metres of octree to load around the track. Must cover the swath: at "
      "70 m altitude a 60 degree fan reaches 40 m either side"
    ),
  )
  parser.add_argument("--refresh", action="store_true")
  return parser.parse_args()


def soundings(
  parameters: np.ndarray,
  positions: np.ndarray,
  rotations: np.ndarray,
  beam_ranges: np.ndarray,
  bearings: np.ndarray,
) -> np.ndarray:
  """World-frame seabed points under a candidate geometry.

  Vectorised over every ping and beam at once -- the optimiser evaluates this
  a few hundred times, so a Python loop over pings is the bottleneck.
  """
  mount = Rotation.from_rotvec(parameters[:3]).as_matrix() @ INITIAL_MOUNT
  gain, offset = parameters[3], parameters[4]
  angles = gain * bearings + offset

  # Beam directions in the sensor frame: the fan opens in the sensor's xy
  # plane about its forward axis, then the mount carries it into the body.
  sensor = np.stack(
    [np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1
  )
  body = sensor @ mount.T

  # (ping, beam, 3): rotate each ping's beams by that ping's attitude.
  offsets = beam_ranges[:, :, None] * body[None, :, :]
  world = np.einsum("pij,pbj->pbi", rotations, offsets)
  return positions[:, None, :] + world


def report(label, residual, bearings, finite, n_beams) -> None:
  median, spread = robust_spread(residual)
  print(f"  {label}: median {median:+.3f} m, MAD-std {spread:.3f} m")

  beam_index = np.tile(np.arange(n_beams), len(finite) // n_beams)[finite]
  centred = residual - median
  edges = np.linspace(0, n_beams, 7).astype(int)
  cells = [
    np.median(centred[(beam_index >= lo) & (beam_index < hi)])
    for lo, hi in itertools.pairwise(edges)
  ]
  degrees = np.degrees(bearings)
  print(
    "    residual by beam sextant (port to starboard, "
    f"{degrees[0]:+.0f} to {degrees[-1]:+.0f} deg): "
    + "  ".join(f"{v:+.2f}" for v in cells)
  )


def main() -> None:
  args = parse_args()
  data = np.load(args.capture)
  images, positions, rotations = (
    data["images"],
    data["positions"],
    data["rotations"],
  )

  ranges = range_bins(
    float(data["range_min"]),
    float(data["range_max"]),
    int(data["range_bins"]),
  )
  bearings = azimuth_angles(float(data["azimuth"]), int(data["azimuth_bins"]))
  beam_ranges = np.array([bottom_return_ranges(im, ranges) for im in images])
  print(f"{len(images)} pings, {beam_ranges.shape[1]} beams")

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
    refresh=args.refresh,
  )
  print(f"octree surface: {len(surface)} cells")
  print()

  def objective(parameters: np.ndarray) -> float:
    points = soundings(parameters, positions, rotations, beam_ranges, bearings)
    residual, _ = surface_residual(points.reshape(-1, 3), surface)
    if len(residual) == 0:
      return 1e6
    return float(np.median(np.abs(residual - np.median(residual))))

  start = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
  points = soundings(start, positions, rotations, beam_ranges, bearings)
  residual, finite = surface_residual(points.reshape(-1, 3), surface)
  print("before fitting:")
  report("start", residual, bearings, finite, beam_ranges.shape[1])
  print()

  result = minimize(
    objective,
    start,
    method="Nelder-Mead",
    options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-5},
  )

  points = soundings(result.x, positions, rotations, beam_ranges, bearings)
  residual, finite = surface_residual(points.reshape(-1, 3), surface)
  print("after fitting:")
  report("fitted", residual, bearings, finite, beam_ranges.shape[1])
  print()

  correction = Rotation.from_rotvec(result.x[:3])
  mount = correction.as_matrix() @ INITIAL_MOUNT
  print(
    f"correction to the start convention: "
    f"{np.degrees(result.x[:3]).round(3)} deg (rotation vector)"
  )
  print(
    f"bearing gain {result.x[3]:.5f}, offset "
    f"{np.degrees(result.x[4]):+.4f} deg "
    f"({np.degrees(result.x[4]) / np.degrees(bearings[1] - bearings[0]):+.2f} bins)"
  )
  print()
  print("fitted sensor-to-body rotation (columns are the sensor axes in body):")
  for row in mount:
    print("   " + "  ".join(f"{v:+.4f}" for v in row))
  print()
  print("beam directions in the body frame:")
  print(f"   fan centre (nadir): {mount[:, 0].round(4)}")
  print(f"   swath axis:         {mount[:, 1].round(4)}")


if __name__ == "__main__":
  main()
