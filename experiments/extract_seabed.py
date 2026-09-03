"""Read the true seabed out of HoloOcean's cached octree. No simulator needed.

    python experiments/extract_seabed.py --out seabed_truth.csv

The simulator caches the octree its sonar raycasts against as JSON on disk, so
the surface the sonar is measuring can be read directly. That makes it ground
truth for the survey rather than an independent estimate of it: a sounding that
disagrees with this is a sonar defect, not terrain.

Writes ``x, y, z`` where ``z`` is world-frame seabed elevation -- **not** the
``x, y, sonar_depth`` the survey writes, which is a range from an unrecorded
vehicle depth.

With ``--check`` it scores a survey CSV against the extracted surface, which is
what validated the reader: the far population of ``map.csv`` agrees to a
0.059 m MAD-std, well below the sonar's own 0.113 m quantisation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from auv_pose.io.logs import CsvLogger
from auv_pose.io.soundings import load_soundings
from auv_pose.mapping.octree import load_surface
from experiments.cli import refuse_overwrite

#: Where the packaged worlds unpack to. ``flake.nix`` exports HOLODECKPATH.
DEFAULT_ROOT = Path(
  os.environ.get("HOLODECKPATH", Path.home() / "data" / "holoocean")
)

#: The cache built for ``octree_min: 0.02`` / ``octree_max: 5.12``, which is what
#: the surveys ran at. ``min50_max800`` is the coarse alternative; measured, the
#: two agree on the seabed, so this is a resolution choice and not a correctness
#: one.
DEFAULT_CACHE = "min2_max512"


def cache_directory(root: Path, version: str, world: str, cache: str) -> Path:
  return (
    root
    / version
    / "worlds"
    / "Ocean"
    / "Linux"
    / "Holodeck"
    / "Octrees"
    / world
    / cache
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, default=Path("seabed_truth.csv"))
  parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
  parser.add_argument("--version", default="2.3.0")
  parser.add_argument("--world", default="Dam")
  parser.add_argument("--cache", default=DEFAULT_CACHE)
  parser.add_argument(
    "--bounds",
    type=float,
    nargs=4,
    metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
    default=[-40.0, 0.0, -20.0, 0.0],
    help="horizontal box to extract, metres; the default is the survey box",
  )
  parser.add_argument(
    "--cell",
    type=float,
    default=0.10,
    help=(
      "horizontal cell for the top-surface reduction, metres. Leaves are 2 cm, "
      "so below that this stops reducing anything"
    ),
  )
  parser.add_argument(
    "--min-normal-z",
    type=float,
    default=0.0,
    help=(
      "drop leaves whose normal points less than this far up. The Dam seabed is "
      "99.8%% upward-facing so the default keeps everything; raise it on a world "
      "with walls or overhangs"
    ),
  )
  parser.add_argument(
    "--check",
    type=Path,
    nargs="*",
    help="survey CSVs to score against the extracted surface",
  )
  parser.add_argument(
    "--force", action="store_true", help="overwrite --out if it exists"
  )
  return parser.parse_args()


def check(surface: np.ndarray, paths: list[Path]) -> None:
  """Score survey soundings against the extracted surface.

  Reported with a median and a MAD-derived std rather than a mean and std: the
  singlebeam's error is bimodal, and the mean of a 53/47 mixture describes
  neither population. That is precisely how the defect stayed hidden.
  """
  frame = load_soundings(paths)
  tree = KDTree(surface[:, :2])
  _, nearest = tree.query(frame[["x", "y"]].to_numpy())

  # The survey never recorded its own z, so this is the seabed it implies if the
  # vehicle was at z = 0. A constant offset here is the survey's true depth.
  residual = surface[nearest, 2] + frame["sonar_depth"].to_numpy()
  median = float(np.median(residual))
  spread = float(np.median(np.abs(residual - median)) * 1.4826)

  print(f"  {len(frame)} soundings scored against the octree")
  print(f"  median offset  {median:+7.3f} m")
  print(f"  MAD-std        {spread:7.3f} m")

  within = np.abs(residual - median) < 1.0
  print(f"  within 1 m of the median: {100 * within.mean():.1f}%")
  if within.mean() < 0.9:
    print(
      "  *** the soundings are not one population. A singlebeam whose error is "
      "bimodal is picking the wrong peak, not measuring rough terrain -- "
      "compare the two clusters against this surface separately ***"
    )


def main() -> None:
  args = parse_args()
  refuse_overwrite(args.out, args.force)

  directory = cache_directory(args.root, args.version, args.world, args.cache)
  if not directory.is_dir():
    raise SystemExit(
      f"no octree cache at {directory}. The simulator builds it on first use "
      "and caches it; run a sonar scenario in this world once, or pass --root."
    )

  print(f"Reading {directory}")
  surface = load_surface(
    directory,
    bounds=tuple(args.bounds),
    cell=args.cell,
    min_normal_z=args.min_normal_z,
  )
  if not len(surface):
    raise SystemExit(f"no geometry in {args.bounds}")

  print(
    f"{len(surface)} surface cells, "
    f"z {surface[:, 2].min():.2f} .. {surface[:, 2].max():.2f} m "
    f"({surface[:, 2].max() - surface[:, 2].min():.2f} m of relief)"
  )

  if args.check:
    check(surface, args.check)

  with CsvLogger(args.out, ("x", "y", "z")) as log:
    for x, y, z in surface:
      log.write(x=x, y=y, z=z)

  print(f"Wrote {args.out}")


if __name__ == "__main__":
  main()
