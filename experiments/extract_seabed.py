"""Read the true seabed out of HoloOcean's cached octree. No simulator needed.

    python experiments/extract_seabed.py --out seabed_truth.csv

The simulator caches the octree its sonar raycasts against as JSON on disk, so
the surface the sonar is measuring can be read directly. That makes it ground
truth for the survey rather than an independent estimate of it: a sounding that
disagrees with this is a sonar defect, not terrain.

Writes ``x, y, z``: world-frame seabed elevation, the same schema the survey
writes.

With ``--check`` it scores a survey CSV against the extracted surface, and
**widens the extracted region to cover those soundings**. That is not a
convenience: a multibeam throws beams tens of metres either side of the track,
so scoring against the survey box alone leaves most soundings outside the
surface, snapping to its edge and reporting a confident, meaningless error.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from auv_pose.io.logs import CsvLogger
from auv_pose.io.soundings import load_soundings
from auv_pose.mapping.octree import (
  load_surface,
  robust_spread,
  surface_residual,
)
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
    help=(
      "horizontal box to extract, metres. The default is the survey box; "
      "--check widens it to cover the soundings being scored"
    ),
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

  This is the number the survey lives or dies by. The octree agrees with a good
  sounding to about 0.06 m, so anything much above that is the sensor.
  """
  frame = load_soundings(paths)
  soundings = frame[["x", "y", "z"]].to_numpy()

  residual, kept = surface_residual(soundings, surface)
  median, spread = robust_spread(residual)

  covered = kept.mean()
  print(
    f"  {len(frame)} soundings, {100 * covered:.1f}% covered by the surface"
  )
  if covered < 0.95:
    print(
      "  *** most soundings fall outside the extracted region. A multibeam "
      "throws beams tens of metres either side of the track, so the surface "
      "has to cover the swath and not just the survey box -- widen --bounds ***"
    )
  print(f"  {int(kept.sum())} soundings scored against the octree")
  print(f"  median offset  {median:+7.3f} m")
  print(f"  MAD-std        {spread:7.3f} m")

  within = np.abs(residual - median) < 1.0
  print(f"  within 1 m of the median: {100 * within.mean():.1f}%")
  if within.mean() < 0.9:
    print(
      "  *** the soundings are not one population. That is a sensor picking "
      "between competing returns, not rough terrain -- score the clusters "
      "against this surface separately before believing either ***"
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

  bounds = list(args.bounds)
  if args.check:
    # Score against a surface that covers the data. The swath reaches far
    # outside the box the vehicle flew.
    frame = load_soundings(args.check)
    margin = 2.0
    bounds = [
      min(bounds[0], float(frame["x"].min()) - margin),
      max(bounds[1], float(frame["x"].max()) + margin),
      min(bounds[2], float(frame["y"].min()) - margin),
      max(bounds[3], float(frame["y"].max()) + margin),
    ]
    print(
      f"Widened bounds to cover the soundings: {[round(b, 1) for b in bounds]}"
    )

  print(f"Reading {directory}")
  surface = load_surface(
    directory,
    bounds=tuple(bounds),
    cell=args.cell,
    min_normal_z=args.min_normal_z,
  )
  if not len(surface):
    raise SystemExit(f"no geometry in {bounds}")

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
