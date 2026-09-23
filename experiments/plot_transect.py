"""A transect across the seabed: soundings, and each map's mean with its band.

    python experiments/plot_transect.py soundings.csv \\
        --map single.pkl "single scale" --map two.pkl "two scales" \\
        --from -20 -25 --to -20 5 --out transect.png

The picture that justifies a kernel choice: across a pipe, a single-scale map
has to compromise between the flat seabed and the pipe's steep flanks, while a
two-scale one can follow both. Soundings within ``--corridor`` of the line are
projected onto it; each map is evaluated along it with a ``+-2 sigma`` band of
its own latent uncertainty.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from auv_pose.io.checkpoints import load_map
from experiments.cli import refuse_overwrite

#: Categorical slots 1 and 2 of the validated palette, in order.
SERIES = ("#2a78d6", "#eb6834")
SOUNDINGS = "#9a9893"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("soundings", type=Path, help="soundings CSV")
  parser.add_argument(
    "--map",
    nargs=2,
    action="append",
    metavar=("CHECKPOINT", "LABEL"),
    required=True,
    help="a map to draw, and its legend label; repeat for each",
  )
  parser.add_argument(
    "--from", dest="start", type=float, nargs=2, required=True
  )
  parser.add_argument("--to", dest="end", type=float, nargs=2, required=True)
  parser.add_argument(
    "--corridor",
    type=float,
    default=1.0,
    help="half-width of the strip of soundings shown, metres",
  )
  parser.add_argument(
    "--true",
    action="store_true",
    help="plot the soundings' true_x/y/z rather than their recorded x/y/z",
  )
  parser.add_argument("--samples", type=int, default=600)
  parser.add_argument("--out", type=Path, default=Path("transect.png"))
  parser.add_argument("--force", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  refuse_overwrite(args.out, args.force)
  if len(args.map) > len(SERIES):
    raise SystemExit(f"at most {len(SERIES)} maps: one categorical slot each")

  columns = ["true_x", "true_y", "true_z"] if args.true else ["x", "y", "z"]
  frame = pd.read_csv(args.soundings, usecols=columns)
  points = frame[columns[:2]].to_numpy(float)
  depth = frame[columns[2]].to_numpy(float)

  start, end = np.asarray(args.start, float), np.asarray(args.end, float)
  length = float(np.linalg.norm(end - start))
  along = (end - start) / length
  across = np.array([-along[1], along[0]])

  offset = points - start
  distance = offset @ along
  side = offset @ across
  near = (
    (np.abs(side) <= args.corridor) & (distance >= 0) & (distance <= length)
  )

  steps = np.linspace(0.0, length, args.samples)
  line = start + steps[:, None] * along

  figure, axes = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
  axes.set_facecolor(SURFACE)
  axes.scatter(
    distance[near],
    depth[near],
    s=4,
    color=SOUNDINGS,
    linewidths=0,
    label=f"soundings within {args.corridor:g} m",
    zorder=1,
  )

  ends: list[tuple[float, str]] = []
  for (checkpoint, label), colour in zip(args.map, SERIES):
    bathymetry = load_map(checkpoint)
    mean, std = bathymetry.predict(line, with_std=True)
    ends.append((float(mean[-1]), label))
    axes.fill_between(
      steps,
      mean - 2 * std,
      mean + 2 * std,
      color=colour,
      alpha=0.18,
      linewidth=0,
      zorder=2,
    )
    axes.plot(steps, mean, color=colour, linewidth=2, label=label, zorder=3)

  # Direct labels at the right end, in text ink beside each line, spread
  # apart vertically where the lines end close together.
  spacing = 11.0  # points
  for rank, (height, label) in enumerate(sorted(ends)):
    lift = (rank - (len(ends) - 1) / 2) * spacing
    axes.annotate(
      label,
      (steps[-1], height),
      xytext=(6, lift),
      textcoords="offset points",
      va="center",
      color=INK,
      fontsize=9,
    )

  axes.set_xlabel(
    f"distance along transect from ({start[0]:g}, {start[1]:g}), m", color=MUTED
  )
  axes.set_ylabel("seabed elevation, m", color=MUTED)
  axes.grid(True, color=GRID, linewidth=0.8)
  axes.set_axisbelow(True)
  for spine in ("top", "right"):
    axes.spines[spine].set_visible(False)
  for spine in ("left", "bottom"):
    axes.spines[spine].set_color(GRID)
  axes.tick_params(colors=MUTED)
  axes.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
  axes.set_xlim(0, length * 1.12)

  figure.tight_layout()
  figure.savefig(args.out, dpi=200, facecolor=SURFACE)
  print(f"Wrote {args.out}: {int(near.sum())} soundings along {length:.1f} m")


if __name__ == "__main__":
  main()
