"""Classify sonar point clouds and match them against the labelled obstacle map.

    python experiments/obstacle_predict.py --clouds pointclouds/ --obstacles a.csv

Each cloud is classified in one batch, then matched per predicted class.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.obstacle_map import ObstacleMap


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--clouds",
    type=Path,
    required=True,
    help="directory of point cloud CSVs (x, y, z[, r, theta])",
  )
  parser.add_argument("--obstacles", type=Path, nargs="+", required=True)
  parser.add_argument(
    "--classifier", type=Path, default=Path("obstacle_classifier.pkl")
  )
  parser.add_argument("--encoder", type=Path, default=Path("label_encoder.pkl"))
  parser.add_argument(
    "--max-distance",
    type=float,
    default=10.0,
    help="drop matches further away than this, metres",
  )
  parser.add_argument("--plot", action="store_true")
  return parser.parse_args()


def match_cloud(cloud: pd.DataFrame, obstacles: ObstacleMap) -> pd.DataFrame:
  """Attach the nearest known obstacle to every row.

  One classifier call for the whole cloud, then one spatial query per predicted
  class -- not one of each per row.
  """
  points = cloud[["x", "y", "z"]].to_numpy(dtype=float)

  matched = cloud.copy()
  matched["predicted_obstacle"] = obstacles.classify(points)

  nearest, distance = obstacles.nearest_many(
    points, matched["predicted_obstacle"].to_numpy()
  )

  matched[["nearest_x", "nearest_y", "nearest_z"]] = nearest
  matched["distance"] = distance
  return matched


def add_projected_position(matched: pd.DataFrame) -> pd.DataFrame:
  """Back out vehicle position from a match and the return's range and bearing."""
  if not {"r", "theta"} <= set(matched.columns):
    return matched

  matched["calc_x"] = matched["x"]
  matched["calc_y"] = matched["nearest_y"] - matched["r"] * np.sin(
    matched["theta"]
  )
  matched["calc_z"] = matched["nearest_z"] + matched["r"] * np.cos(
    matched["theta"]
  )
  return matched


def main() -> None:
  args = parse_args()

  obstacles = ObstacleMap.load(args.obstacles, args.classifier, args.encoder)
  print(f"Obstacle types: {', '.join(obstacles.obstacle_types)}")

  clouds = sorted(args.clouds.glob("*.csv"))
  clouds = [c for c in clouds if not c.name.startswith("predicted_")]
  if not clouds:
    raise SystemExit(f"no point cloud CSVs in {args.clouds}")
  print(f"Found {len(clouds)} clouds")

  all_points = []

  for path in clouds:
    matched = add_projected_position(match_cloud(pd.read_csv(path), obstacles))
    kept = matched[matched["distance"] < args.max_distance]

    out = path.with_name(f"predicted_{path.name}")
    kept.to_csv(out, index=False)
    print(f"{path.name}: kept {len(kept)} of {len(matched)} -> {out.name}")

    if "calc_x" in kept.columns:
      all_points.append(kept[["calc_x", "calc_y"]])

  if args.plot and all_points:
    points = pd.concat(all_points, ignore_index=True)
    figure, axes = plt.subplots(figsize=(10, 8))
    axes.scatter(
      points["calc_x"],
      points["calc_y"],
      c="green",
      s=10,
      label="Projected vehicle position",
    )
    axes.set_xlabel("X (m)")
    axes.set_ylabel("Y (m)")
    axes.set_title("Positions derived from sonar matches")
    axes.grid(True)
    axes.legend()
    axes.set_aspect("equal", adjustable="box")

    out = args.clouds / "all_calculated_xy_points.png"
    figure.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {out}")


if __name__ == "__main__":
  main()
