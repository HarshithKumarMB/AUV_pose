"""Plot a navigation run: tracks in each plane, and error against time.

    python experiments/plot_trajectory.py

Reads the CSV written by ``navigate.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLANES = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}

TRACKS = (
  ("", "Ground truth", "blue", "-", 3),
  ("dr_", "Dead reckoning", "red", "--", 2),
  ("ekf_", "EKF", "green", "-", 2),
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--log", type=Path, default=Path("wp_c.csv"))
  parser.add_argument("--prefix", type=Path, default=Path("trajectory"))
  parser.add_argument(
    "--planes", nargs="+", choices=sorted(PLANES), default=["xy", "xz"]
  )
  return parser.parse_args()


def plot_plane(frame: pd.DataFrame, plane: str, path: Path) -> None:
  horizontal, vertical = PLANES[plane]

  figure, axes = plt.subplots(figsize=(10, 8))
  for prefix, label, colour, style, width in TRACKS:
    columns = (prefix + horizontal, prefix + vertical)
    if not set(columns) <= set(frame.columns):
      continue
    axes.plot(
      frame[columns[0]],
      frame[columns[1]],
      label=label,
      color=colour,
      linestyle=style,
      linewidth=width,
    )

  axes.scatter(
    frame[horizontal].iloc[0],
    frame[vertical].iloc[0],
    color="black",
    s=100,
    marker="o",
    label="Start",
    zorder=5,
  )
  axes.scatter(
    frame[horizontal].iloc[-1],
    frame[vertical].iloc[-1],
    color="purple",
    s=100,
    marker="x",
    label="End",
    zorder=5,
  )

  axes.set_xlabel(f"{horizontal.upper()} position (m)")
  axes.set_ylabel(f"{vertical.upper()} position (m)")
  axes.set_title(f"Trajectory comparison ({plane.upper()} plane)")
  axes.grid(True)
  axes.axis("equal")
  axes.legend()

  figure.savefig(path, dpi=300, bbox_inches="tight")
  plt.close(figure)


def error_norm(frame: pd.DataFrame, prefix: str) -> np.ndarray:
  return np.sqrt(
    sum((frame[axis] - frame[prefix + axis]) ** 2 for axis in ("x", "y", "z"))
  )


def plot_error(frame: pd.DataFrame, path: Path) -> None:
  """Estimator error against ground truth, EKF versus dead reckoning."""
  figure, axes = plt.subplots(figsize=(10, 6))

  steps = frame["step"] if "step" in frame.columns else np.arange(len(frame))

  for prefix, label, colour in (
    ("ekf_", "EKF", "green"),
    ("dr_", "Dead reckoning", "red"),
  ):
    if not {prefix + a for a in "xyz"} <= set(frame.columns):
      continue
    error = error_norm(frame, prefix)
    axes.plot(
      steps,
      error,
      label=f"{label} (RMS {np.sqrt((error**2).mean()):.2f} m)",
      color=colour,
      linewidth=2,
    )

  axes.set_xlabel("Time step")
  axes.set_ylabel("Position error (m)")
  axes.set_title("Position error against ground truth")
  axes.grid(True)
  axes.legend()

  figure.savefig(path, dpi=300, bbox_inches="tight")
  plt.close(figure)


def main() -> None:
  args = parse_args()

  if not args.log.exists():
    raise SystemExit(
      f"{args.log} not found -- generate it with:\n"
      f'  nix run .#sim -- -c "python experiments/navigate.py"'
    )

  frame = pd.read_csv(args.log)
  print(f"{len(frame)} steps from {args.log}")

  for plane in args.planes:
    path = args.prefix.with_name(f"{args.prefix.name}_{plane}.png")
    plot_plane(frame, plane, path)
    print(f"Wrote {path}")

  error_path = args.prefix.with_name(f"{args.prefix.name}_error.png")
  plot_error(frame, error_path)
  print(f"Wrote {error_path}")

  for prefix, label in (("ekf_", "EKF"), ("dr_", "Dead reckoning")):
    if {prefix + a for a in "xyz"} <= set(frame.columns):
      error = error_norm(frame, prefix)
      print(
        f"{label:16s} RMS {error.pow(2).mean() ** 0.5:7.3f} m   "
        f"final {error.iloc[-1]:7.3f} m"
      )


if __name__ == "__main__":
  main()
