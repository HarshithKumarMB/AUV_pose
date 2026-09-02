"""Build ``visuals/navigate_runs.png`` from a directory of navigate.py logs.

    python scripts/plot_runs.py --logs runs/

One 300-step run per configuration. The comparison is the fully-fixed run
against the truth-attitude bound, the frame mismatch that was removed, and the
unaided baseline -- not a replay of every intermediate state. See
``visuals/README.md`` for the commands that produce the logs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HZ = 30
DT = 1 / HZ

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdbd6"

# Canonical palette slots 1-4, in order.
RUNS = [
  ("all fixed", "dvl", "#2a78d6", "-"),
  ("mirrored frames (before)", "legacy", "#eb6834", "-"),
  ("no DVL", "att", "#1baf7a", "-"),
  ("truth attitude (bound)", "truthatt", "#eda100", "--"),
]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--logs",
    type=Path,
    required=True,
    help="directory holding dvl.csv, legacy.csv, att.csv and truthatt.csv",
  )
  parser.add_argument(
    "--out", type=Path, default=Path("visuals/navigate_runs.png")
  )
  return parser.parse_args()


def load(path: Path) -> dict:
  frame = pd.read_csv(path)
  truth = frame[["x", "y", "z"]].to_numpy()
  ekf = frame[["ekf_x", "ekf_y", "ekf_z"]].to_numpy()
  dead_reckoned = frame[["dr_x", "dr_y", "dr_z"]].to_numpy()
  true_velocity = np.gradient(truth, DT, axis=0)

  return {
    "t": frame["step"].to_numpy() / HZ,
    "truth": truth,
    "ekf": ekf,
    "dr": dead_reckoned,
    "pos": np.linalg.norm(truth - ekf, axis=1),
    "dr_err": np.linalg.norm(truth - dead_reckoned, axis=1),
    "vel": np.linalg.norm(
      true_velocity - frame[["ekf_vx", "ekf_vy", "ekf_vz"]].to_numpy(), axis=1
    ),
    "att": frame["att_err_deg"].to_numpy(),
  }


def style(ax, xlabel: str, ylabel: str, title: str) -> None:
  ax.set_facecolor(SURFACE)
  ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)
  ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
  ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
  ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
  ax.set_axisbelow(True)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
  ax.tick_params(colors=INK_2, labelsize=8, length=3)


def label_end(ax, x, y, text: str, dy: float = 0) -> None:
  ax.annotate(
    text,
    xy=(x, y),
    xytext=(5, dy),
    textcoords="offset points",
    color=INK,
    fontsize=8,
    va="center",
  )


def main() -> None:
  args = parse_args()
  data = {name: load(args.logs / f"{key}.csv") for name, key, _, _ in RUNS}

  fig, axes = plt.subplots(2, 2, figsize=(13, 8.8), facecolor=SURFACE)
  fig.suptitle(
    "Terrain-aided navigation on the current code: "
    f"{data['all fixed']['pos'][-1]:.2f} m over 10 s",
    color=INK,
    fontsize=14,
    x=0.055,
    ha="left",
    y=0.982,
  )
  fig.text(
    0.055,
    0.918,
    "One 300-step run per configuration. Dead reckoning is no longer the weak "
    "point: it was 7.5 m before the DVL was used to initialise velocity and to "
    "keep the attitude filter\nfrom mistaking a manoeuvre for a tilt. Guidance "
    "closes on the estimate, so each run flies a different trajectory -- read "
    "sub-metre differences as noise.",
    color=INK_2,
    fontsize=9.5,
    ha="left",
  )

  # The fixed run and its truth-attitude bound now land within centimetres of
  # each other, so their end labels have to be pulled apart by hand.
  apart = {"all fixed": 6, "truth attitude (bound)": -6}

  ax = axes[0][0]
  for name, _, colour, linestyle in RUNS:
    run = data[name]
    ax.plot(
      run["t"], run["pos"], color=colour, linewidth=2, linestyle=linestyle
    )
    label_end(
      ax,
      run["t"][-1],
      run["pos"][-1],
      f"{run['pos'][-1]:.2f} m",
      apart.get(name, 0),
    )
  ax.set_yscale("log")
  ax.set_xlim(0, 11.6)
  style(
    ax,
    "time (s)",
    "EKF position error (m)",
    "a.  Position error against ground truth",
  )

  ax = axes[0][1]
  for name, _, colour, linestyle in RUNS:
    run = data[name]
    ax.plot(
      run["t"], run["dr_err"], color=colour, linewidth=2, linestyle=linestyle
    )
    label_end(
      ax,
      run["t"][-1],
      run["dr_err"][-1],
      f"{run['dr_err'][-1]:.2f} m",
      apart.get(name, 0),
    )
  ax.set_yscale("log")
  ax.set_xlim(0, 11.6)
  # Every run starts at exactly zero error, which on a log axis pulls the floor
  # down through decades that carry nothing.
  ax.set_ylim(bottom=1e-2)
  style(
    ax,
    "time (s)",
    "dead-reckoning error (m)",
    "b.  Dead reckoning, once the DVL also feeds attitude",
  )

  ax = axes[1][0]
  nudge = {"all fixed": 8, "truth attitude (bound)": -8}
  for name, _, colour, linestyle in RUNS:
    run = data[name]
    ax.plot(
      run["t"], run["vel"], color=colour, linewidth=2, linestyle=linestyle
    )
    label_end(
      ax,
      run["t"][-1],
      run["vel"][-1],
      f"{run['vel'][-1]:.2f}",
      nudge.get(name, 0),
    )
  ax.set_xlim(0, 11.6)
  style(
    ax,
    "time (s)",
    "velocity error (m/s)",
    "c.  Velocity error - what the DVL buys",
  )

  ax = axes[1][1]
  run = data["all fixed"]
  ax.plot(
    run["truth"][:, 0],
    run["truth"][:, 1],
    color=INK,
    linewidth=2.5,
    label="truth",
  )
  ax.plot(
    run["ekf"][:, 0],
    run["ekf"][:, 1],
    color="#2a78d6",
    linewidth=2,
    label="EKF estimate",
  )
  ax.plot(
    run["dr"][:, 0],
    run["dr"][:, 1],
    color="#eb6834",
    linewidth=2,
    linestyle="--",
    label="dead reckoning",
  )
  ax.scatter(run["truth"][0, 0], run["truth"][0, 1], color=INK, s=45, zorder=5)
  ax.annotate(
    "start",
    xy=run["truth"][0, :2],
    xytext=(6, -11),
    textcoords="offset points",
    color=INK_2,
    fontsize=8,
  )
  ax.scatter(-10.0, -5.0, color=INK_2, s=90, zorder=5, marker="*")
  ax.annotate(
    "waypoint 1",
    xy=(-10.0, -5.0),
    xytext=(6, 4),
    textcoords="offset points",
    color=INK_2,
    fontsize=8,
  )
  style(ax, "x (m)", "y (m)", "d.  The fully-fixed track, dead reckoning too")
  ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")

  focus = np.vstack(
    [
      run["truth"][:, :2],
      run["ekf"][:, :2],
      run["dr"][:, :2],
      [[-10.0, -5.0]],
    ]
  )
  pad = 3.0
  ax.set_xlim(focus[:, 0].min() - pad, focus[:, 0].max() + pad)
  ax.set_ylim(focus[:, 1].min() - pad, focus[:, 1].max() + pad)
  ax.set_aspect("equal", adjustable="box")

  handles = [
    plt.Line2D(
      [], [], color=colour, linewidth=2, linestyle=linestyle, label=name
    )
    for name, _, colour, linestyle in RUNS
  ]
  fig.legend(
    handles=handles,
    frameon=False,
    fontsize=9,
    labelcolor=INK,
    ncol=4,
    loc="upper left",
    bbox_to_anchor=(0.045, 0.885),
    columnspacing=2.2,
  )

  fig.tight_layout(rect=(0.02, 0.01, 0.99, 0.855))
  args.out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.out, dpi=200, facecolor=SURFACE)
  print(f"Wrote {args.out}")

  for name, _, _, _ in RUNS:
    run = data[name]
    print(
      f"  {name:26s} EKF {run['pos'][-1]:6.2f} m | DR {run['dr_err'][-1]:6.2f} m"
      f" | vel {run['vel'][-1]:5.2f} m/s | att {run['att'].mean():5.2f} deg"
    )


if __name__ == "__main__":
  main()
