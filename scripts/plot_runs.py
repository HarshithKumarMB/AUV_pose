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
  sigma = frame[["ekf_sx", "ekf_sy", "ekf_sz"]].to_numpy()

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
    # Signed, because the point of showing it against a sigma band is whether
    # the error sits inside the uncertainty the filter claims.
    "z_err": ekf[:, 2] - truth[:, 2],
    "sigma_z": sigma[:, 2],
    "sigma_pos": np.linalg.norm(sigma, axis=1),
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


def label_ends(ax, entries, min_gap: float = 11.0) -> None:
  """Label each curve's final value, nudged apart where they would collide.

  Configurations that used to differ by metres now differ by centimetres, so
  hand-tuned offsets need re-tuning on every regeneration. Spread them in
  display space instead, which needs the axis limits and scale to be set first.
  """
  if not entries:
    return

  placed = sorted(
    (ax.transData.transform((x, y))[1], x, y, text) for x, y, text in entries
  )
  offsets = [0.0] * len(placed)
  for i in range(1, len(placed)):
    gap = (placed[i][0] + offsets[i]) - (placed[i - 1][0] + offsets[i - 1])
    if gap < min_gap:
      offsets[i] += min_gap - gap

  # Centre the stack so a crowded panel does not drift its labels upward.
  shift = sum(offsets) / len(offsets)
  for offset, (_, x, y, text) in zip(offsets, placed, strict=True):
    label_end(ax, x, y, text, offset - shift)


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
    y=0.988,
  )
  fig.text(
    0.055,
    0.902,
    "One 300-step run per configuration. Dead reckoning was 7.5 m before the "
    "DVL was used to initialise velocity and to keep the attitude filter from "
    "mistaking a manoeuvre for a tilt;\nwithout a DVL neither fix is available "
    "and it still drifts. Guidance closes on the estimate, so each run flies a "
    "different trajectory -- read sub-metre\ndifferences as noise. The vehicle "
    "holds a constant attitude throughout: none of these runs rotate it.",
    color=INK_2,
    fontsize=9.5,
    ha="left",
  )

  ax = axes[0][0]
  # The filter's own account of how wrong it thinks it is. Error curves sitting
  # well below it mean the filter is pessimistic; above it, overconfident --
  # and nothing in this figure used to say which.
  ax.plot(
    data["all fixed"]["t"],
    data["all fixed"]["sigma_pos"],
    color=INK_2,
    linewidth=1.2,
    linestyle=":",
    label="EKF $\\sigma$, all fixed",
  )
  ends = []
  for name, _, colour, linestyle in RUNS:
    run = data[name]
    ax.plot(
      run["t"], run["pos"], color=colour, linewidth=2, linestyle=linestyle
    )
    ends.append((run["t"][-1], run["pos"][-1], f"{run['pos'][-1]:.2f} m"))
  ax.set_yscale("log")
  ax.set_xlim(0, 11.6)
  ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
  label_ends(ax, ends)
  style(
    ax,
    "time (s)",
    "EKF position error (m)",
    "a.  Position error against ground truth",
  )

  ax = axes[0][1]
  ends = []
  for name, _, colour, linestyle in RUNS:
    run = data[name]
    ax.plot(
      run["t"], run["dr_err"], color=colour, linewidth=2, linestyle=linestyle
    )
    ends.append((run["t"][-1], run["dr_err"][-1], f"{run['dr_err'][-1]:.2f} m"))
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
  label_ends(ax, ends)

  ax = axes[1][0]
  fixed = data["all fixed"]
  ax.fill_between(
    fixed["t"],
    -fixed["sigma_z"],
    fixed["sigma_z"],
    color="#2a78d6",
    alpha=0.13,
    linewidth=0,
    label="EKF $\\pm1\\sigma_z$, all fixed",
  )
  ax.axhline(0, color=INK_2, linewidth=0.8, alpha=0.5)
  ends = []
  for name, _, colour, linestyle in RUNS:
    run = data[name]
    ax.plot(
      run["t"], run["z_err"], color=colour, linewidth=2, linestyle=linestyle
    )
    ends.append((run["t"][-1], run["z_err"][-1], f"{run['z_err'][-1]:+.2f} m"))
  ax.set_xlim(0, 11.6)
  ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
  label_ends(ax, ends)
  style(
    ax,
    "time (s)",
    "EKF depth error (m)",
    "c.  Depth - the only directly observed axis",
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
    bbox_to_anchor=(0.045, 0.862),
    columnspacing=2.2,
  )

  fig.tight_layout(rect=(0.02, 0.01, 0.99, 0.832))
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
