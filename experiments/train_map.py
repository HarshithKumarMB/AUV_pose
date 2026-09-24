"""Fit the Vecchia GP bathymetry map from survey soundings.

    python experiments/train_map.py ~/data/auv_pose/surveys_v4/pass*_smoothed.csv

Reads the survey CSVs, fits the map, writes the checkpoint, and renders the
fitted seabed. The defaults are the configuration the map ships in: every
sounding of every survey, no decimation, §3.2's plane mean, one Matérn-5/2
term, ``m = 30``, 2000 steps. ``--method both`` adds the SVGP baseline.

**The shipped map holds nothing out.** How good it is is measured on an
independently flown test track (``experiments/score_track.py``): soundings the
map never saw, landing between the survey's, which is where the smoother will
query it. The holdouts here remain for quick comparisons. ``--holdout-by ping``
is the one that resembles use; the default blocked cells ask a gap-filling
question the smoother never does, and a random split would leak, since
consecutive soundings are about a centimetre apart.

The score is printed beside the mean of the nearest few training soundings.
That baseline is deliberately unflattering: a Gaussian process that loses to it
is not earning its complexity, and until this reported anything, one that did
went unnoticed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from auv_pose.io.checkpoints import save_map, save_vecchia_map
from auv_pose.io.soundings import (
  COVARIANCE_COLUMNS,
  load_soundings,
  placement_covariance,
  soundings_to_arrays,
)
from auv_pose.mapping.cleaning import object_soundings
from auv_pose.mapping.svgp import BathymetryMap, fit_svgp
from auv_pose.mapping.vecchia import (
  VecchiaMap,
  fit_vecchia,
  fit_vecchia_nigp,
  input_noise_variance,
)
from experiments.cli import refuse_overwrite


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "surveys",
    nargs="+",
    type=Path,
    help="survey CSVs to fit",
  )
  parser.add_argument(
    "--out",
    type=Path,
    default=None,
    help="checkpoint to write; defaults to <method>_bathymetry.pkl",
  )
  parser.add_argument(
    "--plot", type=Path, default=Path("gp_bathymetry_surface.png")
  )
  parser.add_argument(
    "--bounds",
    type=float,
    nargs=4,
    metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
    default=None,
    help=(
      "restrict the fit to this box, metres. Worth setting once the swath is "
      "across-track: at 70 m altitude a 60 degree fan reaches 40 m either side "
      "of the line, so a survey of a 40x20 m box returns soundings over "
      "several times that area and the inducing points spread out over ground "
      "nothing will ever be navigated across"
    ),
  )
  parser.add_argument("--inducing", type=int, default=500)
  parser.add_argument("--epochs", type=int, default=200)
  parser.add_argument(
    "--holdout",
    type=float,
    default=0.0,
    help=(
      "fraction of spatial cells withheld for scoring. The checkpoint is "
      "fitted on the remainder, so the number reported describes the map that "
      "is actually written. 0 fits everything and reports nothing"
    ),
  )
  parser.add_argument(
    "--holdout-by",
    choices=("cell", "ping"),
    default="cell",
    help=(
      "what --holdout withholds. 'ping' holds out whole pings before "
      "decimation and scores their raw soundings, which land between the "
      "remaining pings' soundings -- where the smoother's own beams will "
      "query the map. 'cell' withholds --holdout-cell squares, a gap-filling "
      "test the smoother never poses. Needs the ping column for 'ping'"
    ),
  )
  parser.add_argument(
    "--holdout-cell",
    type=float,
    default=8.0,
    help=(
      "side of the cells the holdout is blocked by, metres. **State it beside "
      "every rmse**: at 1 m a held-out sounding sits a median 0.31 m from "
      "training data and the test barely asks for interpolation; at 8 m it is "
      "1.55 m, and the Vecchia-versus-4-NN verdict reverses between the two"
    ),
  )
  parser.add_argument(
    "--decimate-cell",
    type=float,
    default=0.0,
    help=(
      "take the median sounding per cell of this side, metres; 0, the "
      "default, fits every sounding. Off in the shipped map: decimating "
      "makes the fitted noise the scatter of a cell median rather than of one "
      "sounding, which is what the smoother compares the map against. Must "
      "stay well below --holdout-cell when a blocked holdout is used"
    ),
  )
  parser.add_argument(
    "--max-elevation",
    type=float,
    default=None,
    metavar="Z",
    help=(
      "drop soundings shallower than Z metres, for rejecting water-column "
      "echoes. The corrected surveys need none: the 5%% shallow tail once seen "
      "here was the swath-sign mirror, not echoes"
    ),
  )
  parser.add_argument(
    "--place-by",
    choices=("recorded", "truth"),
    default="recorded",
    help=(
      "which positions and depths the map is fitted to. 'truth' is the "
      "control for a navigated survey: every selection -- bounds, objects, "
      "decimation cells, holdout blocks -- is still made on the recorded "
      "positions, and only then are the true ones swapped in, so the two maps "
      "are fitted and scored on the same soundings. Needs true_x/y/z"
    ),
  )
  parser.add_argument(
    "--decimate-per-line",
    action="store_true",
    help=(
      "keep one point per cell per survey line, told apart by a jump in "
      "ping index, rather than one per cell. Soundings of one line share its "
      "drift, so their covariance averages honestly; soundings of different "
      "lines do not. Needs the ping column"
    ),
  )
  parser.add_argument(
    "--drop-objects",
    action="store_true",
    help=(
      "leave out soundings standing on objects -- the Dam's pipelines -- found "
      "by a morphological opening of the soundings themselves, so a real "
      "survey can use it too. A bathymetric map should not fit a pipe, and "
      "the map's input-noise correction cannot: its first-order slope is "
      "wrong at a sharp edge"
    ),
  )
  parser.add_argument(
    "--object-window",
    type=float,
    default=15.0,
    help="opening window, metres; must exceed the widest object (~10 m here)",
  )
  parser.add_argument(
    "--object-core",
    type=float,
    default=3.5,
    help=(
      "height above the opened seabed that makes a sounding an object on its "
      "own, m: above the Dam's ~3 m mounds, below its 4-6 m pipes"
    ),
  )
  parser.add_argument(
    "--method",
    choices=("vecchia", "svgp", "both"),
    default="vecchia",
    help=(
      "which map to fit. 'both' fits each and scores them side by side, which "
      "is how the claim that local conditioning beats inducing points gets a "
      "number on this seabed rather than only a citation"
    ),
  )
  parser.add_argument(
    "--conditioning",
    type=int,
    default=30,
    metavar="M",
    help="Vecchia conditioning-set size; the paper's m",
  )
  parser.add_argument(
    "--near",
    type=int,
    default=None,
    metavar="M_NEAR",
    help=(
      "how many of --conditioning are nearest neighbours; the rest are "
      "spread across the ordering. Default is all nearest. Stein, Chi and "
      "Welty (2004) find all-nearest the worst design for estimating a range "
      "parameter under a linear mean. At 2 m cells with the linear mean, 22 "
      "nearest + 8 spread made no measurable difference"
    ),
  )
  parser.add_argument(
    "--mean",
    default="linear",
    choices=("linear", "quadratic", "cubic"),
    help=(
      "mean basis; §3.2's plane by default. Near data the kernel carries "
      "the prediction, and on an independent track a spline mean scored the "
      "same as the plane (0.820 against 0.818 m)"
    ),
  )
  parser.add_argument(
    "--steps",
    type=int,
    default=2000,
    help=(
      "Vecchia optimiser steps, run in full. The objective reaches float64 "
      "round-off by about 1500 on the four-heading survey"
    ),
  )
  parser.add_argument(
    "--short-lengthscale",
    type=float,
    default=None,
    metavar="METRES",
    help=(
      "add a second Matern-5/2 term starting at this length -- about the "
      "sonar footprint -- so the map can follow objects' steep flanks as "
      "well as the natural seabed. The single-scale fit is run first, scored, "
      "saved beside --out as *_single, and used as the long term's start"
    ),
  )
  parser.add_argument(
    "--nigp-passes",
    type=int,
    default=0,
    metavar="N",
    help=(
      "carry each sounding's position covariance into its noise over N "
      "fits (McHutchon and Rasmussen's NIGP; the paper's procedure is 2). "
      "Needs soundings placed by georeference.py --pose smoothed. 0 is off; "
      "1 would be one fit with no inflation, so it is refused"
    ),
  )
  parser.add_argument("--batch-size", type=int, default=5000)
  parser.add_argument(
    "--grid", type=int, default=200, help="plot resolution per axis"
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--device",
    default=None,
    help=(
      "torch device to fit on; defaults to cuda when available. The reported "
      "device is printed, because a fit that silently fell back to the CPU "
      "looks identical to one that did not"
    ),
  )
  parser.add_argument("--no-plot", action="store_true")
  parser.add_argument(
    "--force",
    action="store_true",
    help="overwrite --out / --plot if they exist",
  )
  return parser.parse_args()


def decimate(
  X: np.ndarray, y: np.ndarray, cell: float
) -> tuple[np.ndarray, np.ndarray]:
  """Take the median sounding per horizontal cell.

  A multibeam survey produces a few hundred soundings per ping and millions per
  run, which is far more than the GP needs and far more than it can afford.

  **Median, not mean.** A strongest-return picker occasionally lands on the
  wrong feature, and those errors are one-sided rather than symmetric, so a mean
  drags the cell toward them. The median simply ignores a minority of bad beams.

  Keep ``cell`` well below the holdout cell, or decimation merges soundings
  across the boundary the split is meant to hold shut.

  :param X: Sounding positions, ``(n, 2)``.
  :param y: Seabed elevation, ``(n,)``.
  :param cell: Cell side in metres.
  :return: ``(X, y)`` reduced to one point per occupied cell, at the cell's
      median position and elevation.
  """
  groups = cell_groups(X, cell)
  return (
    np.asarray([np.median(X[g], axis=0) for g in groups], dtype=X.dtype),
    np.asarray([np.median(y[g]) for g in groups], dtype=y.dtype),
  )


def cell_groups(X: np.ndarray, cell: float) -> list[np.ndarray]:
  """Indices of the soundings in each occupied horizontal cell.

  In the order :func:`decimate` emits its cells, so anything aggregated over
  these groups lines up with its output row for row.

  :param X: Sounding positions, ``(n, 2)``.
  :param cell: Cell side in metres.
  """
  key = np.floor(X / cell).astype(np.int64)
  _, inverse = np.unique(key, axis=0, return_inverse=True)
  order = np.argsort(inverse.ravel(), kind="stable")
  return np.split(order, np.cumsum(np.bincount(inverse.ravel()))[:-1])


def line_groups(
  X: np.ndarray, ping: np.ndarray, cell: float, gap: int = 50
) -> list[np.ndarray]:
  """Like :func:`cell_groups`, but a cell is split by survey line.

  One pass of the vehicle reaches a cell over a few consecutive pings; another
  line reaches it hundreds of pings later. So within each cell the soundings
  are ordered by ping and split wherever the ping index jumps by more than
  ``gap``. Every group then holds one line's soundings, which share that
  line's drift -- the case in which averaging their covariance is right.

  :param X: Sounding positions, ``(n, 2)``.
  :param ping: Ping index of each sounding, ``(n,)``.
  :param cell: Cell side in metres.
  :param gap: Ping jump that starts a new line.
  """
  key = np.floor(X / cell).astype(np.int64)
  order = np.lexsort((ping, key[:, 1], key[:, 0]))
  key, ping = key[order], np.asarray(ping)[order]
  new_cell = np.any(np.diff(key, axis=0) != 0, axis=1)
  new_line = np.diff(ping) > gap
  starts = np.flatnonzero(np.concatenate([[True], new_cell | new_line]))
  return np.split(order, starts[1:])


def aggregate_covariance(
  cov: np.ndarray, groups: list[np.ndarray]
) -> np.ndarray:
  """Position covariance of each decimated cell: the **mean** of its members'.

  The one place :func:`decimate`'s median convention has to break. A cell's
  soundings are mostly beams of the same ping, and they share the vehicle's
  whole position error -- so the cell is no better placed than any one of
  them, and dividing by the count would treat correlated beams as independent
  fixes. The mean rather than the median because a mean of positive
  semi-definite matrices is one, and an element-wise median need not be.

  :param cov: Per-sounding covariance, ``(n, 2, 2)``.
  :param groups: From :func:`cell_groups`.
  :return: ``(n_cells, 2, 2)``.
  """
  return np.stack([cov[group].mean(axis=0) for group in groups])


def blocked_split(
  X: np.ndarray, fraction: float, cell: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
  """Split by withholding whole spatial cells.

  :param X: Sounding positions, ``(n, 2)``.
  :param fraction: Fraction of occupied cells to withhold.
  :param cell: Cell side in metres.
  :param seed: Seed for choosing cells.
  :return: Boolean ``(train, test)`` masks over the soundings.
  """
  _, index = np.unique(
    np.floor(X / cell).astype(np.int64), axis=0, return_inverse=True
  )
  n_cells = int(index.max()) + 1
  chosen = np.random.default_rng(seed).permutation(n_cells)[
    : round(fraction * n_cells)
  ]
  test = np.isin(index, chosen)
  return ~test, test


def calibration(bathymetry, X, y, test, input_variance=None) -> float:
  """Fraction of held-out soundings inside the map's own 95% interval.

  Accuracy alone cannot tell whether a map's uncertainty is worth propagating
  into a filter. This can: the paper's claim is that ``cov_M`` is the survey's
  posterior uncertainty rather than a tuning parameter, which makes it
  falsifiable. A map reporting 60% coverage is overconfident by exactly the
  amount that would make the smoother's update too sure of itself.

  :param input_variance: Each held-out sounding's own input noise,
      from :func:`input_noise`, for a survey placed by navigation. A held-out sounding
      is misplaced like any other, so its interval must allow for that too;
      leaving it out scores a navigated map as overconfident when it is only
      being compared against misplaced truth.
  """
  predicted, spread = bathymetry.predict(
    X[test], with_std=True, observation_noise=True
  )
  if input_variance is not None:
    spread = np.sqrt(spread**2 + input_variance)
  return float((np.abs(predicted - y[test]) <= 1.96 * spread).mean())


def input_noise(bathymetry, X, cov, test) -> np.ndarray:
  """Placement-induced depth variance at each held-out sounding.

  :func:`~auv_pose.mapping.vecchia.input_noise_variance` at the map's own slope.
  """
  slope = bathymetry.mean_gradient(X[test].astype(np.float64))
  return input_noise_variance(slope, cov[test])


def score(
  bathymetry, X, y, train, test, neighbours: int = 4, input_variance=None
) -> float:
  """Print held-out error beside a nearest-neighbour baseline.

  :return: The map's held-out rmse, so callers comparing two maps need not
      re-predict.
  """
  predicted = bathymetry.predict(X[test])
  gp_rmse = float(np.sqrt(((predicted - y[test]) ** 2).mean()))

  finder = NearestNeighbors(n_neighbors=neighbours).fit(X[train])
  _, nearest = finder.kneighbors(X[test])
  knn_rmse = float(
    np.sqrt(((y[train][nearest].mean(axis=1) - y[test]) ** 2).mean())
  )

  print(f"  held out {test.sum()} soundings in whole cells")
  print(f"  GP rmse                      {gp_rmse:7.3f} m")
  print(f"  mean of {neighbours} nearest soundings  {knn_rmse:7.3f} m")
  print(
    f"  predicting the mean depth    {float(np.sqrt(((y[train].mean() - y[test]) ** 2).mean())):7.3f} m"
  )
  print(
    "  inside its own 95% interval  "
    f"{calibration(bathymetry, X, y, test, input_variance):7.1%}"
  )
  # Where the coverage fails says which term is wrong: on low input noise it is
  # the nugget, on high it is the NIGP inflation. Nothing to split for a survey
  # placed from truth, whose input noise is zero throughout.
  if input_variance is not None and np.ptp(input_variance) > 0.0:
    edges = np.quantile(input_variance, [0.0, 1 / 3, 2 / 3, 1.0])
    thirds = np.digitize(input_variance, edges[1:-1])
    indices = np.flatnonzero(test)
    parts = []
    for third in range(3):
      if not np.any(thirds == third):
        continue
      subset = np.zeros_like(test)
      subset[indices[thirds == third]] = True
      coverage = calibration(
        bathymetry, X, y, subset, input_variance[thirds == third]
      )
      parts.append(f"{coverage:.1%} below {edges[third + 1]:.3g} m^2")
    print(f"  by input noise, in thirds      {'; '.join(parts)}")
  if gp_rmse > knn_rmse:
    print(
      "  *** worse than averaging its neighbours. Either the fit has not "
      "converged -- check the ELBO trace above -- or the soundings carry "
      "structure finer than the survey resolves, which no model recovers. "
      "Fitting a known analytic surface at the same sounding positions tells "
      "the two apart ***"
    )
  return gp_rmse


def describe(vecchia: VecchiaMap) -> None:
  """Print a fitted Vecchia map's convergence and hyperparameters."""
  print(f"  fitted on {vecchia.fit_device}")
  trace = vecchia.loglik_trace
  earlier = trace[max(-len(trace), -(len(trace) // 10) - 1)]
  print(
    f"  restricted log likelihood {trace[0]:.1f} -> {trace[-1]:.1f}; "
    f"last tenth improved by {trace[-1] - earlier:.3f} "
    "(near zero means converged)"
  )
  hyper = vecchia.hyper
  short = hyper.short_lengthscale
  if short is not None:
    print(
      f"  long term:  lengthscales {np.round(hyper.lengthscale, 2)} m, "
      f"amplitude {hyper.amplitude:.3f} m^2"
    )
    print(
      f"  short term: lengthscales {np.round(short, 2)} m, "
      f"amplitude {hyper.short_amplitude:.3f} m^2"
    )
    print(f"  nugget {hyper.noise:.4f} m^2")
  else:
    print(f"  lengthscales {np.round(vecchia.lengthscale, 2)} m")
    print(
      f"  amplitude {hyper.amplitude:.3f} m^2, nugget {hyper.noise:.4f} m^2"
    )
  print(f"  mean coefficients {np.round(vecchia.beta, 4)}")


def report_truth(
  vecchia: VecchiaMap, truth: np.ndarray, test: np.ndarray
) -> None:
  """Score against where the held-out soundings truly lie.

  They are misplaced too, horizontally and in depth, so scoring against them
  measures self-consistency. Against their true position and depth it
  measures the map.
  """
  at_truth = vecchia.predict(truth[test, :2])
  error = np.sqrt(np.mean((at_truth - truth[test, 2]) ** 2))
  print(f"  rmse at the true positions      {error:.3f} m")


def main() -> None:
  args = parse_args()
  if args.place_by == "truth" and args.nigp_passes > 0:
    raise SystemExit(
      "--place-by truth has no placement error for NIGP to model"
    )
  if args.nigp_passes == 1:
    raise SystemExit(
      "--nigp-passes 1 is a plain fit: the first pass has no inflation yet. "
      "Use 0 for off, or 2 for the paper's procedure"
    )
  if args.out is None:
    primary = "svgp" if args.method == "svgp" else "vecchia"
    args.out = Path(f"{primary}_bathymetry.pkl")

  refuse_overwrite(args.out, args.force)
  if not args.no_plot:
    refuse_overwrite(args.plot, args.force)

  frame = load_soundings(args.surveys)
  X, y = soundings_to_arrays(frame)
  # Both ride along with X through every filter below, row for row. The
  # covariance is read whenever the survey carries it, so a map fitted without
  # NIGP is still scored against its misplaced holdout fairly.
  cov = (
    placement_covariance(frame)
    if args.nigp_passes > 0 or set(COVARIANCE_COLUMNS) <= set(frame.columns)
    else None
  )
  ping = frame["ping"].to_numpy() if "ping" in frame.columns else None
  if args.decimate_per_line and ping is None:
    raise SystemExit("--decimate-per-line needs a ping column")
  truth = (
    frame[["true_x", "true_y", "true_z"]].to_numpy(np.float64)
    if {"true_x", "true_y", "true_z"} <= set(frame.columns)
    else None
  )
  print(
    f"Loaded {len(X)} soundings from {', '.join(str(p) for p in args.surveys)}"
  )

  if args.bounds is not None:
    x_min, x_max, y_min, y_max = args.bounds
    inside = (
      (X[:, 0] >= x_min)
      & (X[:, 0] <= x_max)
      & (X[:, 1] >= y_min)
      & (X[:, 1] <= y_max)
    )
    print(
      f"Kept {int(inside.sum())} soundings inside {args.bounds} "
      f"({100 * inside.mean():.1f}%)"
    )
    if not inside.any():
      raise SystemExit("no soundings inside --bounds")
    X, y = X[inside], y[inside]
    cov = None if cov is None else cov[inside]
    truth = None if truth is None else truth[inside]
    ping = None if ping is None else ping[inside]

  if args.max_elevation is not None:
    deep = y <= args.max_elevation
    print(
      f"Kept {int(deep.sum())} soundings at or below {args.max_elevation} m "
      f"({100 * deep.mean():.1f}%); dropped {int((~deep).sum())} shallow returns"
    )
    if not deep.any():
      raise SystemExit("no soundings below --max-elevation")
    X, y = X[deep], y[deep]
    cov = None if cov is None else cov[deep]
    truth = None if truth is None else truth[deep]
    ping = None if ping is None else ping[deep]

  if args.drop_objects:
    objects = object_soundings(
      X, y, window=args.object_window, core=args.object_core
    )
    print(
      f"Dropped {int(objects.sum())} object soundings "
      f"({100 * objects.mean():.1f}%): cores over {args.object_core} m above a "
      f"{args.object_window} m opening of the seabed, and their flanks"
    )
    seabed = ~objects
    X, y = X[seabed], y[seabed]
    cov = None if cov is None else cov[seabed]
    truth = None if truth is None else truth[seabed]
    ping = None if ping is None else ping[seabed]

  # Held-out pings leave before decimation, so none of their soundings is
  # merged into a training cell: they stay raw, at their own positions.
  held = None
  if args.holdout > 0 and args.holdout_by == "ping":
    if ping is None:
      raise SystemExit("--holdout-by ping needs a ping column")
    pings = np.unique(ping)
    chosen = np.random.default_rng(args.seed).permutation(pings)[
      : round(args.holdout * len(pings))
    ]
    out = np.isin(ping, chosen)
    held = (
      X[out],
      y[out],
      None if cov is None else cov[out],
      None if truth is None else truth[out],
    )
    X, y = X[~out], y[~out]
    cov = None if cov is None else cov[~out]
    truth = None if truth is None else truth[~out]
    ping = ping[~out]
    print(
      f"Held out {len(chosen)} of {len(pings)} pings: {int(out.sum())} soundings"
    )

  if args.decimate_cell > 0:
    # Only a blocked split has cells for decimation to merge across.
    if held is None and args.decimate_cell >= args.holdout_cell:
      raise SystemExit(
        f"--decimate-cell {args.decimate_cell} is not smaller than "
        f"--holdout-cell {args.holdout_cell}; decimation would merge "
        "soundings across the boundary the blocked split relies on"
      )
    before = len(X)
    if args.decimate_per_line:
      assert ping is not None  # checked on load
      groups = line_groups(X, ping, args.decimate_cell)
    else:
      groups = cell_groups(X, args.decimate_cell)
    if cov is not None:
      cov = aggregate_covariance(cov, groups)
    if truth is not None:
      truth = np.stack([np.median(truth[g], axis=0) for g in groups])
    X = np.stack([np.median(X[g], axis=0) for g in groups]).astype(X.dtype)
    y = np.asarray([np.median(y[g]) for g in groups], dtype=y.dtype)
    print(
      f"Decimated to {len(X)} soundings "
      f"({before / max(len(X), 1):.1f}x) at {args.decimate_cell} m cells"
    )

  if held is not None:
    held_X, held_y, held_cov, held_truth = held
    train = np.concatenate([np.ones(len(X), bool), np.zeros(len(held_X), bool)])
    test = ~train
    X = np.concatenate([X, held_X.astype(X.dtype)])
    y = np.concatenate([y, held_y.astype(y.dtype)])
    if cov is not None and held_cov is not None:
      cov = np.concatenate([cov, held_cov])
    if truth is not None and held_truth is not None:
      truth = np.concatenate([truth, held_truth])
    from scipy.spatial import KDTree

    gap = KDTree(X[train]).query(X[test])[0]
    print(
      "  held-out soundings from the nearest fitted one: median "
      f"{np.median(gap):.2f} m, 90th {np.percentile(gap, 90):.2f} m"
    )
  elif args.holdout > 0:
    train, test = blocked_split(X, args.holdout, args.holdout_cell, args.seed)
  else:
    train = np.ones(len(X), dtype=bool)
    test = np.zeros(len(X), dtype=bool)

  if args.place_by == "truth":
    # After every selection, never before: selections made on true positions
    # would hand the control a different set of soundings, and a comparison of
    # two maps on different soundings is not a comparison.
    if truth is None:
      raise SystemExit("--place-by truth needs true_x, true_y, true_z columns")
    X, y = truth[:, :2], truth[:, 2]
    cov = None
    print("Fitting the true positions and depths of the same soundings")

  fitted: dict[str, BathymetryMap | VecchiaMap] = {}
  # What save_map needs, and only exists once the SVGP has been fitted.
  svgp_checkpoint: tuple | None = None
  rmse: dict[str, float] = {}

  if args.method in ("svgp", "both"):
    x_scaler = StandardScaler().fit(X[train])
    y_mean, y_std = float(y[train].mean()), float(y[train].std())

    train_x = torch.tensor(x_scaler.transform(X[train]), dtype=torch.float32)
    train_y = torch.tensor((y[train] - y_mean) / y_std, dtype=torch.float32)

    print(
      f"Fitting SVGP on {int(train.sum())} soundings: "
      f"{args.inducing} inducing points, {args.epochs} epochs"
    )
    model, likelihood, inducing_points = fit_svgp(
      train_x,
      train_y,
      n_inducing=args.inducing,
      epochs=args.epochs,
      batch_size=args.batch_size,
      seed=args.seed,
      device=args.device,
    )

    print(f"  fitted on {model.fit_device}")
    trace = model.elbo_trace
    print(
      f"  negative ELBO {trace[0]:.4f} -> {trace[-1]:.4f}; "
      f"last tenth improved by {trace[max(-len(trace), -(len(trace) // 10) - 1)] - trace[-1]:.4f} "
      "(near zero means converged)"
    )
    lengthscale = (
      model.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
      * x_scaler.scale_
    )
    print(f"  lengthscales {np.round(lengthscale, 2)} m")

    svgp = BathymetryMap(
      model, likelihood, x_scaler, y_mean, y_std, device=model.fit_device
    )
    svgp_checkpoint = (
      model,
      likelihood,
      inducing_points,
      x_scaler,
      y_mean,
      y_std,
    )
    fitted["svgp"] = svgp
    if test.any():
      rmse["svgp"] = score(svgp, X, y, train, test)

  if args.method in ("vecchia", "both"):
    split = (
      "all nearest"
      if args.near is None
      else f"{args.near} nearest + {args.conditioning - args.near} spread"
    )
    mean = f"{args.mean} mean"
    print(
      f"Fitting Vecchia on {int(train.sum())} soundings: "
      f"m={args.conditioning} ({split}), {mean}, {args.steps} steps"
    )
    options = {
      "m": args.conditioning,
      "steps": args.steps,
      "device": args.device,
      "near": args.near,
      "mean": args.mean,
    }
    inflation = None
    if args.nigp_passes == 0 and args.short_lengthscale is not None:
      # The single-scale fit is both the two-scale fit's starting point and
      # the baseline it has to beat, so it is scored and kept too.
      single = fit_vecchia(
        X[train].astype(np.float64), y[train].astype(np.float64), **options
      )
      print("  single scale:")
      describe(single)
      if test.any():
        score(single, X, y, train, test)
        if truth is not None:
          report_truth(single, truth, test)
      single_path = args.out.with_name(
        f"{args.out.stem}_single{args.out.suffix}"
      )
      save_vecchia_map(single_path, single)
      print(f"  wrote {single_path}")

      print(
        f"  two scales, short term from {args.short_lengthscale} m, long from "
        "the single-scale fit:"
      )
      vecchia = fit_vecchia(
        X[train].astype(np.float64),
        y[train].astype(np.float64),
        short_lengthscale=args.short_lengthscale,
        initial=single.hyper,
        **options,
      )
    elif args.nigp_passes == 0:
      vecchia = fit_vecchia(
        X[train].astype(np.float64), y[train].astype(np.float64), **options
      )
    else:
      vecchia, inflations = fit_vecchia_nigp(
        X[train].astype(np.float64),
        y[train].astype(np.float64),
        np.asarray(cov)[train],
        passes=args.nigp_passes,
        short_lengthscale=args.short_lengthscale,
        **options,
      )
      # The last fit used inflations[-2]; inflations[-1] is what a further
      # pass would use, so their difference is whether it has settled.
      inflation = inflations[-2]
      print(
        f"  NIGP, {args.nigp_passes} passes: input-noise variance used, median "
        f"{np.median(inflation):.4f}, 95th "
        f"{np.percentile(inflation, 95):.4f} m^2; a further pass would change "
        f"it by {np.abs(inflations[-1] - inflation).max():.4f} m^2 at most"
      )

    describe(vecchia)
    if inflation is not None:
      # Whether NIGP had anything to do: an inflation far below the nugget
      # means the fit would have been the same without it.
      print(
        "  median input noise / nugget "
        f"{np.median(inflation) / vecchia.hyper.noise:.3f}"
      )
    fitted["vecchia"] = vecchia
    if test.any():
      rmse["vecchia"] = score(
        vecchia,
        X,
        y,
        train,
        test,
        input_variance=None
        if cov is None
        else input_noise(vecchia, X, cov, test),
      )
      if truth is not None:
        report_truth(vecchia, truth, test)

  if len(rmse) == 2:
    better, worse = sorted(rmse, key=lambda name: rmse[name])
    margin = 100 * (1 - rmse[better] / rmse[worse])
    print(
      f"\n{better} wins by {margin:.1f}% on held-out rmse "
      f"({rmse[better]:.3f} m against {rmse[worse]:.3f} m)"
    )

  primary = "vecchia" if "vecchia" in fitted else "svgp"
  bathymetry = fitted[primary]

  if isinstance(bathymetry, VecchiaMap):
    save_vecchia_map(args.out, bathymetry)
  else:
    assert svgp_checkpoint is not None  # set wherever an SVGP was fitted
    save_map(args.out, *svgp_checkpoint)
  print(f"Wrote {args.out}")

  if args.no_plot:
    return

  render_surface(bathymetry, frame, args.grid, args.plot)
  print(f"Wrote {args.plot}")


def render_surface(bathymetry, frame, resolution: int, path: Path) -> None:
  """Evaluate the map on a regular grid and save a 3-D surface."""
  xx, yy = np.meshgrid(
    np.linspace(frame["x"].min(), frame["x"].max(), resolution),
    np.linspace(frame["y"].min(), frame["y"].max(), resolution),
  )
  elevation = bathymetry.predict(np.column_stack([xx.ravel(), yy.ravel()]))

  figure = plt.figure(figsize=(14, 10))
  axes = figure.add_subplot(111, projection="3d")
  surface = axes.plot_surface(
    xx,
    yy,
    elevation.reshape(xx.shape),
    cmap="viridis",
    linewidth=0,
    antialiased=True,
  )
  figure.colorbar(surface, shrink=0.6, aspect=15, label="Seabed z (m)")

  axes.set_xlabel("X (m)")
  axes.set_ylabel("Y (m)")
  axes.set_zlabel("Seabed z (m)")
  axes.set_title("Gaussian process bathymetry surface")
  # The map now stores upward elevation rather than downward depth, so the axis
  # is no longer inverted -- doing both would flip the seabed twice.

  figure.tight_layout()
  figure.savefig(path, dpi=300, bbox_inches="tight")
  plt.close(figure)


if __name__ == "__main__":
  main()
