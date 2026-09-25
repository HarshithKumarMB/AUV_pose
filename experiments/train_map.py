"""Fit the Vecchia GP bathymetry map from survey soundings.

    python experiments/train_map.py ~/data/auv_pose/surveys_v4/pass*_smoothed.csv

Reads the survey CSVs, fits the map, writes the checkpoint, and renders the
fitted seabed. The defaults are the configuration the map ships in: every
sounding of every survey, §3.2's plane mean, one Matérn-5/2 term, ``m = 30``,
2000 steps. ``--method both`` adds the SVGP baseline.

**The shipped map holds nothing out.** How good it is is measured on an
independently flown test track (``experiments/score_track.py``): soundings the
map never saw, landing between the survey's, which is where the smoother will
query it. The holdouts here are for quick comparisons. ``--holdout-by ping``
resembles use; blocked cells ask a gap-filling question the smoother never
does, and a random split would leak, since consecutive soundings are about a
centimetre apart.

Scores are printed beside the mean of the nearest few training soundings. A
Gaussian process that loses to that is not earning its complexity.
"""

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from auv_pose.io.checkpoints import save_map, save_vecchia_map
from auv_pose.io.soundings import load_soundings, soundings_to_arrays
from auv_pose.mapping.svgp import BathymetryMap, fit_svgp
from auv_pose.mapping.vecchia import VecchiaMap, fit_vecchia
from experiments.cli import refuse_overwrite


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("surveys", nargs="+", type=Path, help="survey CSVs")
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
    help="restrict the fit to this box, metres",
  )
  parser.add_argument(
    "--holdout",
    type=float,
    default=0.0,
    help="fraction withheld for scoring; 0, the default, fits everything",
  )
  parser.add_argument(
    "--holdout-by",
    choices=("cell", "ping"),
    default="cell",
    help=(
      "what --holdout withholds: whole pings, whose soundings land between "
      "the rest as the smoother's beams will, or --holdout-cell squares"
    ),
  )
  parser.add_argument(
    "--holdout-cell",
    type=float,
    default=8.0,
    help=(
      "side of the withheld squares, metres. State it beside every rmse: at "
      "1 m a held-out sounding sits a median 0.31 m from data, at 8 m 1.55 m"
    ),
  )
  parser.add_argument(
    "--decimate-cell",
    type=float,
    default=0.0,
    help=(
      "median sounding per cell of this side, metres; 0, the default, fits "
      "every sounding. Decimating makes the fitted noise the scatter of a "
      "cell median rather than of the one sounding the smoother compares"
    ),
  )
  parser.add_argument(
    "--max-elevation",
    type=float,
    default=None,
    metavar="Z",
    help="drop soundings shallower than Z metres (water-column echoes)",
  )
  parser.add_argument(
    "--place-by",
    choices=("recorded", "truth"),
    default="recorded",
    help=(
      "'truth' fits the true positions and depths of the same soundings -- "
      "every selection is still made on the recorded ones -- so a navigated "
      "map and its control are scored on identical soundings"
    ),
  )
  parser.add_argument(
    "--method",
    choices=("vecchia", "svgp", "both"),
    default="vecchia",
    help="which map to fit; 'both' scores the SVGP baseline beside it",
  )
  parser.add_argument(
    "--conditioning", type=int, default=30, metavar="M", help="the paper's m"
  )
  parser.add_argument(
    "--near",
    type=int,
    default=None,
    metavar="M_NEAR",
    help=(
      "how many of --conditioning are nearest neighbours; the rest are "
      "spread across the ordering. Default all nearest; see fit_vecchia"
    ),
  )
  parser.add_argument(
    "--mean", default="linear", choices=("linear", "quadratic", "cubic")
  )
  parser.add_argument(
    "--steps", type=int, default=2000, help="optimiser steps, run in full"
  )
  parser.add_argument("--inducing", type=int, default=500, help="SVGP only")
  parser.add_argument("--epochs", type=int, default=200, help="SVGP only")
  parser.add_argument("--batch-size", type=int, default=5000, help="SVGP only")
  parser.add_argument("--grid", type=int, default=200, help="plot resolution")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--device", default=None, help="torch device; cuda when available"
  )
  parser.add_argument("--no-plot", action="store_true")
  parser.add_argument(
    "--force", action="store_true", help="overwrite --out / --plot"
  )
  return parser.parse_args()


@dataclass(frozen=True)
class Soundings:
  """Positions and depths, with whatever optional columns rode along.

  Every selection -- bounds, elevation, a ping holdout -- masks all of them
  together, so a column can never fall out of step with the positions.
  """

  X: np.ndarray
  y: np.ndarray
  ping: np.ndarray | None = None
  truth: np.ndarray | None = None

  def __len__(self) -> int:
    return len(self.y)

  def where(self, mask: np.ndarray) -> Self:
    return type(self)(
      self.X[mask],
      self.y[mask],
      None if self.ping is None else self.ping[mask],
      None if self.truth is None else self.truth[mask],
    )

  def then(self, other: Self) -> Self:
    """These soundings followed by ``other``'s."""

    def join(a, b):
      return None if a is None or b is None else np.concatenate([a, b])

    return type(self)(
      np.concatenate([self.X, other.X]),
      np.concatenate([self.y, other.y]),
      join(self.ping, other.ping),
      join(self.truth, other.truth),
    )


def decimate(data: Soundings, cell: float) -> Soundings:
  """Median of every column per horizontal cell.

  **Median, not mean.** A strongest-return picker occasionally lands on the
  wrong feature, and those errors are one-sided, so a mean drags the cell
  toward them. The median ignores a minority of bad beams.
  """
  key = np.floor(data.X / cell).astype(np.int64)
  _, inverse = np.unique(key, axis=0, return_inverse=True)
  order = np.argsort(inverse.ravel(), kind="stable")
  groups = np.split(order, np.cumsum(np.bincount(inverse.ravel()))[:-1])

  def median(values: np.ndarray) -> np.ndarray:
    return np.stack([np.median(values[g], axis=0) for g in groups])

  return Soundings(
    median(data.X).astype(data.X.dtype),
    median(data.y).astype(data.y.dtype),
    None,
    None if data.truth is None else median(data.truth),
  )


def blocked_split(
  X: np.ndarray, fraction: float, cell: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
  """Withhold whole square cells: boolean ``(train, test)`` masks."""
  _, index = np.unique(
    np.floor(X / cell).astype(np.int64), axis=0, return_inverse=True
  )
  chosen = np.random.default_rng(seed).permutation(int(index.max()) + 1)
  test = np.isin(index, chosen[: round(fraction * len(chosen))])
  return ~test, test


def ping_split(
  ping: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
  """Withhold whole pings: boolean ``(train, test)`` masks."""
  pings = np.unique(ping)
  chosen = np.random.default_rng(seed).permutation(pings)
  test = np.isin(ping, chosen[: round(fraction * len(pings))])
  return ~test, test


def calibration(bathymetry, X, y, test) -> float:
  """Fraction of held-out soundings inside the map's own 95% interval.

  The smoother trusts the map by its uncertainty, so an overconfident map --
  60% coverage, say -- makes every update too sure of itself.
  """
  predicted, spread = bathymetry.predict(
    X[test], with_std=True, observation_noise=True
  )
  return float((np.abs(predicted - y[test]) <= 1.96 * spread).mean())


def score(bathymetry, X, y, train, test, neighbours: int = 4) -> float:
  """Print held-out error beside a nearest-neighbour baseline; return rmse."""
  predicted = bathymetry.predict(X[test])
  gp_rmse = float(np.sqrt(((predicted - y[test]) ** 2).mean()))

  finder = NearestNeighbors(n_neighbors=neighbours).fit(X[train])
  _, nearest = finder.kneighbors(X[test])
  knn = y[train][nearest].mean(axis=1)
  knn_rmse = float(np.sqrt(((knn - y[test]) ** 2).mean()))
  flat_rmse = float(np.sqrt(((y[train].mean() - y[test]) ** 2).mean()))

  print(f"  held out {test.sum()} soundings")
  print(f"  GP rmse                      {gp_rmse:7.3f} m")
  print(f"  mean of {neighbours} nearest soundings  {knn_rmse:7.3f} m")
  print(f"  predicting the mean depth    {flat_rmse:7.3f} m")
  print(
    f"  inside its own 95% interval  {calibration(bathymetry, X, y, test):7.1%}"
  )
  if gp_rmse > knn_rmse:
    print("  *** worse than averaging its neighbours ***")
  return gp_rmse


def last_tenth(trace: list[float]) -> float:
  """How much the objective moved over the last tenth of the run."""
  step = max(1, len(trace) // 10)
  return trace[-1] - trace[max(0, len(trace) - 1 - step)]


def fit_svgp_map(X, y, args) -> tuple[BathymetryMap, tuple]:
  """The SVGP baseline, and what :func:`save_map` needs to write it."""
  x_scaler = StandardScaler().fit(X)
  y_mean, y_std = float(y.mean()), float(y.std())
  model, likelihood, inducing = fit_svgp(
    torch.tensor(x_scaler.transform(X), dtype=torch.float32),
    torch.tensor((y - y_mean) / y_std, dtype=torch.float32),
    n_inducing=args.inducing,
    epochs=args.epochs,
    batch_size=args.batch_size,
    seed=args.seed,
    device=args.device,
  )
  lengthscale = (
    model.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
    * x_scaler.scale_
  )
  print(
    f"  fitted on {model.fit_device}; negative ELBO last tenth improved by "
    f"{-last_tenth(model.elbo_trace):.4f}; lengthscales "
    f"{np.round(lengthscale, 2)} m"
  )
  bathymetry = BathymetryMap(
    model, likelihood, x_scaler, y_mean, y_std, device=model.fit_device
  )
  return bathymetry, (model, likelihood, inducing, x_scaler, y_mean, y_std)


def fit_vecchia_map(X, y, args) -> VecchiaMap:
  fitted = fit_vecchia(
    X.astype(np.float64),
    y.astype(np.float64),
    m=args.conditioning,
    steps=args.steps,
    device=args.device,
    near=args.near,
    mean=args.mean,
  )
  trace = fitted.loglik_trace
  print(
    f"  fitted on {fitted.fit_device}; restricted log likelihood "
    f"{trace[0]:.1f} -> {trace[-1]:.1f}, last tenth improved by "
    f"{last_tenth(trace):.3f}"
  )
  print(
    f"  lengthscales {np.round(fitted.lengthscale, 2)} m, amplitude "
    f"{fitted.hyper.amplitude:.3f} m^2, nugget {fitted.hyper.noise:.4f} m^2"
  )
  return fitted


def main() -> None:
  args = parse_args()
  if args.out is None:
    args.out = Path(
      f"{'svgp' if args.method == 'svgp' else 'vecchia'}_bathymetry.pkl"
    )
  refuse_overwrite(args.out, args.force)
  if not args.no_plot:
    refuse_overwrite(args.plot, args.force)

  frame = load_soundings(args.surveys)
  X, y = soundings_to_arrays(frame)
  data = Soundings(
    X,
    y,
    frame["ping"].to_numpy() if "ping" in frame.columns else None,
    frame[["true_x", "true_y", "true_z"]].to_numpy(np.float64)
    if {"true_x", "true_y", "true_z"} <= set(frame.columns)
    else None,
  )
  print(f"Loaded {len(data)} soundings from {len(args.surveys)} surveys")

  if args.bounds is not None:
    x0, x1, y0, y1 = args.bounds
    inside = (
      (data.X[:, 0] >= x0)
      & (data.X[:, 0] <= x1)
      & (data.X[:, 1] >= y0)
      & (data.X[:, 1] <= y1)
    )
    data = data.where(inside)
    print(f"Kept {len(data)} soundings inside {args.bounds}")
  if args.max_elevation is not None:
    data = data.where(data.y <= args.max_elevation)
    print(f"Kept {len(data)} soundings at or below {args.max_elevation} m")
  if not len(data):
    raise SystemExit("no soundings left to fit")

  # Held-out pings leave before decimation, so none of their soundings is
  # merged into a training cell: they are scored raw, where they landed.
  held = None
  if args.holdout > 0 and args.holdout_by == "ping":
    if data.ping is None:
      raise SystemExit("--holdout-by ping needs a ping column")
    keep, out = ping_split(data.ping, args.holdout, args.seed)
    held, data = data.where(out), data.where(keep)

  if args.decimate_cell > 0:
    blocked = held is None and args.holdout > 0
    if blocked and args.decimate_cell >= args.holdout_cell:
      raise SystemExit(
        "--decimate-cell must be well below --holdout-cell, or decimation "
        "merges soundings across the split"
      )
    before = len(data)
    data = decimate(data, args.decimate_cell)
    print(f"Decimated {before} soundings to {len(data)}")

  if held is not None:
    train = np.arange(len(data) + len(held)) < len(data)
    data = data.then(held)
    gap = KDTree(data.X[train]).query(data.X[~train])[0]
    print(
      f"Held out {len(held)} soundings of whole pings, a median "
      f"{np.median(gap):.2f} m from the nearest fitted one"
    )
  elif args.holdout > 0:
    train, _ = blocked_split(data.X, args.holdout, args.holdout_cell, args.seed)
  else:
    train = np.ones(len(data), dtype=bool)
  test = ~train

  if args.place_by == "truth":
    # After every selection, never before: selections made on true positions
    # would hand the control different soundings from the map it controls.
    if data.truth is None:
      raise SystemExit("--place-by truth needs true_x, true_y, true_z columns")
    data = replace(data, X=data.truth[:, :2], y=data.truth[:, 2])
    print("Fitting the true positions and depths of the same soundings")

  X, y = data.X, data.y
  fitted: dict[str, BathymetryMap | VecchiaMap] = {}
  svgp_checkpoint = None
  if args.method in ("svgp", "both"):
    print(f"Fitting SVGP on {int(train.sum())} soundings")
    fitted["svgp"], svgp_checkpoint = fit_svgp_map(X[train], y[train], args)
  if args.method in ("vecchia", "both"):
    split = (
      "all nearest"
      if args.near is None
      else f"{args.near} nearest + {args.conditioning - args.near} spread"
    )
    print(
      f"Fitting Vecchia on {int(train.sum())} soundings: "
      f"m={args.conditioning} ({split}), {args.mean} mean"
    )
    fitted["vecchia"] = fit_vecchia_map(X[train], y[train], args)

  if test.any():
    for name, bathymetry in fitted.items():
      print(f"\n{name}")
      score(bathymetry, X, y, train, test)
      if data.truth is not None:
        # Held-out soundings are misplaced like any other; against where they
        # truly lie, the score measures the map rather than self-consistency.
        at_truth = bathymetry.predict(data.truth[test, :2])
        error = np.sqrt(np.mean((at_truth - data.truth[test, 2]) ** 2))
        print(f"  rmse at the true positions   {error:7.3f} m")

  bathymetry = fitted.get("vecchia", fitted.get("svgp"))
  assert bathymetry is not None
  if isinstance(bathymetry, VecchiaMap):
    save_vecchia_map(args.out, bathymetry)
  else:
    assert svgp_checkpoint is not None
    save_map(args.out, *svgp_checkpoint)
  print(f"Wrote {args.out}")

  if not args.no_plot:
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
    xx, yy, elevation.reshape(xx.shape), cmap="viridis", linewidth=0
  )
  figure.colorbar(surface, shrink=0.6, aspect=15, label="Seabed z (m)")
  axes.set_xlabel("X (m)")
  axes.set_ylabel("Y (m)")
  axes.set_zlabel("Seabed z (m)")
  axes.set_title("Gaussian process bathymetry surface")
  figure.tight_layout()
  figure.savefig(path, dpi=300, bbox_inches="tight")
  plt.close(figure)


if __name__ == "__main__":
  main()
