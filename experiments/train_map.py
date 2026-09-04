"""Fit the SVGP bathymetry map from survey soundings.

    python experiments/train_map.py

Reads the survey CSVs, fits a sparse variational GP, scores it against held-out
soundings, writes the checkpoint, and renders the fitted seabed.

Refitting is also how you refresh a stale checkpoint: pickled scikit-learn scalers
are not portable across versions, and every depth query passes through one.

**Hold out whole cells, not random soundings.** Consecutive soundings along a
survey track are about a centimetre apart, so a random split leaves every
held-out point with a training point almost on top of it and reports something
close to training error. Withholding whole cells makes the model interpolate
across the gap between tracks, which is what a map is actually asked to do.

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

from auv_pose.io.checkpoints import save_map
from auv_pose.io.soundings import load_soundings, soundings_to_arrays
from auv_pose.mapping.svgp import BathymetryMap, fit_svgp
from experiments.cli import refuse_overwrite


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "surveys",
    nargs="+",
    type=Path,
    help=(
      "survey CSVs to fit. Required: this used to default to map.csv and "
      "map1.csv, which were singlebeam surveys in the pre-a1fd5b1 "
      "'x, y, sonar_depth' schema. They could not be migrated -- the vehicle's "
      "own z was never recorded, so seabed elevation is unrecoverable -- and a "
      "default that always failed was worse than none"
    ),
  )
  parser.add_argument("--out", type=Path, default=Path("svgp_bathymetry.pkl"))
  parser.add_argument(
    "--plot", type=Path, default=Path("gp_bathymetry_surface.png")
  )
  parser.add_argument("--inducing", type=int, default=500)
  parser.add_argument("--epochs", type=int, default=200)
  parser.add_argument(
    "--holdout",
    type=float,
    default=0.2,
    help=(
      "fraction of spatial cells withheld for scoring. The checkpoint is "
      "fitted on the remainder, so the number reported describes the map that "
      "is actually written. 0 fits everything and reports nothing"
    ),
  )
  parser.add_argument(
    "--holdout-cell",
    type=float,
    default=1.0,
    help=(
      "side of the cells the holdout is blocked by, metres. Must exceed the "
      "spacing between consecutive soundings (~1 cm) or the split leaks"
    ),
  )
  parser.add_argument(
    "--decimate-cell",
    type=float,
    default=0.25,
    help=(
      "take the median sounding per cell of this side before splitting, "
      "metres. A multibeam run produces millions of soundings; 0.25 m is "
      "comparable to the across-track beam footprint, so it thins redundancy "
      "rather than resolution. Must stay well below --holdout-cell or "
      "decimation merges soundings across the split boundary. 0 disables it"
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
  key = np.floor(X / cell).astype(np.int64)
  _, inverse = np.unique(key, axis=0, return_inverse=True)
  order = np.argsort(inverse, kind="stable")

  reduced_x, reduced_y = [], []
  for group in np.split(order, np.cumsum(np.bincount(inverse))[:-1]):
    reduced_x.append(np.median(X[group], axis=0))
    reduced_y.append(np.median(y[group]))

  return (
    np.asarray(reduced_x, dtype=X.dtype),
    np.asarray(reduced_y, dtype=y.dtype),
  )


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


def score(bathymetry, X, y, train, test, neighbours: int = 4) -> None:
  """Print held-out error beside a nearest-neighbour baseline."""
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
  if gp_rmse > knn_rmse:
    print(
      "  *** worse than averaging its neighbours. Either the fit has not "
      "converged -- check the ELBO trace above -- or the soundings carry "
      "structure finer than the survey resolves, which no model recovers. "
      "Fitting a known analytic surface at the same sounding positions tells "
      "the two apart ***"
    )


def main() -> None:
  args = parse_args()

  refuse_overwrite(args.out, args.force)
  if not args.no_plot:
    refuse_overwrite(args.plot, args.force)

  frame = load_soundings(args.surveys)
  X, y = soundings_to_arrays(frame)
  print(
    f"Loaded {len(X)} soundings from {', '.join(str(p) for p in args.surveys)}"
  )

  if args.decimate_cell > 0:
    if args.decimate_cell >= args.holdout_cell:
      raise SystemExit(
        f"--decimate-cell {args.decimate_cell} is not smaller than "
        f"--holdout-cell {args.holdout_cell}; decimation would merge "
        "soundings across the boundary the blocked split relies on"
      )
    before = len(X)
    X, y = decimate(X, y, args.decimate_cell)
    print(
      f"Decimated to {len(X)} soundings "
      f"({before / max(len(X), 1):.1f}x) at {args.decimate_cell} m cells"
    )

  if args.holdout > 0:
    train, test = blocked_split(X, args.holdout, args.holdout_cell, args.seed)
  else:
    train = np.ones(len(X), dtype=bool)
    test = np.zeros(len(X), dtype=bool)

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
    f"last tenth improved by {trace[-len(trace) // 10 - 1] - trace[-1]:.4f} "
    "(near zero means converged)"
  )
  lengthscale = (
    model.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
    * x_scaler.scale_
  )
  print(f"  lengthscales {np.round(lengthscale, 2)} m")

  bathymetry = BathymetryMap(
    model, likelihood, x_scaler, y_mean, y_std, device=model.fit_device
  )
  if test.any():
    score(bathymetry, X, y, train, test)

  save_map(
    args.out, model, likelihood, inducing_points, x_scaler, y_mean, y_std
  )
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
