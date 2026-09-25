"""Fit the bathymetry map on every sounding of every survey.

    python experiments/train_map.py ~/data/auv_pose/surveys_v4/pass*_smoothed.csv

Writes the map checkpoint and a render of the fitted seabed beside it. The map
is the Vecchia GP: plane mean, one Matérn-5/2 term, ``m = 30``, 2000 steps.
``--method svgp`` fits the SVGP baseline instead. Nothing is held out: the map
is scored on an independently flown test track by ``score_track.py``.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from auv_pose.io.checkpoints import save_map, save_vecchia_map
from auv_pose.io.soundings import load_soundings, soundings_to_arrays
from auv_pose.mapping.svgp import BathymetryMap, fit_svgp
from auv_pose.mapping.vecchia import VecchiaMap, fit_vecchia
from experiments.cli import refuse_overwrite

CONDITIONING = 30
STEPS = 2000
INDUCING = 500
EPOCHS = 200
BATCH_SIZE = 5000
GRID = 200


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("surveys", nargs="+", type=Path, help="soundings CSVs")
  parser.add_argument("--out", type=Path, default=Path("bathymetry.pkl"))
  parser.add_argument(
    "--method", choices=("vecchia", "svgp"), default="vecchia"
  )
  parser.add_argument(
    "--near",
    type=int,
    default=None,
    metavar="M_NEAR",
    help=(
      f"how many of the {CONDITIONING} conditioning soundings are nearest "
      "neighbours; the rest are spread across the ordering. Default all nearest"
    ),
  )
  parser.add_argument("--device", default=None, help="torch device")
  parser.add_argument("--force", action="store_true", help="overwrite --out")
  return parser.parse_args()


def last_tenth(trace: list[float]) -> float:
  """How much the objective moved over the last tenth of the run."""
  step = max(1, len(trace) // 10)
  return trace[-1] - trace[max(0, len(trace) - 1 - step)]


def fit_vecchia_map(X, y, args) -> VecchiaMap:
  fitted = fit_vecchia(
    X.astype(np.float64),
    y.astype(np.float64),
    m=CONDITIONING,
    steps=STEPS,
    device=args.device,
    near=args.near,
  )
  trace = fitted.loglik_trace
  print(
    f"restricted log likelihood {trace[0]:.1f} -> {trace[-1]:.1f}, last tenth "
    f"{last_tenth(trace):+.3f}; lengthscales {np.round(fitted.lengthscale, 2)} "
    f"m, amplitude {fitted.hyper.amplitude:.3f} m^2, nugget "
    f"{fitted.hyper.noise:.4f} m^2"
  )
  return fitted


def fit_svgp_map(X, y, args) -> tuple[BathymetryMap, tuple]:
  """The SVGP baseline, and what :func:`save_map` needs to write it."""
  x_scaler = StandardScaler().fit(X)
  y_mean, y_std = float(y.mean()), float(y.std())
  model, likelihood, inducing = fit_svgp(
    torch.tensor(x_scaler.transform(X), dtype=torch.float32),
    torch.tensor((y - y_mean) / y_std, dtype=torch.float32),
    n_inducing=INDUCING,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    seed=0,
    device=args.device,
  )
  print(f"negative ELBO last tenth {last_tenth(model.elbo_trace):+.4f}")
  bathymetry = BathymetryMap(
    model, likelihood, x_scaler, y_mean, y_std, device=model.fit_device
  )
  return bathymetry, (model, likelihood, inducing, x_scaler, y_mean, y_std)


def render_surface(bathymetry, X: np.ndarray, path: Path) -> None:
  """Evaluate the map on a regular grid over the survey and save a 3-D surface."""
  low, high = X.min(axis=0), X.max(axis=0)
  xx, yy = np.meshgrid(
    np.linspace(low[0], high[0], GRID), np.linspace(low[1], high[1], GRID)
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
  figure.tight_layout()
  figure.savefig(path, dpi=300, bbox_inches="tight")
  plt.close(figure)


def main() -> None:
  args = parse_args()
  plot = args.out.with_suffix(".png")
  refuse_overwrite(args.out, args.force)
  refuse_overwrite(plot, args.force)

  X, y = soundings_to_arrays(load_soundings(args.surveys))
  print(f"Fitting {args.method} on {len(y)} soundings")
  if args.method == "vecchia":
    bathymetry = fit_vecchia_map(X, y, args)
    save_vecchia_map(args.out, bathymetry)
  else:
    bathymetry, checkpoint = fit_svgp_map(X, y, args)
    save_map(args.out, *checkpoint)
  print(f"Wrote {args.out}")

  render_surface(bathymetry, X, plot)
  print(f"Wrote {plot}")


if __name__ == "__main__":
  main()
