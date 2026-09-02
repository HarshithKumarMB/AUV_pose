"""Fit the SVGP bathymetry map from survey soundings.

    python experiments/train_map.py

Reads the survey CSVs, fits a sparse variational GP, writes the checkpoint, and
renders the fitted seabed.

Refitting is also how you refresh a stale checkpoint: pickled scikit-learn scalers
are not portable across versions, and every depth query passes through one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from auv_pose.io.checkpoints import save_map
from auv_pose.io.soundings import load_soundings, soundings_to_arrays
from auv_pose.mapping.svgp import BathymetryMap, fit_svgp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "surveys", nargs="*", default=["map.csv", "map1.csv"], type=Path,
        help="survey CSVs to fit (default: map.csv map1.csv)",
    )
    parser.add_argument("--out", type=Path, default=Path("svgp_bathymetry.pkl"))
    parser.add_argument("--plot", type=Path, default=Path("gp_bathymetry_surface.png"))
    parser.add_argument("--inducing", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--grid", type=int, default=200, help="plot resolution per axis")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frame = load_soundings(args.surveys)
    X, y = soundings_to_arrays(frame)
    print(f"Loaded {len(X)} soundings from {', '.join(str(p) for p in args.surveys)}")

    x_scaler = StandardScaler().fit(X)
    y_mean, y_std = float(y.mean()), float(y.std())

    train_x = torch.tensor(x_scaler.transform(X), dtype=torch.float32)
    train_y = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)

    print(f"Fitting SVGP: {args.inducing} inducing points, {args.epochs} epochs")
    model, likelihood, inducing_points = fit_svgp(
        train_x,
        train_y,
        n_inducing=args.inducing,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    save_map(args.out, model, likelihood, inducing_points, x_scaler, y_mean, y_std)
    print(f"Wrote {args.out}")

    if args.no_plot:
        return

    bathymetry = BathymetryMap(model, likelihood, x_scaler, y_mean, y_std)
    render_surface(bathymetry, frame, args.grid, args.plot)
    print(f"Wrote {args.plot}")


def render_surface(bathymetry, frame, resolution: int, path: Path) -> None:
    """Evaluate the map on a regular grid and save a 3-D surface."""
    xx, yy = np.meshgrid(
        np.linspace(frame["x"].min(), frame["x"].max(), resolution),
        np.linspace(frame["y"].min(), frame["y"].max(), resolution),
    )
    depth = bathymetry.predict(np.column_stack([xx.ravel(), yy.ravel()]))

    figure = plt.figure(figsize=(14, 10))
    axes = figure.add_subplot(111, projection="3d")
    surface = axes.plot_surface(
        xx, yy, depth.reshape(xx.shape), cmap="viridis", linewidth=0, antialiased=True
    )
    figure.colorbar(surface, shrink=0.6, aspect=15, label="Depth (m)")

    axes.set_xlabel("X (m)")
    axes.set_ylabel("Y (m)")
    axes.set_zlabel("Depth (m)")
    axes.set_title("Gaussian process bathymetry surface")
    axes.invert_zaxis()

    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
