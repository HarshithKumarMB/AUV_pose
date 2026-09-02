"""Query the fitted bathymetry map at one or more positions.

    python experiments/predict_depth.py                    # the default probe point
    python experiments/predict_depth.py -19.41 -9.42
    python experiments/predict_depth.py 0 0 10 5 --std
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from auv_pose.io.checkpoints import load_map

# A fixed probe point, so the reported depth stays comparable across refits.
DEFAULT_POINT = (-19.412034324804928, -9.419920050422297)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "coordinates", nargs="*", type=float,
        help="x y pairs; defaults to the reference probe point",
    )
    parser.add_argument("--map", type=Path, default=Path("svgp_bathymetry.pkl"))
    parser.add_argument("--std", action="store_true", help="also report uncertainty")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.coordinates:
        points = np.array([DEFAULT_POINT])
    elif len(args.coordinates) % 2:
        raise SystemExit("coordinates must be x y pairs")
    else:
        points = np.array(args.coordinates).reshape(-1, 2)

    bathymetry = load_map(args.map)

    if args.std:
        depth, std = bathymetry.predict(points, with_std=True)
        for (x, y), d, s in zip(points, depth, std):
            print(f"({x:.4f}, {y:.4f}) -> {d:.5f} +/- {s:.5f} m")
    else:
        for (x, y), d in zip(points, bathymetry.predict(points)):
            print(f"({x:.4f}, {y:.4f}) -> {d:.5f} m")


if __name__ == "__main__":
    main()
