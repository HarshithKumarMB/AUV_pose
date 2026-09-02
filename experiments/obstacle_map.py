"""Labelled obstacle map: classify a 3-D point, then snap it to a known obstacle.

Shared by ``sidescan_nav.py`` and ``obstacle_predict.py``. Lives in ``experiments``
rather than ``auv_pose`` because it is bound to the specific LightGBM artefacts and
CSV schema of this study, not a general algorithm.

Load once via :meth:`load` -- the classifier is 16 MB, and both it and the CSVs are
queried per sonar tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from numpy.typing import NDArray

OBSTACLE_COLUMNS = ("x", "y", "z", "obstacle_type")


@dataclass(frozen=True)
class Match:
    """The nearest known obstacle point to a query."""

    obstacle_type: str
    position: NDArray[np.float64]
    distance: float


def load_obstacles(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Read and clean labelled obstacle CSVs.

    Coerces coordinates to numeric and drops unusable rows -- including the
    duplicate header rows embedded in the original data, which is why
    ``obstacle_type == "obstacle_type"`` is filtered out.
    """
    frames = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        missing = set(OBSTACLE_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
        frames.append(frame[list(OBSTACLE_COLUMNS)])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["obstacle_type"] != "obstacle_type"]

    for axis in ("x", "y", "z"):
        combined[axis] = pd.to_numeric(combined[axis], errors="coerce")

    return combined.dropna(subset=list(OBSTACLE_COLUMNS)).reset_index(drop=True)


class ObstacleMap:
    """A LightGBM obstacle classifier plus the point cloud it was trained on."""

    def __init__(self, frame: pd.DataFrame, classifier, encoder) -> None:
        self.classifier = classifier
        self.encoder = encoder
        # Grouped after cleaning, so every group holds numeric coordinates.
        self._points = {
            obstacle: group[["x", "y", "z"]].to_numpy(dtype=float)
            for obstacle, group in frame.groupby("obstacle_type")
        }
        # One tree per class, built once: nearest() runs per sonar tick and per
        # point-cloud row, and a linear scan there is O(cloud size) every time.
        self._trees = {
            obstacle: cKDTree(points) for obstacle, points in self._points.items()
        }

    @classmethod
    def load(
        cls,
        obstacle_paths: Iterable[str | Path],
        classifier_path: str | Path,
        encoder_path: str | Path,
    ) -> "ObstacleMap":
        return cls(
            load_obstacles(obstacle_paths),
            joblib.load(classifier_path),
            joblib.load(encoder_path),
        )

    @property
    def obstacle_types(self) -> list[str]:
        return sorted(self._points)

    def classify(self, points: NDArray) -> list[str]:
        """Predict the obstacle type of each ``(x, y, z)`` row.

        Batch whenever you can: each call carries fixed sklearn validation and
        LightGBM booster overhead that dwarfs the per-row prediction cost.
        """
        frame = pd.DataFrame(np.atleast_2d(points), columns=["x", "y", "z"])
        return list(self.encoder.inverse_transform(self.classifier.predict(frame)))

    def nearest(
        self, point: Sequence[float], obstacle_type: str | None = None
    ) -> Match | None:
        """Find the closest known obstacle to ``point``.

        Args:
            point: Query position.
            obstacle_type: Predicted class, if already known. Pass it when the
                caller has classified in batch, to skip a single-row prediction.

        Returns None when the predicted class has no points in the map.
        """
        point = np.asarray(point, dtype=float)
        if obstacle_type is None:
            obstacle_type = self.classify(point[None, :])[0]

        tree = self._trees.get(obstacle_type)
        if tree is None or tree.n == 0:
            return None

        distance, index = tree.query(point)

        return Match(
            obstacle_type=obstacle_type,
            position=self._points[obstacle_type][int(index)].copy(),
            distance=float(distance),
        )

    def nearest_many(
        self, points: NDArray, obstacle_types: Sequence[str]
    ) -> tuple[NDArray, NDArray]:
        """Nearest obstacle for many pre-classified points, one query per class.

        Returns ``(positions, distances)``; rows whose class is absent from the map
        are NaN.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        types = np.asarray(obstacle_types)

        positions = np.full((len(points), 3), np.nan)
        distances = np.full(len(points), np.nan)

        for obstacle_type in np.unique(types):
            tree = self._trees.get(obstacle_type)
            if tree is None or tree.n == 0:
                continue
            rows = np.flatnonzero(types == obstacle_type)
            found, index = tree.query(points[rows])
            positions[rows] = self._points[obstacle_type][index]
            distances[rows] = found

        return positions, distances
