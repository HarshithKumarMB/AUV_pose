"""The raw multibeam capture format, and what it decodes to.

A capture is a ``.npz`` of every ping a flight recorded: the intensity images,
and the pose and attitude each was taken at. It exists because scoring a sonar
needs the *images*, not the soundings -- whether a beam's true echo is present
at all is a question about the profile, and a survey CSV has already thrown
that away.

``survey.py`` and ``check_multibeam.py`` write it; ``check_beam_validity.py``
reads it. The schema lives here rather than in any of them because it was
previously written in two places and unpacked in two more, with the sonar
settings re-derived by hand each time and nothing to disagree with.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from auv_pose.mapping.sonar import (
  azimuth_angles,
  bottom_return_ranges,
  range_bins,
)

__all__ = ["SONAR_KEYS", "Capture", "write_capture"]

#: The sonar settings stored alongside the images. Without them a capture is
#: uninterpretable -- a range bin means nothing without its bounds and count.
SONAR_KEYS = (
  "range_min",
  "range_max",
  "range_bins",
  "azimuth",
  "azimuth_bins",
  "elevation",
)


@dataclass(frozen=True)
class Capture:
  """One flight's worth of pings, with the geometry needed to interpret them.

  Attributes:
      images: ``(pings, range_bins, azimuth_bins)`` intensities.
      positions: ``(pings, 3)`` world-frame vehicle positions.
      rotations: ``(pings, 3, 3)`` body-to-world rotations.
      ranges: Range of each bin, metres.
      bearings: Bearing of each beam from nadir, radians.
      beam_ranges: ``(pings, beams)`` picked bottom range, NaN where a beam had
          no discernible echo.
      settings: The raw sonar block, keyed by :data:`SONAR_KEYS`.
  """

  images: NDArray[np.float64]
  positions: NDArray[np.float64]
  rotations: NDArray[np.float64]
  ranges: NDArray[np.float64]
  bearings: NDArray[np.float64]
  beam_ranges: NDArray[np.float64]
  settings: dict[str, float]

  @property
  def n_pings(self) -> int:
    return len(self.images)

  @property
  def n_beams(self) -> int:
    return len(self.bearings)

  @property
  def bin_width(self) -> float:
    """Range quantisation, metres. The floor on any range measurement."""
    return float(self.ranges[1] - self.ranges[0])

  def describe(self) -> str:
    return (
      f"{self.n_pings} pings, {self.n_beams} beams, "
      f"azimuth {self.settings['azimuth']:.1f} deg, "
      f"{self.settings['range_max']:.1f} m in "
      f"{int(self.settings['range_bins'])} bins ({self.bin_width:.4f} m)"
    )

  @classmethod
  def load(cls, path: str | Path) -> Self:
    """Read a capture and decode its geometry.

    Raises:
        SystemExit: If the file is missing a key, which in practice means it
            was written before the schema settled rather than corrupted.
    """
    data = np.load(path)
    required = ("images", "positions", "rotations", *SONAR_KEYS)
    missing = [key for key in required if key not in data]
    if missing:
      raise SystemExit(f"{path} is missing {missing}; rewrite the capture")

    settings = {key: float(data[key]) for key in SONAR_KEYS}
    ranges = range_bins(
      settings["range_min"],
      settings["range_max"],
      int(settings["range_bins"]),
    )
    bearings = azimuth_angles(
      settings["azimuth"], int(settings["azimuth_bins"])
    )
    images = np.asarray(data["images"], dtype=float)

    return cls(
      images=images,
      positions=np.asarray(data["positions"], dtype=float),
      rotations=np.asarray(data["rotations"], dtype=float),
      ranges=ranges,
      bearings=bearings,
      beam_ranges=np.array(
        [bottom_return_ranges(image, ranges) for image in images]
      ),
      settings=settings,
    )

  def beam_directions(self, ping: int, nadir, swath) -> NDArray[np.float64]:
    """World-frame unit direction of every beam for one ping.

    The fan is defined about the vehicle, so the body-frame directions have to
    be rotated by that ping's attitude before anything can be traced against
    the world.
    """
    body = np.cos(self.bearings)[:, None] * np.asarray(
      nadir, dtype=float
    ) + np.sin(self.bearings)[:, None] * np.asarray(swath, dtype=float)
    return body @ self.rotations[ping].T


def write_capture(
  path: str | Path,
  images: list[np.ndarray],
  positions: list[np.ndarray],
  rotations: list[np.ndarray],
  settings: dict[str, float],
) -> None:
  """Write a capture, compressed.

  Args:
      path: Output ``.npz``.
      images: Per-ping intensity images.
      positions: Per-ping world positions.
      rotations: Per-ping body-to-world rotations.
      settings: The sonar block; must cover :data:`SONAR_KEYS`.

  Raises:
      ValueError: If ``settings`` is missing a key, which would produce a
          capture nothing can decode.
  """
  missing = set(SONAR_KEYS) - set(settings)
  if missing:
    raise ValueError(f"sonar settings missing {sorted(missing)}")

  payload: dict[str, Any] = {
    "images": np.asarray(images),
    "positions": np.asarray(positions),
    "rotations": np.asarray(rotations),
    **{key: settings[key] for key in SONAR_KEYS},
  }
  np.savez_compressed(path, **payload)
