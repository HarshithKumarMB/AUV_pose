"""Extracting ranges and seabed points from multibeam sonar images."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def range_bins(
  range_min: float, range_max: float, n_bins: int
) -> NDArray[np.float64]:
  """Range corresponding to each bin of a sonar intensity profile.

  Bin *endpoints*, where :func:`azimuth_angles` uses centres. Deliberate:
  neither convention fits better, and switching shifts every range by half a
  bin.
  """
  return np.linspace(range_min, range_max, n_bins)


def azimuth_angles(azimuth: float, n_bins: int) -> NDArray[np.float64]:
  """Bearing of each beam's bin centre in a multibeam fan, radians from nadir.

  Args:
      azimuth: Total swath width in degrees.
  """
  half = np.radians(azimuth) / 2.0
  edges = np.linspace(-half, half, n_bins + 1)
  return 0.5 * (edges[:-1] + edges[1:])


def bottom_return_ranges(
  image: ArrayLike, ranges: ArrayLike
) -> NDArray[np.float64]:
  """Bottom range for every beam: the strongest return in each column.

  Args:
      image: Intensity, shape ``(range_bins, azimuth_bins)``.
      ranges: Range for each row, length ``range_bins``.

  Returns:
      Range per beam in metres, NaN where a column is flat (no echo), which is
      normal at the swath edges.
  """
  image = np.asarray(image, dtype=float)
  ranges = np.asarray(ranges, dtype=float)

  if image.ndim != 2:
    raise ValueError(
      f"expected a 2-D (range, azimuth) image, got {image.shape}"
    )
  if image.shape[0] != ranges.shape[0]:
    raise ValueError(
      f"image has {image.shape[0]} range bins but {ranges.shape[0]} ranges"
    )

  picked = ranges[np.argmax(image, axis=0)]
  flat = np.ptp(image, axis=0) == 0
  return np.where(flat, np.nan, picked)


def seabed_points(
  position: ArrayLike,
  rotation: ArrayLike,
  beam_ranges: ArrayLike,
  bearings: ArrayLike,
  swath_axis: ArrayLike = (0.0, 1.0, 0.0),
  nadir_axis: ArrayLike = (0.0, 0.0, 1.0),
) -> NDArray[np.float64]:
  """Where each beam struck the seabed, in world coordinates.

  Args:
      rotation: Body-to-world rotation, shape ``(3, 3)``.
      beam_ranges: Range per beam in metres; NaN gives a NaN row.
      bearings: Bearing per beam in radians from nadir.
      swath_axis: Body-frame unit vector the fan opens along.
      nadir_axis: Body-frame unit vector the fan is centred on; body ``+z`` is
          down in HoloOcean's ``IMUSocket``.

  Returns:
      Seabed points, shape ``(n, 3)``.
  """
  beam_ranges = np.asarray(beam_ranges, dtype=float)
  bearings = np.asarray(bearings, dtype=float)
  if beam_ranges.shape != bearings.shape:
    raise ValueError(
      f"ranges and bearings must match: {beam_ranges.shape} vs {bearings.shape}"
    )

  nadir = np.asarray(nadir_axis, dtype=float)
  across = np.asarray(swath_axis, dtype=float)

  # Unit direction of each beam in the body frame.
  directions = (
    np.cos(bearings)[:, None] * nadir + np.sin(bearings)[:, None] * across
  )
  offsets = beam_ranges[:, None] * directions

  return (
    np.asarray(position, dtype=float)
    + offsets @ np.asarray(rotation, dtype=float).T
  )
