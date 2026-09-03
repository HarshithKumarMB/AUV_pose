"""Extracting ranges from sonar returns.

Pure numpy -- no simulator dependency, so this is testable offline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
  "azimuth_angles",
  "bottom_return_range",
  "bottom_return_ranges",
  "range_bins",
  "seabed_points",
]


def range_bins(
  range_min: float, range_max: float, n_bins: int
) -> NDArray[np.float64]:
  """Range corresponding to each bin of a sonar intensity profile."""
  return np.linspace(range_min, range_max, n_bins)


def bottom_return_range(profile: ArrayLike, ranges: ArrayLike) -> float:
  """Estimate range to the seabed from a singlebeam intensity profile.

  Takes the strongest return as the bottom echo, which is a reasonable model for
  a narrow downward-facing beam over a seabed that reflects more strongly than
  the water column.

  Args:
      profile: Intensity per range bin.
      ranges: Range for each bin, same length as ``profile``.

  Returns:
      Range in metres, or NaN if the profile is empty or entirely flat -- a flat
      profile means there is no discernible echo, and returning bin 0 would
      silently report the minimum range as a real sounding.
  """
  profile = np.asarray(profile, dtype=float)
  ranges = np.asarray(ranges, dtype=float)

  if profile.ndim == 0:
    return float(profile)

  if profile.shape != ranges.shape:
    raise ValueError(
      f"profile and ranges must match: {profile.shape} vs {ranges.shape}"
    )

  if profile.size == 0 or np.ptp(profile) == 0:
    return float("nan")

  return float(ranges[int(np.argmax(profile))])


def azimuth_angles(azimuth: float, n_bins: int) -> NDArray[np.float64]:
  """Bearing of each beam in a multibeam fan, radians from nadir.

  Bin centres rather than edges, so beam ``i`` is the direction the ``i``-th
  column of the intensity image looked along.

  Args:
      azimuth: Total swath width in degrees.
      n_bins: Number of azimuth bins.

  Returns:
      Bearings in radians, ascending, symmetric about zero.
  """
  half = np.radians(azimuth) / 2.0
  edges = np.linspace(-half, half, n_bins + 1)
  return 0.5 * (edges[:-1] + edges[1:])


def bottom_return_ranges(
  image: ArrayLike, ranges: ArrayLike
) -> NDArray[np.float64]:
  """Bottom range for every beam of a multibeam image.

  The multibeam counterpart of :func:`bottom_return_range`: the image is
  ``(range_bins, azimuth_bins)``, so each column is one beam's intensity
  profile and the strongest return in it is that beam's echo.

  Args:
      image: Intensity, shape ``(range_bins, azimuth_bins)``.
      ranges: Range for each row, length ``range_bins``.

  Returns:
      Range per beam in metres, NaN where a beam has no discernible echo. Beams
      angled far off nadir routinely fall beyond ``RangeMax`` and come back
      flat, so NaN is the normal case at the edges of the swath rather than a
      fault.
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

  A multibeam only measures range along a bearing; turning that into a sounding
  needs the vehicle's position *and* attitude, because every beam except nadir
  lands at a horizontal offset of ``range * sin(bearing)`` from the vehicle.
  Recording a sounding at the vehicle's own ``(x, y)`` -- which is what a
  singlebeam survey can get away with -- misplaces every other beam.

  Args:
      position: Vehicle position in the world, shape ``(3,)``.
      rotation: Body-to-world rotation, shape ``(3, 3)``.
      beam_ranges: Range per beam in metres; NaN beams pass through as NaN.
      bearings: Bearing per beam in radians from nadir, same length.
      swath_axis: Body-frame unit vector the fan opens along.
      nadir_axis: Body-frame unit vector the fan is centred on. Defaults to
          body ``+z``, which is down for a sensor in HoloOcean's ``IMUSocket``.

  Returns:
      Seabed points, shape ``(n, 3)``, NaN rows where the beam had no echo.
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
