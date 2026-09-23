"""Extracting ranges from sonar returns.

Pure numpy -- no simulator dependency, so this is testable offline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
  "azimuth_angles",
  "bottom_return_ranges",
  "range_bins",
  "seabed_points",
  "sounding_covariance",
]


def range_bins(
  range_min: float, range_max: float, n_bins: int
) -> NDArray[np.float64]:
  """Range corresponding to each bin of a sonar intensity profile.

  Bin *endpoints*, where :func:`azimuth_angles` uses bin *centres*. The
  inconsistency is deliberate rather than overlooked: measured against altitude
  read from the octree, over three fan configurations, neither convention wins.
  Endpoints came out -0.039, -0.027 and -0.020 m; centres, which are half a bin
  higher, +0.011, +0.023 and +0.030 m. The truth sits between them at about a
  quarter of a bin, both are well inside the 0.0996 m quantisation, and every
  number measured so far was measured under this one. Changing it on symmetry
  alone would shift every range by 0.05 m to no benefit.
  """
  return np.linspace(range_min, range_max, n_bins)


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

  The image is ``(range_bins, azimuth_bins)``, so each column is one beam's
  intensity profile and the strongest return in it is that beam's echo.

  Args:
      image: Intensity, shape ``(range_bins, azimuth_bins)``.
      ranges: Range for each row, length ``range_bins``.

  Returns:
      Range per beam in metres, NaN where a beam has no discernible echo. Beams
      angled far off nadir routinely fall beyond ``RangeMax`` and come back
      flat, so NaN is the normal case at the edges of the swath rather than a
      fault.

  Note:
      **A range that disagrees with the octree is not automatically wrong.**
      Measured against a ray-cast through the octree, these ranges agree to a
      0.035 m MAD-std -- inside the 0.0996 m quantisation. Where they disagree,
      by 4-5 m, the sonar turned out to be right: it sees pipelines lying on the
      Dam seabed that octree generation omits, and they were eventually
      photographed. Three separate explanations were fitted to that discrepancy
      -- beam geometry, an ``atan2`` approximation, fabricated returns -- and
      all three were wrong, because the reference was incomplete rather than the
      sensor. Score with ``experiments/check_beam_validity.py`` and suspect the
      octree first.
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


def sounding_covariance(
  pose_cov: ArrayLike,
  rotation: ArrayLike,
  beam_ranges: ArrayLike,
  bearings: ArrayLike,
  swath_axis: ArrayLike = (0.0, 1.0, 0.0),
  nadir_axis: ArrayLike = (0.0, 0.0, 1.0),
  sigma_range: float = 0.0,
) -> NDArray[np.float64]:
  """Covariance of each sounding :func:`seabed_points` places, from the pose's.

  A sounding is ``q = p + R d`` with ``d`` the beam's body-frame offset. Under
  the right perturbation the navigation state uses, ``R -> R exp(phi)``, the
  first-order error is ``dq = dp - R [d]x phi``, so

      Sigma_q = J P J^T,   J = [ I | -R [d]x ],

  with ``P`` the pose's ``(6, 6)`` position-and-rotation covariance, cross terms
  included. Range noise adds ``sigma_r^2 u u^T`` along the beam.

  **The rotation term is the one that matters, and it is per-beam.** It scales
  with ``|d|``: at 70 m altitude an outer beam lands 40 m out, so one degree of
  heading error moves it 0.7 m sideways while barely moving nadir. That growth
  across the swath is the structure the map's input-noise correction uses; a
  covariance equal for every beam of a ping would reduce it to a constant.

  Args:
      pose_cov: ``(6, 6)`` covariance of ``[position, rotation]``, rotation as a
          right perturbation in radians -- the ``POSITION`` and ``ROTATION``
          blocks of a :class:`~auv_pose.estimation.manifold.ManifoldGaussian`.
      rotation: Body-to-world rotation, shape ``(3, 3)``.
      beam_ranges: Range per beam in metres.
      bearings: Bearing per beam in radians from nadir.
      swath_axis: Body-frame unit vector the fan opens along.
      nadir_axis: Body-frame unit vector the fan is centred on.
      sigma_range: Range noise standard deviation, metres.

  Returns:
      Covariances, shape ``(n, 3, 3)``, world frame; NaN where the range is.
  """
  pose_cov = np.asarray(pose_cov, dtype=float)
  if pose_cov.shape != (6, 6):
    raise ValueError(f"expected a (6, 6) pose covariance, got {pose_cov.shape}")
  rotation = np.asarray(rotation, dtype=float)
  beam_ranges = np.asarray(beam_ranges, dtype=float)
  bearings = np.asarray(bearings, dtype=float)

  directions = np.cos(bearings)[:, None] * np.asarray(
    nadir_axis, dtype=float
  ) + np.sin(bearings)[:, None] * np.asarray(swath_axis, dtype=float)
  offsets = beam_ranges[:, None] * directions

  # [d]x for every beam, then -R [d]x.
  skews = np.zeros((len(offsets), 3, 3))
  skews[:, 0, 1], skews[:, 0, 2] = -offsets[:, 2], offsets[:, 1]
  skews[:, 1, 0], skews[:, 1, 2] = offsets[:, 2], -offsets[:, 0]
  skews[:, 2, 0], skews[:, 2, 1] = -offsets[:, 1], offsets[:, 0]

  jacobian = np.zeros((len(offsets), 3, 6))
  jacobian[:, :, :3] = np.eye(3)
  jacobian[:, :, 3:] = -rotation @ skews

  cov = jacobian @ pose_cov @ jacobian.transpose(0, 2, 1)

  if sigma_range > 0.0:
    along = directions @ rotation.T
    cov = cov + sigma_range**2 * along[:, :, None] * along[:, None, :]

  return 0.5 * (cov + cov.transpose(0, 2, 1))
