"""Sonar range extraction."""

import numpy as np
import pytest

from auv_pose.estimation.quaternion import quat_exp, quat_to_rotmat
from auv_pose.mapping.sonar import (
  bottom_return_ranges,
  range_bins,
  seabed_points,
  sounding_covariance,
)

RANGES = range_bins(0.5, 100.0, 1000)


def image_with_peaks(peaks, n_beams=None):
  """A ``(range_bins, beams)`` image with one unit echo per beam."""
  n_beams = len(peaks) if n_beams is None else n_beams
  image = np.zeros((RANGES.size, n_beams))
  for beam, index in enumerate(peaks):
    image[index, beam] = 1.0
  return image


def test_range_bins_spans_the_configured_range():
  ranges = range_bins(0.5, 100.0, 256)
  assert len(ranges) == 256
  assert ranges[0] == pytest.approx(0.5)
  assert ranges[-1] == pytest.approx(100.0)


def test_each_beam_takes_its_own_strongest_bin():
  peaks = [179, 3, 640]
  np.testing.assert_allclose(
    bottom_return_ranges(image_with_peaks(peaks), RANGES), RANGES[peaks]
  )


@pytest.mark.parametrize("index", [0, 1, 499, 998, 999])
def test_handles_peaks_at_the_array_ends(index):
  picked = bottom_return_ranges(image_with_peaks([index]), RANGES)
  assert picked[0] == pytest.approx(RANGES[index])


def test_a_flat_beam_is_not_a_sounding():
  """A beam with no contrast carries no echo; bin 0 would be a false reading."""
  image = image_with_peaks([10, 20, 30])
  image[:, 1] = 0.0
  image[:, 2] = 7.0
  picked = bottom_return_ranges(image, RANGES)
  assert picked[0] == pytest.approx(RANGES[10])
  assert np.isnan(picked[1:]).all()


def test_first_peak_wins_on_a_tie():
  """argmax semantics: the nearer of two equal returns is the seabed."""
  image = image_with_peaks([200])
  image[100, 0] = 1.0
  assert bottom_return_ranges(image, RANGES)[0] == pytest.approx(RANGES[100])


def test_rejects_a_one_dimensional_profile():
  with pytest.raises(ValueError, match="2-D"):
    bottom_return_ranges(np.zeros(RANGES.size), RANGES)


def test_rejects_mismatched_range_bins():
  with pytest.raises(ValueError, match="range bins"):
    bottom_return_ranges(np.zeros((10, 4)), RANGES)


# -- sounding covariance ----------------------------------------------------

BEARINGS = np.radians(np.linspace(-30.0, 30.0, 7))
SWATH, NADIR = (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
# IMUSocket at rest, yawed 30 degrees: a non-trivial body-to-world rotation.
YAW = np.radians(30.0)
ROTATION = np.array(
  [
    [np.cos(YAW), np.sin(YAW), 0.0],
    [np.sin(YAW), -np.cos(YAW), 0.0],
    [0.0, 0.0, -1.0],
  ]
)


def pose_covariance(rng):
  """Correlated, with metres of position and about a degree of attitude."""
  root = np.array([0.5] * 3 + [0.01] * 3)[:, None] * rng.normal(size=(6, 6))
  return root @ root.T


def test_the_covariance_matches_monte_carlo_through_seabed_points():
  """Push pose draws through the real geometry and compare the scatter."""
  rng = np.random.default_rng(0)
  cov = pose_covariance(rng)
  ranges = np.full(BEARINGS.shape, 75.0)
  position = np.array([3.0, -2.0, -0.4])

  predicted = sounding_covariance(cov, ROTATION, ranges, BEARINGS, SWATH, NADIR)

  draws = rng.multivariate_normal(np.zeros(6), cov, size=20_000)
  points = np.stack(
    [
      seabed_points(
        position + draw[:3],
        ROTATION @ quat_to_rotmat(quat_exp(draw[3:])),
        ranges,
        BEARINGS,
        SWATH,
        NADIR,
      )
      for draw in draws
    ]
  )
  for beam in range(len(BEARINGS)):
    empirical = np.cov(points[:, beam].T)
    np.testing.assert_allclose(
      predicted[beam], empirical, atol=0.03 * np.abs(predicted[beam]).max()
    )


def test_without_attitude_error_every_beam_inherits_the_position_covariance():
  cov = np.zeros((6, 6))
  cov[:3, :3] = np.diag([1.0, 2.0, 0.5])
  predicted = sounding_covariance(
    cov, ROTATION, np.full(BEARINGS.shape, 70.0), BEARINGS, SWATH, NADIR
  )
  np.testing.assert_allclose(predicted, np.broadcast_to(cov[:3, :3], (7, 3, 3)))


def test_heading_error_moves_outer_beams_further_than_nadir():
  cov = np.zeros((6, 6))
  cov[5, 5] = np.radians(1.0) ** 2
  predicted = sounding_covariance(
    cov, ROTATION, np.full(BEARINGS.shape, 70.0), BEARINGS, SWATH, NADIR
  )
  horizontal = np.trace(predicted[:, :2, :2], axis1=1, axis2=2)
  assert horizontal[0] > 100 * horizontal[3]
  # One degree at the edge of a 70 m, 30 degree fan: 35 m out, 0.61 m moved.
  assert np.sqrt(horizontal[0]) == pytest.approx(
    70.0 * np.sin(np.radians(30.0)) * np.radians(1.0), rel=1e-6
  )


def test_range_noise_lies_along_the_beam():
  predicted = sounding_covariance(
    np.zeros((6, 6)),
    ROTATION,
    np.full(BEARINGS.shape, 70.0),
    BEARINGS,
    SWATH,
    NADIR,
    sigma_range=0.1,
  )
  direction = seabed_points(
    np.zeros(3), ROTATION, np.ones_like(BEARINGS), BEARINGS, SWATH, NADIR
  )
  for beam, along in enumerate(direction):
    np.testing.assert_allclose(
      predicted[beam] @ along, 0.01 * along, atol=1e-12
    )
