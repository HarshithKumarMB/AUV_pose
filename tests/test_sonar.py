"""Sonar range extraction."""

import numpy as np
import pytest

from auv_pose.mapping.sonar import (
  bottom_return_ranges,
  range_bins,
  seabed_points,
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


# -- placing the beams --------------------------------------------------------

BEARINGS = np.radians(np.linspace(-60.0, 60.0, 9))


def rotation(yaw, pitch=0.0, roll=0.0):
  cz, sz = np.cos(yaw), np.sin(yaw)
  cy, sy = np.cos(pitch), np.sin(pitch)
  cx, sx = np.cos(roll), np.sin(roll)
  Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
  Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
  Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
  return Rz @ Ry @ Rx


@pytest.mark.parametrize(
  "attitude", [(0, 0, 0), (0.7, 0.1, -0.2), (3.0, -0.3, 0.4)]
)
def test_every_beam_lands_its_range_from_the_vehicle(attitude):
  position = np.array([5.0, -3.0, -10.0])
  ranges = np.linspace(20.0, 90.0, BEARINGS.size)
  points = seabed_points(position, rotation(*attitude), ranges, BEARINGS)
  np.testing.assert_allclose(
    np.linalg.norm(points - position, axis=1), ranges, rtol=1e-12
  )


def test_the_nadir_beam_lands_along_the_rotated_nadir_axis():
  R = rotation(0.7, 0.1, -0.2)
  points = seabed_points(np.zeros(3), R, [50.0], [0.0])
  np.testing.assert_allclose(points[0], 50.0 * R[:, 2], atol=1e-12)


def test_mirrored_bearings_land_mirrored_across_the_track():
  """At level attitude the fan is symmetric about the vertical plane of travel."""
  R = rotation(1.2)
  ranges = np.full(BEARINGS.size, 40.0)
  points = seabed_points(np.zeros(3), R, ranges, BEARINGS)
  np.testing.assert_allclose(points[:, 2], points[::-1, 2], atol=1e-12)
  across = points @ R[:, 1]
  np.testing.assert_allclose(across, -across[::-1], atol=1e-12)
  np.testing.assert_allclose(across, 40.0 * np.sin(BEARINGS), atol=1e-12)


def test_heading_turns_the_fan_but_not_the_depths():
  ranges = np.linspace(30.0, 60.0, BEARINGS.size)
  a = seabed_points(np.zeros(3), rotation(0.0), ranges, BEARINGS)
  b = seabed_points(np.zeros(3), rotation(2.0), ranges, BEARINGS)
  np.testing.assert_allclose(a[:, 2], b[:, 2], atol=1e-12)
  np.testing.assert_allclose(
    np.linalg.norm(a[:, :2], axis=1), np.linalg.norm(b[:, :2], axis=1)
  )


def test_a_beam_without_an_echo_stays_missing_and_alone():
  ranges = np.full(BEARINGS.size, 40.0)
  ranges[3] = np.nan
  points = seabed_points(np.zeros(3), rotation(0.3), ranges, BEARINGS)
  assert np.isnan(points[3]).all()
  assert np.isfinite(np.delete(points, 3, axis=0)).all()


def test_mismatched_ranges_and_bearings_are_refused():
  with pytest.raises(ValueError, match="must match"):
    seabed_points(np.zeros(3), np.eye(3), [1.0, 2.0], BEARINGS)
