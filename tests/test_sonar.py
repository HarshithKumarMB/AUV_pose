"""Sonar range extraction."""

import numpy as np
import pytest

from auv_pose.mapping.sonar import bottom_return_ranges, range_bins

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
