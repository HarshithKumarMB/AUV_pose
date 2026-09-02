"""Sonar range extraction."""

import numpy as np
import pytest

from auv_pose.mapping.sonar import bottom_return_range, range_bins


def test_range_bins_spans_the_configured_range():
    ranges = range_bins(0.5, 100.0, 256)
    assert len(ranges) == 256
    assert ranges[0] == pytest.approx(0.5)
    assert ranges[-1] == pytest.approx(100.0)


def test_picks_the_strongest_bin():
    ranges = range_bins(0.5, 100.0, 256)
    profile = np.zeros(256)
    profile[179] = 1.0
    assert bottom_return_range(profile, ranges) == pytest.approx(ranges[179])


def test_matches_the_value_recorded_in_the_survey_data():
    """map.csv's opening rows read 70.345..., which must be bin 179."""
    ranges = range_bins(0.5, 100.0, 256)
    profile = np.zeros(256)
    profile[179] = 1.0
    assert bottom_return_range(profile, ranges) == pytest.approx(70.34509803921569)


@pytest.mark.parametrize("index", [0, 1, 127, 254, 255])
def test_handles_peaks_at_the_array_ends(index):
    ranges = range_bins(0.5, 100.0, 256)
    profile = np.zeros(256)
    profile[index] = 5.0
    assert bottom_return_range(profile, ranges) == pytest.approx(ranges[index])


def test_flat_profile_is_not_a_sounding():
    """A profile with no contrast carries no echo; bin 0 would be a false reading."""
    ranges = range_bins(0.5, 100.0, 256)
    assert np.isnan(bottom_return_range(np.zeros(256), ranges))
    assert np.isnan(bottom_return_range(np.full(256, 7.0), ranges))


def test_scalar_profile_is_taken_as_a_direct_range():
    """Some HoloOcean builds return a range rather than a profile."""
    assert bottom_return_range(np.float64(12.5), range_bins(0.5, 100.0, 256)) == 12.5


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        bottom_return_range(np.zeros(10), range_bins(0.5, 100.0, 256))


def test_first_peak_wins_on_a_tie():
    """argmax semantics: the nearer of two equal returns is the seabed."""
    ranges = range_bins(0.5, 100.0, 256)
    profile = np.zeros(256)
    profile[100] = profile[200] = 1.0
    assert bottom_return_range(profile, ranges) == pytest.approx(ranges[100])
