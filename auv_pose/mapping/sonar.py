"""Extracting ranges from sonar returns.

Pure numpy -- no simulator dependency, so this is testable offline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["range_bins", "bottom_return_range"]


def range_bins(range_min: float, range_max: float, n_bins: int) -> NDArray[np.float64]:
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
