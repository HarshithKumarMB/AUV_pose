"""Type definitions for state estimation.

These are shared by filters and smoothers, so they live here rather than in
either -- neither module then has to import the other.

State is a value, not filter attributes: ``predict`` and ``condition`` take a
state and return a new one. That is what makes smoothing straightforward, since
a run's history is then just a list of values and the backward pass is a pure
function over it.
"""

from __future__ import annotations

from typing import NamedTuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

NumpyArray: TypeAlias = NDArray[np.floating]

__all__ = ["GaussianState", "Measurement", "NumpyArray", "Step"]


class GaussianState(NamedTuple):
  """A Gaussian belief over the state vector.

  :param mean: State mean, shape ``(n,)``.
  :param cov: State covariance, shape ``(n, n)``.
  """

  mean: NumpyArray
  cov: NumpyArray


class Measurement(NamedTuple):
  """A linear-Gaussian observation of the state.

  :param z: Observed value, shape ``(m,)``.
  :param H: Observation model mapping state to measurement, shape ``(m, n)``.
  :param R: Observation noise covariance, shape ``(m, m)``.
  """

  z: NumpyArray
  H: NumpyArray
  R: NumpyArray


class Step(NamedTuple):
  """One predict/condition cycle of a filter.

  The prior is recorded as well as the posterior because the backward pass
  needs it -- and so does any innovation or normalised-innovation-squared
  diagnostic, which makes this a general trace rather than a smoother-shaped
  carve-out.

  :param prior: Belief after the motion update, before any observation.
  :param posterior: Belief after conditioning on every observation for the step.
  :param transition: State transition matrix used, shape ``(n, n)``.
  """

  prior: GaussianState
  posterior: GaussianState
  transition: NumpyArray
