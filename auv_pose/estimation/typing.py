"""Type definitions for state estimation.

These are shared by filters and smoothers, so they live here rather than in
either -- neither module then has to import the other.

State is a value, not filter attributes: ``predict`` and ``condition`` take a
state and return a new one. That is what makes smoothing straightforward, since
a run's history is then just a list of values and the backward pass is a pure
function over it.
"""

from __future__ import annotations

from typing import Any, Generic, NamedTuple, Protocol, Self, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

NumpyArray: TypeAlias = NDArray[np.floating]


class GaussianBelief(Protocol):
  """What a belief must offer: a mean, a covariance, and a way to replace them.

  Structural, so :class:`GaussianState` and
  :class:`~auv_pose.estimation.manifold.ManifoldGaussian` both satisfy it
  without inheriting from anything.
  """

  @property
  def mean(self) -> Any: ...

  @property
  def cov(self) -> NumpyArray: ...

  def _replace(self, *, mean: Any = ..., cov: Any = ...) -> Self: ...


#: A belief over the state. Any ``(mean, cov)`` pair will do -- the estimators
#: never look inside the mean, they only hand it to a chart.
Belief = TypeVar("Belief", bound=GaussianBelief)

__all__ = [
  "Belief",
  "GaussianBelief",
  "GaussianState",
  "Measurement",
  "NumpyArray",
  "SmootherStep",
  "Step",
]


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


class SmootherStep(NamedTuple, Generic[Belief]):
  """One predict/condition cycle, recorded so a backward pass can use it.

  The sibling of :class:`Step` for a filter with no transition *matrix*. An
  unscented filter never forms one -- it pushes sigma points through the motion
  model instead -- so what it can record is the cross-covariance between the
  previous filtered error state and the predicted one. That is the quantity the
  backward pass actually needs; the linear smoother's ``P F^T`` is the special
  case of it, which is why the two passes agree exactly on a linear problem.

  Generic in the belief because the chart is the caller's business: the same
  record carries a
  :class:`~auv_pose.estimation.manifold.ManifoldGaussian` on the state
  manifold and a :class:`GaussianState` in a vector space. That genericity is
  not decoration -- it is what lets the manifold backward pass be tested
  against :func:`~auv_pose.estimation.smoothers.rts_smooth` on a problem where
  the right answer is known exactly.

  :param prior: Belief after the motion update, before any observation.
  :param posterior: Belief after conditioning on every observation for the step.
  :param cross_cov: ``Cov[xi_prev_posterior, xi_prior]``, shape ``(n, n)``.
  """

  prior: Belief
  posterior: Belief
  cross_cov: NumpyArray
