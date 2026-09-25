"""Types shared by the filter and the smoother."""

from typing import Any, Generic, NamedTuple, Protocol, Self, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

NumpyArray: TypeAlias = NDArray[np.floating]


class GaussianBelief(Protocol):
  """A mean, a covariance, and a way to replace them."""

  @property
  def mean(self) -> Any: ...

  @property
  def cov(self) -> NumpyArray: ...

  def _replace(self, *, mean: Any = ..., cov: Any = ...) -> Self: ...


Belief = TypeVar("Belief", bound=GaussianBelief)


class SmootherStep(NamedTuple, Generic[Belief]):
  """One filter cycle, as the backward pass needs it.

  :param prior: Belief after the motion update.
  :param posterior: Belief after the step's observations.
  :param cross_cov: ``Cov[previous posterior error, prior error]``, ``(n, n)``.
  """

  prior: Belief
  posterior: Belief
  cross_cov: NumpyArray
