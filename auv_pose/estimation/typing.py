"""Types shared by the filter and the smoother."""

from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Protocol, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

NumpyArray: TypeAlias = NDArray[np.floating]


class GaussianBelief(Protocol):
  """A dataclass with a mean and a covariance."""

  __dataclass_fields__: ClassVar[dict[str, Any]]

  @property
  def mean(self) -> Any: ...

  @property
  def cov(self) -> NumpyArray: ...


Belief = TypeVar("Belief", bound=GaussianBelief)


@dataclass(frozen=True, eq=False)
class SmootherStep(Generic[Belief]):
  """One filter cycle, as the backward pass needs it.

  :param prior: Belief after the motion update.
  :param posterior: Belief after the step's observations.
  :param cross_cov: ``Cov[previous posterior error, prior error]``, ``(n, n)``.
  """

  prior: Belief
  posterior: Belief
  cross_cov: NumpyArray
