"""The contract a seabed map must meet for the smoother.

A map must return a *joint* Gaussian over depths: the off-diagonals stop
correlated beams counting as independent constraints on the pose.
"""

from typing import Protocol, runtime_checkable

from numpy.typing import ArrayLike

from auv_pose.estimation.typing import NumpyArray


@runtime_checkable
class DepthMap(Protocol):
  """What the update step needs of a seabed map.

  Depth is world z (z up): a seabed 65 m down answers ``-65``.
  """

  def predict_joint(self, points: ArrayLike) -> tuple[NumpyArray, NumpyArray]:
    """Joint Gaussian over the seabed at a set of horizontal positions.

    :param points: ``(..., b, 2)`` of ``(x, y)``, metres; leading axes batch.
    :return: ``(mean, cov)`` of shapes ``(..., b)`` and ``(..., b, b)``.
    """
    ...

  def mean_gradient(self, points: ArrayLike) -> NumpyArray:
    """Slope ``(n, 2)`` of the mean at ``(n, 2)`` points."""
    ...
