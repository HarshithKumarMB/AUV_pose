"""Scoring soundings against a seabed map.

The map itself is not defined here, and deliberately so. All the smoother
requires of one is that it answers a set of horizontal positions with a joint
Gaussian over the depths there::

    M : Q -> N(z | mu_M(Q), cov_M(Q))

That is the whole contract, stated as :class:`DepthMap` below. Nothing in
:mod:`auv_pose.estimation` imports a concrete map, or torch, so replacing the
map is a change to one constructor call in the experiment that drives the
smoother -- which matters, because the map is expected to be replaced.

**The joint covariance is the part that is easy to under-deliver.** A method
returning only a per-point variance does not satisfy this contract, and neither
does one returning no uncertainty at all. The off-diagonals of ``cov_M(Q)`` are
what stop the update from counting thirty-odd nearby, heavily correlated beams
as thirty independent constraints on the pose; without them the filter is
confidently wrong rather than approximately right.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from numpy.typing import ArrayLike

from auv_pose.estimation.typing import NumpyArray

__all__ = ["DepthMap"]


@runtime_checkable
class DepthMap(Protocol):
  """What the update step needs of a seabed map.

  Depth here is in the **world frame the rest of the package uses**, which is
  z-up: a seabed 65 m down answers ``-65``, not ``65``. The paper writes the
  same quantity as a depth positive downwards; this is the one place that
  difference is recorded, and everywhere below ``[.]_d`` in the paper is
  ``[.]_2`` in the code with the opposite sign.
  """

  def predict_joint(self, points: ArrayLike) -> tuple[NumpyArray, NumpyArray]:
    """Joint Gaussian over the seabed at a set of horizontal positions.

    :param points: ``(..., b, 2)`` of ``(x, y)`` in metres. Leading axes are
        batch dimensions, so a whole sigma-point cloud can be answered at once.
    :return: ``(mean, cov)`` of shapes ``(..., b)`` and ``(..., b, b)``, in
        metres and metres squared.
    """
    ...

  def mean_gradient(self, points: ArrayLike) -> NumpyArray:
    """Slope of the map's mean, in metres per metre.

    Used only to carry the sonar's range noise through the shift in *where*
    the map gets queried, which is a small correction on this data. Finite
    differences are an acceptable implementation.

    :param points: ``(n, 2)`` of ``(x, y)`` in metres.
    :return: ``(n, 2)`` of ``d(depth)/dx, d(depth)/dy``.
    """
    ...
