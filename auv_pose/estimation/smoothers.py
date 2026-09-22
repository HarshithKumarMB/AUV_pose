"""Non-causal estimators.

A smoother uses the whole record, including observations from after the step it
is estimating, so it can only run once a trajectory is complete. That is the
opposite of the filters in :mod:`auv_pose.estimation.filters`, and why the two
live apart: this module imports no filter, only the shared types, so it will
smooth a run recorded by any of them -- or steps built by hand from a log.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from auv_pose.estimation.manifold import (
  boxminus as manifold_boxminus,
)
from auv_pose.estimation.manifold import (
  boxplus as manifold_boxplus,
)
from auv_pose.estimation.manifold import (
  covariance_transport as manifold_transport,
)
from auv_pose.estimation.typing import (
  Belief,
  GaussianState,
  NumpyArray,
  SmootherStep,
  Step,
)

__all__ = ["rts_smooth", "unscented_rts_smooth"]


def rts_smooth(
  initial: GaussianState, history: Sequence[Step]
) -> list[GaussianState]:
  """Rauch-Tung-Striebel fixed-interval smoothing.

  Walks backwards from the final belief, correcting each step with what the
  future turned out to hold::

      C_k = P_k F^T (P_{k+1}^-)^{-1}
      x_k = x_k^f + C_k (x_{k+1}^s - x_{k+1}^-)
      P_k = P_k^f + C_k (P_{k+1}^s - P_{k+1}^-) C_k^T

  Exact rather than approximate here, because the constant-velocity dynamics
  are linear.

  Note:
      Smoothing cannot make an unobservable direction observable. Where the
      measurement model never constrains a state component, the backward pass
      tightens its covariance only through correlation with components that
      are constrained.

  :param initial: Belief before the first step.
  :param history: Recorded steps, oldest first, as produced by
      :meth:`auv_pose.estimation.filters.Filter.step`.
  :return: Smoothed beliefs, oldest first, one longer than ``history``
      because the initial belief is included.
  """
  posteriors = [initial] + [step.posterior for step in history]
  n = len(posteriors)

  smoothed: list[GaussianState] = [posteriors[-1]] * n

  for k in range(n - 2, -1, -1):
    filtered = posteriors[k]
    step = history[k]  # the step leading from k to k + 1

    gain = filtered.cov @ step.transition.T @ np.linalg.inv(step.prior.cov)

    smoothed[k] = GaussianState(
      mean=filtered.mean + gain @ (smoothed[k + 1].mean - step.prior.mean),
      cov=filtered.cov + gain @ (smoothed[k + 1].cov - step.prior.cov) @ gain.T,
    )

  return smoothed


def unscented_rts_smooth(
  initial: Belief,
  history: Sequence[SmootherStep[Belief]],
  boxplus: Callable[..., object] = manifold_boxplus,
  boxminus: Callable[..., NumpyArray] = manifold_boxminus,
  transport: Callable[..., NumpyArray] | None = manifold_transport,
) -> list[Belief]:
  """Fixed-interval smoothing for a filter that records no transition matrix.

  The same recursion as :func:`rts_smooth`, written in a chart::

      G_k = C_{k+1} (P_{k+1}^-)^{-1}
      x_k = x_k^f [+] G_k (x_{k+1}^s [-] x_{k+1}^-)
      P_k = P_k^f + G_k (P_{k+1}^s - P_{k+1}^-) G_k^T

  The gain comes from a recorded cross-covariance rather than from ``P F^T``,
  which is the only reason an unscented forward pass can be smoothed at all:
  it has no ``F``. For a linear filter the two are the same number, so this
  reduces to :func:`rts_smooth` exactly -- and that is worth knowing, because
  it means the backward pass can be tested where the right answer is known.

  **This pass is exact given the forward pass.** Every approximation in the
  smoother lives in the predict and update steps that produced ``history``.

  :param initial: Belief before the first step.
  :param history: Recorded steps, oldest first.
  :param boxplus: Applies a tangent increment to a mean. Defaults to the state
      manifold's; pass ``operator.add`` to smooth in a vector space.
  :param boxminus: The tangent increment between two means.
  :param transport: Maps an increment to the Jacobian carrying a covariance
      along it. ``None`` to skip, which is what a flat chart wants -- see the
      note below.
  :return: Smoothed beliefs, oldest first, one longer than ``history``.

  Note:
      The paper's equations have no ``transport`` term. It is applied here
      because ``P_k^f`` is a covariance in the tangent space at ``x_k^f``,
      while the belief being returned is centred at ``x_k^f [+] correction`` --
      a different tangent space whenever the correction rotates the state. The
      two differ at second order, so it changes nothing on a converged track
      and matters over the first corrections of a run, which can be degrees.

  Note:
      Smoothing cannot make an unobservable direction observable. Where the
      measurement model never constrains a state component, the backward pass
      tightens its covariance only through correlation with components that
      are constrained. On this vehicle that is not hypothetical -- see the
      gyro bias.
  """
  posteriors = [initial] + [step.posterior for step in history]
  n = len(posteriors)

  smoothed: list[Belief] = [posteriors[-1]] * n

  for k in range(n - 2, -1, -1):
    filtered = posteriors[k]
    step = history[k]  # the step leading from k to k + 1
    future = smoothed[k + 1]

    # G = C P^-1, solved rather than inverted: P^T G^T = C^T.
    gain = np.linalg.solve(step.prior.cov.T, step.cross_cov.T).T

    correction = gain @ boxminus(future.mean, step.prior.mean)
    cov = filtered.cov + gain @ (future.cov - step.prior.cov) @ gain.T

    if transport is not None:
      jacobian = transport(correction)
      cov = jacobian @ cov @ jacobian.T

    smoothed[k] = filtered._replace(
      mean=boxplus(filtered.mean, correction), cov=0.5 * (cov + cov.T)
    )

  return smoothed
