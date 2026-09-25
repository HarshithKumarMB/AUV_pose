"""Fixed-interval smoothing of a recorded filter run."""

import operator
from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np

from auv_pose.estimation.manifold import (
  covariance_transport as manifold_transport,
)
from auv_pose.estimation.typing import Belief, NumpyArray, SmootherStep


def unscented_rts_smooth(
  initial: Belief,
  history: Sequence[SmootherStep[Belief]],
  boxplus: Callable[..., object] = operator.add,
  boxminus: Callable[..., NumpyArray] = operator.sub,
  transport: Callable[..., NumpyArray] | None = manifold_transport,
) -> list[Belief]:
  """Fixed-interval smoothing for a filter that records no transition matrix.

  Rauch-Tung-Striebel, written in a chart::

      G_k = C_{k+1} (P_{k+1}^-)^{-1}
      x_k = x_k^f [+] G_k (x_{k+1}^s [-] x_{k+1}^-)
      P_k = P_k^f + G_k (P_{k+1}^s - P_{k+1}^-) G_k^T

  The gain comes from the recorded cross-covariance, since an unscented
  forward pass has no ``F``; for a linear filter it equals ``P F^T``.

  **This pass is exact given the forward pass.** Every approximation in the
  smoother lives in the predict and update steps that produced ``history``.

  :param initial: Belief before the first step.
  :param history: Recorded steps, oldest first.
  :param boxplus: Applies a tangent increment to a mean; ``+`` by default,
      which serves both :class:`~auv_pose.estimation.manifold.NavState` and
      plain vectors.
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

    smoothed[k] = replace(
      filtered,
      mean=boxplus(filtered.mean, correction),
      cov=0.5 * (cov + cov.T),
    )

  return smoothed
