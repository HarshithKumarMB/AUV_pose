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
  """Rauch-Tung-Striebel smoothing in a chart, from recorded cross-covariances::

      G_k = C_{k+1} (P_{k+1}^-)^{-1}
      x_k = x_k^f [+] G_k (x_{k+1}^s [-] x_{k+1}^-)
      P_k = P_k^f + G_k (P_{k+1}^s - P_{k+1}^-) G_k^T

  :param history: Recorded steps, oldest first.
  :param boxplus: Applies a tangent increment to a mean.
  :param boxminus: The tangent increment between two means.
  :param transport: Carries ``P_k`` to the tangent space at the corrected
      mean; ``None`` in a flat chart.
  :return: Smoothed beliefs, oldest first, one longer than ``history``.
  """
  posteriors = [initial] + [step.posterior for step in history]
  n = len(posteriors)

  smoothed: list[Belief] = [posteriors[-1]] * n

  for k in range(n - 2, -1, -1):
    filtered = posteriors[k]
    step = history[k]  # the step leading from k to k + 1
    future = smoothed[k + 1]

    # G = C P^-1 via P^T G^T = C^T.
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
