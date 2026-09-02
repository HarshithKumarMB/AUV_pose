"""Non-causal estimators.

A smoother uses the whole record, including observations from after the step it
is estimating, so it can only run once a trajectory is complete. That is the
opposite of the filters in :mod:`auv_pose.smoothing.filters`, and why the two
live apart: this module imports no filter, only the shared types, so it will
smooth a run recorded by any of them -- or steps built by hand from a log.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from auv_pose.smoothing.typing import GaussianState, Step

__all__ = ["rts_smooth"]


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
      :meth:`auv_pose.smoothing.filters.Filter.step`.
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
