"""Covariance functions for the bathymetry map, and their spatial derivatives.

Separate from the map that uses them because they are pure mathematics with a
closed form, so they can be checked against an independent implementation and
against finite differences without fitting anything.

**Matérn-5/2 rather than squared exponential.** A squared-exponential kernel
assumes the seabed is infinitely differentiable, which smooths over ridges and
then reports a small variance for having done so -- the exact combination that
makes a map dangerous to a filter, since the update trusts what the map says
most confidently. Matérn-5/2 admits a twice-differentiable surface, which is
rough enough for a seabed and smooth enough that the mean still has the gradient
the range-noise term needs.

Lengthscales are **per axis and in metres**. The isotropic alternative is not
the neutral choice it looks like: over a survey box that is not square it is
implicitly anisotropic by the ratio of the sides, which is a fact about the
survey rather than about the seabed. See :class:`~auv_pose.mapping.svgp.SVGPModel`,
whose docstring records the fit that discovered this.
"""

import math

import torch
from torch import Tensor

__all__ = ["matern52", "matern52_gradient"]

_SQRT5 = math.sqrt(5.0)


def _scaled_offsets(
  a: Tensor, b: Tensor, log_lengthscale: Tensor
) -> tuple[Tensor, Tensor]:
  """Pairwise offsets and the Matérn radius, both in lengthscale units.

  :return: ``(offsets, r)`` of shapes ``(..., p, q, d)`` and ``(..., p, q)``,
      where ``offsets`` is ``(a - b) / lengthscale^2`` -- already carrying one
      factor of the lengthscale that the gradient needs -- and ``r`` is the
      Mahalanobis distance under the diagonal lengthscale metric.

  Note:
      The square root is masked, and it has to be. ``sqrt`` is finite at zero
      but its *derivative* is not, and every kernel block this module builds has
      a zero on its diagonal where a point meets itself. Differentiating the
      likelihood with respect to the lengthscales therefore produces ``NaN``
      from a forward pass that looked perfectly healthy.

      Clamping the input would bias every covariance. Masking after the fact
      with a single ``torch.where`` does not work either: both branches are
      evaluated, so the infinite derivative still reaches the sum. The fix is
      the double ``where`` below -- feed ``sqrt`` a harmless ``1`` wherever the
      true value is zero, then discard those entries. The gradient of a
      discarded branch never contributes, so the ``inf`` is never formed.
  """
  lengthscale = torch.exp(log_lengthscale)

  difference = a.unsqueeze(-2) - b.unsqueeze(-3)
  scaled = difference / lengthscale
  squared = (scaled * scaled).sum(-1)

  positive = squared > 0.0
  radius = torch.where(
    positive,
    torch.sqrt(torch.where(positive, squared, torch.ones_like(squared))),
    torch.zeros_like(squared),
  )

  return difference / (lengthscale * lengthscale), radius


def matern52(
  a: Tensor,
  b: Tensor,
  log_amplitude: Tensor,
  log_lengthscale: Tensor,
) -> Tensor:
  """Matérn-5/2 covariance between two sets of positions.

  ``k(a, b) = sigma_f^2 (1 + sqrt5 r + 5 r^2 / 3) exp(-sqrt5 r)``, with
  ``r`` the distance in lengthscale units.

  :param a: Positions, shape ``(..., p, d)``, metres.
  :param b: Positions, shape ``(..., q, d)``, metres.
  :param log_amplitude: ``log(sigma_f^2)``, scalar.
  :param log_lengthscale: ``log`` of the per-axis lengthscale, shape ``(d,)``,
      metres.
  :return: Covariances, shape ``(..., p, q)``.
  """
  _, radius = _scaled_offsets(a, b, log_lengthscale)
  root5r = _SQRT5 * radius

  return torch.exp(log_amplitude) * (
    (1.0 + root5r + root5r * root5r / 3.0) * torch.exp(-root5r)
  )


def matern52_gradient(
  a: Tensor,
  b: Tensor,
  log_amplitude: Tensor,
  log_lengthscale: Tensor,
) -> Tensor:
  """Derivative of :func:`matern52` with respect to its **first** argument.

  ``dk/da_d = -(5/3) sigma_f^2 (1 + sqrt5 r) exp(-sqrt5 r) (a_d - b_d) / l_d^2``

  This is what the map's mean gradient is built from, and what carries the
  sonar's range noise through the shift in where the map gets queried.

  :param a: Positions to differentiate at, shape ``(..., p, d)``, metres.
  :param b: The other positions, shape ``(..., q, d)``, metres.
  :param log_amplitude: ``log(sigma_f^2)``, scalar.
  :param log_lengthscale: ``log`` of the per-axis lengthscale, shape ``(d,)``.
  :return: Gradients, shape ``(..., p, q, d)``, covariance per metre.

  Note:
      **The closed form above is the cancelled one, and that matters.** Writing
      it as (radial derivative) times ``dr/da_d`` leaves a ``1/r`` from
      ``dr/da_d = (a_d - b_d) / (l_d^2 r)``, which is ``0/0`` wherever ``a``
      and ``b`` coincide -- on every diagonal of every kernel block the map
      builds. The ``r`` cancels analytically against the ``r`` in the radial
      term, and the result is finite everywhere and correctly zero at ``r = 0``,
      where the covariance is at its maximum and so has no slope.

      This is not a tidiness point. ``pyproject.toml`` sets
      ``filterwarnings = ["error::RuntimeWarning"]``, so the uncancelled form
      does not return a NaN to be noticed later -- it fails the test suite.
  """
  offsets, radius = _scaled_offsets(a, b, log_lengthscale)
  root5r = _SQRT5 * radius

  decay = (
    (5.0 / 3.0) * torch.exp(log_amplitude) * (1.0 + root5r) * torch.exp(-root5r)
  )

  return -decay.unsqueeze(-1) * offsets
