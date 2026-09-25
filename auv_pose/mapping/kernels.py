"""Matérn-5/2 covariance for the bathymetry map, and its spatial gradient.

Lengthscales are per axis and in metres. Matérn-5/2 rather than squared
exponential, which over-smooths ridges while reporting a small variance.
"""

import math

import torch
from torch import Tensor

_SQRT5 = math.sqrt(5.0)


def _scaled_offsets(
  a: Tensor, b: Tensor, log_lengthscale: Tensor
) -> tuple[Tensor, Tensor]:
  """Pairwise offsets and the Matérn radius, both in lengthscale units.

  :return: ``(offsets, r)`` of shapes ``(..., p, q, d)`` and ``(..., p, q)``,
      where ``offsets`` is ``(a - b) / lengthscale^2``.

  The double ``where`` around ``sqrt`` is required: its derivative is infinite
  at the zero diagonal, and a single ``where`` still lets that ``inf`` reach the
  gradient as ``NaN``.
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
  :param log_amplitude: ``log(sigma_f^2)``.
  :param log_lengthscale: ``log`` of the per-axis lengthscale, ``(d,)``.
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

  Arguments as for :func:`matern52`; returns shape ``(..., p, q, d)``.

  Keep this analytically cancelled form: the chain-rule form has ``1/r``,
  which is ``0/0`` on every kernel diagonal.
  """
  offsets, radius = _scaled_offsets(a, b, log_lengthscale)
  root5r = _SQRT5 * radius

  decay = (
    (5.0 / 3.0) * torch.exp(log_amplitude) * (1.0 + root5r) * torch.exp(-root5r)
  )

  return -decay.unsqueeze(-1) * offsets
