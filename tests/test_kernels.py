"""Matérn-5/2 and its spatial derivative.

Both are checked against something written by someone else: the covariance
against gpytorch's ``MaternKernel(nu=2.5)``, which is already a dependency, and
the gradient against central differences of this module's own covariance. A
closed form checked only against its own rearrangement proves nothing.
"""

import gpytorch
import numpy as np
import pytest
import torch

from auv_pose.mapping.kernels import matern52, matern52_gradient

LOG_AMPLITUDE = torch.tensor(np.log(2.5))
LOG_LENGTHSCALE = torch.log(torch.tensor([3.0, 7.0], dtype=torch.float64))


def points(n, seed, spread=20.0):
  rng = np.random.default_rng(seed)
  return torch.tensor(rng.uniform(-spread, spread, size=(n, 2)))


# -- the covariance ---------------------------------------------------------


def test_it_matches_gpytorchs_matern_kernel():
  """An independent implementation of the same closed form."""
  a, b = points(9, 0), points(6, 1)

  kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2).double()
  with torch.no_grad():
    kernel.lengthscale = torch.exp(LOG_LENGTHSCALE)
    reference = kernel(a, b).to_dense() * torch.exp(LOG_AMPLITUDE)

  np.testing.assert_allclose(
    matern52(a, b, LOG_AMPLITUDE, LOG_LENGTHSCALE).numpy(),
    reference.numpy(),
    rtol=1e-12,
    atol=1e-14,
  )


def test_the_variance_at_zero_separation_is_the_amplitude():
  a = points(5, 2)
  diagonal = torch.diagonal(matern52(a, a, LOG_AMPLITUDE, LOG_LENGTHSCALE))
  np.testing.assert_allclose(
    diagonal.numpy(), np.full(5, float(torch.exp(LOG_AMPLITUDE))), rtol=1e-14
  )


def test_a_gram_matrix_is_symmetric_and_positive_definite():
  a = points(40, 3)
  gram = matern52(a, a, LOG_AMPLITUDE, LOG_LENGTHSCALE)

  np.testing.assert_allclose(gram.numpy(), gram.numpy().T, atol=1e-15)
  assert np.min(np.linalg.eigvalsh(gram.numpy())) > 0.0


def test_covariance_decays_with_separation():
  origin = torch.zeros(1, 2, dtype=torch.float64)
  away = torch.tensor([[1.0, 0.0], [5.0, 0.0], [30.0, 0.0]])

  values = (
    matern52(origin, away, LOG_AMPLITUDE, LOG_LENGTHSCALE).numpy().ravel()
  )
  assert values[0] > values[1] > values[2] > 0.0


def test_the_lengthscales_act_per_axis():
  """A step along the long axis must decorrelate less than along the short."""
  origin = torch.zeros(1, 2, dtype=torch.float64)
  along_x = torch.tensor([[4.0, 0.0]])
  along_y = torch.tensor([[0.0, 4.0]])

  k_x = matern52(origin, along_x, LOG_AMPLITUDE, LOG_LENGTHSCALE)
  k_y = matern52(origin, along_y, LOG_AMPLITUDE, LOG_LENGTHSCALE)

  # lengthscale is 3 m in x and 7 m in y, so y stays more correlated.
  assert float(k_y) > float(k_x)


def test_the_amplitude_scales_the_whole_kernel():
  a, b = points(6, 4), points(4, 5)

  base = matern52(a, b, LOG_AMPLITUDE, LOG_LENGTHSCALE)
  doubled = matern52(a, b, LOG_AMPLITUDE + np.log(2.0), LOG_LENGTHSCALE)
  np.testing.assert_allclose(doubled.numpy(), 2.0 * base.numpy(), rtol=1e-13)


def test_it_batches_over_leading_axes():
  a = points(30, 6).reshape(5, 6, 2)
  b = points(20, 7).reshape(5, 4, 2)

  batched = matern52(a, b, LOG_AMPLITUDE, LOG_LENGTHSCALE)
  assert batched.shape == (5, 6, 4)

  for i in range(5):
    np.testing.assert_allclose(
      batched[i].numpy(),
      matern52(a[i], b[i], LOG_AMPLITUDE, LOG_LENGTHSCALE).numpy(),
      rtol=1e-14,
    )


# -- the gradient -----------------------------------------------------------


def test_the_gradient_matches_central_differences():
  a, b = points(7, 8), points(5, 9)
  step = 1e-6

  analytic = matern52_gradient(a, b, LOG_AMPLITUDE, LOG_LENGTHSCALE).numpy()

  numeric = np.empty_like(analytic)
  for axis in range(2):
    offset = torch.zeros(2, dtype=torch.float64)
    offset[axis] = step
    numeric[..., axis] = (
      (
        matern52(a + offset, b, LOG_AMPLITUDE, LOG_LENGTHSCALE)
        - matern52(a - offset, b, LOG_AMPLITUDE, LOG_LENGTHSCALE)
      )
      / (2 * step)
    ).numpy()

  np.testing.assert_allclose(analytic, numeric, atol=1e-7)


def test_the_gradient_is_exactly_zero_at_zero_separation():
  """And computes it without a RuntimeWarning, which is the real point.

  The uncancelled form -- radial derivative times ``dr/da`` -- divides by ``r``
  here. ``pyproject.toml`` promotes RuntimeWarning to an error, so writing it
  that way fails the suite rather than quietly returning NaN.
  """
  a = points(6, 10)
  gradient = matern52_gradient(a, a, LOG_AMPLITUDE, LOG_LENGTHSCALE).numpy()

  diagonal = gradient[np.arange(6), np.arange(6)]
  assert np.all(diagonal == 0.0)
  assert np.all(np.isfinite(gradient))


def test_the_gradient_is_finite_for_coincident_points_in_general_position():
  """A duplicated sounding is ordinary in a survey, not an edge case."""
  a = torch.tensor([[1.0, 2.0], [1.0, 2.0], [4.0, -1.0]])
  gradient = matern52_gradient(a, a, LOG_AMPLITUDE, LOG_LENGTHSCALE).numpy()

  assert np.all(np.isfinite(gradient))
  np.testing.assert_allclose(gradient[0, 1], np.zeros(2), atol=1e-15)


def test_the_gradient_points_away_from_the_other_point():
  """Covariance falls off with distance, so its slope points back inward."""
  origin = torch.zeros(1, 2, dtype=torch.float64)
  other = torch.tensor([[5.0, 0.0]])

  gradient = matern52_gradient(
    origin, other, LOG_AMPLITUDE, LOG_LENGTHSCALE
  ).numpy()[0, 0]

  # Moving ``origin`` further from ``other`` (negative x) must lower k, so the
  # derivative with respect to x is positive.
  assert gradient[0] > 0.0
  assert gradient[1] == 0.0


def test_the_gradient_is_antisymmetric_in_its_arguments():
  """``dk/da = -dk/db``, since ``k`` depends only on the difference."""
  a, b = points(5, 11), points(5, 12)

  forward = matern52_gradient(a, b, LOG_AMPLITUDE, LOG_LENGTHSCALE).numpy()
  backward = matern52_gradient(b, a, LOG_AMPLITUDE, LOG_LENGTHSCALE).numpy()

  np.testing.assert_allclose(
    forward, -np.swapaxes(backward, 0, 1), rtol=1e-13, atol=1e-15
  )


def test_the_gradient_batches_over_leading_axes():
  a = points(30, 13).reshape(5, 6, 2)
  b = points(20, 14).reshape(5, 4, 2)

  batched = matern52_gradient(a, b, LOG_AMPLITUDE, LOG_LENGTHSCALE)
  assert batched.shape == (5, 6, 4, 2)


@pytest.mark.parametrize("separation", [1e-8, 1e-4, 1e-2])
def test_the_gradient_is_well_behaved_just_off_the_diagonal(separation):
  """Where an uncancelled 1/r would blow up rather than merely divide by zero."""
  origin = torch.zeros(1, 2, dtype=torch.float64)
  near = torch.tensor([[separation, 0.0]])

  gradient = matern52_gradient(
    origin, near, LOG_AMPLITUDE, LOG_LENGTHSCALE
  ).numpy()

  assert np.all(np.isfinite(gradient))
  # The slope vanishes linearly as the separation closes.
  assert abs(gradient[0, 0, 0]) < 1.0
