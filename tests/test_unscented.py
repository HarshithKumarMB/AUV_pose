"""Sigma points and the moments they carry.

The sharpest test available is exactness on an affine map: the unscented
transform reproduces the mean, the covariance and the cross-covariance of
``A x + b`` to machine precision, so any error in the weights, the spread or
the matrix square root shows up immediately and unambiguously.
"""

import numpy as np

from auv_pose.estimation.manifold import DOF
from auv_pose.estimation.unscented import (
  SigmaRule,
  cross_moments,
  matrix_sqrt,
  sigma_offsets,
  tangent_moments,
  weighted_mean,
)

RULES = [
  SigmaRule(),
  SigmaRule(alpha=1e-3, beta=2.0, kappa=0.0),
  SigmaRule(alpha=0.5, beta=2.0, kappa=3.0),
]


def random_cov(rng, n=DOF, scale=1.0):
  """A well-conditioned covariance with no special structure."""
  root = rng.normal(size=(n, n))
  return (root @ root.T + n * np.eye(n)) * scale


# -- weights ----------------------------------------------------------------


def test_mean_weights_sum_to_one():
  for rule in RULES:
    weights_mean, _ = rule.weights(DOF)
    np.testing.assert_allclose(weights_mean.sum(), 1.0, atol=1e-12)


def test_there_are_two_n_plus_one_points():
  for rule in RULES:
    weights_mean, weights_cov = rule.weights(DOF)
    assert weights_mean.shape == (2 * DOF + 1,)
    assert weights_cov.shape == (2 * DOF + 1,)


def test_a_degenerate_rule_is_rejected_rather_than_dividing_by_zero():
  """``alpha=1, kappa=-n`` puts ``n + lambda`` at zero."""
  try:
    SigmaRule(alpha=1.0, kappa=-DOF).weights(DOF)
  except ValueError as error:
    assert "degenerate" in str(error)
    return
  raise AssertionError("expected a ValueError for a degenerate rule")


# -- the matrix square root -------------------------------------------------


def test_matrix_sqrt_factors_a_well_conditioned_covariance():
  cov = random_cov(np.random.default_rng(0))
  factor = matrix_sqrt(cov)
  np.testing.assert_allclose(factor @ factor.T, cov, atol=1e-9)


def test_matrix_sqrt_handles_a_singular_covariance():
  """A perfectly known direction is legitimate; Cholesky alone refuses it."""
  cov = random_cov(np.random.default_rng(1))
  cov[:, 4] = 0.0
  cov[4, :] = 0.0

  factor = matrix_sqrt(cov)
  assert np.all(np.isfinite(factor))
  np.testing.assert_allclose(factor @ factor.T, cov, atol=1e-8)


def test_matrix_sqrt_handles_an_all_zero_covariance():
  factor = matrix_sqrt(np.zeros((DOF, DOF)))
  np.testing.assert_allclose(factor, np.zeros((DOF, DOF)), atol=1e-15)


def test_matrix_sqrt_symmetrises_its_input():
  cov = random_cov(np.random.default_rng(2))
  skewed = cov + np.tril(np.ones_like(cov), -1) * 1e-9

  np.testing.assert_allclose(matrix_sqrt(skewed), matrix_sqrt(cov), atol=1e-7)


# -- offsets ----------------------------------------------------------------


def test_the_first_offset_is_the_mean():
  cov = random_cov(np.random.default_rng(3))
  assert np.all(sigma_offsets(cov)[0] == 0.0)


def test_offsets_reproduce_the_covariance_exactly():
  """The defining property, and what every moment below rests on."""
  rng = np.random.default_rng(4)
  for rule in RULES:
    cov = random_cov(rng)
    offsets = sigma_offsets(cov, rule)
    _, weights_cov = rule.weights(DOF)

    np.testing.assert_allclose(
      tangent_moments(offsets, weights_cov), cov, atol=1e-8
    )


def test_offsets_are_symmetric_about_the_mean():
  cov = random_cov(np.random.default_rng(5))
  offsets = sigma_offsets(cov)

  np.testing.assert_allclose(
    offsets[1 : DOF + 1], -offsets[DOF + 1 :], atol=1e-15
  )


# -- exactness on an affine map ---------------------------------------------


def affine_transform(cov, rule, A, b):
  """Push a zero-mean Gaussian through ``A x + b`` by sigma points."""
  offsets = sigma_offsets(cov, rule)
  weights_mean, weights_cov = rule.weights(cov.shape[0])

  images = offsets @ A.T + b
  mean = weights_mean @ images
  centred = images - mean

  return mean, tangent_moments(centred, weights_cov), offsets, centred


def test_the_transform_is_exact_for_an_affine_mean():
  rng = np.random.default_rng(6)
  cov = random_cov(rng)
  A = rng.normal(size=(7, DOF))
  b = rng.normal(size=7)

  for rule in RULES:
    mean, _, _, _ = affine_transform(cov, rule, A, b)
    np.testing.assert_allclose(mean, b, atol=1e-9)


def test_the_transform_is_exact_for_an_affine_covariance():
  rng = np.random.default_rng(7)
  cov = random_cov(rng)
  A = rng.normal(size=(7, DOF))
  b = rng.normal(size=7)

  for rule in RULES:
    _, moved, _, _ = affine_transform(cov, rule, A, b)
    np.testing.assert_allclose(moved, A @ cov @ A.T, atol=1e-8)


def test_the_transform_is_exact_for_an_affine_cross_covariance():
  """``Cov[x, Ax + b] = P A^T`` -- what the backward pass consumes."""
  rng = np.random.default_rng(8)
  cov = random_cov(rng)
  A = rng.normal(size=(7, DOF))
  b = rng.normal(size=7)

  for rule in RULES:
    _, _, offsets, centred = affine_transform(cov, rule, A, b)
    _, weights_cov = rule.weights(DOF)

    np.testing.assert_allclose(
      cross_moments(offsets, centred, weights_cov), cov @ A.T, atol=1e-8
    )


# -- the generic mean -------------------------------------------------------


def test_weighted_mean_reduces_to_the_arithmetic_mean_in_a_vector_space():
  rng = np.random.default_rng(9)
  points = list(rng.normal(size=(9, 4)))
  weights = rng.uniform(0.1, 1.0, size=9)
  weights /= weights.sum()

  mean = weighted_mean(
    points, weights, boxplus=lambda a, d: a + d, boxminus=lambda a, b: a - b
  )
  np.testing.assert_allclose(mean, weights @ np.array(points), atol=1e-12)


def test_weighted_mean_converges_in_one_pass_in_a_vector_space():
  """Nothing to iterate when the chart is flat, whatever the seed."""
  rng = np.random.default_rng(10)
  points = list(rng.normal(size=(5, 3)))
  weights = np.full(5, 0.2)

  kwargs = {"boxplus": lambda a, d: a + d, "boxminus": lambda a, b: a - b}
  one = weighted_mean(points, weights, initial=points[4], max_iter=1, **kwargs)
  many = weighted_mean(points, weights, initial=points[4], **kwargs)

  np.testing.assert_allclose(one, many, atol=1e-15)


# -- moment helpers ---------------------------------------------------------


def test_tangent_moments_returns_a_symmetric_matrix():
  rng = np.random.default_rng(11)
  offsets = rng.normal(size=(31, DOF))
  weights = rng.normal(size=31)

  cov = tangent_moments(offsets, weights)
  np.testing.assert_allclose(cov, cov.T, atol=1e-15)


def test_cross_moments_has_the_shape_of_its_two_arguments():
  rng = np.random.default_rng(12)
  left = rng.normal(size=(31, DOF))
  right = rng.normal(size=(31, 6))

  assert cross_moments(left, right, np.full(31, 1 / 31)).shape == (DOF, 6)


def test_cross_moments_of_a_set_with_itself_is_its_covariance():
  rng = np.random.default_rng(13)
  offsets = rng.normal(size=(31, DOF))
  weights = np.full(31, 1 / 31)

  np.testing.assert_allclose(
    cross_moments(offsets, offsets, weights),
    tangent_moments(offsets, weights),
    atol=1e-12,
  )
