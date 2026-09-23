"""The Vecchia log marginal likelihood.

Correctness rests on two tests that chain together, and neither is sufficient
alone.

The first says the **approximation is exact when it conditions on everything**.
With every sounding conditioning on all its predecessors the factorisation is
the chain rule, so the answer must equal a dense GP's log density to machine
precision. That pins the block assembly, the noise handling, the ``(g, i)``
arrangement, the last-row identity and the linear mean -- against a reference
that is unambiguously right.

The second says the **batched path equals the naive definition** at realistic
sizes, where exactness no longer holds. The reference is a Python loop over
soundings computing each conditional density straight from the formula.

Together: the code equals the definition, and the definition at full
conditioning equals the truth. Either on its own leaves a gap -- the first only
ever exercises a single batched row, and the second would pass just as happily
if the definition itself had been transcribed wrongly.
"""

import math
from itertools import pairwise

import numpy as np
import pytest
import torch

from auv_pose.estimation.terrain import DepthMap
from auv_pose.mapping.kernels import matern52, matern52_gradient
from auv_pose.mapping.ordering import ordered_neighbours
from auv_pose.mapping.vecchia import (
  LINEAR_MEAN,
  MeanBasis,
  VecchiaHyperparameters,
  VecchiaMap,
  VecchiaStructure,
  build_structure,
  design_matrix,
  draw,
  fit_vecchia,
  fit_vecchia_nigp,
  initial_hyperparameters,
  sparse_factor,
  vecchia_loglik,
  vecchia_reml,
  whiten,
)

LOG_AMPLITUDE = torch.tensor(math.log(4.0), dtype=torch.float64)
LOG_LENGTHSCALE = torch.log(torch.tensor([6.0, 11.0], dtype=torch.float64))


def survey(n, seed, spread=40.0):
  """Scattered soundings and a residual drawn from nothing in particular."""
  rng = np.random.default_rng(seed)
  points = rng.uniform(-spread, spread, size=(n, 2))
  residual = rng.normal(scale=2.0, size=n)
  return points, residual


def dense_loglik(points, residual, noise, jitter=0.0):
  """The exact GP log density. A few lines, and unambiguously right."""
  a = torch.tensor(points, dtype=torch.float64)
  kernel = matern52(a, a, LOG_AMPLITUDE, LOG_LENGTHSCALE).numpy()
  kernel = kernel + np.diag(np.broadcast_to(noise, len(points)) + jitter)

  factor = np.linalg.cholesky(kernel)
  solved = np.linalg.solve(factor, residual)

  return (
    -0.5 * len(points) * math.log(2 * math.pi)
    - np.log(np.diag(factor)).sum()
    - 0.5 * solved @ solved
  )


def naive_loglik(structure, points, residual, noise, jitter=0.0):
  """The Vecchia definition, written out: one conditional density per sounding.

  Deliberately a slow Python loop over the formula, with no batching and no
  Cholesky trickery, so it shares nothing with the implementation but the
  kernel.
  """
  noise = np.broadcast_to(np.asarray(noise, dtype=float), len(points))
  total = 0.0

  def block(indices):
    a = torch.tensor(points[indices], dtype=torch.float64)
    k = matern52(a, a, LOG_AMPLITUDE, LOG_LENGTHSCALE).numpy()
    return k + np.diag(noise[indices] + jitter)

  head = np.arange(structure.n0)
  factor = np.linalg.cholesky(block(head))
  solved = np.linalg.solve(factor, residual[head])
  total += (
    -0.5 * len(head) * math.log(2 * math.pi)
    - np.log(np.diag(factor)).sum()
    - 0.5 * solved @ solved
  )

  for i in range(structure.n0, len(points)):
    g = structure.neighbours[i]
    g = g[g >= 0]

    joint = block(np.append(g, i))
    k_gg, k_gi = joint[:-1, :-1], joint[:-1, -1]

    weights = np.linalg.solve(k_gg, k_gi)
    mean = weights @ residual[g]
    variance = joint[-1, -1] - k_gi @ weights

    total += -0.5 * (
      math.log(2 * math.pi)
      + math.log(variance)
      + (residual[i] - mean) ** 2 / variance
    )

  return total


# -- exactness at full conditioning -----------------------------------------


def test_it_reproduces_a_dense_gp_when_it_conditions_on_everything():
  """The pin. At ``m = N - 1`` the approximation is the chain rule.

  ``n0`` is set to ``m`` so the final sounding goes through the *batched* path
  rather than the dense head, which is what makes this a test of the block
  assembly and not only of the head block.
  """
  for n in (8, 20, 41):
    points, residual = survey(n, seed=n)
    noise = 0.3

    structure = build_structure(points, m=n - 1, n0=n - 1)
    ordered = structure.points
    ordered_residual = residual[structure.order]

    approximate = vecchia_loglik(
      structure,
      torch.tensor(ordered_residual),
      LOG_AMPLITUDE,
      LOG_LENGTHSCALE,
      torch.tensor(noise, dtype=torch.float64),
      jitter=0.0,
    )
    exact = dense_loglik(ordered, ordered_residual, noise)

    np.testing.assert_allclose(float(approximate), exact, rtol=1e-11)


def test_a_wholly_dense_head_reproduces_a_dense_gp():
  """``n0 = N`` short-circuits the chain entirely; it must still be the GP."""
  points, residual = survey(60, seed=1)
  structure = build_structure(points, m=10, n0=60)

  approximate = vecchia_loglik(
    structure,
    torch.tensor(residual[structure.order]),
    LOG_AMPLITUDE,
    LOG_LENGTHSCALE,
    torch.tensor(0.5, dtype=torch.float64),
    jitter=0.0,
  )
  exact = dense_loglik(structure.points, residual[structure.order], 0.5)

  np.testing.assert_allclose(float(approximate), exact, rtol=1e-11)


def test_exactness_holds_for_per_sounding_noise():
  """The NIGP step makes the noise a vector; the exactness must survive it."""
  n = 30
  points, residual = survey(n, seed=2)
  noise = np.random.default_rng(3).uniform(0.1, 2.0, size=n)

  structure = build_structure(points, m=n - 1, n0=n - 1)
  ordered_noise = noise[structure.order]

  approximate = vecchia_loglik(
    structure,
    torch.tensor(residual[structure.order]),
    LOG_AMPLITUDE,
    LOG_LENGTHSCALE,
    torch.tensor(ordered_noise),
    jitter=0.0,
  )
  exact = dense_loglik(
    structure.points, residual[structure.order], ordered_noise
  )

  np.testing.assert_allclose(float(approximate), exact, rtol=1e-11)


# -- the batched path against the definition --------------------------------


def test_it_matches_the_naive_definition_at_realistic_sizes():
  """Where the approximation is a real approximation, and batching is real."""
  for n, m in ((200, 10), (400, 25), (150, 5)):
    points, residual = survey(n, seed=n + m)
    noise = 0.4

    structure = build_structure(points, m=m, n0=max(m, 20))
    approximate = vecchia_loglik(
      structure,
      torch.tensor(residual[structure.order]),
      LOG_AMPLITUDE,
      LOG_LENGTHSCALE,
      torch.tensor(noise, dtype=torch.float64),
      jitter=0.0,
    )
    reference = naive_loglik(
      structure, structure.points, residual[structure.order], noise
    )

    np.testing.assert_allclose(float(approximate), reference, rtol=1e-10)


def test_chunking_does_not_change_the_answer():
  """The likelihood is a sum over soundings, so chunking must be exact."""
  points, residual = survey(300, seed=4)
  structure = build_structure(points, m=15, n0=32)

  def evaluate(chunk):
    return float(
      vecchia_loglik(
        structure,
        torch.tensor(residual[structure.order]),
        LOG_AMPLITUDE,
        LOG_LENGTHSCALE,
        torch.tensor(0.6, dtype=torch.float64),
        chunk=chunk,
      )
    )

  reference = evaluate(8192)
  for chunk in (1, 7, 64, 299):
    np.testing.assert_allclose(evaluate(chunk), reference, rtol=1e-12)


# -- behaviour --------------------------------------------------------------


def test_it_approaches_the_exact_likelihood_as_m_grows():
  """The accuracy claim, and the test that catches a plausible-but-wrong g(i)."""
  points, residual = survey(250, seed=5)
  noise = 0.4
  ordered = build_structure(points, m=2, n0=2)

  exact = dense_loglik(ordered.points, residual[ordered.order], noise)

  errors = []
  for m in (2, 5, 15, 40):
    structure = build_structure(points, m=m, n0=m)
    # Same ordering every time, so only the conditioning set changes.
    assert np.array_equal(structure.order, ordered.order)

    value = float(
      vecchia_loglik(
        structure,
        torch.tensor(residual[structure.order]),
        LOG_AMPLITUDE,
        LOG_LENGTHSCALE,
        torch.tensor(noise, dtype=torch.float64),
      )
    )
    errors.append(abs(value - exact))

  assert all(a > b for a, b in pairwise(errors)), errors


def test_maximin_beats_an_arbitrary_ordering_at_the_same_m():
  """Why the ordering is stated in the paper rather than left implicit.

  A bad ordering still yields a valid, positive-definite Gaussian -- just a
  worse approximation of the intended one. Nothing else in the suite notices.
  """
  points, residual = survey(400, seed=6)
  noise = 0.4
  m = 8

  maximin = build_structure(points, m=m, n0=m)
  exact = dense_loglik(maximin.points, residual[maximin.order], noise)

  shuffled = np.random.default_rng(7).permutation(len(points))
  arbitrary = VecchiaStructure(
    points=points[shuffled],
    neighbours=ordered_neighbours(points[shuffled], m=m),
    order=shuffled,
    n0=m,
  )

  def error(structure):
    value = float(
      vecchia_loglik(
        structure,
        torch.tensor(residual[structure.order]),
        LOG_AMPLITUDE,
        LOG_LENGTHSCALE,
        torch.tensor(noise, dtype=torch.float64),
      )
    )
    return abs(value - exact)

  assert error(maximin) < error(arbitrary)


def test_it_is_differentiable_in_the_hyperparameters():
  """Autograd through the batched Cholesky is what the fit rests on."""
  points, residual = survey(200, seed=8)
  structure = build_structure(points, m=10, n0=16)

  amplitude = LOG_AMPLITUDE.clone().requires_grad_(True)
  lengthscale = LOG_LENGTHSCALE.clone().requires_grad_(True)
  noise = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

  value = vecchia_loglik(
    structure,
    torch.tensor(residual[structure.order]),
    amplitude,
    lengthscale,
    noise,
  )
  value.backward()

  gradients = (amplitude.grad, lengthscale.grad, noise.grad)
  assert all(g is not None for g in gradients)
  for gradient in gradients:
    assert torch.all(torch.isfinite(gradient)), gradient

  # Regression: the lengthscale gradient came back NaN, from ``sqrt(0)`` on the
  # diagonal of every kernel block. The forward value was finite throughout.
  assert float(torch.abs(lengthscale.grad).max()) > 0.0


def test_the_likelihood_peaks_near_the_truth():
  """A draw from a known GP should score best at the hyperparameters it used."""
  rng = np.random.default_rng(9)
  points = rng.uniform(-30, 30, size=(300, 2))

  a = torch.tensor(points)
  kernel = matern52(a, a, LOG_AMPLITUDE, LOG_LENGTHSCALE).numpy()
  draw = np.linalg.cholesky(kernel + 0.05 * np.eye(len(points))) @ rng.normal(
    size=len(points)
  )

  structure = build_structure(points, m=20, n0=32)
  ordered = torch.tensor(draw[structure.order])

  def score(log_lengthscale):
    return float(
      vecchia_loglik(
        structure,
        ordered,
        LOG_AMPLITUDE,
        log_lengthscale,
        torch.tensor(0.05, dtype=torch.float64),
      )
    )

  truth = score(LOG_LENGTHSCALE)
  assert truth > score(LOG_LENGTHSCALE - math.log(4.0))
  assert truth > score(LOG_LENGTHSCALE + math.log(4.0))


# -- structure --------------------------------------------------------------


def test_the_structure_reorders_into_maximin_order():
  points, _ = survey(120, seed=10)
  structure = build_structure(points, m=8)

  np.testing.assert_array_equal(structure.points, points[structure.order])
  np.testing.assert_array_equal(np.sort(structure.order), np.arange(120))


def test_the_head_block_is_never_smaller_than_the_conditioning_set():
  """Otherwise the batched rows would be ragged."""
  points, _ = survey(200, seed=11)
  assert build_structure(points, m=30, n0=5).n0 == 30
  assert build_structure(points, m=30).n0 == 64


def test_the_head_block_is_capped_at_the_survey_size():
  points, _ = survey(10, seed=12)
  assert build_structure(points, m=4, n0=1000).n0 == 10


def test_every_batched_row_has_a_full_conditioning_set():
  points, _ = survey(300, seed=13)
  structure = build_structure(points, m=12, n0=20)
  assert np.all(structure.neighbours[structure.n0 :] >= 0)


def test_the_design_matrix_is_one_and_the_coordinates():
  points = np.array([[1.0, 2.0], [3.0, -4.0]])
  np.testing.assert_array_equal(
    design_matrix(points), [[1.0, 1.0, 2.0], [1.0, 3.0, -4.0]]
  )


def test_it_rejects_malformed_inputs():
  for bad in (np.zeros((5, 3)), np.zeros(5)):
    try:
      build_structure(bad)
    except ValueError:
      continue
    raise AssertionError(f"expected a ValueError for shape {bad.shape}")


# -- fitting ----------------------------------------------------------------


def gp_draw(points, log_amplitude, log_lengthscale, noise, seed):
  """An exact draw from the GP the fit is supposed to recover."""
  a = torch.tensor(points, dtype=torch.float64)
  kernel = matern52(a, a, log_amplitude, log_lengthscale).numpy()
  factor = np.linalg.cholesky(kernel + noise * np.eye(len(points)))
  return factor @ np.random.default_rng(seed).normal(size=len(points))


def test_it_recovers_the_hyperparameters_of_a_known_draw():
  """The statistical test, and the counterpart of the SVGP's lengthscale check.

  Draw a field with known amplitude, lengthscales and nugget; fit; see whether
  the fit finds them. This replaces reaching into a library's internals with a
  question the model can actually be wrong about.
  """
  rng = np.random.default_rng(20)
  points = rng.uniform(-40, 40, size=(700, 2))

  truth_amplitude = math.log(9.0)
  truth_lengthscale = torch.log(torch.tensor([5.0, 12.0], dtype=torch.float64))
  truth_noise = 0.25

  depth = gp_draw(
    points,
    torch.tensor(truth_amplitude, dtype=torch.float64),
    truth_lengthscale,
    truth_noise,
    seed=21,
  )

  fitted = fit_vecchia(points, depth, m=20, steps=400, device="cpu")

  assert 0.5 < fitted.hyper.amplitude / 9.0 < 2.0, fitted.hyper.amplitude
  assert 0.5 < fitted.hyper.noise / truth_noise < 2.0, fitted.hyper.noise
  ratio = fitted.lengthscale / np.array([5.0, 12.0])
  assert np.all((ratio > 0.5) & (ratio < 2.0)), fitted.lengthscale


def test_it_learns_a_lengthscale_per_axis():
  """A field stretched along one axis must come back with unequal lengthscales.

  The direct counterpart of ``test_lengthscales_are_learned_per_axis`` in the
  SVGP suite, but read off a fitted parameter in metres rather than out of
  gpytorch's internals.
  """
  rng = np.random.default_rng(22)
  points = rng.uniform(-40, 40, size=(600, 2))
  depth = gp_draw(
    points,
    torch.tensor(math.log(4.0), dtype=torch.float64),
    torch.log(torch.tensor([3.0, 15.0], dtype=torch.float64)),
    0.1,
    seed=23,
  )

  fitted = fit_vecchia(points, depth, m=20, steps=400, device="cpu")
  assert fitted.lengthscale[1] > 2.0 * fitted.lengthscale[0], fitted.lengthscale


def test_the_likelihood_climbs_over_the_fit():
  rng = np.random.default_rng(24)
  points = rng.uniform(-30, 30, size=(400, 2))
  depth = gp_draw(
    points,
    torch.tensor(math.log(4.0), dtype=torch.float64),
    torch.log(torch.tensor([6.0, 6.0], dtype=torch.float64)),
    0.2,
    seed=25,
  )

  fitted = fit_vecchia(points, depth, m=15, steps=120, device="cpu")

  assert len(fitted.loglik_trace) == 120
  assert fitted.loglik_trace[-1] > fitted.loglik_trace[0]
  # And the last tenth should be flattening out, not still climbing steeply.
  tail = fitted.loglik_trace[-12:]
  assert tail[-1] - tail[0] < fitted.loglik_trace[12] - fitted.loglik_trace[0]


def test_the_fit_is_deterministic():
  """No subsampling, no random initialisation, so a refit must agree exactly.

  Which matters because the ordering is stored in a checkpoint, and a refit
  that disagreed with it would be a confusing thing to chase.
  """
  points, depth = survey(300, seed=26)
  kwargs = {"m": 12, "steps": 40, "device": "cpu"}

  first = fit_vecchia(points, depth, **kwargs)
  second = fit_vecchia(points, depth, **kwargs)

  np.testing.assert_array_equal(first.structure.order, second.structure.order)
  np.testing.assert_allclose(first.beta, second.beta, rtol=1e-14)
  np.testing.assert_allclose(first.lengthscale, second.lengthscale, rtol=1e-14)
  assert first.loglik_trace == second.loglik_trace


def test_the_linear_mean_absorbs_a_plane():
  """``beta`` is the regional trend, so a plane should leave nothing behind."""
  rng = np.random.default_rng(27)
  points = rng.uniform(-30, 30, size=(400, 2))
  depth = -60.0 + 0.3 * points[:, 0] - 0.15 * points[:, 1]

  fitted = fit_vecchia(points, depth, m=10, steps=30, device="cpu")

  np.testing.assert_allclose(fitted.beta, [-60.0, 0.3, -0.15], atol=1e-8)
  assert np.abs(fitted.residual).max() < 1e-8


def test_tying_the_lengthscales_keeps_them_tied():
  rng = np.random.default_rng(28)
  points = rng.uniform(-40, 40, size=(400, 2))
  depth = gp_draw(
    points,
    torch.tensor(math.log(4.0), dtype=torch.float64),
    torch.log(torch.tensor([3.0, 15.0], dtype=torch.float64)),
    0.1,
    seed=29,
  )

  fitted = fit_vecchia(points, depth, m=15, steps=60, ard=False, device="cpu")
  np.testing.assert_allclose(
    fitted.lengthscale[0], fitted.lengthscale[1], rtol=1e-12
  )


def test_the_initial_guess_comes_from_the_data():
  points = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 50.0], [100.0, 50.0]])
  residual = np.array([1.0, -1.0, 2.0, -2.0])

  start = initial_hyperparameters(points, residual)

  np.testing.assert_allclose(start.lengthscale, [10.0, 5.0], rtol=1e-12)
  np.testing.assert_allclose(start.amplitude, residual.var(), rtol=1e-12)
  np.testing.assert_allclose(start.noise, 0.01 * residual.var(), rtol=1e-12)


def test_the_noise_is_per_sounding_once_inflated():
  points, depth = survey(200, seed=30)
  inflation = np.linspace(0.0, 5.0, 200)

  fitted = fit_vecchia(
    points, depth, m=10, steps=20, noise_inflation=inflation, device="cpu"
  )

  assert fitted.noise.shape == (200,)
  # Stored in the structure's order, and still the nugget plus the inflation.
  np.testing.assert_allclose(
    fitted.noise,
    fitted.hyper.noise + inflation[fitted.structure.order],
    rtol=1e-12,
  )


def test_it_rejects_mismatched_inputs():
  points, _ = survey(20, seed=31)
  try:
    fit_vecchia(points, np.zeros(19), m=4, steps=1, device="cpu")
  except ValueError as error:
    assert "20" in str(error) and "19" in str(error)
    return
  raise AssertionError("expected a ValueError for mismatched lengths")


# -- REML -------------------------------------------------------------------


def dense_reml(points, depth, noise):
  """Restricted log likelihood and GLS mean, computed densely. The reference."""
  a = torch.tensor(points, dtype=torch.float64)
  kernel = matern52(a, a, LOG_AMPLITUDE, LOG_LENGTHSCALE).numpy()
  kernel = kernel + np.diag(np.broadcast_to(noise, len(points)))

  basis = design_matrix(points)
  n, p = basis.shape

  inverse_basis = np.linalg.solve(kernel, basis)
  information = basis.T @ inverse_basis
  beta = np.linalg.solve(information, inverse_basis.T @ depth)

  residual = depth - basis @ beta
  factor = np.linalg.cholesky(kernel)

  value = (
    -0.5 * (n - p) * math.log(2 * math.pi)
    - np.log(np.diag(factor)).sum()
    - 0.5 * np.linalg.slogdet(information)[1]
    - 0.5 * residual @ np.linalg.solve(kernel, residual)
  )
  return value, beta


def test_whitening_reproduces_the_precision_it_stands_for():
  """``H' K^-1 H = (U'H)' (U'H)`` -- the identity REML is built on.

  At ``m = N - 1`` the approximation is exact, so the whitened form must equal
  a dense solve. This is what makes the extra REML terms cost three matvecs
  rather than an inverse.
  """
  n = 40
  points, _ = survey(n, seed=40)
  noise = 0.3

  structure = build_structure(points, m=n - 1, n0=n - 1)
  basis = torch.tensor(design_matrix(structure.points))

  whitened, _ = whiten(
    structure,
    basis,
    LOG_AMPLITUDE,
    LOG_LENGTHSCALE,
    torch.tensor(noise, dtype=torch.float64),
    jitter=0.0,
  )

  a = torch.tensor(structure.points)
  kernel = matern52(
    a, a, LOG_AMPLITUDE, LOG_LENGTHSCALE
  ).numpy() + noise * np.eye(n)
  dense = basis.numpy().T @ np.linalg.solve(kernel, basis.numpy())

  np.testing.assert_allclose((whitened.T @ whitened).numpy(), dense, rtol=1e-9)


def test_whitening_treats_a_vector_and_a_one_column_matrix_alike():
  points, residual = survey(80, seed=41)
  structure = build_structure(points, m=12, n0=16)
  ordered = torch.tensor(residual[structure.order])

  args = (
    LOG_AMPLITUDE,
    LOG_LENGTHSCALE,
    torch.tensor(0.4, dtype=torch.float64),
  )
  flat, first = whiten(structure, ordered, *args)
  column, second = whiten(structure, ordered[:, None], *args)

  assert flat.shape == (80,)
  assert column.shape == (80, 1)
  np.testing.assert_allclose(flat.numpy(), column.numpy().ravel(), rtol=1e-14)
  np.testing.assert_allclose(float(first), float(second), rtol=1e-14)


def test_reml_reproduces_a_dense_reml_at_full_conditioning():
  """The exactness pin, carried over to the restricted likelihood."""
  for n in (20, 35):
    points, depth = survey(n, seed=42 + n)
    noise = 0.3

    structure = build_structure(points, m=n - 1, n0=n - 1)
    ordered_depth = depth[structure.order]

    value, beta = vecchia_reml(
      structure,
      torch.tensor(ordered_depth),
      torch.tensor(design_matrix(structure.points)),
      LOG_AMPLITUDE,
      LOG_LENGTHSCALE,
      torch.tensor(noise, dtype=torch.float64),
      jitter=0.0,
    )
    expected, expected_beta = dense_reml(structure.points, ordered_depth, noise)

    np.testing.assert_allclose(float(value), expected, rtol=1e-9)
    np.testing.assert_allclose(beta.numpy(), expected_beta, rtol=1e-8)


def test_reml_recovers_the_amplitude_better_than_plain_likelihood():
  """The reason REML is the default.

  Estimating the mean shortens the residual, and a plug-in likelihood reads
  that as a smaller amplitude. Measured over eight draws from a known field:
  about -9% under ML against about -3% under REML.
  """
  rng = np.random.default_rng(44)
  truth = 9.0

  ml, reml = [], []
  for seed in range(4):
    points = rng.uniform(-40, 40, size=(350, 2))
    depth = gp_draw(
      points,
      torch.tensor(math.log(truth), dtype=torch.float64),
      torch.log(torch.tensor([5.0, 12.0], dtype=torch.float64)),
      0.25,
      seed=seed,
    )
    ml.append(
      fit_vecchia(
        points, depth, m=20, steps=300, method="ml", device="cpu"
      ).hyper.amplitude
    )
    reml.append(
      fit_vecchia(
        points, depth, m=20, steps=300, method="reml", device="cpu"
      ).hyper.amplitude
    )

  ml_bias = abs(np.mean(ml) / truth - 1.0)
  reml_bias = abs(np.mean(reml) / truth - 1.0)
  assert reml_bias < ml_bias, (np.mean(ml), np.mean(reml))


def test_reml_refits_beta_rather_than_keeping_the_least_squares_one():
  """What makes it REML and not maximum likelihood with a plug-in mean."""
  rng = np.random.default_rng(45)
  points = rng.uniform(-40, 40, size=(300, 2))
  depth = gp_draw(
    points,
    torch.tensor(math.log(9.0), dtype=torch.float64),
    torch.log(torch.tensor([14.0, 14.0], dtype=torch.float64)),
    0.2,
    seed=46,
  )

  fitted = fit_vecchia(points, depth, m=20, steps=200, device="cpu")

  ordinary = np.linalg.lstsq(
    design_matrix(fitted.structure.points),
    depth[fitted.structure.order],
    rcond=None,
  )[0]

  # A long lengthscale correlates the residuals strongly, which is exactly
  # where generalised and ordinary least squares part company.
  assert not np.allclose(fitted.beta, ordinary, rtol=1e-3)


def test_the_stored_residual_matches_the_stored_beta():
  points, depth = survey(250, seed=47)
  fitted = fit_vecchia(points, depth, m=12, steps=30, device="cpu")

  basis = design_matrix(fitted.structure.points)
  np.testing.assert_allclose(
    fitted.residual,
    depth[fitted.structure.order] - basis @ fitted.beta,
    atol=1e-12,
  )


def test_reml_is_differentiable():
  points, depth = survey(200, seed=48)
  structure = build_structure(points, m=10, n0=16)

  amplitude = LOG_AMPLITUDE.clone().requires_grad_(True)
  lengthscale = LOG_LENGTHSCALE.clone().requires_grad_(True)

  value, beta = vecchia_reml(
    structure,
    torch.tensor(depth[structure.order]),
    torch.tensor(design_matrix(structure.points)),
    amplitude,
    lengthscale,
    torch.tensor(0.4, dtype=torch.float64),
  )
  value.backward()

  assert beta.shape == (3,)
  assert torch.all(torch.isfinite(amplitude.grad))
  assert torch.all(torch.isfinite(lengthscale.grad))


def test_the_fit_rejects_an_unknown_method():
  points, depth = survey(20, seed=49)
  try:
    fit_vecchia(points, depth, m=4, steps=1, method="mle", device="cpu")
  except ValueError as error:
    assert "reml" in str(error)
    return
  raise AssertionError("expected a ValueError for an unknown method")


# -- prediction -------------------------------------------------------------


def dense_kriging(points, depth, noise, queries, observation_noise=False):
  """Universal kriging, computed densely. The reference for prediction.

  GLS mean, the usual conditional covariance, and the term that accounts for
  ``beta`` having been estimated rather than known.
  """
  a = torch.tensor(points, dtype=torch.float64)
  b = torch.tensor(queries, dtype=torch.float64)

  kernel = matern52(a, a, LOG_AMPLITUDE, LOG_LENGTHSCALE).numpy()
  kernel = kernel + np.diag(np.broadcast_to(noise, len(points)))
  cross = matern52(b, a, LOG_AMPLITUDE, LOG_LENGTHSCALE).numpy()
  query_cov = matern52(b, b, LOG_AMPLITUDE, LOG_LENGTHSCALE).numpy()

  basis = design_matrix(points)
  query_basis = design_matrix(queries)

  inverse_basis = np.linalg.solve(kernel, basis)
  information = basis.T @ inverse_basis
  beta = np.linalg.solve(information, inverse_basis.T @ depth)

  weights = np.linalg.solve(kernel, cross.T).T
  mean = query_basis @ beta + weights @ (depth - basis @ beta)

  residual_basis = query_basis - weights @ basis
  covariance = (
    query_cov
    - weights @ cross.T
    + residual_basis @ np.linalg.solve(information, residual_basis.T)
  )
  if observation_noise:
    covariance = covariance + math.exp(LOG_NOISE) * np.eye(len(queries))

  return mean, 0.5 * (covariance + covariance.T)


LOG_NOISE = math.log(0.3)


def fitted_at(points, depth, noise, m, n0=None):
  """A map with hyperparameters pinned, so prediction is tested on its own."""
  structure = build_structure(points, m=m, n0=n0)
  basis = design_matrix(structure.points)
  ordered_depth = depth[structure.order]

  _, beta = vecchia_reml(
    structure,
    torch.tensor(ordered_depth),
    torch.tensor(basis),
    LOG_AMPLITUDE,
    LOG_LENGTHSCALE,
    torch.tensor(noise, dtype=torch.float64),
    jitter=0.0,
  )
  beta = beta.numpy()

  whitened, _ = whiten(
    structure,
    torch.tensor(basis),
    LOG_AMPLITUDE,
    LOG_LENGTHSCALE,
    torch.tensor(noise, dtype=torch.float64),
    jitter=0.0,
  )

  return VecchiaMap(
    structure=structure,
    residual=ordered_depth - basis @ beta,
    beta=beta,
    hyper=VecchiaHyperparameters(
      log_amplitude=float(LOG_AMPLITUDE),
      log_lengthscale=(float(LOG_LENGTHSCALE[0]), float(LOG_LENGTHSCALE[1])),
      log_noise=LOG_NOISE,
    ),
    noise=np.full(len(points), noise),
    loglik_trace=[],
    information=(whitened.T @ whitened).numpy(),
  )


def test_a_single_query_at_full_conditioning_is_exact_kriging():
  """The prediction pin.

  With ``m = N`` and one query, the query conditions on the whole survey, so
  there is no approximation left -- the answer must be universal kriging. This
  pins the block assembly, the latent-versus-response noise mask, the mean, the
  conditional variance and the ``beta``-uncertainty term together.
  """
  for n in (18, 30):
    points, depth = survey(n, seed=60 + n)
    noise = 0.3
    fitted = fitted_at(points, depth, noise, m=n, n0=n)

    queries = np.random.default_rng(61).uniform(-25, 25, size=(4, 2))
    for query in queries:
      mean, covariance = fitted.predict_joint(
        query[None], observation_noise=False, jitter=0.0
      )
      expected_mean, expected_cov = dense_kriging(
        fitted.structure.points,
        depth[fitted.structure.order],
        noise,
        query[None],
      )
      np.testing.assert_allclose(mean, expected_mean, rtol=1e-8)
      np.testing.assert_allclose(covariance, expected_cov, rtol=1e-8)


def test_the_observation_noise_adds_the_nugget_to_the_diagonal():
  points, depth = survey(25, seed=62)
  fitted = fitted_at(points, depth, 0.3, m=25, n0=25)
  query = np.array([[2.0, -3.0]])

  _, latent = fitted.predict_joint(query, observation_noise=False, jitter=0.0)
  _, sounding = fitted.predict_joint(query, observation_noise=True, jitter=0.0)

  np.testing.assert_allclose(
    sounding - latent, math.exp(LOG_NOISE) * np.eye(1), atol=1e-12
  )


def test_the_beta_uncertainty_term_widens_the_map():
  """It is the difference between knowing the trend and having estimated it."""
  points, depth = survey(40, seed=63)
  fitted = fitted_at(points, depth, 0.3, m=20)

  query = np.random.default_rng(64).uniform(-25, 25, size=(6, 2))
  _, with_term = fitted.predict_joint(query, beta_uncertainty=True)
  _, without = fitted.predict_joint(query, beta_uncertainty=False)

  assert np.all(np.diag(with_term) > np.diag(without))


def test_the_joint_covariance_is_symmetric_and_positive_definite():
  points, depth = survey(300, seed=65)
  fitted = fitted_at(points, depth, 0.3, m=25)
  query = np.random.default_rng(66).uniform(-30, 30, size=(20, 2))

  _, covariance = fitted.predict_joint(query)
  np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
  assert np.min(np.linalg.eigvalsh(covariance)) > 0.0


def test_permuting_the_queries_permutes_the_answer():
  """Guards the internal maximin reorder and its inverse.

  The queries are reordered before conditioning and must be put back. Getting
  the inverse permutation wrong scrambles which beam is which, silently.
  """
  points, depth = survey(300, seed=67)
  fitted = fitted_at(points, depth, 0.3, m=25)

  query = np.random.default_rng(68).uniform(-30, 30, size=(12, 2))
  shuffle = np.random.default_rng(69).permutation(12)

  mean, covariance = fitted.predict_joint(query)
  shuffled_mean, shuffled_cov = fitted.predict_joint(query[shuffle])

  np.testing.assert_allclose(shuffled_mean, mean[shuffle], rtol=1e-10)
  np.testing.assert_allclose(
    shuffled_cov, covariance[np.ix_(shuffle, shuffle)], rtol=1e-10
  )


def test_nearby_queries_covary_and_distant_ones_do_not():
  """The off-diagonals are why the joint exists rather than a set of marginals."""
  points, depth = survey(400, seed=70)
  fitted = fitted_at(points, depth, 0.3, m=25)

  query = np.array([[0.0, 0.0], [0.4, 0.0], [34.0, 34.0]])
  _, covariance = fitted.predict_joint(query, observation_noise=False)

  assert covariance[0, 1] > 0.0
  assert abs(covariance[0, 2]) < abs(covariance[0, 1])


def test_it_answers_a_whole_sigma_cloud_in_one_call():
  points, depth = survey(400, seed=71)
  fitted = fitted_at(points, depth, 0.3, m=25)
  cloud = np.random.default_rng(72).uniform(-25, 25, size=(7, 5, 2))

  mean, covariance = fitted.predict_joint(cloud)
  assert mean.shape == (7, 5)
  assert covariance.shape == (7, 5, 5)


def test_batched_and_separate_calls_agree():
  points, depth = survey(400, seed=73)
  fitted = fitted_at(points, depth, 0.3, m=25)
  cloud = np.random.default_rng(74).uniform(-25, 25, size=(4, 6, 2))

  mean, covariance = fitted.predict_joint(cloud)
  for i in range(4):
    one_mean, one_cov = fitted.predict_joint(cloud[i])
    np.testing.assert_allclose(mean[i], one_mean, rtol=1e-10)
    np.testing.assert_allclose(covariance[i], one_cov, rtol=1e-10)


def test_predict_returns_elevations_and_spreads():
  points, depth = survey(300, seed=75)
  fitted = fitted_at(points, depth, 0.3, m=25)
  query = np.random.default_rng(76).uniform(-25, 25, size=(9, 2))

  elevation = fitted.predict(query)
  assert elevation.shape == (9,)

  also, spread = fitted.predict(query, with_std=True)
  np.testing.assert_allclose(also, elevation, rtol=1e-12)
  assert np.all(spread > 0.0)


def test_predict_is_chunk_invariant():
  points, depth = survey(300, seed=77)
  fitted = fitted_at(points, depth, 0.3, m=25)
  query = np.random.default_rng(78).uniform(-25, 25, size=(40, 2))

  reference = fitted.predict(query, chunk_size=5000)
  for chunk in (1, 7, 39):
    np.testing.assert_allclose(
      fitted.predict(query, chunk_size=chunk), reference, rtol=1e-12
    )


def test_uncertainty_grows_away_from_the_survey():
  points, depth = survey(400, seed=79, spread=20.0)
  fitted = fitted_at(points, depth, 0.3, m=25)

  _, near = fitted.predict([[0.0, 0.0]], with_std=True)
  _, far = fitted.predict([[400.0, 400.0]], with_std=True)
  assert far[0] > near[0]


def test_more_queries_than_the_conditioning_set_is_refused():
  """Beyond ``m`` the queries stop conditioning on each other, silently."""
  points, depth = survey(300, seed=80)
  fitted = fitted_at(points, depth, 0.3, m=10)

  try:
    fitted.predict_joint(np.zeros((11, 2)))
  except ValueError as error:
    assert "conditioning set" in str(error)
    return
  raise AssertionError("expected a ValueError for B > m")


def test_prediction_rejects_points_of_the_wrong_width():
  points, depth = survey(200, seed=81)
  fitted = fitted_at(points, depth, 0.3, m=20)

  try:
    fitted.predict_joint(np.zeros((4, 3)))
  except ValueError as error:
    assert "(..., b, 2)" in str(error)
  else:
    raise AssertionError("expected a ValueError")

  try:
    fitted.predict(np.zeros((4, 3)))
  except ValueError as error:
    assert "(n, 2)" in str(error)
    return
  raise AssertionError("expected a ValueError")


# -- the mean's gradient ----------------------------------------------------


def dense_gradient(points, residual, noise, beta, queries):
  """``grad mu = (beta_n, beta_e) + [dk/dq]' K^-1 r``, computed densely."""
  a = torch.tensor(points, dtype=torch.float64)
  b = torch.tensor(queries, dtype=torch.float64)

  kernel = matern52(a, a, LOG_AMPLITUDE, LOG_LENGTHSCALE).numpy()
  kernel = kernel + np.diag(np.broadcast_to(noise, len(points)))
  weights = np.linalg.solve(kernel, residual)

  slope = matern52_gradient(
    b[:, None, :], a[None], LOG_AMPLITUDE, LOG_LENGTHSCALE
  ).numpy()[:, 0]

  return beta[1:] + np.einsum("qnd,n->qd", slope, weights)


def test_it_satisfies_the_depth_map_contract():
  """Structural, not nominal -- nothing declares the Protocol as a base."""
  points, depth = survey(200, seed=90)
  assert isinstance(fitted_at(points, depth, 0.3, m=20), DepthMap)


def test_the_gradient_is_exact_at_full_conditioning():
  """The pin: with ``m = N`` the local GP is the whole GP."""
  for n in (20, 32):
    points, depth = survey(n, seed=91 + n)
    noise = 0.3
    fitted = fitted_at(points, depth, noise, m=n, n0=n)

    queries = np.random.default_rng(92).uniform(-25, 25, size=(6, 2))
    expected = dense_gradient(
      fitted.structure.points, fitted.residual, noise, fitted.beta, queries
    )
    # jitter=0.0 so this compares like with like. At the default 1e-8 the two
    # agree only to ~1e-6 relative, which is the regulariser doing its job
    # rather than an error -- worth knowing it is not a no-op.
    np.testing.assert_allclose(
      fitted.mean_gradient(queries, jitter=0.0), expected, rtol=1e-9
    )


def test_the_gradient_matches_central_differences_at_full_conditioning():
  """Where the conditioning set cannot change, so the mean really is smooth.

  Away from full conditioning the mean is only piecewise smooth -- a query
  crossing between soundings swaps a neighbour -- so a difference taken across
  such a boundary would disagree with the analytic slope and neither would be
  wrong.
  """
  n = 30
  points, depth = survey(n, seed=93)
  fitted = fitted_at(points, depth, 0.3, m=n, n0=n)

  queries = np.random.default_rng(94).uniform(-20, 20, size=(5, 2))
  step = 1e-5

  analytic = fitted.mean_gradient(queries)
  numeric = np.empty_like(analytic)
  for axis in range(2):
    offset = np.zeros(2)
    offset[axis] = step
    numeric[:, axis] = (
      fitted.predict(queries + offset) - fitted.predict(queries - offset)
    ) / (2 * step)

  np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_the_gradient_recovers_a_plane_exactly():
  """A plane is entirely the linear mean, so the residual carries nothing."""
  rng = np.random.default_rng(95)
  points = rng.uniform(-30, 30, size=(400, 2))
  depth = -60.0 + 0.2 * points[:, 0] - 0.07 * points[:, 1]

  fitted = fit_vecchia(points, depth, m=20, steps=40, device="cpu")
  queries = rng.uniform(-25, 25, size=(8, 2))

  np.testing.assert_allclose(
    fitted.mean_gradient(queries),
    np.tile([0.2, -0.07], (8, 1)),
    atol=1e-8,
  )


def test_the_gradient_is_in_metres_per_metre():
  """Double the slope of the surface and the gradient must double.

  The check that catches a missing or spurious scale factor, which no shape or
  finiteness test would see. There is no input standardisation here to get
  wrong, which is part of why the map fits in physical units.
  """
  rng = np.random.default_rng(96)
  points = rng.uniform(-30, 30, size=(500, 2))
  probe = rng.uniform(-20, 20, size=(6, 2))

  def slope_of(gain):
    depth = -60.0 + gain * np.sin(points[:, 0] / 8.0)
    fitted = fit_vecchia(points, depth, m=20, steps=200, device="cpu")
    return fitted.mean_gradient(probe)[:, 0]

  np.testing.assert_allclose(slope_of(2.0) / slope_of(1.0), 2.0, rtol=0.05)


def test_the_gradient_follows_a_known_ridge():
  """Sign and magnitude against a surface whose slope is known in closed form."""
  rng = np.random.default_rng(97)
  points = rng.uniform(-30, 30, size=(1200, 2))
  depth = -60.0 + 3.0 * np.sin(points[:, 0] / 7.0)

  fitted = fit_vecchia(points, depth, m=25, steps=300, device="cpu")

  probe = np.array([[0.0, 0.0], [11.0, 5.0], [-11.0, -5.0]])
  expected = (3.0 / 7.0) * np.cos(probe[:, 0] / 7.0)

  gradient = fitted.mean_gradient(probe)
  np.testing.assert_allclose(gradient[:, 0], expected, atol=0.08)
  np.testing.assert_allclose(gradient[:, 1], np.zeros(3), atol=0.08)


def test_the_gradient_can_be_taken_twice():
  """Trivially true without autograd, and worth pinning that it stays so.

  The SVGP needed a cache clear between calls or the second one raised. This
  implementation is analytic and holds no graph, so the property is free --
  but it is the kind of thing a later rewrite could quietly lose.
  """
  points, depth = survey(200, seed=98)
  fitted = fitted_at(points, depth, 0.3, m=20)
  queries = np.random.default_rng(99).uniform(-20, 20, size=(5, 2))

  np.testing.assert_array_equal(
    fitted.mean_gradient(queries), fitted.mean_gradient(queries)
  )


def test_the_gradient_is_chunk_invariant():
  points, depth = survey(300, seed=100)
  fitted = fitted_at(points, depth, 0.3, m=20)
  queries = np.random.default_rng(101).uniform(-25, 25, size=(30, 2))

  reference = fitted.mean_gradient(queries, chunk_size=5000)
  for chunk in (1, 7, 29):
    np.testing.assert_allclose(
      fitted.mean_gradient(queries, chunk_size=chunk), reference, rtol=1e-12
    )


def test_the_gradient_accepts_a_single_point():
  points, depth = survey(200, seed=102)
  fitted = fitted_at(points, depth, 0.3, m=20)
  assert fitted.mean_gradient([[1.0, 2.0]]).shape == (1, 2)


def test_the_gradient_rejects_points_of_the_wrong_width():
  points, depth = survey(200, seed=103)
  fitted = fitted_at(points, depth, 0.3, m=20)
  try:
    fitted.mean_gradient(np.zeros((4, 3)))
  except ValueError as error:
    assert "(n, 2)" in str(error)
    return
  raise AssertionError("expected a ValueError")


def test_the_survey_tree_is_built_once_and_kept():
  """``navigate.py`` calls the map every ping; a rebuilt tree would dominate."""
  points, depth = survey(300, seed=104)
  fitted = fitted_at(points, depth, 0.3, m=20)

  first = fitted.tree
  fitted.predict([[0.0, 0.0]])
  fitted.mean_gradient([[0.0, 0.0]])
  assert fitted.tree is first


# -- the explicit sparse factor, and drawing from it -------------------------


def test_the_factor_applies_like_whiten():
  """The pin on ``sparse_factor``.

  ``whiten`` applies ``U^T`` without ever forming ``U``, by a different route
  -- a triangular solve against each block. If the assembled matrix is right,
  multiplying by it must give the same answer. Two independent derivations of
  the same operator, which is the only reason either is trustworthy.
  """
  points, residual = survey(400, 60)
  structure = build_structure(points, m=12, n0=20)

  noise = torch.tensor(0.3, dtype=torch.float64)
  values = torch.tensor(residual[structure.order], dtype=torch.float64)

  expected, _ = whiten(structure, values, LOG_AMPLITUDE, LOG_LENGTHSCALE, noise)
  factor = sparse_factor(structure, LOG_AMPLITUDE, LOG_LENGTHSCALE, noise)

  np.testing.assert_allclose(
    factor.T @ values.numpy(), expected.numpy(), rtol=1e-9, atol=1e-9
  )


def test_the_factor_is_upper_triangular():
  """The Vecchia condition, read off the matrix itself."""
  points, _ = survey(300, 61)
  structure = build_structure(points, m=10, n0=16)

  factor = sparse_factor(
    structure, LOG_AMPLITUDE, LOG_LENGTHSCALE, torch.tensor(0.2)
  ).tocoo()

  assert np.all(factor.row <= factor.col)
  assert np.all(np.asarray(factor.tocsc().diagonal()) > 0.0)


def test_the_factor_inverts_the_kernel_when_it_conditions_on_everything():
  """``U U^T = K^-1`` exactly, once nothing is approximated away."""
  n = 45
  points, _ = survey(n, 62, spread=12.0)
  structure = build_structure(points, m=n - 1, n0=n - 1)

  noise = 0.4
  factor = sparse_factor(
    structure,
    LOG_AMPLITUDE,
    LOG_LENGTHSCALE,
    torch.tensor(noise, dtype=torch.float64),
    jitter=0.0,
  )

  ordered = torch.tensor(structure.points, dtype=torch.float64)
  kernel = matern52(ordered, ordered, LOG_AMPLITUDE, LOG_LENGTHSCALE).numpy()
  kernel = kernel + noise * np.eye(n)

  dense = factor.toarray()
  np.testing.assert_allclose(
    dense @ dense.T, np.linalg.inv(kernel), rtol=1e-7, atol=1e-9
  )


def test_draws_have_the_covariance_they_should():
  """Monte Carlo against the kernel the draw was asked for.

  This is what makes the generator usable as a reference: it validates the
  covariance of what comes out, not merely that something came out.
  """
  n = 60
  points, _ = survey(n, 63, spread=15.0)
  structure = build_structure(points, m=n - 1, n0=n - 1)

  noise = 0.25
  samples = draw(
    structure,
    LOG_AMPLITUDE,
    LOG_LENGTHSCALE,
    torch.tensor(noise, dtype=torch.float64),
    count=40000,
    seed=7,
    jitter=0.0,
  )

  ordered = torch.tensor(structure.points, dtype=torch.float64)
  expected = matern52(
    ordered, ordered, LOG_AMPLITUDE, LOG_LENGTHSCALE
  ).numpy() + noise * np.eye(n)

  # Back into the structure's order to compare against its own kernel.
  empirical = np.cov(samples[structure.order])

  # Monte Carlo error on a covariance entry is ~ sigma_ii sigma_jj / sqrt(S).
  scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
  np.testing.assert_allclose(empirical / scale, expected / scale, atol=0.05)


def test_a_draw_comes_back_in_the_callers_order():
  """The permutation, which is silent and wrong-looking when inverted."""
  points, _ = survey(200, 64)
  structure = build_structure(points, m=10, n0=16)

  values = draw(
    structure,
    LOG_AMPLITUDE,
    LOG_LENGTHSCALE,
    torch.tensor(0.2, dtype=torch.float64),
    count=1,
    seed=3,
  )
  assert values.shape == (200,)

  # Nearby points must have similar values; that is only true in the right
  # order. Compare the spread of differences between neighbours in the
  # caller's frame against the spread over arbitrary pairs.
  from scipy.spatial import KDTree

  _, partner = KDTree(points).query(points, k=2)
  close = np.abs(values - values[partner[:, 1]]).mean()
  rng = np.random.default_rng(0)
  far = np.abs(values - values[rng.permutation(len(values))]).mean()

  assert close < 0.5 * far


def test_the_draw_count_shapes_the_output():
  points, _ = survey(80, 65)
  structure = build_structure(points, m=8, n0=12)
  noise = torch.tensor(0.2, dtype=torch.float64)

  single = draw(structure, LOG_AMPLITUDE, LOG_LENGTHSCALE, noise, seed=1)
  many = draw(structure, LOG_AMPLITUDE, LOG_LENGTHSCALE, noise, count=5, seed=1)
  assert single.shape == (80,)
  assert many.shape == (80, 5)


# -- the mean basis ----------------------------------------------------------


def basis_points(n=400, seed=70, spread=50.0):
  return np.random.default_rng(seed).uniform(-spread, spread, size=(n, 2))


def test_the_default_basis_is_still_the_linear_one():
  """Nothing may change for a map that did not ask for a richer mean."""
  points = basis_points(20)
  np.testing.assert_allclose(design_matrix(points), LINEAR_MEAN(points))
  np.testing.assert_allclose(
    design_matrix(points),
    np.column_stack([np.ones(len(points)), points]),
  )


def test_each_basis_has_the_size_it_claims():
  """``size`` must match what the basis actually produces, and be full rank.

  ``min_support=0`` keeps every spline function, so the sizes below are the
  complete grids. A pruned basis is smaller and carries an intercept instead;
  that is covered by its own test.
  """
  fitted = basis_points()
  for kind, extra, expected in (
    ("linear", {}, 3),
    ("quadratic", {}, 6),
    ("cubic", {}, 10),
    ("spline", {"knots": 8, "min_support": 0.0}, 100),
    ("spline", {"knots": 12, "min_support": 0.0}, 196),
  ):
    basis = MeanBasis.build(fitted, kind=kind, **extra)
    assert basis.size == expected

    # Comfortably more points than basis functions, or the rank check below
    # is bounded by the sample rather than by the basis.
    sample = max(2000, 8 * expected)
    values = basis(basis_points(sample, seed=71, spread=49.0))
    assert values.shape == (sample, basis.size)
    assert np.linalg.matrix_rank(values) == basis.size


def test_the_basis_gradient_matches_finite_differences():
  """The pin on ``gradient``. A wrong spline derivative is otherwise silent.

  It would not show up as an error -- only as a terrain update that pulls the
  vehicle slightly the wrong way, which is the hardest kind of bug to find
  downstream.
  """
  fitted = basis_points()
  queries = basis_points(40, seed=72, spread=40.0)
  step = 1e-5

  for kind, extra in (
    ("linear", {}),
    ("quadratic", {}),
    ("cubic", {}),
    ("spline", {"knots": 8}),
  ):
    basis = MeanBasis.build(fitted, kind=kind, **extra)
    analytic = basis.gradient(queries)

    numeric = np.stack(
      [
        (basis(queries + step * axis) - basis(queries - step * axis))
        / (2.0 * step)
        for axis in np.eye(2)
      ],
      axis=-1,
    )
    np.testing.assert_allclose(analytic, numeric, atol=1e-6)


def test_the_linear_gradient_reduces_to_the_slope_coefficients():
  """The identity the map used to hardcode, now a consequence rather than an
  assumption."""
  basis = MeanBasis.build(basis_points(), kind="linear")
  beta = np.array([3.0, -0.7, 0.4])

  slope = np.einsum("npd,p->nd", basis.gradient(basis_points(6, seed=73)), beta)
  np.testing.assert_allclose(slope, np.broadcast_to(beta[1:], (6, 2)))


def test_a_spline_basis_goes_flat_outside_its_extent():
  """A polynomial mean diverges where the survey stops; this must not."""
  fitted = np.array([[0.0, 0.0], [100.0, 100.0]])
  basis = MeanBasis.build(fitted, kind="spline", knots=8, min_support=0.0)

  edge = basis(np.array([[100.0, 100.0]]))
  beyond = basis(np.array([[5000.0, 5000.0]]))
  np.testing.assert_allclose(beyond, edge)

  assert np.all(basis.gradient(np.array([[5000.0, 5000.0]])) == 0.0)


def test_a_complete_spline_basis_is_a_partition_of_unity():
  """B-splines sum to one, so a constant depth is representable without help."""
  basis = MeanBasis.build(
    basis_points(4000), kind="spline", knots=10, min_support=0.0
  )
  assert not basis.intercept

  values = basis(basis_points(80, seed=74, spread=40.0))
  np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-10)


def test_a_pruned_spline_basis_carries_an_intercept_instead():
  """The fix for what pruning breaks, and why it is conditional.

  Dropping unsupported functions destroys the partition of unity, so the mean
  would decay toward *zero* away from the survey -- 0 m against a seabed at
  -65 m on the real data. An intercept restores a constant. But adding one to a
  *complete* basis duplicates the sum of the others exactly and leaves the
  design rank deficient, so it is added only when something was dropped.
  """
  rng = np.random.default_rng(76)
  # A diamond, like four survey headings: its bounding box has empty corners.
  points = rng.uniform(-50.0, 50.0, size=(6000, 2))
  points = points[np.abs(points).sum(axis=1) < 50.0]

  basis = MeanBasis.build(points, kind="spline", knots=10, min_support=5.0)
  assert basis.intercept
  assert basis.size < (10 + 3 - 1) ** 2

  values = basis(points)
  assert values.shape[1] == basis.size
  np.testing.assert_allclose(values[:, 0], 1.0)
  assert np.linalg.matrix_rank(values) == basis.size

  # Far outside the surveyed diamond the mean must revert to the intercept,
  # not decay to zero.
  depth = -65.0 + 0.5 * points[:, 0] / 50.0
  beta, *_ = np.linalg.lstsq(values, depth, rcond=None)
  corner = basis(np.array([[49.0, 49.0]])) @ beta
  assert -80.0 < corner[0] < -50.0


def test_the_basis_gradient_is_one_sided_on_the_clamp_boundary():
  """At the extent the basis has a kink, and the interior slope is the answer.

  A central difference straddles the clamp and returns half of it, so this is
  asserted against the interior derivative directly rather than numerically.
  """
  fitted = basis_points(2000, spread=50.0)
  basis = MeanBasis.build(fitted, kind="spline", knots=8, min_support=0.0)

  upper = np.array(basis.upper)
  edge = basis.gradient(upper[None, :])
  just_inside = basis.gradient((upper - 1e-6)[None, :])

  np.testing.assert_allclose(edge, just_inside, atol=1e-6)
  assert np.abs(edge).max() > 0.0

  beyond = basis.gradient((upper + 10.0)[None, :])
  assert np.all(beyond == 0.0)


def test_it_rejects_an_unknown_basis():
  try:
    MeanBasis.build(basis_points(10), kind="fourier")
  except ValueError as error:
    assert "fourier" in str(error)
    return
  raise AssertionError("expected a ValueError")


def test_a_spline_mean_fits_and_predicts():
  """End to end, because the basis has to survive the whole pipeline."""
  rng = np.random.default_rng(75)
  points = rng.uniform(-30.0, 30.0, size=(600, 2))
  depth = -60.0 + 0.02 * points[:, 0] + 3.0 * np.sin(points[:, 1] / 9.0)

  fitted = fit_vecchia(
    points, depth, m=12, steps=25, mean="spline", mean_knots=5, device="cpu"
  )
  assert fitted.basis.kind == "spline"
  assert fitted.beta.shape == (fitted.basis.size,)

  queries = rng.uniform(-25.0, 25.0, size=(40, 2))
  predicted = fitted.predict(queries)
  assert predicted.shape == (40,)
  assert np.all(np.isfinite(predicted))

  gradient = fitted.mean_gradient(queries)
  assert gradient.shape == (40, 2)
  assert np.all(np.isfinite(gradient))


# -- noisy inputs (NIGP) ----------------------------------------------------


def misplaced_survey(seed=30, n=700):
  """A known field sampled at true positions, recorded at jittered ones.

  Each sounding's position covariance grows with a stand-in for its offset
  across the swath, as the heading term makes it.
  """
  rng = np.random.default_rng(seed)
  truth = rng.uniform(-40, 40, size=(n, 2))
  depth = gp_draw(
    truth,
    torch.tensor(math.log(9.0), dtype=torch.float64),
    torch.log(torch.tensor([6.0, 6.0], dtype=torch.float64)),
    0.04,
    seed=seed + 1,
  )
  spread = rng.uniform(0.05, 1.2, size=n)
  cov = spread[:, None, None] ** 2 * np.eye(2)
  recorded = truth + spread[:, None] * rng.normal(size=(n, 2))
  return recorded, depth, cov


def test_nigp_gives_the_position_error_back_to_the_positions():
  """Plain, the nugget absorbs the misplacement; with NIGP it need not."""
  points, depth, cov = misplaced_survey()
  kwargs = {"m": 20, "steps": 300, "device": "cpu"}

  plain = fit_vecchia(points, depth, **kwargs)
  nigp, _ = fit_vecchia_nigp(points, depth, cov, passes=2, **kwargs)

  assert nigp.hyper.noise < 0.5 * plain.hyper.noise, (
    nigp.hyper.noise,
    plain.hyper.noise,
  )


def test_the_nigp_inflation_settles_across_passes():
  points, depth, cov = misplaced_survey(seed=32)
  _, inflations = fit_vecchia_nigp(
    points, depth, cov, passes=3, m=20, steps=300, device="cpu"
  )

  first = np.abs(inflations[1] - inflations[0]).max()
  second = np.abs(inflations[2] - inflations[1]).max()
  assert second < first, (first, second)


def test_nigp_refuses_a_covariance_of_the_wrong_shape():
  points, depth, cov = misplaced_survey(n=40)
  with pytest.raises(ValueError, match="position covariance"):
    fit_vecchia_nigp(points, depth, cov[:-1], m=5, steps=1, device="cpu")
