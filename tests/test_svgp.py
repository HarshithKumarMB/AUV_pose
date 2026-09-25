"""SVGP bathymetry map. Fits are tiny: wiring and shapes, not map quality."""

import numpy as np
import pytest
import torch

from auv_pose.estimation.terrain import DepthMap
from auv_pose.mapping.svgp import BathymetryMap, fit_svgp


@pytest.fixture
def synthetic():
  """A gentle sloping seabed."""
  rng = np.random.default_rng(0)
  X = rng.uniform(-20, 20, size=(200, 2))
  return X, -60.0 + 0.1 * X[:, 0] - 0.05 * X[:, 1]


def inducing(bathymetry):
  return bathymetry.model.variational_strategy.inducing_points.detach()


def test_fit_returns_a_trained_map(synthetic):
  bathymetry = fit_svgp(*synthetic, n_inducing=20, epochs=2, seed=0)
  assert isinstance(bathymetry, BathymetryMap)
  assert inducing(bathymetry).shape == (20, 2)
  assert all(torch.isfinite(p).all() for p in bathymetry.model.parameters())


def test_fit_is_reproducible_given_a_seed(synthetic):
  a = fit_svgp(*synthetic, n_inducing=10, epochs=1, seed=7)
  b = fit_svgp(*synthetic, n_inducing=10, epochs=1, seed=7)
  torch.testing.assert_close(inducing(a), inducing(b))


def test_fit_rejects_more_inducing_points_than_data(synthetic):
  with pytest.raises(ValueError, match="exceeds"):
    fit_svgp(*synthetic, n_inducing=10_000, epochs=1)


def test_the_map_does_not_depend_on_where_the_survey_sits(synthetic):
  """Shifting the survey by kilometres shifts the map, nothing more."""
  X, y = synthetic
  shift = np.array([5000.0, -3000.0])
  here = fit_svgp(X, y, n_inducing=20, epochs=5, seed=0)
  there = fit_svgp(X + shift, y, n_inducing=20, epochs=5, seed=0)
  probe = np.array([[0.0, 0.0], [7.0, -3.0]])
  np.testing.assert_allclose(
    here.predict(probe), there.predict(probe + shift), atol=1e-3
  )


def _fitted_map(synthetic, epochs=15):
  return fit_svgp(*synthetic, n_inducing=32, epochs=epochs, seed=0)


def test_predict_shape_and_finiteness(synthetic):
  bathymetry = _fitted_map(synthetic, epochs=2)
  depth = bathymetry.predict(np.array([[0.0, 0.0], [5.0, -5.0], [10.0, 10.0]]))

  assert depth.shape == (3,)
  assert np.all(np.isfinite(depth))


def test_predict_accepts_a_single_point(synthetic):
  bathymetry = _fitted_map(synthetic, epochs=2)
  assert bathymetry.predict([[0.0, 0.0]]).shape == (1,)


def test_predict_rejects_wrong_dimensionality(synthetic):
  bathymetry = _fitted_map(synthetic, epochs=1)
  with pytest.raises(ValueError, match=r"\(n, 2\)"):
    bathymetry.predict(np.zeros((4, 3)))


def test_chunking_does_not_change_the_answer(synthetic):
  bathymetry = _fitted_map(synthetic, epochs=2)
  points = np.random.default_rng(1).uniform(-20, 20, size=(50, 2))

  np.testing.assert_allclose(
    bathymetry.predict(points, chunk_size=1000),
    bathymetry.predict(points, chunk_size=7),
    rtol=1e-5,
    atol=1e-5,
  )


def test_predict_recovers_the_synthetic_slope(synthetic):
  bathymetry = _fitted_map(synthetic, epochs=40)
  points = np.array([[0.0, 0.0], [10.0, 0.0], [-10.0, 5.0]])
  expected = -60.0 + 0.1 * points[:, 0] - 0.05 * points[:, 1]

  np.testing.assert_allclose(bathymetry.predict(points), expected, atol=2.0)


def test_with_std_returns_positive_uncertainty(synthetic):
  bathymetry = _fitted_map(synthetic, epochs=2)
  depth, std = bathymetry.predict(np.zeros((3, 2)), with_std=True)

  assert depth.shape == std.shape == (3,)
  assert np.all(std > 0)


def _fit_surface(surface, n=1500, epochs=300, n_inducing=120, seed=0):
  """Fit an analytic surface; return the map, inputs, truth and model."""
  rng = np.random.default_rng(seed)
  X = rng.uniform(-20, 20, size=(n, 2))
  y = surface(X)
  bathymetry = fit_svgp(
    X, y, n_inducing=n_inducing, epochs=epochs, batch_size=512, seed=seed
  )
  return bathymetry, X, y, bathymetry.model


def test_recovers_a_known_curved_surface():
  """Curvature, unlike a plane, exercises the kernel."""

  def surface(X):
    return -60.0 + 2.0 * np.sin(X[:, 0] / 6.0) + 1.5 * np.cos(X[:, 1] / 5.0)

  bathymetry, _, y, _ = _fit_surface(surface)

  probe = np.random.default_rng(1).uniform(-18, 18, size=(300, 2))
  error = bathymetry.predict(probe) - surface(probe)
  rmse = float(np.sqrt((error**2).mean()))

  assert rmse < 0.4 * y.std(), f"rmse {rmse:.3f} against relief {y.std():.3f}"


def test_lengthscales_are_learned_per_axis():
  """ARD: a surface flat in y gets the longer y lengthscale."""

  def ridges(X):
    return -60.0 + 2.0 * np.sin(X[:, 0] / 2.0)  # varies in x, flat in y

  _, _, _, model = _fit_surface(ridges, epochs=300)
  lengthscale = model.covar_module.base_kernel.lengthscale.detach().numpy()

  assert lengthscale.size == 2, "kernel is not ARD"
  x_scale, y_scale = lengthscale.ravel()
  assert y_scale > 2.0 * x_scale, (
    f"expected a longer lengthscale along the flat axis, got {lengthscale}"
  )


def test_with_std_is_the_map_not_the_sounding(synthetic):
  """By default the std excludes the likelihood's noise."""
  bathymetry = _fitted_map(synthetic, epochs=5)
  points = np.array([[0.0, 0.0], [5.0, -5.0]])

  _, latent = bathymetry.predict(points, with_std=True)
  _, noisy = bathymetry.predict(points, with_std=True, observation_noise=True)

  assert np.all(latent < noisy)
  assert np.all(latent > 0)


def test_uncertainty_grows_away_from_the_data(synthetic):
  bathymetry = _fitted_map(synthetic, epochs=30)

  _, near = bathymetry.predict(np.array([[0.0, 0.0]]), with_std=True)
  _, far = bathymetry.predict(np.array([[400.0, 400.0]]), with_std=True)

  assert far[0] > near[0]


def test_fit_records_its_elbo_trace(synthetic):
  bathymetry = fit_svgp(*synthetic, n_inducing=20, epochs=8, seed=0)
  assert len(bathymetry.elbo_trace) == 8
  assert bathymetry.elbo_trace[-1] < bathymetry.elbo_trace[0]


# -- the DepthMap contract --------------------------------------------------


def test_it_satisfies_the_depth_map_contract(synthetic):
  assert isinstance(_fitted_map(synthetic, epochs=2), DepthMap)


def test_the_joint_diagonal_is_the_marginal_variance(synthetic):
  bathymetry = _fitted_map(synthetic, epochs=5)
  points = np.array([[0.0, 0.0], [5.0, -5.0], [-8.0, 3.0], [12.0, 9.0]])

  _, stds = bathymetry.predict(points, with_std=True, observation_noise=True)
  _, cov = bathymetry.predict_joint(points)

  np.testing.assert_allclose(np.diag(cov), stds**2, rtol=1e-4)


def test_the_joint_mean_matches_predict(synthetic):
  bathymetry = _fitted_map(synthetic, epochs=5)
  points = np.array([[0.0, 0.0], [5.0, -5.0], [-8.0, 3.0]])

  mean, _ = bathymetry.predict_joint(points)
  np.testing.assert_allclose(mean, bathymetry.predict(points), rtol=1e-4)


def test_the_joint_covariance_is_symmetric_and_positive_definite(synthetic):
  bathymetry = _fitted_map(synthetic, epochs=5)
  points = np.random.default_rng(0).uniform(-15, 15, size=(12, 2))

  _, cov = bathymetry.predict_joint(points)
  np.testing.assert_allclose(cov, cov.T, atol=1e-12)
  assert np.min(np.linalg.eigvalsh(cov)) > 0.0


def test_nearby_soundings_covary_and_distant_ones_do_not(synthetic):
  bathymetry = _fitted_map(synthetic, epochs=20)
  points = np.array([[0.0, 0.0], [0.5, 0.0], [18.0, 18.0]])

  _, cov = bathymetry.predict_joint(points, observation_noise=False)

  assert cov[0, 1] > 0.0
  assert abs(cov[0, 2]) < abs(cov[0, 1])


def test_the_joint_takes_a_whole_sigma_cloud_in_one_call(synthetic):
  bathymetry = _fitted_map(synthetic, epochs=2)
  cloud = np.random.default_rng(1).uniform(-10, 10, size=(7, 5, 2))

  mean, cov = bathymetry.predict_joint(cloud)
  assert mean.shape == (7, 5)
  assert cov.shape == (7, 5, 5)


def test_batched_and_separate_calls_agree(synthetic):
  bathymetry = _fitted_map(synthetic, epochs=5)
  cloud = np.random.default_rng(2).uniform(-10, 10, size=(3, 4, 2))

  mean, cov = bathymetry.predict_joint(cloud)
  for i in range(3):
    one_mean, one_cov = bathymetry.predict_joint(cloud[i])
    np.testing.assert_allclose(mean[i], one_mean, rtol=1e-5)
    np.testing.assert_allclose(cov[i], one_cov, rtol=1e-4, atol=1e-8)


def test_observation_noise_only_ever_widens_the_covariance(synthetic):
  bathymetry = _fitted_map(synthetic, epochs=5)
  points = np.random.default_rng(3).uniform(-15, 15, size=(6, 2))

  _, latent = bathymetry.predict_joint(points, observation_noise=False)
  _, with_noise = bathymetry.predict_joint(points, observation_noise=True)

  assert np.all(np.diag(with_noise) > np.diag(latent))
  # Independent noise lands on the diagonal only.
  off = ~np.eye(len(points), dtype=bool)
  np.testing.assert_allclose(with_noise[off], latent[off], atol=1e-6)


def test_the_joint_rejects_points_of_the_wrong_width(synthetic):
  bathymetry = _fitted_map(synthetic, epochs=2)
  with pytest.raises(ValueError, match=r"\(\.\.\., b, 2\)"):
    bathymetry.predict_joint(np.zeros((4, 3)))


# -- the mean's gradient ----------------------------------------------------


def test_the_gradient_recovers_the_synthetic_slope(synthetic):
  bathymetry = _fitted_map(synthetic, epochs=300)
  points = np.array([[0.0, 0.0], [4.0, -4.0], [-6.0, 2.0]])

  gradient = bathymetry.mean_gradient(points)
  assert gradient.shape == (3, 2)
  np.testing.assert_allclose(gradient, np.tile([0.1, -0.05], (3, 1)), atol=0.02)


def test_the_gradient_matches_central_differences(synthetic):
  bathymetry = _fitted_map(synthetic, epochs=20)
  points = np.array([[1.0, 2.0], [-7.0, 5.0], [11.0, -3.0]])
  step = 1e-2

  analytic = bathymetry.mean_gradient(points)
  numeric = np.empty_like(analytic)
  for axis in range(2):
    offset = np.zeros(2)
    offset[axis] = step
    numeric[:, axis] = (
      bathymetry.predict(points + offset) - bathymetry.predict(points - offset)
    ) / (2 * step)

  np.testing.assert_allclose(analytic, numeric, atol=2e-3)


def test_the_gradient_can_be_taken_twice(synthetic):
  """Regression: the variational strategy's memo must be cleared."""
  bathymetry = _fitted_map(synthetic, epochs=2)
  points = np.array([[0.0, 0.0], [3.0, 3.0]])

  first = bathymetry.mean_gradient(points)
  second = bathymetry.mean_gradient(points)
  np.testing.assert_allclose(first, second, rtol=1e-6)


def test_the_gradient_is_in_metres_per_metre_not_standardised_units():
  """Doubling the slope doubles the gradient; catches a missing ``x_scale``."""
  narrow, _, _, _ = _fit_surface(
    lambda X: -60.0 + 0.2 * X[:, 0], n=800, epochs=200, n_inducing=60
  )
  wide, _, _, _ = _fit_surface(
    lambda X: -60.0 + 0.1 * X[:, 0], n=800, epochs=200, n_inducing=60
  )

  probe = np.array([[0.0, 0.0], [5.0, 5.0], [-5.0, -5.0]])
  ratio = (
    narrow.mean_gradient(probe)[:, 0].mean()
    / wide.mean_gradient(probe)[:, 0].mean()
  )
  np.testing.assert_allclose(ratio, 2.0, rtol=0.1)


def test_the_gradient_rejects_points_of_the_wrong_width(synthetic):
  bathymetry = _fitted_map(synthetic, epochs=2)
  with pytest.raises(ValueError, match=r"\(n, 2\)"):
    bathymetry.mean_gradient(np.zeros((4, 3)))
