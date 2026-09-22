"""SVGP bathymetry surrogate.

Fits are deliberately tiny -- these check wiring and shapes, not map quality.
"""

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from auv_pose.estimation.terrain import DepthMap
from auv_pose.io.checkpoints import load_map, save_map
from auv_pose.mapping.svgp import BathymetryMap, SVGPModel, fit_svgp


@pytest.fixture
def synthetic():
  """A gentle sloping seabed, standardised the way train_map.py does it."""
  rng = np.random.default_rng(0)
  X = rng.uniform(-20, 20, size=(200, 2)).astype(np.float32)
  y = (-60.0 + 0.1 * X[:, 0] - 0.05 * X[:, 1]).astype(np.float32)

  x_scaler = StandardScaler().fit(X)
  y_mean, y_std = float(y.mean()), float(y.std())

  return {
    "X": X,
    "y": y,
    "x_scaler": x_scaler,
    "y_mean": y_mean,
    "y_std": y_std,
    "train_x": torch.tensor(x_scaler.transform(X), dtype=torch.float32),
    "train_y": torch.tensor((y - y_mean) / y_std, dtype=torch.float32),
  }


def test_fit_returns_a_trained_model(synthetic):
  model, likelihood, inducing = fit_svgp(
    synthetic["train_x"], synthetic["train_y"], n_inducing=20, epochs=2, seed=0
  )
  assert isinstance(model, SVGPModel)
  assert inducing.shape == (20, 2)
  assert all(torch.isfinite(p).all() for p in model.parameters())
  assert all(torch.isfinite(p).all() for p in likelihood.parameters())


def test_fit_is_reproducible_given_a_seed(synthetic):
  a = fit_svgp(
    synthetic["train_x"], synthetic["train_y"], n_inducing=10, epochs=1, seed=7
  )[2]
  b = fit_svgp(
    synthetic["train_x"], synthetic["train_y"], n_inducing=10, epochs=1, seed=7
  )[2]
  torch.testing.assert_close(a, b)


def test_fit_rejects_more_inducing_points_than_data(synthetic):
  with pytest.raises(ValueError, match="exceeds"):
    fit_svgp(
      synthetic["train_x"], synthetic["train_y"], n_inducing=10_000, epochs=1
    )


def _fitted_map(synthetic, epochs=15):
  model, likelihood, _ = fit_svgp(
    synthetic["train_x"],
    synthetic["train_y"],
    n_inducing=32,
    epochs=epochs,
    seed=0,
  )
  return BathymetryMap(
    model,
    likelihood,
    synthetic["x_scaler"],
    synthetic["y_mean"],
    synthetic["y_std"],
  )


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
  """A short fit on a plane should land near the true depth."""
  bathymetry = _fitted_map(synthetic, epochs=40)
  points = np.array([[0.0, 0.0], [10.0, 0.0], [-10.0, 5.0]])
  expected = -60.0 + 0.1 * points[:, 0] - 0.05 * points[:, 1]

  np.testing.assert_allclose(bathymetry.predict(points), expected, atol=2.0)


def test_with_std_returns_positive_uncertainty(synthetic):
  bathymetry = _fitted_map(synthetic, epochs=2)
  depth, std = bathymetry.predict(np.zeros((3, 2)), with_std=True)

  assert depth.shape == std.shape == (3,)
  assert np.all(std > 0)


def test_checkpoint_roundtrip(tmp_path, synthetic):
  model, likelihood, inducing = fit_svgp(
    synthetic["train_x"], synthetic["train_y"], n_inducing=16, epochs=3, seed=0
  )
  path = tmp_path / "map.pkl"
  save_map(
    path,
    model,
    likelihood,
    inducing,
    synthetic["x_scaler"],
    synthetic["y_mean"],
    synthetic["y_std"],
  )

  original = BathymetryMap(
    model,
    likelihood,
    synthetic["x_scaler"],
    synthetic["y_mean"],
    synthetic["y_std"],
  )
  restored = load_map(path)

  points = np.array([[0.0, 0.0], [3.0, -2.0]])
  np.testing.assert_allclose(
    original.predict(points), restored.predict(points), rtol=1e-5, atol=1e-5
  )


def test_load_rejects_a_foreign_pickle(tmp_path):
  import pickle

  path = tmp_path / "not-a-map.pkl"
  with open(path, "wb") as handle:
    pickle.dump({"something": "else"}, handle)

  with pytest.raises(ValueError, match="not a bathymetry checkpoint"):
    load_map(path)


def _fit_surface(surface, n=1500, epochs=300, n_inducing=120, seed=0):
  """Fit a known analytic surface and return the map, its inputs and truth."""
  rng = np.random.default_rng(seed)
  X = rng.uniform(-20, 20, size=(n, 2)).astype(np.float32)
  y = surface(X).astype(np.float32)

  x_scaler = StandardScaler().fit(X)
  y_mean, y_std = float(y.mean()), float(y.std())
  model, likelihood, _ = fit_svgp(
    torch.tensor(x_scaler.transform(X), dtype=torch.float32),
    torch.tensor((y - y_mean) / y_std, dtype=torch.float32),
    n_inducing=n_inducing,
    epochs=epochs,
    batch_size=512,
    seed=seed,
  )
  return BathymetryMap(model, likelihood, x_scaler, y_mean, y_std), X, y, model


def test_recovers_a_known_curved_surface():
  """The check that a plane cannot make: does it fit *structure*?

  ``test_predict_recovers_the_synthetic_slope`` fits a plane, which a constant
  mean plus almost any kernel gets right, so it passes whatever the covariance
  is doing. Curvature is what actually exercises the kernel -- and a map that
  cannot beat the mean of a few nearby soundings is not earning its complexity,
  which is the failure this exists to catch.
  """

  def surface(X):
    return -60.0 + 2.0 * np.sin(X[:, 0] / 6.0) + 1.5 * np.cos(X[:, 1] / 5.0)

  bathymetry, _, y, _ = _fit_surface(surface)

  probe = np.random.default_rng(1).uniform(-18, 18, size=(300, 2))
  error = bathymetry.predict(probe) - surface(probe)
  rmse = float(np.sqrt((error**2).mean()))

  assert rmse < 0.4 * y.std(), f"rmse {rmse:.3f} against relief {y.std():.3f}"


def test_lengthscales_are_learned_per_axis():
  """ARD: structure that varies fast in x and slowly in y must be seen as such.

  Without ``ard_num_dims`` there is a single lengthscale shared by both axes,
  and the only anisotropy the model can express is whatever ratio the input
  scaler happens to impose -- a fact about the survey's bounding box rather
  than about the seabed.
  """

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
  """Uncertainty about the seabed, not about a future sonar return.

  Adding the likelihood's noise is right when predicting a *sounding* and wrong
  when asking how well the seabed is known. It matters because a poorly fitted
  GP parks its misfit in that noise term, so the noisy figure comes back large
  and almost flat across the map -- which is worse than useless to a filter
  trying to decide how far to trust the map here rather than there.
  """
  bathymetry = _fitted_map(synthetic, epochs=5)
  points = np.array([[0.0, 0.0], [5.0, -5.0]])

  _, latent = bathymetry.predict(points, with_std=True)
  _, noisy = bathymetry.predict(points, with_std=True, observation_noise=True)

  assert np.all(latent < noisy)
  assert np.all(latent > 0)


def test_uncertainty_grows_away_from_the_data(synthetic):
  """The property that makes the std usable as a measurement variance."""
  bathymetry = _fitted_map(synthetic, epochs=30)

  _, near = bathymetry.predict(np.array([[0.0, 0.0]]), with_std=True)
  _, far = bathymetry.predict(np.array([[400.0, 400.0]]), with_std=True)

  assert far[0] > near[0]


def test_fit_records_its_elbo_trace(synthetic):
  """Convergence should be observable rather than assumed."""
  model, _, _ = fit_svgp(
    synthetic["train_x"], synthetic["train_y"], n_inducing=20, epochs=8, seed=0
  )
  assert len(model.elbo_trace) == 8
  assert model.elbo_trace[-1] < model.elbo_trace[0]


# -- the DepthMap contract --------------------------------------------------
#
# These cover what the smoother's update step needs, which ``predict`` alone
# does not provide: a joint covariance over a whole ping's soundings, and the
# slope of the map's mean.


def test_it_satisfies_the_depth_map_contract(synthetic):
  """Structural, not nominal -- nothing declares the Protocol as a base."""
  assert isinstance(_fitted_map(synthetic, epochs=2), DepthMap)


def test_the_joint_diagonal_is_the_marginal_variance(synthetic):
  """The two paths through gpytorch must not disagree about one point."""
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
  """The off-diagonals are the whole reason this method exists.

  Without them the update counts a fan of adjacent beams as that many
  independent constraints on the pose. The kernel is RBF, so covariance must
  fall away with separation.
  """
  bathymetry = _fitted_map(synthetic, epochs=20)
  points = np.array([[0.0, 0.0], [0.5, 0.0], [18.0, 18.0]])

  _, cov = bathymetry.predict_joint(points, observation_noise=False)

  assert cov[0, 1] > 0.0
  assert abs(cov[0, 2]) < abs(cov[0, 1])


def test_the_joint_takes_a_whole_sigma_cloud_in_one_call(synthetic):
  """``(31, 32, 2)`` in, ``(31, 32)`` and ``(31, 32, 32)`` out.

  One forward pass per ping rather than one per sigma point is the difference
  between the update costing milliseconds and costing a second.
  """
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
  # The likelihood's noise is independent per sounding, so it lands on the
  # diagonal and leaves the correlations alone.
  off = ~np.eye(len(points), dtype=bool)
  np.testing.assert_allclose(with_noise[off], latent[off], atol=1e-6)


def test_the_joint_rejects_points_of_the_wrong_width(synthetic):
  bathymetry = _fitted_map(synthetic, epochs=2)
  with pytest.raises(ValueError, match=r"\(\.\.\., b, 2\)"):
    bathymetry.predict_joint(np.zeros((4, 3)))


# -- the mean's gradient ----------------------------------------------------


def test_the_gradient_recovers_the_synthetic_slope(synthetic):
  """The fixture is ``-60 + 0.1 x - 0.05 y``, so the slope is known exactly."""
  bathymetry = _fitted_map(synthetic, epochs=300)
  points = np.array([[0.0, 0.0], [4.0, -4.0], [-6.0, 2.0]])

  gradient = bathymetry.mean_gradient(points)
  assert gradient.shape == (3, 2)
  np.testing.assert_allclose(gradient, np.tile([0.1, -0.05], (3, 1)), atol=0.02)


def test_the_gradient_matches_central_differences(synthetic):
  """Against the map itself, so it holds whatever the fit came out as."""
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
  """Regression: gpytorch memoises the variational strategy across calls.

  Without clearing that cache the second call raises *"Trying to backward
  through the graph a second time"*, from a traceback that points into torch's
  autograd engine and never mentions gpytorch. The smoother calls this once per
  ping, so it would have failed on the second ping of every run.
  """
  bathymetry = _fitted_map(synthetic, epochs=2)
  points = np.array([[0.0, 0.0], [3.0, 3.0]])

  first = bathymetry.mean_gradient(points)
  second = bathymetry.mean_gradient(points)
  np.testing.assert_allclose(first, second, rtol=1e-6)


def test_the_gradient_is_in_metres_per_metre_not_standardised_units():
  """Stretch the surface by two and the slope must halve.

  This is what catches a missing ``x_scaler.scale_``. Omitting it leaves a
  silent constant factor on every gradient -- and so on every range-noise
  variance the update computes -- that no shape or finiteness check would see.
  """
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
