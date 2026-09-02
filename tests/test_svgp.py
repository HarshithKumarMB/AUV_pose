"""SVGP bathymetry surrogate.

Fits are deliberately tiny -- these check wiring and shapes, not map quality.
"""

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

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
