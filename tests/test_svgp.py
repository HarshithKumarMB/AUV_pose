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
