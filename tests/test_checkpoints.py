"""Persisting a fitted map, in either format.

``load_map`` has to read two formats and refuse everything else, and the
refusals matter as much as the successes: this module has already been through
one silent format change whose diagnosis had to be reverse-engineered from a
state-dict shape mismatch. The tests below pin that a stale or foreign file
says what is wrong with it.
"""

import pickle

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from auv_pose.io.checkpoints import (
  VECCHIA_VERSION,
  load_map,
  save_map,
  save_vecchia_map,
)
from auv_pose.mapping.svgp import BathymetryMap, fit_svgp
from auv_pose.mapping.vecchia import VecchiaMap, fit_vecchia


@pytest.fixture
def vecchia():
  rng = np.random.default_rng(0)
  points = rng.uniform(-20.0, 20.0, size=(300, 2))
  depth = -60.0 + 0.1 * points[:, 0] + rng.normal(scale=0.3, size=300)
  return fit_vecchia(points, depth, m=20, steps=30, device="cpu")


@pytest.fixture
def queries():
  return np.random.default_rng(1).uniform(-15.0, 15.0, size=(6, 2))


# -- the Vecchia format -----------------------------------------------------


def test_a_vecchia_map_survives_a_round_trip_exactly(
  tmp_path, vecchia, queries
):
  """No float32 weights anywhere, so this is equality rather than closeness."""
  path = tmp_path / "vecchia.pkl"
  save_vecchia_map(path, vecchia)
  loaded = load_map(path)

  assert isinstance(loaded, VecchiaMap)
  np.testing.assert_array_equal(
    loaded.predict(queries), vecchia.predict(queries)
  )
  np.testing.assert_array_equal(
    loaded.mean_gradient(queries), vecchia.mean_gradient(queries)
  )

  mean, covariance = vecchia.predict_joint(queries)
  loaded_mean, loaded_covariance = loaded.predict_joint(queries)
  np.testing.assert_array_equal(loaded_mean, mean)
  np.testing.assert_array_equal(loaded_covariance, covariance)


def test_the_checkpoint_carries_everything_prediction_needs(
  tmp_path, vecchia, queries
):
  """Reconstructed from the file alone, with no refit and no recomputation.

  In particular ``information`` -- recomputing it would mean whitening the
  whole survey again, which is most of a fit.
  """
  path = tmp_path / "vecchia.pkl"
  save_vecchia_map(path, vecchia)

  loaded = load_map(path)
  assert isinstance(loaded, VecchiaMap)
  assert loaded.information is not None
  np.testing.assert_array_equal(loaded.information, vecchia.information)
  np.testing.assert_array_equal(loaded.beta, vecchia.beta)
  np.testing.assert_array_equal(loaded.noise, vecchia.noise)
  assert loaded.lengthscale.shape == (2,)


def test_the_ordering_is_stored_rather_than_rebuilt(tmp_path, vecchia):
  """A recomputed ordering that disagreed would be a miserable bug."""
  path = tmp_path / "vecchia.pkl"
  save_vecchia_map(path, vecchia)
  loaded = load_map(path)
  assert isinstance(loaded, VecchiaMap)

  np.testing.assert_array_equal(loaded.structure.order, vecchia.structure.order)
  np.testing.assert_array_equal(
    loaded.structure.neighbours, vecchia.structure.neighbours
  )
  assert loaded.structure.n0 == vecchia.structure.n0


def test_a_stale_version_is_refused_with_an_explanation(tmp_path, vecchia):
  path = tmp_path / "vecchia.pkl"
  save_vecchia_map(path, vecchia)

  with open(path, "rb") as handle:
    payload = pickle.load(handle)
  payload["version"] = VECCHIA_VERSION + 1
  with open(path, "wb") as handle:
    pickle.dump(payload, handle)

  with pytest.raises(ValueError, match="version"):
    load_map(path)


def test_a_truncated_vecchia_checkpoint_says_what_is_missing(tmp_path, vecchia):
  path = tmp_path / "vecchia.pkl"
  save_vecchia_map(path, vecchia)

  with open(path, "rb") as handle:
    payload = pickle.load(handle)
  del payload["beta"]
  with open(path, "wb") as handle:
    pickle.dump(payload, handle)

  with pytest.raises(ValueError, match="not a bathymetry checkpoint"):
    load_map(path)


def test_the_checkpoint_holds_no_framework_objects(tmp_path, vecchia):
  """No torch tensors, no sklearn scaler -- nothing with a version to break.

  The SVGP format cannot say this, and its docstring carries a warning about
  scikit-learn pickle portability as a result.
  """
  path = tmp_path / "vecchia.pkl"
  save_vecchia_map(path, vecchia)

  with open(path, "rb") as handle:
    payload = pickle.load(handle)

  for key, value in payload.items():
    assert not isinstance(value, torch.Tensor), key
    assert not isinstance(value, StandardScaler), key


# -- the SVGP format still loads --------------------------------------------


def svgp_checkpoint(tmp_path):
  rng = np.random.default_rng(2)
  points = rng.uniform(-20.0, 20.0, size=(200, 2)).astype(np.float32)
  depth = (-60.0 + 0.1 * points[:, 0]).astype(np.float32)

  scaler = StandardScaler().fit(points)
  mean, spread = float(depth.mean()), float(depth.std())

  model, likelihood, inducing = fit_svgp(
    torch.tensor(scaler.transform(points), dtype=torch.float32),
    torch.tensor((depth - mean) / spread, dtype=torch.float32),
    n_inducing=16,
    epochs=2,
    seed=0,
  )
  path = tmp_path / "svgp.pkl"
  save_map(path, model, likelihood, inducing, scaler, mean, spread)
  return path


def test_a_legacy_svgp_checkpoint_without_a_format_field_still_loads(tmp_path):
  """``save_map`` writes no ``format`` key, and existing files on disk have none.

  The fallback that recognises an SVGP by its ``model_state_dict`` is therefore
  permanent, not transitional.
  """
  path = svgp_checkpoint(tmp_path)

  with open(path, "rb") as handle:
    payload = pickle.load(handle)
  assert "format" not in payload

  assert isinstance(load_map(path), BathymetryMap)


def test_load_map_dispatches_on_the_format_field(tmp_path, vecchia):
  svgp = svgp_checkpoint(tmp_path)
  vecchia_path = tmp_path / "vecchia.pkl"
  save_vecchia_map(vecchia_path, vecchia)

  assert isinstance(load_map(svgp), BathymetryMap)
  assert isinstance(load_map(vecchia_path), VecchiaMap)


def test_both_formats_answer_the_same_query_interface(
  tmp_path, vecchia, queries
):
  """A caller querying depths need not know which map it was handed."""
  svgp = load_map(svgp_checkpoint(tmp_path))

  vecchia_path = tmp_path / "vecchia.pkl"
  save_vecchia_map(vecchia_path, vecchia)
  loaded = load_map(vecchia_path)

  for bathymetry in (svgp, loaded):
    assert bathymetry.predict(queries).shape == (6,)
    elevation, spread = bathymetry.predict(queries, with_std=True)
    assert elevation.shape == (6,) and spread.shape == (6,)
    assert np.all(spread > 0.0)


# -- refusals ---------------------------------------------------------------


def test_a_foreign_pickle_is_refused(tmp_path):
  path = tmp_path / "foreign.pkl"
  with open(path, "wb") as handle:
    pickle.dump({"something": "else"}, handle)

  with pytest.raises(ValueError, match="not a bathymetry checkpoint"):
    load_map(path)


def test_a_pickle_that_is_not_even_a_mapping_is_refused(tmp_path):
  path = tmp_path / "list.pkl"
  with open(path, "wb") as handle:
    pickle.dump([1, 2, 3], handle)

  with pytest.raises(ValueError, match="not a bathymetry checkpoint"):
    load_map(path)


def test_a_version_one_checkpoint_loads_as_a_linear_mean(tmp_path):
  """Backward compatibility, and it is not merely cosmetic.

  Version 1 predates the mean basis being stored. Every such map was fitted
  with the linear mean, so defaulting to it is correct rather than merely
  convenient -- and loading one as a spline would change every prediction it
  makes without erroring.
  """
  from auv_pose.mapping.vecchia import LINEAR_MEAN

  rng = np.random.default_rng(0)
  points = rng.uniform(-10.0, 10.0, size=(40, 2))
  fitted = fit_vecchia(
    points,
    -50.0 + 0.1 * points[:, 0],
    m=6,
    steps=5,
    device="cpu",
  )

  path = tmp_path / "old.pkl"
  save_vecchia_map(path, fitted)

  with open(path, "rb") as handle:
    payload = pickle.load(handle)
  payload["version"] = 1
  del payload["basis"]
  with open(path, "wb") as handle:
    pickle.dump(payload, handle)

  loaded = load_map(path)
  assert isinstance(loaded, VecchiaMap)
  assert loaded.basis == LINEAR_MEAN
  np.testing.assert_allclose(
    loaded.predict(points[:5]), fitted.predict(points[:5])
  )
