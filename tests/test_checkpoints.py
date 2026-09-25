"""Saving and loading a fitted map."""

import numpy as np
import pytest

from auv_pose.io.checkpoints import VERSION, load_map, save_map
from auv_pose.mapping.svgp import BathymetryMap, fit_svgp
from auv_pose.mapping.vecchia import VecchiaMap, fit_vecchia


def survey(n=300, seed=0):
  rng = np.random.default_rng(seed)
  points = rng.uniform(-20.0, 20.0, size=(n, 2))
  return points, -60.0 + 0.1 * points[:, 0] + rng.normal(scale=0.3, size=n)


@pytest.fixture(scope="module")
def vecchia():
  return fit_vecchia(*survey(), m=20, steps=30, device="cpu")


@pytest.fixture(scope="module")
def svgp():
  return fit_svgp(*survey(), n_inducing=16, epochs=3, seed=0, device="cpu")


QUERIES = np.random.default_rng(1).uniform(-25.0, 25.0, size=(6, 2))


def round_trip(tmp_path, bathymetry):
  path = tmp_path / "map.npz"
  save_map(path, bathymetry)
  return load_map(path)


def test_a_vecchia_map_comes_back_bit_for_bit(tmp_path, vecchia):
  loaded = round_trip(tmp_path, vecchia)
  assert isinstance(loaded, VecchiaMap)
  assert loaded.hyper == vecchia.hyper
  assert loaded.basis == vecchia.basis
  np.testing.assert_array_equal(loaded.structure.order, vecchia.structure.order)
  np.testing.assert_array_equal(
    loaded.structure.neighbours, vecchia.structure.neighbours
  )
  for query in (
    lambda m: m.predict(QUERIES),
    lambda m: m.mean_gradient(QUERIES),
  ):
    np.testing.assert_array_equal(query(loaded), query(vecchia))
  np.testing.assert_array_equal(
    loaded.predict_joint(QUERIES)[1], vecchia.predict_joint(QUERIES)[1]
  )


def test_the_stored_indices_keep_their_integer_type(tmp_path, vecchia):
  loaded = round_trip(tmp_path, vecchia)
  assert isinstance(loaded, VecchiaMap)
  assert loaded.structure.neighbours.dtype == np.int64
  assert loaded.structure.order.dtype == np.int64


def test_an_svgp_map_comes_back_bit_for_bit(tmp_path, svgp):
  loaded = round_trip(tmp_path, svgp)
  assert isinstance(loaded, BathymetryMap)
  np.testing.assert_array_equal(loaded.predict(QUERIES), svgp.predict(QUERIES))
  np.testing.assert_array_equal(
    loaded.mean_gradient(QUERIES), svgp.mean_gradient(QUERIES)
  )


def test_the_file_holds_only_plain_arrays(tmp_path, vecchia, svgp):
  """Loadable with pickling off: nothing with a framework version inside."""
  for bathymetry in (vecchia, svgp):
    path = tmp_path / "map.npz"
    save_map(path, bathymetry)
    with np.load(path, allow_pickle=False) as stored:
      assert all(stored[key].dtype != object for key in stored.files)


def test_a_file_of_another_version_is_refused(tmp_path, vecchia):
  path = tmp_path / "map.npz"
  save_map(path, vecchia)
  with np.load(path) as stored:
    arrays = dict(stored)
  arrays["version"] = np.array(VERSION + 1)
  np.savez(path, **arrays)
  with pytest.raises(ValueError, match="not a version"):
    load_map(path)


@pytest.mark.parametrize(
  "arrays",
  [
    {"points": np.zeros((3, 2))},
    {"version": np.array(VERSION), "kind": np.array("rbf")},
  ],
  ids=["foreign", "unknown kind"],
)
def test_a_file_that_is_not_a_map_is_refused(tmp_path, arrays):
  path = tmp_path / "other.npz"
  np.savez(path, **arrays)
  with pytest.raises(ValueError):
    load_map(path)


def test_a_pickle_inside_the_file_is_never_executed(tmp_path):
  path = tmp_path / "evil.npz"
  np.savez(
    path, version=np.array(VERSION), kind=np.array({"a": 1}, dtype=object)
  )
  with pytest.raises(ValueError):
    load_map(path)
