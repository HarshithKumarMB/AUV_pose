"""Registering survey passes: recover each pass's offset from the overlaps."""

import numpy as np
import pytest

from auv_pose.mapping.registration import grid_surface, register_passes


def seabed(xy):
  """Gentle slope, two mounds and a pipe-like ridge: relief that pins a shift."""
  x, y = xy[:, 0], xy[:, 1]
  return (
    -70.0
    + 0.02 * x
    + 3.0 * np.exp(-((x - 30) ** 2 + (y - 60) ** 2) / 120.0)
    + 2.0 * np.exp(-((x - 70) ** 2 + (y - 25) ** 2) / 60.0)
    + 4.0 * np.exp(-((y - 45) ** 2) / 8.0)
  )


def survey_pass(rng, offset, strip, surface=seabed, n=40_000):
  """Soundings in a strip, recorded ``offset`` away from where they struck."""
  x0, x1, y0, y1 = strip
  truth = np.column_stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)])
  depth = surface(truth) + rng.normal(scale=0.05, size=n)
  return truth + offset, depth


OFFSETS = np.array([[-1.1, -0.7], [0.0, 1.4], [0.0, 1.0], [1.8, -3.1]])
STRIPS = [
  (0, 100, 0, 70),
  (0, 100, 30, 100),
  (0, 70, 0, 100),
  (30, 100, 0, 100),
]
EXPECTED = -(OFFSETS - OFFSETS.mean(axis=0))


def survey(offsets=OFFSETS, surface=seabed, seed=0):
  rng = np.random.default_rng(seed)
  return [survey_pass(rng, o, s, surface) for o, s in zip(offsets, STRIPS)]


@pytest.fixture(scope="module")
def registered():
  passes = survey()
  return passes, register_passes(passes)


def test_it_recovers_each_pass_offset_relative_to_the_mean(registered):
  _, (shifts, biases) = registered
  np.testing.assert_allclose(shifts, EXPECTED, atol=0.05)
  np.testing.assert_allclose(biases, 0.0, atol=0.02)


def test_the_shifts_and_biases_sum_to_zero(registered):
  """Only differences are observable, so the gauge is fixed at the mean."""
  _, (shifts, biases) = registered
  np.testing.assert_allclose(shifts.sum(axis=0), 0.0, atol=1e-9)
  assert abs(biases.sum()) < 1e-9


def test_a_vertical_bias_is_separated_from_the_shift():
  passes = survey(seed=2)
  passes[1] = (passes[1][0], passes[1][1] + 0.3)
  shifts, biases = register_passes(passes)
  np.testing.assert_allclose(shifts, EXPECTED, atol=0.05)
  np.testing.assert_allclose(
    biases, -(np.array([0, 0.3, 0, 0]) - 0.075), atol=0.02
  )


def test_aligned_passes_are_left_where_they_are():
  shifts, biases = register_passes(survey(offsets=np.zeros((4, 2)), seed=3))
  np.testing.assert_allclose(shifts, 0.0, atol=0.05)
  np.testing.assert_allclose(biases, 0.0, atol=0.02)


def test_moving_the_whole_survey_changes_nothing(registered):
  passes, (shifts, _) = registered
  moved = [(xy + [500.0, -300.0], z) for xy, z in passes]
  np.testing.assert_allclose(register_passes(moved)[0], shifts, atol=1e-9)


def test_reordering_the_passes_reorders_the_shifts(registered):
  passes, (shifts, _) = registered
  order = [2, 0, 3, 1]
  reordered, _ = register_passes([passes[i] for i in order])
  np.testing.assert_allclose(reordered, shifts[order], atol=0.05)


def test_flat_seabed_is_refused_rather_than_fitted_to_noise():
  flat = survey(surface=lambda xy: np.full(len(xy), -70.0), seed=4)
  with pytest.raises(ValueError, match="cannot be registered in x and y"):
    register_passes(flat)


def test_relief_along_one_axis_leaves_that_axis_unregistered():
  """A ridge running along x pins y, and says nothing about x."""
  ridge = survey(
    surface=lambda xy: -70.0 + 4.0 * np.exp(-((xy[:, 1] - 45) ** 2) / 8.0),
    seed=5,
  )
  with pytest.raises(ValueError, match="cannot be registered in x:"):
    register_passes(ridge)


def test_a_single_pass_is_refused():
  with pytest.raises(ValueError, match="at least two"):
    register_passes(survey()[:1])


def test_passes_that_do_not_overlap_are_refused():
  rng = np.random.default_rng(7)
  apart = [
    survey_pass(rng, np.zeros(2), (0, 40, 0, 40)),
    survey_pass(rng, np.zeros(2), (60, 100, 60, 100)),
  ]
  with pytest.raises(ValueError, match="overlap"):
    register_passes(apart)


# -- the gridded surface ------------------------------------------------------


def test_the_grid_is_the_median_per_cell():
  points = np.array([[0.1, 0.1], [0.2, 0.3], [0.4, 0.2], [1.5, 0.5]])
  z = np.array([1.0, 5.0, 2.0, 7.0])
  grid = grid_surface(points, z, cell=1.0)
  assert grid.z[0, 0] == 2.0 and grid.z[0, 1] == 7.0


def test_sampling_a_plane_is_exact_in_value_and_slope():
  """Bilinear interpolation reproduces a plane given its values at the cells."""
  centres = np.stack(np.meshgrid(np.arange(20) + 0.5, np.arange(20) + 0.5), -1)
  centres = centres.reshape(-1, 2)

  def plane(xy):
    return 3.0 + 0.4 * xy[:, 0] - 0.25 * xy[:, 1]

  grid = grid_surface(centres, plane(centres), cell=1.0)
  queries = np.random.default_rng(6).uniform(1, 19, size=(50, 2))
  value, slope = grid.sample(queries)
  np.testing.assert_allclose(value, plane(queries), atol=1e-12)
  np.testing.assert_allclose(
    slope, np.broadcast_to([0.4, -0.25], (50, 2)), atol=1e-12
  )


def test_sampling_off_the_grid_is_nan():
  grid = grid_surface(np.array([[0.0, 0.0], [5.0, 5.0]]), np.zeros(2), cell=1.0)
  value, slope = grid.sample(np.array([[-3.0, 2.0], [2.0, 50.0]]))
  assert np.isnan(value).all() and np.isnan(slope).all()
