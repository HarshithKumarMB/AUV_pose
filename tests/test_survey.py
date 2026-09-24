"""The survey track, which decides what the map can resolve."""

import numpy as np
import pytest

from experiments.survey import SURVEY_BOX, figure_eight, lawnmower


def legs(waypoints):
  """Direction of each *traverse*, as unit vectors in the horizontal plane.

  A boustrophedon alternates long traverses with short steps across to the next
  line, so select by length: only the traverses carry the heading.
  """
  points = np.asarray(waypoints)[:, :2]
  steps = np.diff(points, axis=0)
  lengths = np.linalg.norm(steps, axis=1)
  traverses = steps[lengths > 0.5 * lengths.max()]
  return traverses / np.linalg.norm(traverses, axis=1)[:, None]


@pytest.mark.parametrize("heading", [0.0, 45.0, 90.0, 135.0])
def test_traverses_run_along_the_heading(heading):
  """The fan opens along body -y, so the track must run along body x.

  A track parallel to the fan sweeps the same strip 240 times and leaves the
  across-track sampling at the line spacing -- which is how a 240-beam fan
  bought no more coverage than one beam.
  """
  forward = np.array([np.cos(np.radians(heading)), np.sin(np.radians(heading))])

  for direction in legs(lawnmower(heading=heading)):
    assert abs(abs(direction @ forward) - 1.0) < 1e-9


@pytest.mark.parametrize("heading", [0.0, 30.0, 45.0, 90.0, 135.0])
def test_every_heading_covers_the_box(heading):
  """A rotated pattern must still span the box, not an inscribed square."""
  waypoints = np.asarray(lawnmower(heading=heading, spacing=2.0))[:, :2]
  x_min, x_max, y_min, y_max = SURVEY_BOX

  assert waypoints[:, 0].min() <= x_min and waypoints[:, 0].max() >= x_max
  assert waypoints[:, 1].min() <= y_min and waypoints[:, 1].max() >= y_max


def test_lines_are_spaced_as_asked():
  waypoints = np.asarray(lawnmower(heading=0.0, spacing=5.0))
  # Heading 0 steps across in y; consecutive lines differ by the spacing.
  offsets = np.unique(np.round(waypoints[:, 1], 6))
  assert np.allclose(np.diff(offsets), 5.0)


def test_the_pattern_reverses_each_line():
  """Boustrophedon, not a raster: flying back to the start wastes the leg."""
  waypoints = np.asarray(lawnmower(heading=0.0, spacing=5.0))[:, :2]
  directions = legs(waypoints)

  # Consecutive traverses alternate sign along the heading.
  along = directions @ np.array([1.0, 0.0])
  assert np.allclose(np.abs(along), 1.0)
  assert np.all(along[:-1] * along[1:] < 0)


def test_headings_that_differ_by_180_degrees_cover_the_same_ground():
  forward = np.asarray(lawnmower(heading=0.0, spacing=5.0))[:, :2]
  reverse = np.asarray(lawnmower(heading=180.0, spacing=5.0))[:, :2]

  assert forward[:, 0].min() == pytest.approx(reverse[:, 0].min(), abs=1e-6)
  assert forward[:, 1].min() == pytest.approx(reverse[:, 1].min(), abs=1e-6)


def test_depth_is_held():
  waypoints = np.asarray(lawnmower(heading=45.0, z=-12.0))
  assert np.allclose(waypoints[:, 2], -12.0)


# -- the figure-eight test track --------------------------------------------


def test_each_loop_is_a_circle_at_its_own_depth():
  track = np.asarray(figure_eight((0.0, 0.0), 10.0, (0.0, -30.0), 24))
  one, two = track[1:25], track[26:]
  np.testing.assert_allclose(
    np.linalg.norm(one[:, :2] - [-10, 0], axis=1), 10.0
  )
  np.testing.assert_allclose(np.linalg.norm(two[:, :2] - [10, 0], axis=1), 10.0)
  assert np.all(one[:, 2] == 0.0) and np.all(two[:, 2] == -30.0)


def test_the_loops_turn_opposite_ways_and_meet_at_the_centre():
  """A figure eight, not two circles flown the same way."""
  track = np.asarray(figure_eight((5.0, -3.0), 10.0, (0.0, -30.0), 24))
  np.testing.assert_allclose(track[0], [5.0, -3.0, 0.0])
  np.testing.assert_allclose(track[-1], [5.0, -3.0, -30.0], atol=1e-9)

  def turning(points, centre):
    offset = points[:, :2] - centre
    a, b = offset[:-1], offset[1:]
    return np.sign(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

  assert np.all(turning(track[1:25], [-5.0, -3.0]) > 0)  # anticlockwise
  assert np.all(turning(track[26:], [15.0, -3.0]) < 0)  # clockwise


def test_the_depth_change_happens_at_the_crossing():
  track = np.asarray(figure_eight((0.0, 0.0), 10.0, (0.0, -30.0), 24))
  np.testing.assert_allclose(track[25], [0.0, 0.0, -30.0], atol=1e-9)
