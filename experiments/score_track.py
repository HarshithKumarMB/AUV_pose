"""Score maps on a test track the survey never saw.

    python experiments/score_track.py track8_smoothed.csv \\
        --track-log surveys_v3/track8 --survey-log surveys_v3/pass0 \\
        --map spline.pkl spline --map plane.pkl "plane, two terms"

The test that matches how the map is used. The track -- a figure eight from
``survey.py --route figure8`` -- was flown after the survey, on its own
navigation with its own drift, so its beams land between the survey's
soundings on every heading and at two altitudes. Each map is asked for the
seabed where every beam truly struck it, and scored against the depth there.

**Frames.** Each flight's map frame is the world shifted by that flight's own
surface-fix error. A map fitted on several passes sits at their mean offset,
and the track's ``true_`` columns are in the track's own frame, so the track's
truth is carried into the map's frame through the recorded start offsets
before any map is queried. The passes' offsets differ from each other too --
about a metre, as separate dives' fixes do -- and the map carries that as
disagreement where passes overlap.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from auv_pose.io.checkpoints import load_map
from auv_pose.io.raw_survey import RawSurvey, load_raw_survey
from auv_pose.mapping.cleaning import object_soundings
from auv_pose.mapping.vecchia import VecchiaMap


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "track", type=Path, help="track soundings from georeference.py"
  )
  parser.add_argument("--track-log", type=Path, required=True)
  parser.add_argument(
    "--survey-log",
    type=Path,
    nargs="+",
    required=True,
    help=(
      "raw log of each survey pass the map was fitted on. Each pass starts "
      "from its own surface fix, so a map fitted on several sits at their "
      "mean offset from the world, and that is where the track is carried"
    ),
  )
  parser.add_argument(
    "--map",
    nargs=2,
    action="append",
    metavar=("CHECKPOINT", "LABEL"),
    required=True,
    help="a map fitted on the survey, and its label; repeat for each",
  )
  return parser.parse_args()


def loop_of_each_ping(log: RawSurvey) -> np.ndarray:
  """Which loop a ping was taken on, from the vehicle's true depth.

  The loops are flown at two depths; the vehicle is on the first while it is
  nearer that depth than the second.
  """
  depths = log.meta["waypoints"]
  first, second = depths[1][2], depths[-1][2]
  ticks = log.ticks.set_index("tick").loc[log.ping_ticks, "true_z"].to_numpy()
  nearer_first = np.abs(ticks - first) < np.abs(ticks - second)
  return np.where(nearer_first, 0, 1)


def main() -> None:
  args = parse_args()
  track = pd.read_csv(args.track)
  track_log = load_raw_survey(args.track_log.expanduser())
  survey_offset = np.mean(
    [load_raw_survey(log.expanduser()).start_offset for log in args.survey_log],
    axis=0,
  )

  # Track truth is in the track's frame; the maps are in the survey's.
  shift = survey_offset - track_log.start_offset
  where = track[["true_x", "true_y"]].to_numpy(float) + shift
  depth = track["true_z"].to_numpy(float)

  loop = loop_of_each_ping(track_log)[track["ping"].to_numpy(int)]
  objects = object_soundings(
    track[["x", "y"]].to_numpy(float), track["z"].to_numpy(float)
  )
  groups = {
    "all": np.ones(len(track), bool),
    "loop 1 (higher)": loop == 0,
    "loop 2 (lower)": loop == 1,
    "objects": objects,
    "seabed": ~objects,
  }
  print(
    f"{len(track)} track soundings: "
    + ", ".join(
      f"{name} {int(mask.sum())}"
      for name, mask in groups.items()
      if name != "all"
    )
  )
  print(f"frame shift survey - track: {np.round(shift, 3)} m")

  for checkpoint, label in args.map:
    bathymetry = load_map(checkpoint)
    mean, std = bathymetry.predict(where, with_std=True, observation_noise=True)
    error = mean - depth
    inside = np.abs(error) <= 1.96 * std

    gap = (
      KDTree(bathymetry.structure.points).query(where)[0]
      if isinstance(bathymetry, VecchiaMap)
      else None
    )
    print(f"\n{label}")
    if gap is not None:
      print(f"  nearest fitted sounding: median {np.median(gap):.2f} m")
    for name, mask in groups.items():
      if not mask.any():
        continue
      rmse = float(np.sqrt(np.mean(error[mask] ** 2)))
      print(
        f"  {name:16s} rmse {rmse:6.3f} m   inside 95% {inside[mask].mean():6.1%}"
      )


if __name__ == "__main__":
  main()
