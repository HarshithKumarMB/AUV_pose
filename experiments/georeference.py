"""Place a raw survey's soundings, from the smoothed pose or from truth.

    python experiments/georeference.py ~/data/auv_pose/surveys_v3/pass0 \\
        --pose smoothed --out pass0_smoothed.csv
    python experiments/georeference.py ~/data/auv_pose/surveys_v3/pass0 \\
        --pose truth --out pass0_truth.csv

Replays the log's ticks through the same
:class:`~auv_pose.estimation.navigation.InertialNavigator` the vehicle steered
on, smooths the record with
:func:`~auv_pose.estimation.smoothers.unscented_rts_smooth`, and places every
beam from the pose at its ping. Each sounding carries its own position
covariance (:func:`~auv_pose.mapping.sonar.sounding_covariance`) -- the input
uncertainty the map's NIGP step feeds on -- and where the same range would have
landed from the true pose, for scoring.

``--pose truth`` is the control: the same pings, placed exactly, with zero
covariance. Fitting a map to both separates "the map is distorted by
navigation" from everything else.

The replay is checked against the flight: it must reproduce the filter's own
innovations, which it does exactly when nothing in the navigation has changed
since the survey was flown.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from auv_pose.estimation.inertial import ImuNoise
from auv_pose.estimation.manifold import (
  POSITION,
  ROTATION,
  ManifoldGaussian,
  NavState,
  boxminus,
)
from auv_pose.estimation.navigation import (
  AidingNoise,
  InertialNavigator,
  dvl_noise_covariance,
)
from auv_pose.estimation.quaternion import quat_normalize, quat_to_rotmat
from auv_pose.estimation.smoothers import unscented_rts_smooth
from auv_pose.io.raw_survey import RawSurvey, load_raw_survey
from auv_pose.io.soundings import OPTIONAL_COLUMNS, SOUNDING_COLUMNS
from auv_pose.mapping.sonar import seabed_points, sounding_covariance
from experiments.cli import refuse_overwrite

#: The six tangent coordinates a sounding's placement depends on.
POSE = np.r_[np.arange(15)[POSITION], np.arange(15)[ROTATION]]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("log", type=Path, help="raw survey log directory")
  parser.add_argument(
    "--pose",
    choices=("smoothed", "filtered", "truth"),
    default="smoothed",
    help=(
      "which pose places the soundings. 'filtered' is the causal estimate the "
      "vehicle steered on, kept to show what smoothing buys"
    ),
  )
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument(
    "--sigma-range",
    type=float,
    default=0.0,
    help=(
      "range noise, metres, added along each beam. Zero by default: the "
      "simulated multibeam is quantised at 0.0995 m and otherwise exact"
    ),
  )
  parser.add_argument("--force", action="store_true")
  return parser.parse_args()


def initial_belief(meta: dict) -> ManifoldGaussian:
  """The belief the flight started from, as its metadata recorded it."""
  mean = meta["initial_mean"]
  sigma = meta["initial_sigma"]
  state = NavState(
    position=np.asarray(mean["position"], float),
    attitude=quat_normalize(mean["attitude"]),
    velocity=np.asarray(mean["velocity"], float),
    gyro_bias=np.asarray(mean["gyro_bias"], float),
    accel_bias=np.asarray(mean["accel_bias"], float),
  )
  spread = np.concatenate(
    [
      sigma["position"],
      np.radians(sigma["attitude_deg"]),
      sigma["velocity"],
      sigma["gyro_bias"],
      sigma["accel_bias"],
    ]
  )
  return ManifoldGaussian(state, np.diag(np.asarray(spread, float) ** 2))


def replay(survey: RawSurvey) -> InertialNavigator:
  """Run the flight's forward pass again, tick for tick."""
  meta = survey.meta
  navigator = InertialNavigator(
    initial_belief(meta),
    1.0 / meta["tick_rate_hz"],
    AidingNoise(
      dvl=dvl_noise_covariance(
        meta["dvl_beam_sigma"], meta["dvl_elevation_deg"]
      ),
      depth=meta["depth_sigma"],
      magnetometer=meta["compass_sigma"],
    ),
    imu_noise=ImuNoise(**meta["imu_noise"]),
    field=meta["magnetic_field"],
  )

  ticks = survey.ticks
  gyro, accel = survey.readings("gyro"), survey.readings("accel")
  dvl, compass = survey.readings("dvl"), survey.readings("mag")
  depth = ticks["depth"].to_numpy(float)
  pinged = set(survey.ping_ticks.tolist())

  for row, index in enumerate(ticks["tick"].to_numpy(int)):
    navigator.tick(
      index,
      gyro[row],
      accel[row],
      dvl=dvl[row] if np.isfinite(dvl[row]).all() else None,
      depth=depth[row] if np.isfinite(depth[row]) else None,
      magnetometer=compass[row] if np.isfinite(compass[row]).all() else None,
      close=index in pinged,
    )
  return navigator


def truth_at(survey: RawSurvey) -> dict[int, NavState]:
  """The logged true pose at each ping tick."""
  frame = survey.ticks.set_index("tick").loc[survey.ping_ticks]
  position = frame[["true_x", "true_y", "true_z"]].to_numpy(float)
  attitude = frame[["true_qw", "true_qx", "true_qy", "true_qz"]].to_numpy(float)
  return {
    int(tick): NavState.at_rest(position=p, attitude=q)
    for tick, p, q in zip(survey.ping_ticks, position, attitude)
  }


def main() -> None:
  args = parse_args()
  refuse_overwrite(args.out, args.force)

  survey = load_raw_survey(args.log.expanduser())
  meta = survey.meta
  bearings = np.asarray(meta["bearings"], float)
  swath, nadir = meta["swath_axis"], meta["nadir_axis"]
  truth = truth_at(survey)

  if args.pose == "truth":
    poses = {
      tick: ManifoldGaussian(state, np.zeros((15, 15)))
      for tick, state in truth.items()
    }
  else:
    navigator = replay(survey)
    print(
      f"Replayed {len(survey.ticks)} ticks into {len(navigator.history)} cycles"
    )
    for name, values in navigator.nis.items():
      dof = {"dvl": 3, "depth": 1, "compass": 3}[name]
      print(f"  {name:8s} mean NIS {np.mean(values):.2f} against {dof}")

    beliefs = (
      [step.posterior for step in navigator.history]
      if args.pose == "filtered"
      else unscented_rts_smooth(navigator.initial, navigator.history)[1:]
    )
    poses = dict(zip(navigator.cycle_ticks, beliefs))

    # How honest the pose is where it matters: at the pings, in the six
    # coordinates that place a sounding.
    errors, nees = [], []
    for tick, state in truth.items():
      belief = poses[tick]
      error = boxminus(state, belief.mean)[POSE]
      errors.append(np.linalg.norm(error[:2]))
      block = belief.cov[np.ix_(POSE, POSE)]
      nees.append(error @ np.linalg.solve(block, error))
    print(
      f"  {args.pose} pose at the pings: horizontal error median "
      f"{np.median(errors):.2f} m, max {np.max(errors):.2f} m; "
      f"mean NEES {np.mean(nees):.2f} against 6"
    )

  columns = [*SOUNDING_COLUMNS, *OPTIONAL_COLUMNS]
  frames = []
  for ping, (tick, ranges) in enumerate(zip(survey.ping_ticks, survey.ranges)):
    belief = poses[int(tick)]
    rotation = quat_to_rotmat(belief.mean.attitude)
    points = seabed_points(
      belief.mean.position, rotation, ranges, bearings, swath, nadir
    )
    cov = sounding_covariance(
      belief.cov[np.ix_(POSE, POSE)],
      rotation,
      ranges,
      bearings,
      swath,
      nadir,
      sigma_range=args.sigma_range,
    )
    true = truth[int(tick)]
    true_points = seabed_points(
      true.position, true.rotation, ranges, bearings, swath, nadir
    )
    finite = np.isfinite(points).all(axis=1)
    frames.append(
      pd.DataFrame(
        {
          "x": points[finite, 0],
          "y": points[finite, 1],
          "z": points[finite, 2],
          "cov_xx": cov[finite, 0, 0],
          "cov_xy": cov[finite, 0, 1],
          "cov_yy": cov[finite, 1, 1],
          "cov_zz": cov[finite, 2, 2],
          "ping": ping,
          "true_x": true_points[finite, 0],
          "true_y": true_points[finite, 1],
          "true_z": true_points[finite, 2],
        },
        columns=columns,
      )
    )

  soundings = pd.concat(frames, ignore_index=True)
  soundings.to_csv(args.out, index=False)
  print(
    f"Wrote {len(soundings)} soundings from {len(frames)} pings to {args.out}"
  )

  if args.pose != "truth":
    offset = (
      soundings[["x", "y"]].to_numpy()
      - soundings[["true_x", "true_y"]].to_numpy()
    )
    distance = np.linalg.norm(offset, axis=1)
    sigma = np.sqrt(soundings["cov_xx"] + soundings["cov_yy"]).to_numpy()
    print(
      f"  placement error median {np.median(distance):.2f} m, "
      f"95th {np.percentile(distance, 95):.2f} m; "
      f"predicted 1-sigma median {np.median(sigma):.2f} m"
    )


if __name__ == "__main__":
  main()
