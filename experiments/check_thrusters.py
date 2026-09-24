"""Measure what each horizontal thruster does to the BlueROV2, from rest.

    nix run .#sim -- -c "python -u experiments/check_thrusters.py --headless"

The vendored client documents the thruster geometry only in a comment, and the
comment does not survive a check: taken at face value, the forward command
guidance has always used would push the vehicle backwards. So the mixing is
measured rather than derived. Each horizontal thruster is fired alone from
rest, and the body-frame displacement and yaw it produces are recorded; the
pattern that yaws the vehicle with no net force is then solved for, and
flown to check it.

Control scheme 0 takes eight thrusts: four vertical, then the four angled
horizontal ones, indices 4-7.
"""

from __future__ import annotations

import argparse

import holoocean
import numpy as np

from experiments.cli import configure_sdl
from experiments.scenarios import (
  ocean_scenario,
  orientation_sensor,
  pose_sensor,
)

HORIZONTAL = (4, 5, 6, 7)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--thrust", type=float, default=10.0)
  parser.add_argument("--ticks", type=int, default=15, help="ticks of thrust")
  parser.add_argument("--settle", type=int, default=90, help="ticks at rest")
  parser.add_argument("--headless", action="store_true")
  return parser.parse_args()


def heading(rotation: np.ndarray) -> float:
  """Yaw of the body x axis in the world, radians."""
  return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def trial(env, command: np.ndarray, ticks: int, settle: int) -> np.ndarray:
  """``(forward, starboard-y, yaw)`` change from ``command``, in the body frame."""
  env.reset()
  state = None
  for _ in range(settle):
    state = env.step(np.zeros(8))
  assert state is not None
  start = np.array(state["pose"])[:3, 3]
  yaw0 = heading(np.array(state["orient"]))

  for _ in range(ticks):
    state = env.step(command)
  moved = np.array(state["pose"])[:3, 3] - start
  yaw1 = heading(np.array(state["orient"]))

  c, s = np.cos(yaw0), np.sin(yaw0)
  body = np.array([c * moved[0] + s * moved[1], -s * moved[0] + c * moved[1]])
  turned = np.angle(np.exp(1j * (yaw1 - yaw0)))
  return np.array([body[0], body[1], turned])


def main() -> None:
  args = parse_args()
  configure_sdl(args.headless)
  env = holoocean.make(
    scenario_cfg=ocean_scenario(
      "thrusters",
      start=[-20.0, -10.0, -5.0],
      sensors=[pose_sensor(), orientation_sensor()],
    ),
    show_viewport=not args.headless,
  )

  effects = []
  for index in HORIZONTAL:
    command = np.zeros(8)
    command[index] = args.thrust
    effect = trial(env, command, args.ticks, args.settle)
    effects.append(effect)
    print(
      f"thruster {index}: forward {effect[0]:+.4f} m, y {effect[1]:+.4f} m, "
      f"yaw {np.degrees(effect[2]):+.3f} deg"
    )

  # Columns: what one unit of each thruster does. Pure yaw is the pattern
  # that moves nothing and turns the vehicle.
  response = np.array(effects).T / args.thrust
  target = np.array([0.0, 0.0, 1.0])
  pattern = np.linalg.lstsq(response, target, rcond=None)[0]
  pattern /= np.abs(pattern).max()
  print(f"\npure-yaw pattern over thrusters 4-7: {np.round(pattern, 3)}")

  for label, horizontal in (
    ("forward, as guidance mixes it", [1, 1, 0, 0]),
    ("starboard, as guidance mixes it", [1, -1, 1, -1]),
    ("pure yaw, solved", pattern),
    ("pure yaw, reversed", -pattern),
  ):
    command = np.zeros(8)
    command[4:] = args.thrust * np.asarray(horizontal, dtype=float)
    effect = trial(env, command, args.ticks, args.settle)
    print(
      f"{label:32s}: forward {effect[0]:+.4f} m, y {effect[1]:+.4f} m, "
      f"yaw {np.degrees(effect[2]):+.3f} deg"
    )


if __name__ == "__main__":
  main()
