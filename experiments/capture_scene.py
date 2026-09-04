"""Photograph a patch of seabed, to see what the sonar is ranging against.

    nix run .#sim -- -c "python -u experiments/capture_scene.py --out shots/"

The multibeam and the octree disagree by 4-5 m over compact patches whose world
position and ~10 m size hold across altitudes of 17, 40 and 69 m. That is the
signature of an object standing on the seabed, and the octree contains no
geometry there -- but "an object the octree omits" and "a sonar artefact that
happens to be world-locked" predict the same ranges. A picture separates them:
either something is visibly there or nothing is.

Uses ``ViewportCapture`` with :meth:`move_viewport`, so the camera flies free of
the vehicle and can be put wherever the view is best. Water fog is turned down,
without which 70 m down is opaque and the frame says nothing either way.

``--mark`` draws a wireframe box where the sonar puts the surface, so the render
can be compared against the measurement rather than eyeballed on its own.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import holoocean
import numpy as np

from experiments.cli import configure_sdl
from experiments.scenarios import (
  ocean_scenario,
  orientation_sensor,
  pose_sensor,
  viewport_capture,
)

#: Where the sonar says something stands that the octree does not contain:
#: centred (x, y), apex and seabed elevation, and its measured width.
SUSPECT = {"x": -21.0, "y": -1.7, "apex": -65.1, "seabed": -69.7, "width": 10.0}

#: Ground the sonar and the octree agree on to 0.035 m, for a control shot. If
#: the suspect looks like an object and this looks like bare seabed, that is the
#: comparison; if both look the same, the render is not resolving the question.
CONTROL = {
  "x": -24.0,
  "y": -25.0,
  "apex": -68.7,
  "seabed": -68.7,
  "width": 10.0,
}


def views(target: dict, standoff: float, height: float) -> list[tuple]:
  """(label, location, rotation) triples looking at ``target`` from several sides.

  One angle is not enough: a dome is unmistakable in silhouette against the
  water and nearly invisible from directly above, and a sonar artefact would
  show in none of them.
  """
  x, y = target["x"], target["y"]
  mid = 0.5 * (target["apex"] + target["seabed"])
  return [
    # Low and to the side: puts anything standing proud against open water.
    ("side_south", [x, y - standoff, mid + 1.0], [0.0, 0.0, 90.0]),
    ("side_west", [x - standoff, y, mid + 1.0], [0.0, 0.0, 0.0]),
    # Raised three-quarter view, the most readable of a mound.
    (
      "oblique",
      [x - standoff * 0.7, y - standoff * 0.7, target["seabed"] + height],
      [0.0, -np.degrees(np.arctan2(height, standoff)), 45.0],
    ),
    # Straight down, which gives the footprint rather than the profile.
    ("overhead", [x, y, target["seabed"] + height * 1.6], [0.0, -90.0, 0.0]),
  ]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, default=Path("scene"))
  parser.add_argument(
    "--target",
    choices=("suspect", "control", "both"),
    default="both",
    help="which patch to photograph; 'both' gives the comparison",
  )
  parser.add_argument("--standoff", type=float, default=16.0)
  parser.add_argument("--height", type=float, default=14.0)
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height-px", type=int, default=720)
  parser.add_argument(
    "--fog",
    type=float,
    default=0.0,
    help=(
      "water fog density, 0-10. The default of 0 is deliberately unphysical: "
      "the question is what geometry is there, and holoocean's default makes "
      "70 m of water opaque"
    ),
  )
  parser.add_argument(
    "--mark",
    action="store_true",
    help="draw a box where the sonar puts the surface",
  )
  parser.add_argument(
    "--settle",
    type=int,
    default=30,
    help=(
      "ticks between moving the camera and grabbing the frame. The teleport "
      "applies on the next tick and the water surface takes a few more to "
      "settle, so a frame grabbed immediately can be of the old viewpoint"
    ),
  )
  parser.add_argument("--octree-min", type=float, default=0.02)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  configure_sdl(headless=True)

  from PIL import Image

  targets = {"suspect": SUSPECT, "control": CONTROL}
  wanted = list(targets) if args.target == "both" else [args.target]

  # Spawn somewhere harmless; the vehicle is only a carrier for the sensor, and
  # the camera is moved off it immediately.
  start = [targets[wanted[0]]["x"], targets[wanted[0]]["y"], -40.0]
  env = holoocean.make(
    scenario_cfg=ocean_scenario(
      "scene_capture",
      start=start,
      octree_min=args.octree_min,
      sensors=[
        pose_sensor(),
        orientation_sensor(),
        viewport_capture(width=args.width, height=args.height_px),
      ],
    ),
    show_viewport=False,
    window_res=(args.height_px, args.width),
  )
  env.set_render_quality(3)
  env.water_fog(args.fog)
  env.should_render_viewport(True)

  written = []
  for name in wanted:
    target = targets[name]
    for label, location, rotation in views(target, args.standoff, args.height):
      env.move_viewport(location, rotation)
      if args.mark:
        centre = [
          target["x"],
          target["y"],
          0.5 * (target["apex"] + target["seabed"]),
        ]
        half = 0.5 * (target["seabed"] - target["apex"])
        env.draw_box(
          centre,
          [target["width"] / 2, target["width"] / 2, abs(half)],
          color=[255, 0, 0],
          thickness=6.0,
          lifetime=0.0,
        )

      frame = None
      for _ in range(args.settle):
        state = env.tick()
        if "ViewportCapture" in state:
          frame = state["ViewportCapture"]

      if frame is None:
        print(f"  {name}/{label}: no frame returned")
        continue

      path = args.out / f"{name}_{label}.png"
      Image.fromarray(np.asarray(frame)[..., :3]).save(path)
      written.append(path)
      print(f"  wrote {path}  camera {location} rot {rotation}")

  print(f"\n{len(written)} frames in {args.out}")


if __name__ == "__main__":
  main()
