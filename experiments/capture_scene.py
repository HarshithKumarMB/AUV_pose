"""Photograph a patch of seabed, to see what the sonar is ranging against.

    nix run .#sim -- -c "python -u experiments/capture_scene.py --out shots/"

Sonar and octree disagree by 4-5 m over compact patches of the Dam seabed. Two
explanations predicted identical ranges -- a sonar artefact that happens to be
world-locked, or an object the octree does not contain -- and no amount of
further ranging could separate them. A photograph could, and did: **they are
bolted pipelines lying on the seabed**, with a valve manifold and the dam wall
behind. The sonar was right; octree generation voxelises only landscape.

That is worth keeping as a tool rather than a one-off. It is the cheapest way to
answer "is the reference wrong or is the sensor wrong", and the calibration
worlds have to be checked the same way before anything measured in them is
trusted -- an octree that silently omits geometry invalidates every score taken
against it.

Uses ``ViewportCapture`` with :meth:`move_viewport`, so the camera flies free of
the vehicle. Water fog is turned off, without which 70 m down is opaque and the
frame says nothing either way. ``--mark`` draws a box where the sonar puts the
surface, so the render can be compared against the measurement rather than
eyeballed.
"""

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


def views(
  x: float, y: float, seabed: float, apex: float, standoff: float, height: float
) -> list[tuple[str, list[float], list[float]]]:
  """``(label, location, rotation)`` looking at the target from several sides.

  One angle is not enough: something standing proud of the seabed is
  unmistakable in silhouette from low down and nearly invisible from overhead,
  and a sensor artefact would show in none of them.
  """
  mid = 0.5 * (apex + seabed)
  pitch = float(np.degrees(np.arctan2(height, standoff)))
  return [
    # Low and to the side, which puts anything standing proud against water.
    ("side_south", [x, y - standoff, mid + 1.0], [0.0, 0.0, 90.0]),
    ("side_west", [x - standoff, y, mid + 1.0], [0.0, 0.0, 0.0]),
    # Raised three-quarter view, the most readable of a mound or a pipe.
    (
      "oblique",
      [x - standoff * 0.7, y - standoff * 0.7, seabed + height],
      [0.0, -pitch, 45.0],
    ),
    # Looking down, which gives the footprint rather than the profile.
    ("overhead", [x, y, seabed + height * 1.6], [0.0, -90.0, 0.0]),
  ]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, default=Path("scene"))
  parser.add_argument(
    "--at",
    type=float,
    nargs=2,
    metavar=("X", "Y"),
    default=[-21.0, -1.7],
    help=(
      "world position to photograph. The default is the Dam pipeline that "
      "started this -- pass the centre of whatever check_beam_validity.py "
      "reports a disagreement over"
    ),
  )
  parser.add_argument(
    "--seabed",
    type=float,
    default=-69.7,
    help="seabed elevation there, metres; sets where the camera sits",
  )
  parser.add_argument(
    "--apex",
    type=float,
    default=-65.1,
    help=(
      "elevation the sonar reports, metres. Only frames the shot and places "
      "--mark; pass the seabed value if nothing is expected to stand there"
    ),
  )
  parser.add_argument("--width-m", type=float, default=10.0, help="--mark size")
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
      "applies on the next tick and the water takes a few more to settle, so "
      "a frame grabbed immediately can be of the old viewpoint"
    ),
  )
  parser.add_argument("--world", default="Dam")
  parser.add_argument("--octree-min", type=float, default=0.02)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  configure_sdl(headless=True)

  from PIL import Image

  x, y = args.at
  env = holoocean.make(
    scenario_cfg=ocean_scenario(
      "scene_capture",
      start=[x, y, args.seabed + 30.0],
      world=args.world,
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
  for label, location, rotation in views(
    x, y, args.seabed, args.apex, args.standoff, args.height
  ):
    env.move_viewport(location, rotation)
    if args.mark:
      half = 0.5 * abs(args.apex - args.seabed)
      env.draw_box(
        [x, y, 0.5 * (args.apex + args.seabed)],
        [args.width_m / 2, args.width_m / 2, max(half, 0.25)],
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
      print(f"  {label}: no frame returned")
      continue

    path = args.out / f"{label}.png"
    Image.fromarray(np.asarray(frame)[..., :3]).save(path)
    written.append(path)
    print(f"  wrote {path}  camera {location} rot {rotation}")

  print(f"\n{len(written)} frames in {args.out}")


if __name__ == "__main__":
  main()
