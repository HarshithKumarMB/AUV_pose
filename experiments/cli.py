"""Argument and environment helpers shared by the drivers."""

import argparse
import os
from pathlib import Path

__all__ = [
  "DEFAULT_ROOT",
  "add_octree_args",
  "configure_sdl",
  "octree_directory",
  "refuse_overwrite",
]

#: Where the simulator keeps its world binaries and octree caches. The flake
#: sets ``HOLODECKPATH``; the fallback is what holoocean would use unaided.
DEFAULT_ROOT = Path(
  os.environ.get("HOLODECKPATH", Path.home() / "data" / "holoocean")
)


def add_octree_args(parser: argparse.ArgumentParser) -> None:
  """Add the arguments :func:`octree_directory` consumes.

  Every script that scores against the octree needs the same four, and they
  were copied into each one separately -- which is how three of them ended up
  with their own ``DEFAULT_ROOT``.
  """
  group = parser.add_argument_group("octree cache")
  group.add_argument("--root", type=Path, default=DEFAULT_ROOT)
  group.add_argument("--version", default="2.3.0")
  group.add_argument("--world", default="Dam")
  group.add_argument(
    "--cache",
    default="min2_max512",
    help=(
      "octree cache directory, named for its voxel bounds in centimetres. "
      "holoocean rounds the coarsest node up to octree_min * 2^n, so "
      "octree_max = 5.0 with octree_min = 0.02 gives 5.12 m -- 'max512', not "
      "'max500'. Scoring against the wrong cache compares a run to a "
      "differently resolved world"
    ),
  )


def octree_directory(args: argparse.Namespace) -> Path:
  """The octree cache directory named by :func:`add_octree_args`.

  Raises:
      SystemExit: If it does not exist, with the reason it usually does not --
          the world has never been run with a sonar at these settings.
  """
  directory = (
    args.root
    / args.version
    / "worlds/Ocean/Linux/Holodeck/Octrees"
    / args.world
    / args.cache
  )
  if not directory.is_dir():
    raise SystemExit(
      f"no octree cache at {directory}. The simulator writes one the first "
      f"time a sonar runs in {args.world} at these octree bounds; run a "
      "scenario there once, or pass --root/--world/--cache."
    )
  return directory


def configure_sdl(headless: bool) -> None:
  """Point SDL at a video backend the FHS environment actually provides.

  Unreal initialises SDL, and SDL prefers Wayland whenever ``WAYLAND_DISPLAY``
  is set. The simulator environment ships the X11 stack but not Wayland's, so
  on a Wayland session SDL reports "wayland not available" and Unreal exits
  with ``InitSDL() failed, cannot create application instance`` before the
  client's loading semaphore is ever posted -- which surfaces only as a 30
  second timeout.

  XWayland covers the windowed case. Headless has no display at all, so it gets
  SDL's offscreen backend.

  This **overrides** any inherited ``SDL_VIDEODRIVER``. Wayland sessions
  commonly export ``SDL_VIDEODRIVER=wayland`` globally, and that value is
  inherited into the simulator environment where it is simply wrong -- the
  environment ships no Wayland libraries. An inherited session default is not
  an instruction. To force a backend deliberately, set
  ``AUV_POSE_SDL_VIDEODRIVER``.

  :param headless: Whether the simulator will run with ``-RenderOffScreen``.
  """
  override = os.environ.get("AUV_POSE_SDL_VIDEODRIVER")
  os.environ["SDL_VIDEODRIVER"] = override or (
    "offscreen" if headless else "x11"
  )


def refuse_overwrite(path: Path, force: bool) -> None:
  """Stop rather than clobber an existing output file.

  A survey and the map fitted on it are the only record of a simulator run that
  takes minutes and tens of GB of octree to reproduce, and they are the default
  output paths of the scripts that would overwrite them. Refuse by default.

  :param path: Output path about to be written.
  :param force: Overwrite anyway.
  :raises SystemExit: If ``path`` exists and ``force`` is not set.
  """
  if path.exists() and not force:
    raise SystemExit(
      f"{path} already exists. Pass --force to overwrite it, or --out to write "
      "somewhere else."
    )
