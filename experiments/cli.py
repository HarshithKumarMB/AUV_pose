"""Argument and environment helpers shared by the drivers."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["configure_sdl", "refuse_overwrite"]


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

  The committed ``map*.csv`` and ``svgp_bathymetry.pkl`` are the only record of
  a survey that takes a simulator run to reproduce, and they are the default
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
