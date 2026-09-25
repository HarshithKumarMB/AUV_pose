"""Environment and output helpers shared by the drivers."""

import os
from pathlib import Path


def configure_sdl(headless: bool) -> None:
  """Set ``SDL_VIDEODRIVER`` to x11, or offscreen when ``headless``.

  Overrides any inherited value, since the simulator environment has no
  Wayland libraries; set ``AUV_POSE_SDL_VIDEODRIVER`` to force a backend.
  """
  override = os.environ.get("AUV_POSE_SDL_VIDEODRIVER")
  os.environ["SDL_VIDEODRIVER"] = override or (
    "offscreen" if headless else "x11"
  )


def refuse_overwrite(path: Path, force: bool) -> None:
  """Exit if ``path`` exists and ``force`` is not set."""
  if path.exists() and not force:
    raise SystemExit(
      f"{path} already exists. Pass --force to overwrite it, or --out to write "
      "somewhere else."
    )
