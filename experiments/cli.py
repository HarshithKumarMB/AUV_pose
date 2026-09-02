"""Argument helpers shared by the drivers."""

from __future__ import annotations

from pathlib import Path

__all__ = ["refuse_overwrite"]


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
