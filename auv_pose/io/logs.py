"""Writing run logs as CSV with a fixed schema."""

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Self


class CsvWriter:
  """Write rows to a CSV with a fixed header, refusing any that do not match.

  Use as a context manager::

      with CsvWriter("run.csv", ("step", "x")) as log:
          log.write(step=0, x=1.0)

  Line-buffered, so a run killed partway leaves every finished row on disk.
  """

  def __init__(self, path: str | Path, columns: Sequence[str]) -> None:
    self.path = Path(path)
    self.columns = tuple(columns)
    self._handle = None
    self._writer = None

  def __enter__(self) -> Self:
    self._handle = open(self.path, "w", newline="", buffering=1)
    self._writer = csv.DictWriter(self._handle, self.columns)
    self._writer.writeheader()
    return self

  def __exit__(self, *exc_info) -> None:
    if self._handle is not None:
      self._handle.close()
    self._handle = self._writer = None

  def write(self, **values) -> None:
    """Write one row. Every column must be supplied, and no extras."""
    if self._writer is None:
      raise RuntimeError("CsvWriter must be used as a context manager")
    # DictWriter refuses extras itself, but fills a missing column with "".
    missing = set(self.columns) - set(values)
    if missing:
      raise ValueError(f"row lacks {sorted(missing)}")
    self._writer.writerow(values)
