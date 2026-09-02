"""Writing run logs as CSV with a fixed schema."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

__all__ = ["CsvLogger"]


class CsvLogger:
    """Append rows to a CSV with a fixed header.

    Use as a context manager::

        with CsvLogger("wp_c.csv", NAVIGATION_COLUMNS) as log:
            log.write(step=0, x=1.0, ...)
    """

    def __init__(self, path: str | Path, columns: Sequence[str]) -> None:
        self.path = Path(path)
        self.columns = tuple(columns)
        self._handle = None
        self._writer = None

    def __enter__(self) -> "CsvLogger":
        self._handle = open(self.path, "w", newline="")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(self.columns)
        return self

    def __exit__(self, *exc_info) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None

    def write(self, **values) -> None:
        """Write one row. Every column must be supplied, and no extras."""
        if self._writer is None:
            raise RuntimeError("CsvLogger must be used as a context manager")

        missing = set(self.columns) - set(values)
        extra = set(values) - set(self.columns)
        if missing or extra:
            raise ValueError(
                f"row does not match schema; missing={sorted(missing)} "
                f"unexpected={sorted(extra)}"
            )

        self._writer.writerow([values[column] for column in self.columns])
