"""lab_utils.logging.csv_logger — append-safe per-epoch scalar CSV.

One row per ``write()`` call.  Columns are union-of-all-keys seen so far;
missing values in a row are written as empty strings.  Safe to resume: if the
file already exists the header is read from it and new columns are appended.

Usage::

    logger = CsvLogger(rd.metrics_path)
    logger.write(epoch=1, loss=0.342, bce=0.210, simpos=0.587, ...)
"""

from __future__ import annotations

import csv
import os
import threading
from pathlib import Path
from typing import Any, Union


class CsvLogger:
    """Append-safe CSV file for scalar training metrics."""

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = str(path)
        self._lock = threading.Lock()
        self._columns: list[str] = []

        if os.path.exists(self._path) and os.path.getsize(self._path) > 0:
            with open(self._path, newline='') as f:
                reader = csv.reader(f)
                try:
                    self._columns = next(reader)
                except StopIteration:
                    pass

    def write(self, **kwargs: Any) -> None:
        """Append one row.  Key order is stable (insertion order of first appearance)."""
        with self._lock:
            new_cols = [k for k in kwargs if k not in self._columns]
            need_rewrite = bool(new_cols) and bool(self._columns)
            self._columns.extend(new_cols)

            if need_rewrite and os.path.exists(self._path):
                # New columns seen mid-run — rewrite old rows with empty cells for new cols.
                with open(self._path, newline='') as f:
                    old_rows = list(csv.DictReader(f))
                with open(self._path, 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=self._columns, extrasaction='ignore')
                    w.writeheader()
                    for row in old_rows:
                        w.writerow(row)
            elif not os.path.exists(self._path) or os.path.getsize(self._path) == 0:
                # Fresh file — write header.
                with open(self._path, 'w', newline='') as f:
                    csv.writer(f).writerow(self._columns)

            # Append the new row.
            with open(self._path, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self._columns, extrasaction='ignore')
                w.writerow({k: ('' if v is None else v) for k, v in kwargs.items()})
