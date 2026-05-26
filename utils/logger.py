from __future__ import annotations

from pathlib import Path
import csv


class CSVLogger:
    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()

    def log(self, row: dict) -> None:
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._file.flush()

    def close(self) -> None:
        self._file.close()
