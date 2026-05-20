"""Results persistence utilities.

All experiment outputs are appended to a single CSV file ``res/results.csv``.
The file is created with a header row on first write and never overwritten.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "res" / "results.csv"

# Canonical column order — every appended row must contain these fields.
# ``tabpfn_version`` records which internal TabPFN ModelVersion produced the
# row (``"v2_6"`` or ``"v3"``); it is blank for non-TabPFN models.
RESULT_COLUMNS = [
    "timestamp",
    "experiment_id",
    "dataset",
    "model",
    "task",
    "fold",
    "metric",
    "value",
    "tabpfn_version",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_results(rows: list[dict], output_path: Path | None = None) -> None:
    """Append one or more result rows to a results CSV file.

    Creates the file with a header row if it does not exist.  Never
    overwrites existing data.

    Args:
        rows: list of dicts.  Each dict must contain all keys in
            ``RESULT_COLUMNS``.  A ``timestamp`` key is added automatically
            if absent.
        output_path: path to the CSV file.  Defaults to ``res/results.csv``.

    Raises:
        ValueError: if any required column is missing from a row.
    """
    if not rows:
        return

    path = output_path if output_path is not None else RESULTS_PATH

    # Validate and fill defaults.
    # ``tabpfn_version`` is auto-filled with "" so non-TabPFN appends do not
    # need to thread it through every call site.
    _optional_defaults = {"tabpfn_version": ""}
    enriched = []
    for row in rows:
        missing = [
            c for c in RESULT_COLUMNS
            if c not in row and c != "timestamp" and c not in _optional_defaults
        ]
        if missing:
            raise ValueError(f"Result row missing required columns: {missing}. Row: {row}")
        enriched.append({
            "timestamp": row.get("timestamp", _now_iso()),
            **{
                c: row.get(c, _optional_defaults.get(c))
                for c in RESULT_COLUMNS if c != "timestamp"
            },
        })

    df_new = pd.DataFrame(enriched, columns=RESULT_COLUMNS)

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    df_new.to_csv(path, mode="a", header=write_header, index=False)
    logger.info("Appended %d result row(s) to %s", len(rows), path)


def build_result_row(
    experiment_id: str,
    dataset: str,
    model: str,
    task: str,
    fold: int | str,
    metric: str,
    value: float,
    tabpfn_version: str = "",
) -> dict:
    """Construct a single result dict with all required fields.

    Args:
        fold: integer fold index, or the string ``"pooled"`` for the OOF score.
        tabpfn_version: internal TabPFN ModelVersion (``"v2_6"`` / ``"v3"``);
            leave blank for non-TabPFN models.
    """
    return {
        "timestamp": _now_iso(),
        "experiment_id": experiment_id,
        "dataset": dataset,
        "model": model,
        "task": task,
        "fold": fold,
        "metric": metric,
        "value": float(value),
        "tabpfn_version": tabpfn_version,
    }
