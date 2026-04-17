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
RESULT_COLUMNS = [
    "timestamp",
    "experiment_id",
    "dataset",
    "model",
    "task",
    "fold",
    "metric",
    "value",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_results(rows: list[dict]) -> None:
    """Append one or more result rows to ``res/results.csv``.

    Creates the file with a header row if it does not exist.  Never
    overwrites existing data.

    Args:
        rows: list of dicts.  Each dict must contain all keys in
            ``RESULT_COLUMNS``.  A ``timestamp`` key is added automatically
            if absent.

    Raises:
        ValueError: if any required column is missing from a row.
    """
    if not rows:
        return

    # Validate and fill timestamp
    enriched = []
    for row in rows:
        missing = [c for c in RESULT_COLUMNS if c not in row and c != "timestamp"]
        if missing:
            raise ValueError(f"Result row missing required columns: {missing}. Row: {row}")
        enriched.append({
            "timestamp": row.get("timestamp", _now_iso()),
            **{c: row[c] for c in RESULT_COLUMNS if c != "timestamp"},
        })

    df_new = pd.DataFrame(enriched, columns=RESULT_COLUMNS)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_PATH.exists()

    df_new.to_csv(RESULTS_PATH, mode="a", header=write_header, index=False)
    logger.info(
        "Appended %d result row(s) to %s", len(rows), RESULTS_PATH
    )


def build_result_row(
    experiment_id: str,
    dataset: str,
    model: str,
    task: str,
    fold: int | str,
    metric: str,
    value: float,
) -> dict:
    """Construct a single result dict with all required fields.

    Args:
        fold: integer fold index, or the string ``"pooled"`` for the OOF score.
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
    }
