"""OpenRCA-style difficulty bands from task_index strings (e.g. task_3)."""

from __future__ import annotations

import re

_TASK_INDEX_RE = re.compile(r"^task_(\d+)$", re.IGNORECASE)


def parse_openrca_task_number(task_index: str | None) -> int | None:
    """Parse N from ``task_N``; return None if missing or invalid."""
    if not task_index or not isinstance(task_index, str):
        return None
    match = _TASK_INDEX_RE.match(task_index.strip())
    if not match:
        return None
    return int(match.group(1))


def difficulty_from_openrca_number(n: int) -> str:
    """Same thresholds as OpenRCA ``run_agent_standard.py`` (task_id from task_<N>)."""
    if n <= 3:
        return "easy"
    if n <= 6:
        return "middle"
    if n <= 7:
        return "hard"
    return "hard"


def difficulty_from_task_index(task_index: str | None) -> str:
    """Derive difficulty from ``task_index``; defaults to ``easy`` if unparseable."""
    n = parse_openrca_task_number(task_index)
    if n is None:
        return "easy"
    return difficulty_from_openrca_number(n)
