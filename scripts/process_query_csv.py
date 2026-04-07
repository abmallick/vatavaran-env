#!/usr/bin/env python3
"""Read query.csv and write query_processed.csv with task_id and difficulty columns.

task_id does not encode difficulty (e.g. Bank_00000). difficulty follows OpenRCA rules
on task_<N>: N<=3 easy, N<=6 middle, N<=7 hard (see vatavaran/openrca_difficulty.py).
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


def _load_openrca_difficulty():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "vatavaran" / "openrca_difficulty.py"
    spec = importlib.util.spec_from_file_location("openrca_difficulty", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_diff = _load_openrca_difficulty()
difficulty_from_task_index = _diff.difficulty_from_task_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "query_csv",
        type=Path,
        nargs="?",
        default=None,
        help="Path to query.csv (default: <repo>/data/Bank/query.csv; use Bank_filtered after filtering)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: next to input as query_processed.csv)",
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        default=None,
        help="Prefix for task_id (default: parent folder name, e.g. Bank)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    query_path = args.query_csv or (repo_root / "data" / "Bank" / "query.csv")
    if not query_path.is_file():
        raise FileNotFoundError(f"Missing query file: {query_path}")

    out_path = args.output or (query_path.parent / "query_processed.csv")
    dataset_key = args.dataset_key or query_path.parent.name

    df = pd.read_csv(query_path)
    required = {"task_index", "instruction", "scoring_points"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"query.csv missing columns: {sorted(missing)}")

    rows = []
    for i, row in df.iterrows():
        task_index = str(row["task_index"]).strip()
        difficulty = difficulty_from_task_index(task_index)
        task_id = f"{dataset_key}_{int(i):05d}"
        rows.append(
            {
                "task_id": task_id,
                "difficulty": difficulty,
                "task_index": task_index,
                "instruction": row["instruction"],
                "scoring_points": row["scoring_points"],
            }
        )

    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
