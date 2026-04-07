#!/usr/bin/env python3
"""Smoke-test task loading: VatavaranEnvironment task CSV and reset() match rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=4)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    processed = repo / "data" / "Bank_filtered" / "queries.csv"
    if not processed.is_file():
        print(
            f"Missing {processed}; run scripts/filter_bank_by_telemetry_day.py or provide tasks CSV",
            file=sys.stderr,
        )
        return 1

    df = pd.read_csv(processed)
    from vatavaran.server.rca_environment import VatavaranEnvironment

    env = VatavaranEnvironment()
    if len(env.tasks) != len(df):
        print(f"Mismatch: env has {len(env.tasks)} tasks, CSV has {len(df)}", file=sys.stderr)
        return 1

    end = min(args.end, len(df) - 1)
    for idx in range(args.start, end + 1):
        obs = env.reset(task_list_index=idx, seed=idx)
        row = df.iloc[idx]
        if obs.task_id != row["task_id"] or obs.difficulty != row["difficulty"]:
            print(
                f"FAIL idx={idx}: obs={obs.task_id},{obs.difficulty} row={row['task_id']},{row['difficulty']}",
                file=sys.stderr,
            )
            return 1
        print(f"ok idx={idx} task_id={obs.task_id} difficulty={obs.difficulty}")

    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
