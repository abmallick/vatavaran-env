#!/usr/bin/env python3
"""Build data/Bank_filtered from data/Bank for one telemetry calendar day.

Keeps only rows whose instruction/scoring text mentions the selected day, so
`queries.csv` matches the chosen telemetry folder.

Writes queries.csv (same columns as query_processed), aligned record.csv rows, and
copies data/Bank/telemetry/<DAY>/ into Bank_filtered/telemetry/.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_MONTH_NAME_DATE_RE = re.compile(
    r"\b("
    r"January|February|March|April|May|June|July|August|September|October|November|December"
    r")\s+(\d{1,2}),\s+(\d{4})\b"
)

def _extract_dates_from_text(text: object) -> set[str]:
    if text is None:
        return set()
    s = str(text)
    out: set[str] = set()
    for month, day, year in _MONTH_NAME_DATE_RE.findall(s):
        dt = datetime.strptime(f"{month} {day}, {year}", "%B %d, %Y")
        out.add(dt.strftime("%Y_%m_%d"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--day",
        default="2021_03_05",
        help="Telemetry folder name under telemetry/ (default: 2021_03_05)",
    )
    parser.add_argument(
        "--bank-dir",
        type=Path,
        default=None,
        help="Bank dataset directory (default: <repo>/data/Bank)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/data/Bank_filtered)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    bank_dir = args.bank_dir or (repo_root / "data" / "Bank")
    out_dir = args.out or (repo_root / "data" / "Bank_filtered")

    folder_day = args.day.strip()
    try:
        datetime.strptime(folder_day, "%Y_%m_%d")
    except ValueError:
        sys.exit(f"Invalid --day (expected YYYY_MM_DD): {folder_day!r}")

    query_path = bank_dir / "query_processed.csv"
    record_path = bank_dir / "record.csv"
    telemetry_src = bank_dir / "telemetry" / folder_day

    if not query_path.is_file():
        sys.exit(f"Missing {query_path}")
    if not record_path.is_file():
        sys.exit(f"Missing {record_path}")
    if not telemetry_src.is_dir():
        sys.exit(f"Missing telemetry directory: {telemetry_src}")

    q_df = pd.read_csv(query_path)
    r_df = pd.read_csv(record_path)
    if len(q_df) != len(r_df):
        sys.exit(
            f"Row count mismatch: query_processed ({len(q_df)}) vs record ({len(r_df)})"
        )

    if "instruction" not in q_df.columns:
        sys.exit("query_processed.csv must contain instruction column")
    mask = []
    for _, row in q_df.iterrows():
        row_dates = _extract_dates_from_text(row["instruction"])
        mask.append(folder_day in row_dates)

    filtered_q = q_df.loc[mask].reset_index(drop=True)
    filtered_r = r_df.loc[mask].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    filtered_q.to_csv(out_dir / "queries.csv", index=False)
    filtered_r.to_csv(out_dir / "record.csv", index=False)

    telemetry_dst = out_dir / "telemetry" / folder_day
    if telemetry_dst.exists():
        shutil.rmtree(telemetry_dst)
    shutil.copytree(telemetry_src, telemetry_dst)

    n_out = len(filtered_q)
    print(f"Wrote {n_out} rows to {out_dir / 'queries.csv'} and {out_dir / 'record.csv'}")
    print(f"Copied telemetry to {telemetry_dst}")


if __name__ == "__main__":
    main()
