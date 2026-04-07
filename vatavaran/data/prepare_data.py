"""Optional dev script: generate synthetic telemetry and a small tasks.json.

Not used by VatavaranEnvironment (runtime loads tasks from env_config and dataset_root).
"""

from __future__ import annotations

import csv
import json
import random
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

random.seed(7)

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_ROOT = REPO_ROOT / "data" / "synthetic_dev"
TELEMETRY_ROOT = SYNTHETIC_ROOT / "telemetry" / "Bank"

COMPONENTS = [
    "Tomcat01",
    "Tomcat02",
    "MG01",
    "MG02",
    "Mysql01",
    "Redis01",
]

KPI_CPU = "OSLinux-CPU_CPU_CPUCpuUtil"
KPI_MEM = "OSLinux-MEM_MEM_MemUsedPercent"
KPI_LAT = "OSLinux-NET_NET_NicLatency"

INCIDENTS = [
    {
        "date": "2021_03_05",
        "anomaly_dt": "2021-03-05 10:12:00",
        "component": "Tomcat01",
        "reason": "high CPU usage",
        "kpi": KPI_CPU,
        "boost": 70.0,
    },
    {
        "date": "2021_03_06",
        "anomaly_dt": "2021-03-06 14:20:00",
        "component": "MG02",
        "reason": "network latency",
        "kpi": KPI_LAT,
        "boost": 120.0,
    },
    {
        "date": "2021_03_07",
        "anomaly_dt": "2021-03-07 16:05:00",
        "component": "Mysql01",
        "reason": "high memory usage",
        "kpi": KPI_MEM,
        "boost": 40.0,
    },
]


def _base_value(kpi: str) -> float:
    if kpi == KPI_CPU:
        return 22.0
    if kpi == KPI_MEM:
        return 48.0
    return 8.0


def _write_date_telemetry(incident: dict):
    date = incident["date"]
    anomaly_ts = int(datetime.strptime(incident["anomaly_dt"], "%Y-%m-%d %H:%M:%S").timestamp())
    start_ts = anomaly_ts - 3600
    end_ts = anomaly_ts + 3600

    day_root = TELEMETRY_ROOT / date
    metric_dir = day_root / "metric"
    trace_dir = day_root / "trace"
    log_dir = day_root / "log"
    metric_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    metric_app_rows = []
    metric_container_rows = []
    trace_rows = []
    log_rows = []

    for ts in range(start_ts, end_ts + 1, 60):
        # Metric app rows.
        metric_app_rows.append(
            {
                "timestamp": ts,
                "rr": round(100 + random.uniform(-2, 2), 2),
                "sr": round(98 + random.uniform(-1, 1), 2),
                "cnt": int(18 + random.uniform(-3, 3)),
                "mrt": round(60 + random.uniform(-10, 10), 2),
                "tc": "ServiceTest1",
            }
        )

        # Metric container rows.
        for comp in COMPONENTS:
            for kpi in [KPI_CPU, KPI_MEM, KPI_LAT]:
                value = _base_value(kpi) + random.uniform(-3, 3)
                if comp == incident["component"] and kpi == incident["kpi"] and abs(ts - anomaly_ts) <= 300:
                    value += incident["boost"] + random.uniform(-4, 4)
                metric_container_rows.append(
                    {
                        "timestamp": ts,
                        "cmdb_id": comp,
                        "kpi_name": kpi,
                        "value": round(value, 4),
                    }
                )

        # Trace spans (ms).
        trace_id = f"trace-{date}-{ts}"
        span_a = f"{date}-A-{ts}"
        span_b = f"{date}-B-{ts}"
        span_c = f"{date}-C-{ts}"
        trace_rows.extend(
            [
                {
                    "timestamp": ts * 1000,
                    "cmdb_id": "apache01",
                    "parent_id": "",
                    "span_id": span_a,
                    "trace_id": trace_id,
                    "duration": 20 + int(random.uniform(0, 8)),
                },
                {
                    "timestamp": ts * 1000 + 5,
                    "cmdb_id": "MG02" if incident["component"] == "MG02" else "Tomcat02",
                    "parent_id": span_a,
                    "span_id": span_b,
                    "trace_id": trace_id,
                    "duration": 30 + int(random.uniform(0, 10)),
                },
                {
                    "timestamp": ts * 1000 + 10,
                    "cmdb_id": incident["component"],
                    "parent_id": span_b,
                    "span_id": span_c,
                    "trace_id": trace_id,
                    "duration": 45 + int(random.uniform(0, 16)),
                },
            ]
        )

        # Service logs.
        if abs(ts - anomaly_ts) <= 300:
            msg = (
                f"Anomaly detected around {incident['anomaly_dt']}; "
                f"component={incident['component']}; reason={incident['reason']}"
            )
        else:
            msg = f"Healthy service heartbeat at {datetime.fromtimestamp(ts, UTC)}"
        log_rows.append(
            {
                "log_id": uuid.uuid4().hex,
                "timestamp": ts,
                "cmdb_id": incident["component"],
                "log_name": "service.log",
                "value": msg,
            }
        )

    _write_csv(metric_dir / "metric_app.csv", metric_app_rows)
    _write_csv(metric_dir / "metric_container.csv", metric_container_rows)
    _write_csv(trace_dir / "trace_span.csv", trace_rows)
    _write_csv(log_dir / "log_service.csv", log_rows)


def _build_tasks() -> list[dict]:
    tasks = []
    for idx, incident in enumerate(INCIDENTS, start=1):
        date = incident["date"]
        dt = incident["anomaly_dt"]
        comp = incident["component"]
        reason = incident["reason"]
        window_start = (
            datetime.strptime(dt, "%Y-%m-%d %H:%M:%S") - timedelta(minutes=15)
        ).strftime("%Y-%m-%d %H:%M:%S")
        window_end = (
            datetime.strptime(dt, "%Y-%m-%d %H:%M:%S") + timedelta(minutes=15)
        ).strftime("%Y-%m-%d %H:%M:%S")

        tasks.append(
            {
                "task_id": f"easy_{idx}",
                "difficulty": "easy",
                "task_index": "task_3",
                "system": "Bank",
                "date": date,
                "instruction": (
                    f"Between {window_start} and {window_end}, one banking-service failure occurred. "
                    "Identify the root cause component only."
                ),
                "scoring_points": f"The only predicted root cause component is {comp}",
            }
        )
        tasks.append(
            {
                "task_id": f"middle_{idx}",
                "difficulty": "middle",
                "task_index": "task_6",
                "system": "Bank",
                "date": date,
                "instruction": (
                    f"During {window_start} to {window_end}, one failure happened. "
                    "Identify both the root cause component and reason."
                ),
                "scoring_points": (
                    f"The only predicted root cause component is {comp}\n"
                    f"The only predicted root cause reason is {reason}"
                ),
            }
        )
        tasks.append(
            {
                "task_id": f"hard_{idx}",
                "difficulty": "hard",
                "task_index": "task_7",
                "system": "Bank",
                "date": date,
                "instruction": (
                    f"Within {window_start} to {window_end}, diagnose the failure and provide "
                    "the exact root cause occurrence datetime, component, and reason."
                ),
                "scoring_points": (
                    f"The only root cause occurrence time is within 1 minutes (i.e., <=1min) of {dt}\n"
                    f"The only predicted root cause component is {comp}\n"
                    f"The only predicted root cause reason is {reason}"
                ),
            }
        )
    return tasks


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    TELEMETRY_ROOT.mkdir(parents=True, exist_ok=True)
    for incident in INCIDENTS:
        _write_date_telemetry(incident)

    tasks = _build_tasks()
    tasks_path = SYNTHETIC_ROOT / "tasks.json"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    with tasks_path.open("w", encoding="utf-8") as handle:
        json.dump({"tasks": tasks}, handle, indent=2)
    print(f"Wrote {len(tasks)} tasks to {tasks_path}")


if __name__ == "__main__":
    main()
