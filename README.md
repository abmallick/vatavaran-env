---
title: Vatavaran
emoji: 🍃
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# Vatavaran Environment (OpenEnv)

Vatavaran is an OpenEnv-compatible benchmark environment for cloud root cause analysis (RCA).  
It converts an RCA workflow into an interactive environment where an external agent performs telemetry investigation via `reset()`, `step()`, and `state()`.

The environment simulates realistic DevOps RCA work on metrics, traces, and logs using a persistent IPython sandbox.

## Motivation

Most toy environments do not reflect real RCA workflows. This environment targets:

- multi-step diagnosis over heterogeneous telemetry,
- strict, deterministic grading of RCA outputs,
- partial progress rewards for agent training,
- reproducible evaluation across easy/middle/hard tasks.

## OpenEnv Compliance

This project includes:

- typed Pydantic models (`VatavaranAction`, `VatavaranObservation`, `VatavaranState`),
- `reset()`, `step()`, and `state`,
- `openenv.yaml`,
- root `server/app.py` compatibility entrypoint,
- Dockerfile and lockfile (`uv.lock`),
- additional hackathon endpoints: `/tasks`, `/grader`, `/baseline`.

Validation command:

```bash
.venv/bin/openenv validate
```

## Project Layout

```text
├── data/
│   └── Bank_filtered/                  # repo-root dataset (env_config dataset_root)
│       ├── queries.csv
│       ├── record.csv
│       └── telemetry/{YYYY_MM_DD}/...
├── openenv.yaml
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── inference.py
├── vatavaran/
│   ├── __init__.py
│   ├── client.py
│   ├── models.py
│   ├── config/
│   │   ├── env_config.yaml
│   │   └── reward_config.yaml
│   ├── data/
│   │   └── prepare_data.py             # optional dev generator (not used by the server)
│   └── server/
│       ├── app.py
│       ├── rca_environment.py
│       ├── code_sandbox.py
│       ├── evaluator.py
│       ├── reward_engine.py
│       └── domain_knowledge.py
└── server/
    └── app.py
```

## Action Space

`VatavaranAction`:

- `action_type`: `"execute_code" | "list_files" | "submit_answer"`
- `content`: payload string (code, path, or answer JSON)

## Observation Space

`VatavaranObservation` includes:

- `result`: textual output from action execution,
- `success`: action success flag,
- `last_action_error`: error message or `null`,
- `task_id`, `task_description`, `difficulty`,
- `domain_knowledge`: schema + candidate lists,
- `step_count`, `max_steps`,
- OpenEnv reward/done fields.

## Tasks and Difficulty

Tasks are loaded from the CSV path in `vatavaran/config/env_config.yaml` (`tasks.path`, default `data/Bank_filtered/queries.csv`). Telemetry for each episode comes from `tasks.dataset_root` (default `data/Bank_filtered`) under `telemetry/<YYYY_MM_DD>/`. Override paths with `VATAVARAN_TASKS_FILE` and `VATAVARAN_DATASET_ROOT` if needed.

Difficulty follows OpenRCA rules on `task_index` (e.g. **Easy** `task_3`: component only; **Middle** `task_6`: component + reason; **Hard** `task_7`: datetime + component + reason).

## Grader

The grader (`vatavaran/server/evaluator.py`) is RCA-style:

- component/reason: exact string match,
- datetime: within 60 seconds,
- permutation-aware matching for multi-root-cause formatting,
- deterministic score in `[0.0, 1.0]`.

You can call:

- `GET /grader` for last episode grade,
- `POST /grader` with `{ "prediction": "...", "scoring_points": "..." }` for explicit grading.

## Reward Shaping (Config Driven)

All reward logic is parameterized in `vatavaran/config/reward_config.yaml`:

- code execution success reward / error penalty,
- step efficiency penalty,
- multi-modal exploration bonus (`metric`/`trace`/`log`),
- cross-validation bonus,
- final answer score weight,
- per-difficulty max step limits.

The runtime/sandbox policy is in `vatavaran/config/env_config.yaml`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e .
```

Ensure the dataset in `data/Bank_filtered/` exists (see `env_config.yaml`: `queries.csv`, `record.csv`, and `telemetry/`). To build it from the full Bank export, use `scripts/filter_bank_by_telemetry_day.py`.

## Run Locally

Start server:

```bash
.venv/bin/uvicorn vatavaran.server.app:app --host 127.0.0.1 --port 8000
```

Basic checks:

```bash
curl -s -X POST -H "Content-Type: application/json" -d '{}' http://127.0.0.1:8000/reset
curl -s http://127.0.0.1:8000/tasks
curl -s http://127.0.0.1:8000/baseline
```

## Baseline Inference Script

The required script is root-level `inference.py`.

Required env vars:

- `API_BASE_URL` (default: `https://router.huggingface.co/v1`)
- `MODEL_NAME` (default: `Qwen/Qwen2.5-72B-Instruct`)
- `HF_TOKEN` or `API_KEY`
- `IMAGE_NAME` or `LOCAL_IMAGE_NAME` when using `from_docker_image()`

Local run against a running server:

```bash
RCA_USE_BASE_URL=true RCA_BASE_URL=http://127.0.0.1:8000 .venv/bin/python inference.py
```

The script emits only:

- `[START] ...`
- `[STEP] ...`
- `[END] ...`

## Reproducible Baseline Scores

Scores depend on the loaded task CSV, model, and agent policy. Tune `RCA_EPISODE_COUNT` and task selection env vars to match your evaluation harness.

## Docker

Build:

```bash
docker build -t vatavaran .
```

Run:

```bash
docker run --rm -p 8000:8000 vatavaran
```

## Pre-Submission Validation

Use your provided checker:

```bash
chmod +x pre-validation.sh
./pre-validation.sh <your_hf_space_url> .
```

This checks:

1. `POST /reset` returns `200`,
2. Docker build passes,
3. `openenv validate` passes.
