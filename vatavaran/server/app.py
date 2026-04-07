"""FastAPI entrypoint for the Vatavaran environment."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import Body, Query, Request
from fastapi.responses import RedirectResponse
from openenv.core.env_server import create_app

from ..models import VatavaranAction, VatavaranObservation

from .evaluator import evaluate_prediction
from .rca_environment import VatavaranEnvironment

# Main OpenEnv app (session-aware instances managed by OpenEnv runtime).
app = create_app(
    VatavaranEnvironment,
    VatavaranAction,
    VatavaranObservation,
    env_name="vatavaran",
)

# Utility singleton for auxiliary HTTP endpoints.
_service_env = VatavaranEnvironment()


@app.get("/")
def root() -> RedirectResponse:
    """Redirect root to interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/tasks")
def tasks() -> dict[str, Any]:
    """List available tasks and action schema."""

    return _service_env.get_tasks_payload()


@app.api_route("/grader", methods=["GET", "POST"])
def grader(payload: dict | None = Body(default=None)) -> dict[str, Any]:
    """Return last score, or evaluate explicit prediction payload."""

    if payload and payload.get("prediction") and payload.get("scoring_points"):
        details = evaluate_prediction(payload["prediction"], payload["scoring_points"])
        return {
            "score": details["score"],
            "passed_criteria": details["passed_criteria"],
            "failed_criteria": details["failed_criteria"],
        }
    return _service_env.get_last_grader_result()


@app.get("/baseline")
def baseline(
    request: Request,
    mode: str = Query(default="inference_script"),
) -> dict[str, Any]:
    """Trigger baseline scoring and return difficulty-wise scores."""

    if mode == "rule_based":
        return {
            "mode": "rule_based",
            "result": _service_env.run_rule_based_baseline(),
        }
    if mode != "inference_script":
        return {
            "mode": mode,
            "status": "error",
            "message": "Unsupported mode. Use 'inference_script' or 'rule_based'.",
        }

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "inference.py"
    if not script_path.exists():
        return {
            "mode": "inference_script",
            "status": "error",
            "message": "inference.py not found at repo root.",
        }

    cmd = [sys.executable, str(script_path)]
    env = dict(os.environ)
    env.setdefault("API_BASE_URL", "https://router.huggingface.co/v1")
    env.setdefault("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    env.setdefault("RCA_BASE_URL", str(request.base_url).rstrip("/"))
    env.setdefault("RCA_USE_BASE_URL", "true")
    env.setdefault("RCA_EPISODE_COUNT", "3")

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    start_tasks = re.findall(r"^\[START\]\s+task=([^\s]+)", proc.stdout, flags=re.M)
    end_scores = [float(value) for value in re.findall(r"^\[END\].*score=([0-9]*\.?[0-9]+)", proc.stdout, flags=re.M)]
    rows = []
    for idx, score in enumerate(end_scores):
        task_name = start_tasks[idx] if idx < len(start_tasks) else f"episode_{idx+1}"
        difficulty = "unknown"
        if task_name.startswith("easy_"):
            difficulty = "easy"
        elif task_name.startswith("middle_"):
            difficulty = "middle"
        elif task_name.startswith("hard_"):
            difficulty = "hard"
        rows.append({"task": task_name, "difficulty": difficulty, "score": score})

    by_difficulty: dict[str, list[float]] = {"easy": [], "middle": [], "hard": []}
    for row in rows:
        if row["difficulty"] in by_difficulty:
            by_difficulty[row["difficulty"]].append(row["score"])
    difficulty_summary = {
        key: round(sum(values) / len(values), 2) if values else None
        for key, values in by_difficulty.items()
    }
    overall = round(sum(end_scores) / len(end_scores), 2) if end_scores else None

    return {
        "mode": "inference_script",
        "status": "ok" if proc.returncode == 0 else "error",
        "returncode": proc.returncode,
        "summary": {
            "episodes": rows,
            "by_difficulty": difficulty_summary,
            "overall_score": overall,
        },
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main():
    """Main entrypoint for local server execution."""

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
