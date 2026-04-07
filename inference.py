"""
Vatavaran OpenEnv inference script (task-scoped LLM agent loop).

MANDATORY / environment variables
- API_BASE_URL     LLM API base URL (default: OpenAI https://api.openai.com/v1).
- MODEL_NAME       Model id for inference (default: gpt-4o-mini).
- OPENAI_API_KEY or HF_TOKEN or API_KEY  API key for the LLM.

Optional
- IMAGE_NAME / LOCAL_IMAGE_NAME  Docker image for from_docker_image().
- RCA_ENV_MODE     Environment bootstrap mode: "client" (default) or "local".
- RCA_BASE_URL + RCA_USE_BASE_URL=true  Remote env server URL.
- RCA_BENCHMARK    Label for [START] line (default: vatavaran).
- RCA_MAX_STEPS    Upper bound on steps per episode (default: 32), capped by env max_steps.
- RCA_SEED         Optional int passed to reset(seed=...).
- RCA_TASK_ID      Optional task id override; if unset, /reset decides task.
- RCA_MESSAGE_TIMEOUT_S  Max seconds to wait for each env WebSocket reply (default: 600).
- RCA_WS_PING_INTERVAL   WebSocket keepalive ping interval in seconds (default: 60). Set to
                         "none" to disable client pings (can avoid keepalive timeouts on very
                         long server-side work; less ideal through proxies).
- RCA_WS_PING_TIMEOUT    Seconds to wait for pong after a ping (default: 120).

STDOUT: [START], one [STEP] per env.step(), then [END] (always, including on error).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from openai import OpenAI
from websockets.exceptions import ConnectionClosedError

from vatavaran import VatavaranAction, VatavaranEnv
from vatavaran.server.rca_environment import VatavaranEnvironment

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_NAME = os.getenv("IMAGE_NAME", "vatavaran") or os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("RCA_BENCHMARK", "vatavaran")
RCA_TASK_ID = (os.getenv("RCA_TASK_ID") or "").strip() or None


RCA_BASE_URL = "https://abmallick-vatavaran-env.hf.space"
RCA_USE_BASE_URL = (os.getenv("RCA_USE_BASE_URL") or "true").lower() == "true"
RCA_ENV_MODE = (os.getenv("RCA_ENV_MODE") or "client").strip().lower()
RCA_MAX_STEPS = int(os.getenv("RCA_MAX_STEPS", "4"))
RCA_SEED = os.getenv("RCA_SEED")


def _parse_message_timeout_s() -> float:
    return float(os.getenv("RCA_MESSAGE_TIMEOUT_S", "600"))


def _parse_ws_ping_interval() -> float | None:
    env = os.getenv("RCA_WS_PING_INTERVAL")
    if env is None:
        return None
    raw = env.strip().lower()
    if raw in ("none", "off", "disable"):
        return None
    return float(raw)


def _parse_ws_ping_timeout() -> float | None:
    env = os.getenv("RCA_WS_PING_TIMEOUT")
    if env is None:
        return 120.0
    raw = env.strip().lower()
    if raw == "none":
        return None
    return float(raw)

TEMPERATURE = float(os.getenv("RCA_TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("RCA_MAX_TOKENS", "2048"))
SUCCESS_SCORE_THRESHOLD = 0.5

_JSON_PARSE_RETRIES = 2
_RAW_LOG_CONVERSATION_PATH = os.getenv("RCA_CONVERSATION_LOG_PATH", "log_conversation.json")


def _resolve_log_path(raw_path: str) -> str:
    """Resolve conversation log path relative to repo root, not process cwd."""
    expanded = os.path.expanduser(raw_path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.join(REPO_ROOT, expanded)


LOG_CONVERSATION_PATH = _resolve_log_path(_RAW_LOG_CONVERSATION_PATH)

ALLOWED_ACTIONS = frozenset({"list_files", "execute_code", "submit_answer"})

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an agent solving a root cause analysis (RCA) task in a production system.
    You are provided the a time range of the incident and the telemetry data for that day. 
    The telemetry data is organized into three directories: metric, trace, and log.
    
    metric folder contains values and timestamps for various metrics at app level and container level.
    log folder contains failure logs for different components by timestamp and log name
    

    PATH RULES (critical): The sandbox process starts **inside** the incident-day telemetry directory
    (it already contains `metric/`, `trace/`, `log/`). For `list_files`, `content` must be a path
    **relative to that directory** — e.g. `"."`, `"metric"`, `"trace"`, `"log"`, or
    `"metric/metric_app.csv"`. Wrong: any path starting with `data/`, or repeating
    `.../telemetry/<date>/...` — that double-joins paths and fails. For `execute_code`, open and
    read files with the same relative paths (cwd is already the telemetry day folder).

    Each turn you must choose exactly one action by replying with a single JSON object only
    (no markdown fences, no extra text), with this shape:
    {"action_type":"<type>","content":"<string>","reasoning":"<string>"}

    action_type must be one of:
    - list_files — content is the path to list (e.g. "." or "metric").
    - execute_code — content is Python source to run in the task sandbox.
    - submit_answer — content is a JSON string for the final diagnosis (format required by the task).
    - reasoning — required short explanation for why this action is chosen. Include all the learnings you have made from the previous actions and observations which led to this action along with any other relevant information.


    Use the conversation history: previous assistant messages are your past actions; user messages
    labeled "Environment result" are the outcomes of those actions (like tool returns).
    """
).strip()


def _clean_action(action: str, limit: int = 140) -> str:
    return action
    # action = action.replace("\n", " ").strip()
    # if len(action) <= limit:
    #     return action
    # return action[: limit - 3] + "..."


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={_clean_action(action)} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{value:.2f}" for value in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _extract_json_object(text: str) -> str:
    raw = (text or "").strip()
    if "```" in raw:
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.S)
        if fence:
            raw = fence.group(1).strip()
    return raw


def _parse_action_json(text: str) -> VatavaranAction:
    raw = _extract_json_object(text)
    data: Any = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    action_type = data.get("action_type")
    content = data.get("content", "")
    reasoning = data.get("reasoning")
    if action_type not in ALLOWED_ACTIONS:
        raise ValueError(f"Invalid action_type: {action_type!r}")
    if not isinstance(content, str):
        content = json.dumps(content) if content is not None else ""
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("Missing or empty `reasoning`")
    return VatavaranAction(action_type=action_type, content=content, reasoning=reasoning.strip())


def _score_from_submit_result(result_text: str, fallback: float) -> float:
    try:
        payload = json.loads(result_text)
        score = float(payload.get("score", fallback))
    except Exception:
        score = fallback
    return min(max(score, 0.01), 0.99)


def _env_ws_kwargs() -> dict[str, Any]:
    return {
        "message_timeout_s": _parse_message_timeout_s(),
        "ws_ping_interval": _parse_ws_ping_interval(),
        "ws_ping_timeout": _parse_ws_ping_timeout(),
    }


@dataclass
class _EnvStepResult:
    observation: Any
    reward: float | None
    done: bool


class LocalVatavaranEnvAdapter:
    """Async-compatible adapter for in-process VatavaranEnvironment."""

    def __init__(self, env: VatavaranEnvironment):
        self._env = env

    async def reset(self, **kwargs: Any) -> _EnvStepResult:
        obs = self._env.reset(**kwargs)
        return _EnvStepResult(observation=obs, reward=getattr(obs, "reward", 0.0), done=bool(obs.done))

    async def step(self, action: VatavaranAction) -> _EnvStepResult:
        obs = self._env.step(action)
        return _EnvStepResult(observation=obs, reward=getattr(obs, "reward", 0.0), done=bool(obs.done))

    async def close(self) -> None:
        return None


async def _build_env() -> Any:
    if RCA_ENV_MODE == "local":
        return LocalVatavaranEnvAdapter(VatavaranEnvironment())

    ws_kw = _env_ws_kwargs()
    if RCA_USE_BASE_URL and RCA_BASE_URL:
        return VatavaranEnv(base_url=RCA_BASE_URL, **ws_kw)
    if IMAGE_NAME:
        return await VatavaranEnv.from_docker_image(IMAGE_NAME, **ws_kw)
    return VatavaranEnv(base_url="http://localhost:8000", **ws_kw)


def _safe_reward(value: float | None) -> float:
    return min(max(float(value or 0.0), 0.01), 0.99)


def _initial_user_message(
    task_id: str,
    task_description: str,
    domain_knowledge: str,
    max_steps: int,
) -> str:
    return textwrap.dedent(
        f"""
        Task id: {task_id}

        Task description:
        {task_description}

        When you choose execute_code, you have to write the code which is passed to the sandbox to execute.
        Use the code to analyze the telemetry data and find the root cause.

        Use list_files to figure out what files are available to you.


        First try to understand the overall structure of the telemetry data by writing a script analyze the data distributions in all the files. 
        Your job is to first figure out what anomalous behavior was observed in the telemetry data in the query time range. 
        Write scripts to analyze metrics, logs and traces to do so and corroborate across different modalities where relevant. 

        One you figure our the anomalous behaviour, try to find the root cause by tracking which component started exhibiting the anomalous behaviour first or started getting errors in logs and traces first. 
        The root cause component should be one of the components in cmdb_id column of the logs or traces.

        Root cause can consist of three elements: component, reason and datetime of occurrence.

        It may not always be possible to find the reason and datetime of occurrence. In that case, you can leave it blank. Always try to find the most plausible component at least. 

        Once you have the root cause, use the submit_answer action to submit your answer.

        the format of the answer is which should be passed as the content of the submit_answer action:
   
        "root cause occurrence datetime":"<datetime>"
        "root cause component":"<component>"
        "root cause reason":"<reason>"

        Domain knowledge (reference):
        {domain_knowledge}


        Respond with your first action as JSON only: {{"action_type":"...","content":"...","reasoning":"..."}}

        All the timestamps are in UTC+8 timezone. So always use that for convertions and comparisons and not the local timezone.
        Make sure that root cause component is a valid component name from the domain knowledge.
        Make sure that root cause reason is a valid reason from the domain knowledge.
        DO NOT prematurely jump to conclusions. 
        Run anomaly detection, causal analysis, time series analysis and any other relevant algorithms to find the root cause.
        Make sure you understand the data content and format before you start implementing analysis code to make sure you dont write buggy code.
        """
    ).strip()


def _env_result_user_message(obs: Any, reward: float, done: bool) -> str:
    err = obs.last_action_error if obs.last_action_error else "null"
    body = textwrap.dedent(
        f"""
        Environment result:
        success: {str(obs.success).lower()}
        reward: {reward:.4f}
        done: {str(done).lower()}
        last_action_error: {err}

        Result (main output):
        {obs.result}
        """
    ).strip()
    return body


def _append_conversation_event(
    path: str,
    task_id: str,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    event: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "content": content,
    }
    if metadata:
        event["metadata"] = metadata

    payload: dict[str, Any]
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except Exception:
            payload = {"tasks": {}}
    else:
        payload = {"tasks": {}}

    tasks = payload.setdefault("tasks", {})
    task_events = tasks.setdefault(task_id, [])
    task_events.append(event)

    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _reset_conversation_log(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump({"tasks": {}}, fp, indent=2)


def get_model_action(client: OpenAI, messages: list[dict[str, str]]) -> str:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        #max_tokens=MAX_TOKENS,
        stream=False,
    )
    return (completion.choices[0].message.content or "").strip()


async def _run_episode(client: OpenAI, env: VatavaranEnv, task_id: str) -> None:
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    seed: int | None = None
    if RCA_SEED is not None and str(RCA_SEED).strip() != "":
        try:
            seed = int(RCA_SEED)
        except ValueError:
            seed = None

    reset_kwargs: dict[str, Any] = {}
    if task_id:
        reset_kwargs["task_id"] = task_id
    if seed is not None:
        reset_kwargs["seed"] = seed

    try:
        reset_result = await env.reset(**reset_kwargs)
        obs = reset_result.observation
        task_name = obs.task_id or task_id or "episode"
        env_max = max(1, int(obs.max_steps or RCA_MAX_STEPS))
        step_limit = min(env_max, RCA_MAX_STEPS)

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _initial_user_message(
                    task_name,
                    obs.task_description,
                    obs.domain_knowledge,
                    step_limit,
                ),
            },
        ]
        _append_conversation_event(
            LOG_CONVERSATION_PATH,
            task_name,
            "system",
            SYSTEM_PROMPT,
            {"event_type": "episode_start"},
        )
        _append_conversation_event(
            LOG_CONVERSATION_PATH,
            task_name,
            "user",
            messages[1]["content"],
            {"event_type": "initial_instruction"},
        )

        if reset_result.done:
            score = _score_from_submit_result(obs.result, 0.0)
            success = score >= SUCCESS_SCORE_THRESHOLD
            return

        for step in range(1, step_limit + 1):
            parse_failures = 0
            action: VatavaranAction | None = None
            assistant_text = ""

            while action is None and parse_failures < _JSON_PARSE_RETRIES:
                request_payload = [
                    {"role": str(msg.get("role", "")), "content": str(msg.get("content", ""))}
                    for msg in messages
                ]
                _append_conversation_event(
                    LOG_CONVERSATION_PATH,
                    task_name,
                    "system",
                    json.dumps(request_payload, ensure_ascii=False, indent=2),
                    {
                        "event_type": "llm_request",
                        "step": step,
                        "attempt": parse_failures + 1,
                        "message_count": len(request_payload),
                    },
                )
                try:
                    assistant_text = get_model_action(client, messages)
                    _append_conversation_event(
                        LOG_CONVERSATION_PATH,
                        task_name,
                        "assistant",
                        assistant_text or "(empty)",
                        {
                            "event_type": "llm_response",
                            "step": step,
                            "attempt": parse_failures + 1,
                        },
                    )
                except Exception as exc:
                    parse_failures += 1
                    messages.append({"role": "assistant", "content": assistant_text or "(empty)"})
                    _append_conversation_event(
                        LOG_CONVERSATION_PATH,
                        task_name,
                        "assistant",
                        assistant_text or "(empty)",
                        {"event_type": "action_parse_retry", "parse_failures": parse_failures},
                    )
                    _append_conversation_event(
                        LOG_CONVERSATION_PATH,
                        task_name,
                        "system",
                        f"Model API call failed: {exc}",
                        {
                            "event_type": "api_call_failure",
                            "step": step,
                            "parse_failures": parse_failures,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        },
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Invalid or missing JSON action. Reply with a single JSON object only: "
                                '{"action_type":"list_files"|"execute_code"|"submit_answer","content":"...","reasoning":"..."} '
                                f"Parse error: {exc}"
                            ),
                        }
                    )
                    _append_conversation_event(
                        LOG_CONVERSATION_PATH,
                        task_name,
                        "user",
                        messages[-1]["content"],
                        {"event_type": "parse_feedback"},
                    )
                    continue

                try:
                    action = _parse_action_json(assistant_text)
                except Exception as exc:
                    parse_failures += 1
                    messages.append({"role": "assistant", "content": assistant_text or "(empty)"})
                    _append_conversation_event(
                        LOG_CONVERSATION_PATH,
                        task_name,
                        "assistant",
                        assistant_text or "(empty)",
                        {"event_type": "action_parse_retry", "parse_failures": parse_failures},
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Invalid or missing JSON action. Reply with a single JSON object only: "
                                '{"action_type":"list_files"|"execute_code"|"submit_answer","content":"...","reasoning":"..."} '
                                f"Parse error: {exc}"
                            ),
                        }
                    )
                    _append_conversation_event(
                        LOG_CONVERSATION_PATH,
                        task_name,
                        "user",
                        messages[-1]["content"],
                        {"event_type": "parse_feedback"},
                    )

            if action is None:
                print("No action found in step", step, flush=True)
                continue

            messages.append({"role": "assistant", "content": assistant_text or json.dumps(action.model_dump())})
            _append_conversation_event(
                LOG_CONVERSATION_PATH,
                task_name,
                "assistant",
                messages[-1]["content"],
                {
                    "event_type": "agent_action",
                    "step": step,
                    "action_type": action.action_type,
                    "reasoning": action.reasoning,
                },
            )
            try:
                step_result = await env.step(action)
            except ConnectionClosedError as exc:
                print(f"[ERROR] env.step connection closed at step {step}: {exc}", flush=True)
                _append_conversation_event(
                    LOG_CONVERSATION_PATH,
                    task_name,
                    "system",
                    str(exc),
                    {
                        "event_type": "env_step_connection_closed",
                        "step": step,
                        "error_type": type(exc).__name__,
                    },
                )
                break

            obs = step_result.observation
            reward = _safe_reward(step_result.reward)
            done = bool(step_result.done)
            err = obs.last_action_error

            rewards.append(reward)
            steps_taken = step

            action_str = json.dumps(
                {
                    "action_type": action.action_type,
                    "content": action.content,
                    "reasoning": action.reasoning,
                }
            )
            log_step(step=step, action=action_str, reward=reward, done=done, error=err)

            messages.append(
                {"role": "user", "content": _env_result_user_message(obs, reward, done)}
            )
            _append_conversation_event(
                LOG_CONVERSATION_PATH,
                task_name,
                "user",
                messages[-1]["content"],
                {
                    "event_type": "environment_result",
                    "step": step,
                    "reward": reward,
                    "done": done,
                },
            )

            if done:
                score = _score_from_submit_result(
                    obs.result,
                    min(max(reward, 0.0), 1.0),
                )
                success = score >= SUCCESS_SCORE_THRESHOLD
                break

        if steps_taken > 0 and not success:
            score = _score_from_submit_result(
                obs.result,
                min(max(rewards[-1], 0.0), 1.0) if rewards else 0.0,
            )

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=min(max(score, 0.01), 0.99),
            rewards=rewards,
        )


async def main() -> None:
    _reset_conversation_log(LOG_CONVERSATION_PATH)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "missing-key")
    env = await _build_env()

    try:
        await _run_episode(client, env, RCA_TASK_ID)
    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
