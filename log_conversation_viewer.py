from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st


DEFAULT_LOG_PATH = "log_conversation.json"


@dataclass
class ConversationEvent:
    timestamp: str
    role: str
    content: str
    metadata: dict[str, Any]

    @property
    def step(self) -> int | None:
        value = self.metadata.get("step")
        if isinstance(value, int):
            return value
        return None

    @property
    def event_type(self) -> str:
        value = self.metadata.get("event_type")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return "unknown"

    @property
    def timestamp_dt(self) -> datetime | None:
        raw = self.timestamp.strip()
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None

    @property
    def reasoning(self) -> str:
        value = self.metadata.get("reasoning")
        if isinstance(value, str) and value.strip():
            return value.strip()

        # Backward-compatible fallback for logs where reasoning only lived in JSON content.
        try:
            payload = json.loads(self.content)
        except Exception:
            return ""
        if isinstance(payload, dict):
            raw = payload.get("reasoning")
            if isinstance(raw, str):
                return raw.strip()
        return ""


def _load_conversation(path: Path) -> dict[str, list[ConversationEvent]]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    tasks_payload = payload.get("tasks", {})
    if not isinstance(tasks_payload, dict):
        raise ValueError("`tasks` must be an object mapping task ids to arrays.")

    tasks: dict[str, list[ConversationEvent]] = {}
    for task_id, events in tasks_payload.items():
        if not isinstance(events, list):
            continue
        normalized: list[ConversationEvent] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            normalized.append(
                ConversationEvent(
                    timestamp=str(event.get("timestamp", "")),
                    role=str(event.get("role", "unknown")),
                    content=str(event.get("content", "")),
                    metadata=event.get("metadata", {}) if isinstance(event.get("metadata"), dict) else {},
                )
            )
        tasks[str(task_id)] = normalized
    return tasks


def _extract_step_from_content(content: str) -> int | None:
    # Parse "step: <n>" style text in environment result messages.
    marker = "step:"
    lower = content.lower()
    idx = lower.find(marker)
    if idx < 0:
        return None
    start = idx + len(marker)
    digits: list[str] = []
    for ch in content[start:]:
        if ch.isdigit():
            digits.append(ch)
        elif digits:
            break
    if not digits:
        return None
    return int("".join(digits))


def _extract_score_from_content(content: str) -> float | None:
    # JSON-like score: "score": 0.75
    json_match = re.search(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)', content)
    if json_match:
        try:
            return float(json_match.group(1))
        except ValueError:
            pass

    # Text score: score: 0.75
    text_match = re.search(r"\bscore\s*:\s*(-?\d+(?:\.\d+)?)", content, re.IGNORECASE)
    if text_match:
        try:
            return float(text_match.group(1))
        except ValueError:
            pass
    return None


def _event_score(event: ConversationEvent, step_to_score: dict[int, float]) -> float | None:
    for key in ("score", "reward"):
        value = event.metadata.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    parsed = _extract_score_from_content(event.content)
    if parsed is not None:
        return parsed

    event_step = event.step
    if event_step is not None:
        return step_to_score.get(event_step)
    return None


def _build_step_score_map(events: list[ConversationEvent]) -> dict[int, float]:
    mapping: dict[int, float] = {}
    for event in events:
        step = event.step
        if step is None:
            step = _extract_step_from_content(event.content)
        if step is None:
            continue
        score = _event_score(event, {})
        if score is not None:
            mapping[step] = score
    return mapping


def _filter_events(
    events: list[ConversationEvent],
    selected_roles: list[str],
    selected_event_types: list[str],
    step_min: int,
    step_max: int,
    search_text: str,
    score_min: float,
    score_max: float,
    include_no_score: bool,
    step_to_score: dict[int, float],
) -> list[ConversationEvent]:
    q = search_text.lower().strip()
    filtered: list[ConversationEvent] = []
    for event in events:
        if selected_roles and event.role not in selected_roles:
            continue
        if selected_event_types and event.event_type not in selected_event_types:
            continue

        event_step = event.step
        if event_step is None:
            event_step = _extract_step_from_content(event.content)
        if event_step is not None and not (step_min <= event_step <= step_max):
            continue

        score = _event_score(event, step_to_score)
        if score is None:
            if not include_no_score:
                continue
        elif not (score_min <= score <= score_max):
            continue

        if q:
            haystack = f"{event.role}\n{event.event_type}\n{event.content}".lower()
            if q not in haystack:
                continue
        filtered.append(event)
    return filtered


def _is_llm_message(event: ConversationEvent) -> bool:
    """
    True when an event is part of the model-facing conversation,
    not an environment/tool execution result.
    """
    if event.event_type == "environment_result":
        return False

    # Safety fallback for older logs without reliable metadata.
    return "environment result:" not in event.content.lower()


def _try_parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def _timestamp_label(event: ConversationEvent) -> str:
    if event.timestamp_dt is None:
        return event.timestamp or "unknown-time"
    return event.timestamp_dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def _render_event(event: ConversationEvent, idx: int, score: float | None) -> None:
    role_color = {
        "assistant": "#e8f3ff",
        "user": "#f7f7f8",
        "system": "#fff8e6",
    }.get(event.role, "#f3f3f3")
    label = f"{event.role} - {event.event_type}"
    if event.step is not None:
        label += f" - step {event.step}"
    if score is not None:
        label += f" - score {score:.4f}"

    st.markdown(
        (
            f"<div style='background:{role_color};padding:0.75rem 1rem;border-radius:0.6rem;"
            "margin-bottom:0.75rem;border:1px solid #d7d7d7;'>"
            f"<div style='font-size:0.85rem;color:#555;'><b>{idx + 1}. {label}</b> "
            f"<span style='margin-left:0.5rem'>{_timestamp_label(event)}</span></div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    if event.event_type == "llm_request":
        st.markdown("**LLM input (messages passed in)**")
        parsed = _try_parse_json(event.content)
        if isinstance(parsed, list):
            st.json(parsed)
        elif event.content.strip():
            st.code(event.content, language="json", wrap_lines=True)
        else:
            st.caption("(empty request)")
    elif event.event_type == "llm_response":
        st.markdown("**LLM raw output (response text)**")
        if event.content.strip():
            st.code(event.content, language="json", wrap_lines=True)
        else:
            st.caption("(empty response)")
    else:
        if event.content.strip():
            st.code(event.content, language="text", wrap_lines=True)
        else:
            st.caption("(empty message)")

    if event.event_type == "agent_action" and event.reasoning:
        st.markdown("**Reasoning**")
        st.code(event.reasoning, language="text", wrap_lines=True)

    if event.metadata:
        with st.expander("metadata", expanded=False):
            st.json(event.metadata)


def main() -> None:
    st.set_page_config(page_title="Conversation Log Viewer", page_icon=":speech_balloon:", layout="wide")
    st.title("Conversation Log Viewer")
    st.caption("Inspect agent and environment turns from log_conversation.json")

    default_path = str(Path.cwd() / DEFAULT_LOG_PATH)
    log_path_input = st.sidebar.text_input("Log file path", value=default_path)
    path = Path(log_path_input).expanduser()

    if not path.exists():
        st.error(f"Log file not found: {path}")
        st.stop()

    try:
        tasks = _load_conversation(path)
    except Exception as exc:
        st.error(f"Failed to parse log file: {exc}")
        st.stop()

    if not tasks:
        st.warning("No task conversations found in the log file.")
        st.stop()

    task_ids = sorted(tasks.keys())
    selected_task = st.sidebar.selectbox("Task", task_ids, index=0)
    events = tasks.get(selected_task, [])

    if not events:
        st.info("No events for this task.")
        st.stop()

    step_to_score = _build_step_score_map(events)

    all_roles = sorted({event.role for event in events})
    all_event_types = sorted({event.event_type for event in events})
    step_values = sorted({event.step for event in events if event.step is not None})

    view_mode = st.sidebar.radio(
        "View",
        options=["Full conversation", "LLM messages only"],
        index=0,
        help="Show all events, or only messages exchanged with the LLM.",
    )
    if view_mode == "LLM messages only":
        llm_io_events = [
            event
            for event in events
            if event.event_type in {"llm_request", "llm_response"}
        ]
        if llm_io_events:
            events = llm_io_events
        else:
            # Backward compatibility for logs generated before explicit llm_request/llm_response.
            events = [event for event in events if _is_llm_message(event)]
        if not events:
            st.info("No LLM-only messages found for this task.")
            st.stop()
        all_roles = sorted({event.role for event in events})
        all_event_types = sorted({event.event_type for event in events})
        step_values = sorted({event.step for event in events if event.step is not None})

    selected_roles = st.sidebar.multiselect("Role filter", options=all_roles, default=all_roles)
    selected_event_types = st.sidebar.multiselect(
        "Event type filter",
        options=all_event_types,
        default=all_event_types,
    )

    event_scores = [_event_score(event, step_to_score) for event in events]
    numeric_scores = sorted({score for score in event_scores if score is not None})
    if numeric_scores:
        score_min_all, score_max_all = float(min(numeric_scores)), float(max(numeric_scores))
        if score_min_all < score_max_all:
            score_range = st.sidebar.slider(
                "Score range",
                min_value=score_min_all,
                max_value=score_max_all,
                value=(score_min_all, score_max_all),
                step=0.0001,
            )
        else:
            st.sidebar.caption(f"Score range: fixed at {score_min_all:.4f}")
            score_range = (score_min_all, score_max_all)
        include_no_score = st.sidebar.checkbox("Include events with no score", value=True)
    else:
        score_range = (-1e9, 1e9)
        include_no_score = True

    if step_values:
        min_step, max_step = min(step_values), max(step_values)
        if min_step < max_step:
            step_range = st.sidebar.slider(
                "Step range",
                min_value=min_step,
                max_value=max_step,
                value=(min_step, max_step),
            )
        else:
            st.sidebar.caption(f"Step range: fixed at {min_step}")
            step_range = (min_step, max_step)
    else:
        step_range = (0, 10_000)

    search_text = st.sidebar.text_input("Search text", value="", help="Match against role, event type, and content.")
    newest_first = st.sidebar.checkbox("Newest first", value=False)
    show_stats = st.sidebar.checkbox("Show summary stats", value=True)

    filtered = _filter_events(
        events=events,
        selected_roles=selected_roles,
        selected_event_types=selected_event_types,
        step_min=step_range[0],
        step_max=step_range[1],
        search_text=search_text,
        score_min=score_range[0],
        score_max=score_range[1],
        include_no_score=include_no_score,
        step_to_score=step_to_score,
    )
    if newest_first:
        filtered = list(reversed(filtered))

    header_left, header_right = st.columns([2, 1])
    with header_left:
        st.subheader(f"Task: {selected_task}")
        st.caption(f"Showing {len(filtered)} of {len(events)} events")
    with header_right:
        if st.button("Reload log", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    if show_stats:
        role_counts: dict[str, int] = {}
        event_type_counts: dict[str, int] = {}
        for event in filtered:
            role_counts[event.role] = role_counts.get(event.role, 0) + 1
            event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1

        m1, m2, m3 = st.columns(3)
        m1.metric("Events", len(filtered))
        m2.metric("Roles", len(role_counts))
        m3.metric("Event Types", len(event_type_counts))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Role distribution**")
            st.json(role_counts)
        with c2:
            st.markdown("**Event type distribution**")
            st.json(event_type_counts)

    if not filtered:
        st.info("No events match the active filters.")
        st.stop()

    for idx, event in enumerate(filtered):
        _render_event(event, idx, _event_score(event, step_to_score))


if __name__ == "__main__":
    main()
