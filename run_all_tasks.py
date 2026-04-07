"""
Run inference across the selected default Bank tasks.

Connects directly to https://abmallick-vatavaran-env.hf.space and iterates
the configured task IDs sequentially in a single process so the conversation
log is preserved across all tasks (one reset at the very start, then one
episode per task appended to the same log file).
"""

from __future__ import annotations

import asyncio

from openai import OpenAI

from vatavaran import VatavaranEnv
from inference import (
    API_BASE_URL,
    API_KEY,
    LOG_CONVERSATION_PATH,
    _env_ws_kwargs,
    _reset_conversation_log,
    _run_episode,
)

HF_SPACE_URL = "https://abmallick-vatavaran-env.hf.space"

# Task IDs aligned with the default inference list.
TASK_IDS = [
    "Bank_00018",
    "Bank_00020",
    "Bank_00057",
]


async def run_all() -> None:
    _reset_conversation_log(LOG_CONVERSATION_PATH)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "missing-key")

    for task_id in TASK_IDS:
        print(f"\n{'='*60}", flush=True)
        print(f"Starting task: {task_id}", flush=True)
        print(f"{'='*60}", flush=True)

        env = VatavaranEnv(base_url=HF_SPACE_URL, **_env_ws_kwargs())
        try:
            await _run_episode(client, env, task_id)
        finally:
            try:
                await env.close()
            except Exception as exc:
                print(f"[DEBUG] env.close() error for {task_id}: {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(run_all())
