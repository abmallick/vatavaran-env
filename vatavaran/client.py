"""Async EnvClient for the Vatavaran environment."""

from __future__ import annotations

import os
from typing import Any, Optional, TYPE_CHECKING

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from websockets.asyncio.client import connect as ws_connect

from .models import VatavaranAction, VatavaranObservation, VatavaranState

if TYPE_CHECKING:
    from openenv.core.containers.runtime import ContainerProvider


class VatavaranEnv(EnvClient[VatavaranAction, VatavaranObservation, VatavaranState]):
    """Client wrapper for interacting with the Vatavaran environment server.

    WebSocket keepalive: ``websockets`` defaults (ping every 20s, fail if no pong in 20s) are easy
    to trip when the server is busy running ``execute_code``. This client uses more conservative
    defaults and passes ``ping_interval`` / ``ping_timeout`` into :func:`websockets.asyncio.client.connect`.
    Set ``ws_ping_interval=None`` to disable client-side pings (not recommended behind proxies).
    """

    def __init__(
        self,
        base_url: str,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 300.0,
        max_message_size_mb: float = 100.0,
        provider: Optional["ContainerProvider"] = None,
        mode: Optional[str] = None,
        *,
        ws_ping_interval: float | None = 60.0,
        ws_ping_timeout: float | None = 120.0,
    ) -> None:
        super().__init__(
            base_url=base_url,
            connect_timeout_s=connect_timeout_s,
            message_timeout_s=message_timeout_s,
            max_message_size_mb=max_message_size_mb,
            provider=provider,
            mode=mode,
        )
        self._ws_ping_interval = ws_ping_interval
        self._ws_ping_timeout = ws_ping_timeout

    async def connect(self) -> "VatavaranEnv":
        if self._ws is not None:
            return self

        ws_url_lower = self._ws_url.lower()
        is_localhost = "localhost" in ws_url_lower or "127.0.0.1" in ws_url_lower

        old_no_proxy = os.environ.get("NO_PROXY")
        if is_localhost:
            current_no_proxy = old_no_proxy or ""
            if "localhost" not in current_no_proxy.lower():
                os.environ["NO_PROXY"] = (
                    f"{current_no_proxy},localhost,127.0.0.1"
                    if current_no_proxy
                    else "localhost,127.0.0.1"
                )

        try:
            self._ws = await ws_connect(
                self._ws_url,
                open_timeout=self._connect_timeout,
                max_size=self._max_message_size,
                ping_interval=self._ws_ping_interval,
                ping_timeout=self._ws_ping_timeout,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self._ws_url}: {e}") from e
        finally:
            if is_localhost:
                if old_no_proxy is None:
                    os.environ.pop("NO_PROXY", None)
                else:
                    os.environ["NO_PROXY"] = old_no_proxy

        return self

    @classmethod
    async def from_docker_image(
        cls,
        image: str,
        provider: Optional["ContainerProvider"] = None,
        *,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 300.0,
        max_message_size_mb: float = 100.0,
        ws_ping_interval: float | None = 60.0,
        ws_ping_timeout: float | None = 120.0,
        **kwargs: Any,
    ) -> "VatavaranEnv":
        """Same as :meth:`EnvClient.from_docker_image` but forwards WebSocket keepalive settings."""

        from openenv.core.containers.runtime import LocalDockerProvider

        if provider is None:
            provider = LocalDockerProvider()

        base_url = provider.start_container(image, **kwargs)
        provider.wait_for_ready(base_url)

        client = cls(
            base_url=base_url,
            connect_timeout_s=connect_timeout_s,
            message_timeout_s=message_timeout_s,
            max_message_size_mb=max_message_size_mb,
            provider=provider,
            ws_ping_interval=ws_ping_interval,
            ws_ping_timeout=ws_ping_timeout,
        )
        await client.connect()
        return client

    def _step_payload(self, action: VatavaranAction) -> dict:
        return {
            "action_type": action.action_type,
            "content": action.content,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: dict) -> StepResult[VatavaranObservation]:
        observation = VatavaranObservation(**payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> VatavaranState:
        return VatavaranState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            current_task_id=payload.get("current_task_id"),
            current_difficulty=payload.get("current_difficulty"),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            modalities_explored=payload.get("modalities_explored", []),
            last_score=payload.get("last_score"),
        )
