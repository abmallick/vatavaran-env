"""Typed action/observation/state models for Vatavaran."""

from __future__ import annotations

from typing import Literal

from openenv.core.env_server.interfaces import Action, Observation, State
from pydantic import Field


class VatavaranAction(Action):
    """Action sent by the agent to the environment."""

    action_type: Literal["execute_code", "list_files", "submit_answer"] = Field(
        ...,
        description="Type of action to execute.",
    )
    content: str = Field(
        default="",
        description="Payload for the action. Python code, path, or JSON answer.",
    )
    reasoning: str = Field(
        default="",
        description="Reasoning for the action. This is the reasoning for the action that the agent has taken.",
    )


class VatavaranObservation(Observation):
    """Observation returned after each environment step."""

    result: str = Field(default="", description="Main textual result from the action.")
    success: bool = Field(
        default=True,
        description="Whether the requested action executed successfully.",
    )
    last_action_error: str | None = Field(
        default=None,
        description="Error message from the last action if any.",
    )
    task_id: str = Field(default="", description="Current task identifier.")
    task_description: str = Field(
        default="",
        description="Natural language RCA task objective.",
    )
    difficulty: str = Field(default="", description="Task difficulty: easy/middle/hard.")
    domain_knowledge: str = Field(
        default="",
        description="System schema and candidate root cause values.",
    )
    step_count: int = Field(default=0, description="Current step index in the episode.")
    max_steps: int = Field(default=0, description="Maximum steps allowed in the episode.")
    available_actions: list[str] = Field(
        default_factory=lambda: ["execute_code", "list_files", "submit_answer"],
        description="Allowed action types.",
    )


class VatavaranState(State):
    """Extended state for the Vatavaran environment."""

    current_task_id: str | None = Field(default=None)
    current_difficulty: str | None = Field(default=None)
    cumulative_reward: float = Field(default=0.0)
    modalities_explored: list[str] = Field(default_factory=list)
    last_score: float | None = Field(default=None)
