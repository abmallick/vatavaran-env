"""Config-driven reward shaping for the Vatavaran environment."""

from __future__ import annotations

from typing import Iterable


class RewardEngine:
    """Computes rewards from a configurable reward policy."""

    def __init__(self, config: dict):
        self.config = config

    @staticmethod
    def _clamp01(value: float) -> float:
        return min(max(value, 0.01), 0.99)

    def step_penalty(self) -> float:
        step_cfg = self.config.get("step_efficiency", {})
        if not step_cfg.get("enabled", True):
            return 0.0
        return float(step_cfg.get("penalty_per_step", 0.0))

    def on_code_execution(self, success: bool) -> float:
        code_cfg = self.config.get("code_execution", {})
        base = (
            float(code_cfg.get("success_reward", 0.0))
            if success
            else float(code_cfg.get("error_penalty", 0.0))
        )
        return base + self.step_penalty()

    def on_list_files(self) -> float:
        return self.step_penalty()

    def _exploration_bonus(self, modalities_explored: Iterable[str]) -> float:
        cfg = self.config.get("exploration_bonus", {})
        if not cfg.get("enabled", True):
            return 0.0
        tracked = set(cfg.get("tracked_modalities", []))
        explored = set(modalities_explored)
        count = len(explored.intersection(tracked))
        per_modality = float(cfg.get("per_modality_bonus", 0.0))
        max_bonus = float(cfg.get("max_bonus", per_modality * len(tracked)))
        return min(count * per_modality, max_bonus)

    def _cross_validation_bonus(self, modalities_explored: Iterable[str]) -> float:
        cfg = self.config.get("cross_validation", {})
        if not cfg.get("enabled", True):
            return 0.0
        min_modalities = int(cfg.get("min_modalities", 2))
        return (
            float(cfg.get("bonus", 0.0))
            if len(set(modalities_explored)) >= min_modalities
            else 0.0
        )

    def on_submit(self, eval_score: float, modalities_explored: Iterable[str]) -> float:
        final_cfg = self.config.get("final_answer", {})
        weighted_eval = float(final_cfg.get("weight", 1.0)) * float(eval_score)
        reward = (
            weighted_eval
            + self._exploration_bonus(modalities_explored)
            + self._cross_validation_bonus(modalities_explored)
            + self.step_penalty()
        )
        return self._clamp01(reward)

    def on_max_steps(self, modalities_explored: Iterable[str]) -> float:
        final_cfg = self.config.get("final_answer", {})
        reward = (
            float(final_cfg.get("no_submission_reward", 0.0))
            + self._exploration_bonus(modalities_explored)
            + self._cross_validation_bonus(modalities_explored)
        )
        return self._clamp01(reward)

    def max_steps_for_difficulty(self, difficulty: str) -> int:
        episode_cfg = self.config.get("episode", {})
        if difficulty == "easy":
            return int(episode_cfg.get("max_steps_easy", 15))
        if difficulty == "middle":
            return int(episode_cfg.get("max_steps_middle", 20))
        return int(episode_cfg.get("max_steps_hard", 25))
