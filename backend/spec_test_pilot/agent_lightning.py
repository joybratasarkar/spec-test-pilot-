"""
Compatibility wrapper for Agent Lightning integration.

This module keeps the legacy `spec_test_pilot.agent_lightning` import path working
while delegating execution to the current v2 implementation.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Dict, Optional

from spec_test_pilot.agent_lightning_v2 import (
    create_agent_lightning_system,
    train_with_agent_lightning,
)


class AgentLightningTrainer:
    """
    Backward-compatible trainer facade used by legacy scripts.

    Public methods preserved:
    - train_on_task(...)
    - get_stats()
    """

    def __init__(
        self,
        gam_memory_system=None,
        max_workers: int = 1,
        enable_torch: bool = True,
        sandbox_mode: bool = False,
        **_: Any,
    ):
        self.max_workers = max_workers
        self.sandbox_mode = sandbox_mode

        # v2 trainer already registers "spec_test_pilot" agent via adapter.
        self.server = create_agent_lightning_system(gam_memory_system=gam_memory_system)
        self.server.training_enabled = bool(enable_torch)

        # Legacy callers expect `trainer.server.rl_algorithm.training_step` to exist.
        if not hasattr(self.server.rl_algorithm, "training_step"):
            self.server.rl_algorithm.training_step = getattr(
                self.server.rl_algorithm, "train_step", None
            )

        self._recent_rewards: deque[float] = deque(maxlen=100)

    def train_on_task(self, **task_data: Any) -> Dict[str, Any]:
        """Run one task through the v2 async trainer using legacy sync API."""

        result = self._run_async(
            self.server.train_agent("spec_test_pilot", dict(task_data))
        )

        task_result = result.get("result", {})
        task_success = bool(task_result.get("success", result.get("success", False)))
        final_reward = self._derive_reward(result, task_result, task_success)
        self._recent_rewards.append(final_reward)

        return {
            "success": result.get("success", False),
            "task_success": task_success,
            "final_reward": final_reward,
            "training_enabled": result.get("training_enabled", self.server.training_enabled),
            "execution_time": result.get("execution_time", 0.0),
            "session_id": result.get("session_id"),
            "traces_collected": result.get("traces_collected", 0),
            "task_result": task_result,
            "raw_result": result,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return legacy-formatted stats dictionary."""

        stats = self.server.get_training_stats()
        recent_avg_reward = (
            sum(self._recent_rewards) / len(self._recent_rewards)
            if self._recent_rewards
            else 0.0
        )

        return {
            "registered_agents": stats.get("registered_agents", 0),
            "total_traces": stats.get("total_traces", 0),
            "active_sessions": stats.get("active_sessions", 0),
            "total_transitions": stats.get("rl_buffer_size", 0),
            "training_steps": stats.get("rl_training_steps", 0),
            "training_enabled": stats.get("training_enabled", False),
            "recent_avg_reward": recent_avg_reward,
        }

    def _derive_reward(
        self, result: Dict[str, Any], task_result: Dict[str, Any], task_success: bool
    ) -> float:
        if "quality_score" in task_result:
            return float(task_result["quality_score"])
        if task_success:
            return 1.0
        if result.get("success", False):
            return 0.5
        return 0.0

    def _run_async(self, coroutine):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        raise RuntimeError(
            "train_on_task() cannot be called from an active event loop. "
            "Use: await trainer.server.train_agent('spec_test_pilot', task_data)"
        )


__all__ = [
    "AgentLightningTrainer",
    "create_agent_lightning_system",
    "train_with_agent_lightning",
]
