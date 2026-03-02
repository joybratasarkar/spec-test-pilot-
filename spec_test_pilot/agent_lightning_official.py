#!/usr/bin/env python3
"""Official Agent Lightning integration for SpecTestPilot.

This module intentionally uses the real `agentlightning` package API.
It does not re-implement the trainer/algorithm loop locally.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

from openai import AsyncOpenAI

from agentlightning import APO, PromptTemplate, Trainer, prompt_rollout
from agentlightning.adapter.messages import TraceToMessages

logger = logging.getLogger(__name__)

# Kept for compatibility with previous imports in this repo.
Task = Dict[str, Any]

DEFAULT_PROMPT_TEMPLATE = """You are a senior QA test architect.

Goal: {nlp_prompt}
API: {spec_title}

Generate high-value test scenarios that maximize bug discovery:
1. Positive and negative contract tests
2. Security and abuse cases
3. Boundary and malformed input tests
4. AuthN/AuthZ behavior checks
5. Error and recovery validation

Return practical, executable test cases.
"""


@dataclass(frozen=True)
class OfficialLightningConfig:
    """Runtime configuration for official Agent Lightning integration."""

    openai_api_key: Optional[str] = None
    n_runners: int = 1
    gradient_model: str = "gpt-5-mini"
    apply_edit_model: str = "gpt-4.1-mini"
    gradient_batch_size: int = 4
    val_batch_size: int = 16
    beam_width: int = 4
    branch_factor: int = 2
    beam_rounds: int = 2
    rollout_batch_timeout: float = 180.0
    run_initial_validation: bool = True
    seed_prompt_template: str = DEFAULT_PROMPT_TEMPLATE


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _compute_reward(result: Mapping[str, Any]) -> float:
    """Compute scalar reward from the SpecTestPilot sandbox output.

    The official runner can consume a direct numeric reward from rollout.
    """
    reward = 0.1

    if bool(result.get("success", False)):
        reward += 0.55

    quality_score = result.get("quality_score")
    if isinstance(quality_score, (int, float)):
        reward += 0.25 * _clamp01(float(quality_score))

    generated_tests = result.get("generated_tests")
    if isinstance(generated_tests, list):
        reward += min(0.1, len(generated_tests) * 0.01)
    elif generated_tests:
        reward += 0.05

    errors = result.get("errors")
    if isinstance(errors, list) and errors:
        reward -= 0.15

    return _clamp01(reward)


@prompt_rollout
def spec_test_pilot_rollout(task: Task, prompt_template: PromptTemplate) -> float:
    """Official LitAgent rollout function using PromptTemplate resource.

    The returned float is treated as final reward by Agent Lightning.
    """
    from .sandbox import AgentLightningSandbox

    payload: Dict[str, Any] = dict(task)
    payload.setdefault("spec_title", payload.get("spec_title", "Unknown API"))
    payload.setdefault(
        "nlp_prompt",
        "Generate comprehensive API test scenarios with security, negative, and boundary coverage",
    )

    rendered_prompt = prompt_template.format(**payload)
    payload["nlp_prompt"] = rendered_prompt

    sandbox = AgentLightningSandbox()
    result = sandbox.execute_agent_task(payload)
    return _compute_reward(result)


def create_official_spec_test_pilot_agent():
    """Return a LitAgent backed by the official prompt_rollout decorator."""
    return spec_test_pilot_rollout


def create_official_agent_lightning(
    openai_api_key: Optional[str] = None,
    **overrides: Any,
) -> Trainer:
    """Create a real Agent Lightning Trainer using official package components.

    Returns:
        agentlightning.Trainer configured with:
        - APO algorithm (official)
        - TraceToMessages adapter (required by APO)
        - PromptTemplate initial resource
        - shared-memory execution strategy
    """
    config = OfficialLightningConfig(openai_api_key=openai_api_key, **overrides)

    api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required for official APO training. "
            "Set env var OPENAI_API_KEY or pass openai_api_key=..."
        )

    try:
        algorithm = APO(
            async_openai_client=AsyncOpenAI(api_key=api_key),
            gradient_model=config.gradient_model,
            apply_edit_model=config.apply_edit_model,
            gradient_batch_size=config.gradient_batch_size,
            val_batch_size=config.val_batch_size,
            beam_width=config.beam_width,
            branch_factor=config.branch_factor,
            beam_rounds=config.beam_rounds,
            rollout_batch_timeout=config.rollout_batch_timeout,
            run_initial_validation=config.run_initial_validation,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "poml":
            raise RuntimeError(
                "Official APO dependency missing (`poml`). Install the official APO extra with: "
                "`venv/bin/pip install 'agentlightning[apo]>=0.1.0'` "
                "or reinstall from `requirements.txt`."
            ) from exc
        raise

    initial_prompt = PromptTemplate(template=config.seed_prompt_template, engine="f-string")

    # APO requires TraceToMessages adapter and both train/val datasets.
    trainer = Trainer(
        algorithm=algorithm,
        adapter=TraceToMessages(),
        initial_resources={"qa_prompt_template": initial_prompt},
        n_runners=config.n_runners,
        strategy={"type": "shm", "main_thread": "algorithm"},
    )

    logger.info(
        "Initialized official Agent Lightning Trainer (algorithm=%s, adapter=%s, strategy=shm)",
        type(algorithm).__name__,
        type(trainer.adapter).__name__,
    )
    return trainer


def train_spec_test_pilot_official(
    train_dataset: Iterable[Task],
    val_dataset: Optional[Iterable[Task]] = None,
    **trainer_kwargs: Any,
) -> Dict[str, Any]:
    """Run one official Agent Lightning training session and return summary."""
    train_data: List[Task] = list(train_dataset)
    if not train_data:
        raise ValueError("train_dataset must contain at least one task")

    val_data: List[Task] = list(val_dataset) if val_dataset is not None else list(train_data)
    if not val_data:
        raise ValueError("val_dataset must contain at least one task")

    trainer = create_official_agent_lightning(**trainer_kwargs)
    agent = create_official_spec_test_pilot_agent()

    trainer.fit(agent=agent, train_dataset=train_data, val_dataset=val_data)

    best_prompt = None
    if trainer.algorithm is not None and hasattr(trainer.algorithm, "get_best_prompt"):
        try:
            best_prompt = trainer.algorithm.get_best_prompt()
        except Exception:
            best_prompt = None

    return {
        "trainer": trainer,
        "algorithm": type(trainer.algorithm).__name__ if trainer.algorithm is not None else None,
        "adapter": type(trainer.adapter).__name__,
        "n_runners": trainer.n_runners,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "best_prompt": best_prompt,
    }


__all__ = [
    "Task",
    "PromptTemplate",
    "OfficialLightningConfig",
    "create_official_agent_lightning",
    "create_official_spec_test_pilot_agent",
    "spec_test_pilot_rollout",
    "train_spec_test_pilot_official",
]
