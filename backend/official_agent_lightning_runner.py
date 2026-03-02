#!/usr/bin/env python3
"""Run SpecTestPilot with the official Microsoft Agent Lightning library."""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

from spec_test_pilot.agent_lightning_official import (
    create_official_agent_lightning,
    create_official_spec_test_pilot_agent,
)


def _build_demo_dataset() -> List[Dict[str, Any]]:
    return [
        {
            "spec_title": "Banking Security API",
            "nlp_prompt": "Generate high-risk security and negative API tests",
            "openapi_spec": "examples/banking_api.yaml",
            "tenant_id": "security_banking",
        },
        {
            "spec_title": "E-commerce API",
            "nlp_prompt": "Generate test cases for auth, input validation, and order flows",
            "openapi_spec": "examples/ecommerce_api.yaml",
            "tenant_id": "ecommerce_qa",
        },
        {
            "spec_title": "Healthcare Data API",
            "nlp_prompt": "Generate data privacy and strict validation tests",
            "openapi_spec": "examples/healthcare_api.yaml",
            "tenant_id": "healthcare_qa",
        },
        {
            "spec_title": "Social Media API",
            "nlp_prompt": "Generate abuse, moderation, and authorization test scenarios",
            "openapi_spec": "examples/social_api.yaml",
            "tenant_id": "social_qa",
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official Agent Lightning runner")
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--n-runners", type=int, default=1)
    parser.add_argument("--beam-rounds", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--branch-factor", type=int, default=2)
    parser.add_argument("--gradient-model", default="gpt-5-mini")
    parser.add_argument("--apply-edit-model", default="gpt-4.1-mini")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    train_dataset = _build_demo_dataset()
    val_dataset = list(train_dataset)

    print("OFFICIAL AGENT LIGHTNING RUN")
    print("=")

    try:
        trainer = create_official_agent_lightning(
            openai_api_key=args.openai_api_key,
            n_runners=args.n_runners,
            beam_rounds=args.beam_rounds,
            beam_width=args.beam_width,
            branch_factor=args.branch_factor,
            gradient_model=args.gradient_model,
            apply_edit_model=args.apply_edit_model,
        )
        agent = create_official_spec_test_pilot_agent()

        print(f"Trainer type: {type(trainer).__name__}")
        print(f"Algorithm type: {type(trainer.algorithm).__name__ if trainer.algorithm else 'None'}")
        print(f"Adapter type: {type(trainer.adapter).__name__}")
        print(f"Runners: {trainer.n_runners}")
        print(f"Train tasks: {len(train_dataset)} | Val tasks: {len(val_dataset)}")

        trainer.fit(agent=agent, train_dataset=train_dataset, val_dataset=val_dataset)

        best_prompt_text = None
        if trainer.algorithm is not None and hasattr(trainer.algorithm, "get_best_prompt"):
            try:
                best_prompt = trainer.algorithm.get_best_prompt()
                best_prompt_text = best_prompt.template
            except Exception:
                best_prompt_text = None

        print("Training completed through official agentlightning package.")
        if best_prompt_text:
            print("Best prompt (first 300 chars):")
            print(best_prompt_text[:300])

        return 0
    except Exception as exc:
        print(f"Run failed: {exc}")
        print("If this is an API/auth/network issue, set OPENAI_API_KEY and retry.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
