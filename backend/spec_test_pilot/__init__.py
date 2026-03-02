"""
SpecTestPilot: RL-trainable API Spec to Test Case JSON Agent

This package implements an agent that:
- Parses OpenAPI specifications
- Uses GAM-style deep-research memory
- Generates comprehensive test case JSON
- Is trainable via reinforcement learning
"""

__version__ = "0.1.0"
__author__ = "SpecTestPilot Team"

from spec_test_pilot.schemas import (
    SpecTestPilotOutput,
    SpecSummary,
    DeepResearch,
    TestCase,
    CoverageChecklist,
)
from spec_test_pilot.openapi_parse import parse_openapi_spec
from spec_test_pilot.reward import compute_reward

__all__ = [
    "SpecTestPilotOutput",
    "SpecSummary",
    "DeepResearch",
    "TestCase",
    "CoverageChecklist",
    "parse_openapi_spec",
    "compute_reward",
]
