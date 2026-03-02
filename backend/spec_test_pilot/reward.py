"""
Reward computation for SpecTestPilot outputs.

This module provides:
- Hard-gate contract checks
- Endpoint coverage scoring
- Lightweight quality scoring
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Set, Tuple

from pydantic import ValidationError

from spec_test_pilot.openapi_parse import ParsedSpec
from spec_test_pilot.schemas import SpecTestPilotOutput

EndpointTuple = Tuple[str, str]


@dataclass
class RewardBreakdown:
    """Detailed reward diagnostics."""

    valid_json: bool
    pydantic_valid: bool
    no_invented_endpoints: bool
    endpoint_coverage: float
    test_density: float
    assertion_quality: float
    final_reward: float


def _normalize_endpoint(method: str, path: str) -> EndpointTuple:
    return (str(method).upper(), str(path))


def _extract_spec_endpoints(parsed_spec: ParsedSpec) -> Set[EndpointTuple]:
    return {_normalize_endpoint(ep.method, ep.path) for ep in parsed_spec.endpoints}


def _extract_output_test_endpoints(output_dict: Dict[str, Any]) -> Set[EndpointTuple]:
    endpoints: Set[EndpointTuple] = set()

    for test in output_dict.get("test_suite", []):
        endpoint = test.get("endpoint", {})
        method = endpoint.get("method")
        path = endpoint.get("path")
        if method and path:
            endpoints.add(_normalize_endpoint(method, path))

    return endpoints


def _extract_output_detected_endpoints(output_dict: Dict[str, Any]) -> Set[EndpointTuple]:
    endpoints: Set[EndpointTuple] = set()

    spec_summary = output_dict.get("spec_summary", {})
    for endpoint in spec_summary.get("endpoints_detected", []):
        method = endpoint.get("method")
        path = endpoint.get("path")
        if method and path:
            endpoints.add(_normalize_endpoint(method, path))

    return endpoints


def _compute_endpoint_coverage(
    test_endpoints: Iterable[EndpointTuple], spec_endpoints: Iterable[EndpointTuple]
) -> float:
    """
    Compute endpoint coverage ratio in [0, 1].

    Coverage is based on unique endpoints in test suite intersected with spec endpoints.
    """

    spec_set = set(spec_endpoints)
    if not spec_set:
        return 0.0

    test_set = set(test_endpoints)
    covered = len(test_set & spec_set)
    return covered / len(spec_set)


def _compute_test_density(test_count: int, endpoint_count: int) -> float:
    """
    Compute test density score in [0, 1].

    Score reaches 1.0 at 2+ tests per endpoint.
    """

    if endpoint_count <= 0:
        return 0.0

    tests_per_endpoint = test_count / endpoint_count
    return min(1.0, tests_per_endpoint / 2.0)


def _compute_assertion_quality(output_dict: Dict[str, Any]) -> float:
    """
    Compute assertion quality in [0, 1].

    Score reaches 1.0 at 2+ assertions per test on average.
    """

    test_suite = output_dict.get("test_suite", [])
    if not test_suite:
        return 0.0

    total_assertions = 0
    for test in test_suite:
        assertions = test.get("assertions", [])
        if isinstance(assertions, list):
            total_assertions += len(assertions)

    avg_assertions = total_assertions / len(test_suite)
    return min(1.0, avg_assertions / 2.0)


def compute_reward(output_dict: Dict[str, Any], parsed_spec: ParsedSpec) -> Tuple[float, RewardBreakdown]:
    """
    Compute final reward and a diagnostic breakdown.

    Hard gates:
    - Output must be JSON-serializable
    - Output must validate with Pydantic schema
    - Test endpoints must not invent endpoints absent from parsed spec
    """

    # Gate 1: JSON-serializable output
    try:
        json.dumps(output_dict)
        valid_json = True
    except (TypeError, ValueError):
        valid_json = False

    # Extract endpoint sets
    spec_endpoints = _extract_spec_endpoints(parsed_spec)
    test_endpoints = _extract_output_test_endpoints(output_dict)
    detected_endpoints = _extract_output_detected_endpoints(output_dict)

    # Gate 2: Pydantic contract validation
    try:
        SpecTestPilotOutput.model_validate(output_dict)
        pydantic_valid = True
    except ValidationError:
        pydantic_valid = False

    # Gate 3: no invented endpoints relative to parsed spec
    no_invented_endpoints = len(test_endpoints - spec_endpoints) == 0

    # If hard gates fail, score is 0
    if not (valid_json and pydantic_valid and no_invented_endpoints):
        breakdown = RewardBreakdown(
            valid_json=valid_json,
            pydantic_valid=pydantic_valid,
            no_invented_endpoints=no_invented_endpoints,
            endpoint_coverage=0.0,
            test_density=0.0,
            assertion_quality=0.0,
            final_reward=0.0,
        )
        return 0.0, breakdown

    # Soft metrics
    endpoint_coverage = _compute_endpoint_coverage(test_endpoints, spec_endpoints)
    test_density = _compute_test_density(
        test_count=len(output_dict.get("test_suite", [])),
        endpoint_count=len(spec_endpoints),
    )
    assertion_quality = _compute_assertion_quality(output_dict)

    # Consistency bonus if detected endpoints align with parsed spec
    detected_consistency = _compute_endpoint_coverage(detected_endpoints, spec_endpoints)

    # Weighted final score in [0, 1]
    final_reward = (
        0.50 * endpoint_coverage
        + 0.25 * test_density
        + 0.15 * assertion_quality
        + 0.10 * detected_consistency
    )
    final_reward = max(0.0, min(1.0, final_reward))

    breakdown = RewardBreakdown(
        valid_json=valid_json,
        pydantic_valid=pydantic_valid,
        no_invented_endpoints=no_invented_endpoints,
        endpoint_coverage=endpoint_coverage,
        test_density=test_density,
        assertion_quality=assertion_quality,
        final_reward=final_reward,
    )

    return final_reward, breakdown


__all__ = [
    "RewardBreakdown",
    "compute_reward",
    "_compute_endpoint_coverage",
]
