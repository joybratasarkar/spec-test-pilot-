"""Tests for adaptive QA scenario selection policy."""

import pytest

from spec_test_pilot.adaptive_policy import AdaptiveScenarioPolicy, scenario_fingerprint


def test_scenario_fingerprint_is_stable() -> None:
    kwargs = {
        "test_type": "authentication",
        "method": "post",
        "endpoint": "/orders",
        "expected_status": 401,
        "has_body": True,
        "has_params": False,
    }
    assert scenario_fingerprint(**kwargs) == scenario_fingerprint(**kwargs)


def test_policy_round_trip_preserves_scoring() -> None:
    policy = AdaptiveScenarioPolicy(feature_dim=32, alpha=0.2, regularization=0.7)

    policy.observe(
        test_type="input_validation",
        method="POST",
        endpoint="/products",
        expected_status=400,
        has_body=True,
        has_params=False,
        reward=-1.1,
        passed=False,
    )
    policy.observe(
        test_type="authentication",
        method="GET",
        endpoint="/products",
        expected_status=401,
        has_body=False,
        has_params=False,
        reward=1.2,
        passed=True,
    )

    expected = policy.score(
        test_type="input_validation",
        method="POST",
        endpoint="/products",
        expected_status=400,
        has_body=True,
        has_params=False,
        rl_risk=0.05,
        novelty_bonus=0.1,
        legacy_weight_bonus=0.02,
    )

    restored = AdaptiveScenarioPolicy.from_state(policy.to_state())
    actual = restored.score(
        test_type="input_validation",
        method="POST",
        endpoint="/products",
        expected_status=400,
        has_body=True,
        has_params=False,
        rl_risk=0.05,
        novelty_bonus=0.1,
        legacy_weight_bonus=0.02,
    )

    assert actual["score"] == pytest.approx(expected["score"], rel=1e-8, abs=1e-8)
    assert actual["uncertainty"] == pytest.approx(
        expected["uncertainty"], rel=1e-8, abs=1e-8
    )


def test_failure_focus_bonus_increases_after_failures() -> None:
    policy = AdaptiveScenarioPolicy(feature_dim=48, alpha=0.15)
    before = policy.score(
        test_type="error_handling",
        method="POST",
        endpoint="/orders",
        expected_status=400,
        has_body=True,
        has_params=False,
    )

    for _ in range(3):
        policy.observe(
            test_type="error_handling",
            method="POST",
            endpoint="/orders",
            expected_status=400,
            has_body=True,
            has_params=False,
            reward=-1.2,
            passed=False,
        )

    after = policy.score(
        test_type="error_handling",
        method="POST",
        endpoint="/orders",
        expected_status=400,
        has_body=True,
        has_params=False,
    )
    fp = scenario_fingerprint(
        test_type="error_handling",
        method="POST",
        endpoint="/orders",
        expected_status=400,
        has_body=True,
        has_params=False,
    )

    assert policy.scenario_stats[fp]["attempts"] == 3
    assert policy.scenario_stats[fp]["failure_rate"] == 1.0
    assert after["failure_focus_bonus"] > before["failure_focus_bonus"]
