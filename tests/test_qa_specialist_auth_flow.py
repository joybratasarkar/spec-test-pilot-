"""Tests for auth-aware scenario execution behavior in QA specialist agent."""

from __future__ import annotations

from pathlib import Path

import yaml

from spec_test_pilot.multi_language_tester import (
    TestScenario as ScenarioModel,
    TestType as ScenarioType,
)
from spec_test_pilot.qa_specialist_agent import QASpecialistAgent


def _write_spec(path: Path) -> None:
    spec = {
        "openapi": "3.0.3",
        "info": {"title": "Auth API", "version": "1.0.0"},
        "paths": {
            "/orders/{orderId}": {
                "get": {
                    "security": [{"bearerAuth": []}],
                    "responses": {"200": {"description": "ok"}, "401": {"description": "unauthorized"}},
                }
            }
        },
        "components": {
            "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer"}}
        },
    }
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


def test_non_auth_negative_scenario_gets_default_bearer_token(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )
    spec = agent._load_spec()
    agent._auth_required_ops = agent._build_auth_requirement_map(spec)

    scenario = ScenarioModel(
        name="test_get_orders_sql_injection",
        description="security test for SQL injection handling",
        test_type=ScenarioType.SECURITY,
        endpoint="/orders/{orderId}",
        method="GET",
        expected_status=400,
    )

    headers = agent._normalize_auth_headers_for_execution(
        scenario=scenario,
        method="GET",
        headers={},
    )
    assert headers.get("Authorization") == "Bearer valid_token_123"


def test_auth_negative_scenario_accepts_401_for_expected_403(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )
    spec = agent._load_spec()
    agent._auth_required_ops = agent._build_auth_requirement_map(spec)

    scenario = ScenarioModel(
        name="test_get_orders_forbidden_invalid_token",
        description="authorization test with invalid token",
        test_type=ScenarioType.AUTHORIZATION,
        endpoint="/orders/{orderId}",
        method="GET",
        expected_status=403,
        headers={"Authorization": "Bearer invalid_token_12345"},
    )

    headers = agent._normalize_auth_headers_for_execution(
        scenario=scenario,
        method="GET",
        headers=scenario.headers,
    )
    assert headers.get("Authorization") == "Bearer invalid"
    assert agent._status_matches_expectation(scenario, 401) is True


def test_selection_expands_to_cover_all_uncertain_scenarios(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=1,
    )

    scenarios = [
        ScenarioModel(
            name="s1",
            description="high uncertainty one",
            test_type=ScenarioType.SECURITY,
            endpoint="/u1",
            method="GET",
            expected_status=400,
        ),
        ScenarioModel(
            name="s2",
            description="high uncertainty two",
            test_type=ScenarioType.SECURITY,
            endpoint="/u2",
            method="GET",
            expected_status=400,
        ),
        ScenarioModel(
            name="s3",
            description="high uncertainty three",
            test_type=ScenarioType.SECURITY,
            endpoint="/u3",
            method="GET",
            expected_status=400,
        ),
        ScenarioModel(
            name="s4",
            description="low uncertainty",
            test_type=ScenarioType.HAPPY_PATH,
            endpoint="/stable",
            method="GET",
            expected_status=200,
        ),
    ]

    uncertainty_by_endpoint = {
        "/u1": 0.9,
        "/u2": 0.9,
        "/u3": 0.9,
        "/stable": 0.1,
    }

    def fake_score(
        *,
        test_type: str,
        method: str,
        endpoint: str,
        expected_status: int,
        has_body: bool,
        has_params: bool,
        rl_risk: float = 0.0,
        novelty_bonus: float = 0.0,
        legacy_weight_bonus: float = 0.0,
        diversity_penalty: float = 0.0,
    ) -> dict:
        uncertainty = float(uncertainty_by_endpoint.get(endpoint, 0.0))
        return {
            "score": uncertainty + novelty_bonus + legacy_weight_bonus + rl_risk,
            "expected_reward": 0.0,
            "uncertainty": uncertainty,
            "exploration_bonus": 0.0,
            "failure_focus_bonus": 0.0,
            "historical_reward": 0.0,
        }

    agent.adaptive_policy.score = fake_score  # type: ignore[assignment]
    selected = agent._select_scenarios_with_learning(scenarios)

    selected_endpoints = {scenario.endpoint for scenario in selected}
    assert selected_endpoints == {"/u1", "/u2", "/u3"}
    assert len(selected) == 3
    assert agent._last_selection_summary.get("budget_expanded_for_uncertainty") is True
    assert agent._last_selection_summary.get("effective_budget") == 3
    assert agent._last_selection_summary.get("uncertain_candidate_count") == 3


def test_refresh_repair_rules_from_repeated_failures(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )
    fingerprint = "POST|/orders|happy_path|201|body=1|params=0"
    agent.learning_state["scenario_stats"] = {
        fingerprint: {
            "attempts": 5,
            "passes": 0,
            "failures": 5,
            "avg_reward": -1.2,
            "failure_rate": 1.0,
            "test_type": "happy_path",
            "method": "POST",
            "endpoint": "/orders",
            "expected_status": 201,
            "actual_status_counts": {"400": 5},
        }
    }

    agent._refresh_scenario_repair_rules()
    rules = agent.learning_state.get("scenario_repair_rules", {})
    assert fingerprint in rules
    rule = rules[fingerprint]
    assert rule.get("repair_request_body") is True
    assert "override_expected_status" not in rule


def test_apply_scenario_repairs_adds_required_fields_and_status_override(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    spec = {
        "openapi": "3.0.3",
        "info": {"title": "Repair API", "version": "1.0.0"},
        "paths": {
            "/orders": {
                "post": {
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["productId", "quantity"],
                                    "properties": {
                                        "productId": {"type": "string"},
                                        "quantity": {"type": "integer"},
                                    },
                                }
                            }
                        },
                    },
                    "responses": {"201": {"description": "created"}, "400": {"description": "bad request"}},
                }
            }
        },
    }
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )

    scenario = ScenarioModel(
        name="test_post_orders_success",
        description="happy path create order",
        test_type=ScenarioType.HAPPY_PATH,
        endpoint="/orders",
        method="POST",
        body={"name": "partial"},
        expected_status=201,
    )
    fingerprint = agent._scenario_fingerprint(scenario)
    agent.learning_state["scenario_repair_rules"] = {
        fingerprint: {
            "override_expected_status": 400,
            "repair_request_body": True,
            "attempts": 4,
            "failure_rate": 1.0,
            "dominant_actual_status": 400,
        }
    }

    repaired, summary = agent._apply_scenario_repairs(spec, [scenario])
    assert repaired[0].expected_status == 400
    assert repaired[0].body is not None
    assert "productId" in repaired[0].body
    assert "quantity" in repaired[0].body
    assert summary["applied_repairs"] >= 1
