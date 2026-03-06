"""Tests for auth-aware scenario execution behavior in QA specialist agent."""

from __future__ import annotations

from pathlib import Path

import yaml

from spec_test_pilot.multi_language_tester import (
    TestScenario as ScenarioModel,
    TestType as ScenarioType,
)
from spec_test_pilot.qa_specialist_agent import (
    QASpecialistAgent,
    ScenarioExecutionResult,
)


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


def _mutation_strategies(scenarios: list[ScenarioModel]) -> set[str]:
    strategies: set[str] = set()
    for scenario in scenarios:
        for assertion in list(scenario.assertions or []):
            text = str(assertion)
            if text.startswith("rl_mutation:"):
                strategies.add(text.split(":", 1)[1])
    return strategies


class _FakeResponse:
    def __init__(self, payload, text: str = "") -> None:
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


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


def test_auth_negative_scenario_requires_exact_status_match(tmp_path: Path) -> None:
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
    assert agent._status_matches_expectation(scenario, 401) is False
    assert agent._status_matches_expectation(scenario, 403) is True


def test_auth_negative_401_with_bearer_token_forces_invalid_token(tmp_path: Path) -> None:
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
        name="test_get_orders_authorization_failure",
        description="authorization failure with non-privileged token",
        test_type=ScenarioType.AUTHORIZATION,
        endpoint="/orders/{orderId}",
        method="GET",
        expected_status=401,
        headers={"Authorization": "Bearer valid_but_unauthorized_token"},
    )

    headers = agent._normalize_auth_headers_for_execution(
        scenario=scenario,
        method="GET",
        headers=scenario.headers,
    )
    assert headers.get("Authorization") == "Bearer invalid"


def test_boundary_401_expectation_is_treated_as_auth_negative(tmp_path: Path) -> None:
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
        name="test_get_orders_boundary_probe",
        description="boundary probe without auth marker text",
        test_type=ScenarioType.BOUNDARY_TESTING,
        endpoint="/orders/{orderId}",
        method="GET",
        expected_status=401,
        headers={"Authorization": "Bearer valid_token_123"},
    )

    headers = agent._normalize_auth_headers_for_execution(
        scenario=scenario,
        method="GET",
        headers=scenario.headers,
    )
    assert headers.get("Authorization") == "Bearer invalid"


def test_boundary_unconstrained_path_400_normalizes_to_404(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )

    scenario = ScenarioModel(
        name="test_get_orders_boundary_invalid_id",
        description="boundary testing invalid id format",
        test_type=ScenarioType.BOUNDARY_TESTING,
        endpoint="/orders/{orderId}",
        method="GET",
        expected_status=400,
    )
    op_meta = {
        "response_statuses": [200, 401, 404],
        "path_param_schemas": {"orderId": {"type": "string"}},
    }

    normalized = agent._normalize_expected_status_for_execution(
        scenario=scenario,
        method="GET",
        op_meta=op_meta,
    )
    assert normalized == 404


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


def test_selection_respects_max_scenarios_when_budget_not_expanded(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=2,
    )
    agent._operation_index = {}
    agent.learning_state["run_count"] = 1

    scenarios = [
        ScenarioModel(
            name="auth_case",
            description="auth case",
            test_type=ScenarioType.AUTHENTICATION,
            endpoint="/a1",
            method="GET",
            expected_status=401,
        ),
        ScenarioModel(
            name="input_case",
            description="input case",
            test_type=ScenarioType.INPUT_VALIDATION,
            endpoint="/a2",
            method="GET",
            expected_status=400,
        ),
        ScenarioModel(
            name="error_case",
            description="error case",
            test_type=ScenarioType.ERROR_HANDLING,
            endpoint="/a3",
            method="GET",
            expected_status=400,
        ),
        ScenarioModel(
            name="boundary_case",
            description="boundary case",
            test_type=ScenarioType.BOUNDARY_TESTING,
            endpoint="/a4",
            method="GET",
            expected_status=400,
        ),
    ]

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
        return {
            "score": 1.0 + novelty_bonus + legacy_weight_bonus + rl_risk,
            "expected_reward": 0.5,
            "uncertainty": 0.0,
            "exploration_bonus": 0.0,
            "failure_focus_bonus": 0.0,
            "historical_reward": 0.0,
        }

    agent.adaptive_policy.score = fake_score  # type: ignore[assignment]
    selected = agent._select_scenarios_with_learning(scenarios)

    assert len(selected) <= 2
    assert int(agent._last_selection_summary.get("effective_budget", 0)) == 2
    assert bool(agent._last_selection_summary.get("budget_expanded_for_uncertainty", False)) is False


def test_filters_gam_memo_excerpts_to_current_spec(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )
    spec = agent._load_spec()
    agent._spec_scope_key = agent._compute_spec_scope_key(spec)
    agent._spec_memory_tags = set(agent._build_spec_memory_tags("Auth API"))

    excerpts = [
        {
            "source": "memo",
            "title": "Spec Context Signals: Auth API",
            "tags": ["memo", "spec_context", "auth_api", agent._spec_scope_key],
            "excerpt": "Spec: Auth API Auth type: bearer",
        },
        {
            "source": "memo",
            "title": "Spec Context Signals: E-commerce API",
            "tags": ["memo", "spec_context", "e_commerce_api", "spec_deadbeefdeadbeef"],
            "excerpt": "Spec: E-commerce API Auth type: bearer",
        },
        {
            "source": "convention",
            "title": "REST API Testing Conventions",
            "tags": ["convention", "rest", "testing"],
            "excerpt": "Include auth, validation, and error tests.",
        },
    ]

    filtered = agent._filter_memory_excerpts_for_current_spec(
        excerpts, spec_title="Auth API"
    )
    titles = [str(item.get("title", "")) for item in filtered]
    assert "Spec Context Signals: Auth API" in titles
    assert "REST API Testing Conventions" in titles
    assert "Spec Context Signals: E-commerce API" not in titles


def test_focus_points_skip_stable_zero_failure_trend_excerpt(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )

    excerpts = [
        {
            "source": "memo",
            "title": "RL Trend Signals: Auth API",
            "excerpt": (
                "- GET /products | authentication | expected=401 | attempts=20 | "
                "failure_rate=0.0 | avg_reward=1.12"
            ),
        },
        {
            "source": "memo",
            "title": "RL Trend Signals: Auth API",
            "excerpt": (
                "- GET /orders/{orderId} | error_handling | expected=404 | attempts=6 | "
                "failure_rate=0.67 | avg_reward=-0.42"
            ),
        },
    ]

    focus_points = agent._extract_focus_points_from_excerpts(excerpts, limit=2)
    joined = " || ".join(focus_points)
    assert "Weak RL pattern: GET /orders/{orderId}" in joined
    assert "GET /products" not in joined
    assert "failure_rate=0.0" not in joined


def test_gam_focus_points_include_run_to_run_delta_signal(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )
    agent.learning_state["run_count"] = 12
    agent.learning_state["decision_history"] = [
        {
            "timestamp": "2026-03-03T10:00:00Z",
            "run_reward": 0.40,
            "average_decision_reward": 0.30,
        },
        {
            "timestamp": "2026-03-03T10:05:00Z",
            "run_reward": 0.35,
            "average_decision_reward": 0.22,
        },
    ]

    excerpts = [
        {
            "source": "trusted_docs_fallback",
            "title": "Spec risk",
            "excerpt": "Spec risk action: prioritize dependency-order producer->consumer checks.",
        },
    ]

    points = agent._build_gam_prompt_focus_points(excerpts, limit=2)

    assert len(points) == 2
    assert "Trend delta:" in points[0]
    assert "run_reward" in points[0]


def test_gam_contract_requires_real_weak_pattern_signal(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )

    excerpts = [
        {
            "source": "memo",
            "title": "RL Trend Signals: Auth API",
            "excerpt": (
                "- GET /products | authentication | expected=401 | attempts=20 | "
                "failure_rate=0.0 | avg_reward=1.12"
            ),
        },
        {
            "source": "trusted_docs_fallback",
            "title": "Spec risk",
            "excerpt": "Spec: Auth API risk: verify token handling on protected operations.",
        },
        {
            "source": "trusted_docs_fallback",
            "title": "Action",
            "excerpt": "Next action: reinforce dependency-order variants for auth checks.",
        },
        {
            "source": "trusted_docs_fallback",
            "title": "Trend",
            "excerpt": "Trend delta action: increase exploration for uncertain endpoints.",
        },
    ]

    pack = agent._build_gam_context_pack(
        memory_excerpts=excerpts,
        diagnostics={"quality_score": 1.0, "warnings": []},
        reflection="unit test",
    )

    assert pack["contract_checks"]["has_weak_pattern"] is False
    assert pack["status"] == "needs_retry"


def test_format_focus_point_text_strips_leading_dash(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )

    formatted = agent._format_focus_point_text(
        "- GET /products | authentication | expected=401 | attempts=20 | failure_rate=0.3 | avg_reward=-0.2",
        source="memo",
        max_chars=200,
    )
    assert not formatted.startswith("- ")


def test_happy_path_guardrail_adds_missing_operation_coverage(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    spec = {
        "openapi": "3.0.3",
        "info": {"title": "Coverage API", "version": "1.0.0"},
        "paths": {
            "/products": {
                "get": {
                    "security": [{"bearerAuth": []}],
                    "responses": {"200": {"description": "ok"}, "401": {"description": "unauthorized"}},
                }
            },
            "/orders": {
                "post": {
                    "security": [{"bearerAuth": []}],
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
                    "responses": {"201": {"description": "created"}, "401": {"description": "unauthorized"}},
                }
            },
        },
        "components": {
            "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer"}}
        },
    }
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=8,
    )
    loaded = agent._load_spec()
    agent._auth_required_ops = agent._build_auth_requirement_map(loaded)
    agent._operation_index = agent._build_operation_index(loaded)

    base = [
        ScenarioModel(
            name="test_get_products_no_auth",
            description="negative auth",
            test_type=ScenarioType.AUTHENTICATION,
            endpoint="/products",
            method="GET",
            expected_status=401,
        )
    ]

    covered, summary = agent._ensure_happy_path_coverage(base)
    happy_ops = {
        f"{item.method.upper()} {item.endpoint}"
        for item in covered
        if item.test_type == ScenarioType.HAPPY_PATH and int(item.expected_status) < 300
    }
    assert "GET /products" in happy_ops
    assert "POST /orders" in happy_ops
    assert summary.get("added", 0) >= 2


def test_rl_mutations_do_not_inject_rl_case_query_param(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    spec = {
        "openapi": "3.0.3",
        "info": {"title": "Mutation API", "version": "1.0.0"},
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
                                        "quantity": {"type": "integer", "minimum": 1},
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
        max_scenarios=8,
    )
    loaded = agent._load_spec()
    agent._operation_index = agent._build_operation_index(loaded)
    op_meta = agent._operation_index["POST /orders"]

    scenario = ScenarioModel(
        name="test_post_orders_type_mismatch",
        description="input validation check",
        test_type=ScenarioType.INPUT_VALIDATION,
        endpoint="/orders",
        method="POST",
        expected_status=400,
        body={"productId": "123", "quantity": "invalid"},
    )

    variants = agent._build_rl_mutations_for_scenario(
        scenario,
        max_variants=6,
        operation_meta=op_meta,
        target_context={
            "failure_rate": 1.0,
            "op_failure_rate": 1.0,
            "uncertainty": 0.2,
            "op_attempts": 10,
        },
    )
    assert variants
    assert all("_rl_case" not in dict(item.params or {}) for item in variants)


def test_selection_keeps_happy_path_coverage_when_available(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=1,
    )
    loaded = agent._load_spec()
    agent._operation_index = agent._build_operation_index(loaded)

    scenarios = [
        ScenarioModel(
            name="test_get_orders_success",
            description="happy path",
            test_type=ScenarioType.HAPPY_PATH,
            endpoint="/orders/{orderId}",
            method="GET",
            expected_status=200,
        ),
        ScenarioModel(
            name="test_get_orders_no_auth",
            description="auth negative",
            test_type=ScenarioType.AUTHENTICATION,
            endpoint="/orders/{orderId}",
            method="GET",
            expected_status=401,
        ),
    ]

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
        if test_type == ScenarioType.AUTHENTICATION.value:
            score = 1.0
        else:
            score = 0.2
        return {
            "score": score,
            "expected_reward": score,
            "uncertainty": 0.0,
            "exploration_bonus": 0.0,
            "failure_focus_bonus": 0.0,
            "historical_reward": 0.0,
        }

    agent.adaptive_policy.score = fake_score  # type: ignore[assignment]
    selected = agent._select_scenarios_with_learning(scenarios)
    assert any(item.test_type == ScenarioType.HAPPY_PATH for item in selected)


def test_workflow_guardrail_injects_integration_dependency_scenarios(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    spec = {
        "openapi": "3.0.3",
        "info": {"title": "Workflow API", "version": "1.0.0"},
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
                    "responses": {"201": {"description": "created"}},
                }
            },
            "/orders/{orderId}": {
                "get": {
                    "parameters": [
                        {"name": "orderId", "in": "path", "required": True, "schema": {"type": "string"}}
                    ],
                    "responses": {"200": {"description": "ok"}, "404": {"description": "not found"}},
                }
            },
        },
    }
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=8,
    )
    loaded = agent._load_spec()
    agent._operation_index = agent._build_operation_index(loaded)
    agent._auth_required_ops = agent._build_auth_requirement_map(loaded)

    base = []
    spec_intelligence = {
        "workflow_candidates": [
            {"sequence": ["POST /orders", "GET /orders/{orderId}"], "reason": "path_parameter_dependency"}
        ]
    }
    scenarios, summary = agent._inject_workflow_sequence_scenarios(
        base,
        spec_intelligence=spec_intelligence,
        limit=4,
    )
    integration = [s for s in scenarios if s.test_type == ScenarioType.INTEGRATION]
    assert len(integration) >= 1
    assert summary.get("added_sequence_probes", 0) >= 1


def test_policy_movement_metrics_and_persist_state(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )
    prior = "GET|/orders/{orderId}|authentication|401|body=0|params=0"
    agent.learning_state["last_selected_fingerprints"] = [prior]

    current = [
        ScenarioModel(
            name="test_get_orders_auth",
            description="auth negative",
            test_type=ScenarioType.AUTHENTICATION,
            endpoint="/orders/{orderId}",
            method="GET",
            expected_status=401,
        ),
        ScenarioModel(
            name="test_get_orders_success",
            description="happy path",
            test_type=ScenarioType.HAPPY_PATH,
            endpoint="/orders/{orderId}",
            method="GET",
            expected_status=200,
        ),
    ]
    metrics = agent._build_policy_movement_metrics(current)
    assert metrics.get("current_count", 0) >= 1
    assert metrics.get("status") in {"stable", "shifted", "cold_start"}
    agent._persist_policy_movement_state(current)
    assert isinstance(agent.learning_state.get("last_selected_fingerprints"), list)


def test_prod_safe_blocks_unsafe_methods_without_network(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        base_url="http://127.0.0.1:9",
        max_scenarios=2,
        environment_profile="prod_safe",
    )
    scenario = ScenarioModel(
        name="test_post_orders_blocked",
        description="unsafe write in prod_safe",
        test_type=ScenarioType.ERROR_HANDLING,
        endpoint="/orders",
        method="POST",
        expected_status=400,
        body={"x": "y"},
    )
    results = agent._execute_against_live_api([scenario])
    assert len(results) == 1
    assert results[0].passed is False
    assert "unsafe_action_blocked_prod_safe" in str(results[0].error)


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


def test_prepare_scenarios_normalizes_schema_valid_boundary_payload_to_happy_status(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "spec.yaml"
    spec = {
        "openapi": "3.0.3",
        "info": {"title": "Boundary API", "version": "1.0.0"},
        "paths": {
            "/products": {
                "post": {
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["name", "price"],
                                    "properties": {
                                        "name": {"type": "string", "minLength": 1},
                                        "price": {"type": "number", "minimum": 0},
                                        "stock": {"type": "integer", "minimum": 0},
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "201": {"description": "created"},
                        "400": {"description": "bad request"},
                    },
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
    loaded = agent._load_spec()
    agent._operation_index = agent._build_operation_index(loaded)

    scenario = ScenarioModel(
        name="test_post_products_boundary_price_zero",
        description="boundary payload at minimum should still be valid",
        test_type=ScenarioType.BOUNDARY_TESTING,
        endpoint="/products",
        method="POST",
        expected_status=400,
        body={"name": "Free Product", "price": 0, "stock": 10},
    )

    prepared = agent._prepare_scenarios_for_execution_and_scripts([scenario])
    assert len(prepared) == 1
    assert int(prepared[0].expected_status) == 201


def test_prepare_scenarios_normalizes_unconstrained_path_input_validation_to_404(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "spec.yaml"
    spec = {
        "openapi": "3.0.3",
        "info": {"title": "Path Input API", "version": "1.0.0"},
        "paths": {
            "/orders/{orderId}": {
                "get": {
                    "parameters": [
                        {
                            "name": "orderId",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {"description": "ok"},
                        "404": {"description": "not found"},
                    },
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
    loaded = agent._load_spec()
    agent._operation_index = agent._build_operation_index(loaded)

    scenario = ScenarioModel(
        name="test_get_orders_invalid_id_format",
        description="input validation with unconstrained string path id",
        test_type=ScenarioType.INPUT_VALIDATION,
        endpoint="/orders/{orderId}",
        method="GET",
        expected_status=400,
        params={"orderId": "!nv@l!d"},
    )

    prepared = agent._prepare_scenarios_for_execution_and_scripts([scenario])
    assert len(prepared) == 1
    assert int(prepared[0].expected_status) == 404
    assert str(prepared[0].params.get("orderId")) == "nonexistent_dependency_id"


def test_prepare_scenarios_normalizes_unconstrained_edge_case_path_status_to_happy(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "spec.yaml"
    spec = {
        "openapi": "3.0.3",
        "info": {"title": "Edge Case API", "version": "1.0.0"},
        "paths": {
            "/orders/{orderId}": {
                "get": {
                    "parameters": [
                        {
                            "name": "orderId",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {"description": "ok"},
                        "404": {"description": "not found"},
                    },
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
    loaded = agent._load_spec()
    agent._operation_index = agent._build_operation_index(loaded)

    scenario = ScenarioModel(
        name="test_get_orders_huge_id_edge_case",
        description="edge case with huge unconstrained id",
        test_type=ScenarioType.EDGE_CASES,
        endpoint="/orders/{orderId}",
        method="GET",
        expected_status=400,
        params={"orderId": "a1234567890123456789012345678901234567890"},
    )

    prepared = agent._prepare_scenarios_for_execution_and_scripts([scenario])
    assert len(prepared) == 1
    assert int(prepared[0].expected_status) == 200


def test_history_seed_stage_adds_learning_driven_candidates(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )
    spec = agent._load_spec()
    agent._auth_required_ops = agent._build_auth_requirement_map(spec)

    weak_fp = "GET|/orders/{orderId}|error_handling|404|body=0|params=1"
    agent.learning_state["scenario_stats"] = {
        weak_fp: {
            "attempts": 5,
            "passes": 0,
            "failures": 5,
            "avg_reward": -1.2,
            "failure_rate": 1.0,
            "test_type": "error_handling",
            "method": "GET",
            "endpoint": "/orders/{orderId}",
            "expected_status": 404,
            "actual_status_counts": {"200": 5},
        }
    }

    base = [
        ScenarioModel(
            name="test_get_orders_no_auth",
            description="auth baseline",
            test_type=ScenarioType.AUTHENTICATION,
            endpoint="/orders/{orderId}",
            method="GET",
            expected_status=401,
        )
    ]

    candidates, mutation_summary = agent._augment_scenarios_with_rl_mutation(spec, base)
    assert mutation_summary.get("history_seed_candidates_added", 0) >= 1
    assert any("rl_history_seed" in scenario.name for scenario in candidates)


def test_history_seed_boundary_auth_status_is_remapped(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)
    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=8,
    )

    scenario = agent._build_history_seed_scenario(
        {
            "endpoint": "/orders/{orderId}",
            "method": "GET",
            "test_type": "boundary_testing",
            "expected_status": 401,
            "op_meta": {
                "response_statuses": [200, 401, 404],
                "path_param_names": ["orderId"],
            },
            "has_body": False,
            "has_params": True,
            "attempts": 2,
            "failure_rate": 0.5,
        }
    )

    assert scenario.expected_status == 404


def test_rl_mutation_auth_negative_keeps_auth_focused_variants(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )
    spec = agent._load_spec()
    agent._auth_required_ops = agent._build_auth_requirement_map(spec)
    op_meta = agent._build_operation_index(spec).get("GET /orders/{orderId}", {})

    scenario = ScenarioModel(
        name="test_get_orders_no_auth",
        description="auth scenario without token",
        test_type=ScenarioType.AUTHENTICATION,
        endpoint="/orders/{orderId}",
        method="GET",
        expected_status=401,
    )

    variants = agent._build_rl_mutations_for_scenario(
        scenario,
        max_variants=8,
        operation_meta=op_meta,
        target_context={
            "failure_rate": 0.8,
            "op_failure_rate": 0.8,
            "uncertainty": 0.3,
            "op_attempts": 8,
            "dominant_actual_status": 401,
        },
    )
    strategies = _mutation_strategies(variants)
    assert "invalid_auth" in strategies
    assert "path_not_found" not in strategies
    assert "path_type_fuzz" not in strategies
    assert "query_fuzz" not in strategies
    assert all(int(item.expected_status) == 401 for item in variants)


def test_rl_mutation_method_not_allowed_avoids_path_variants(tmp_path: Path) -> None:
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
        name="test_delete_orders_invalid_method",
        description="invalid method should return 405",
        test_type=ScenarioType.ERROR_HANDLING,
        endpoint="/orders/{orderId}",
        method="DELETE",
        expected_status=405,
    )

    variants = agent._build_rl_mutations_for_scenario(
        scenario,
        max_variants=8,
        operation_meta={},
        target_context={
            "failure_rate": 0.7,
            "op_failure_rate": 0.7,
            "uncertainty": 0.4,
            "op_attempts": 10,
            "dominant_actual_status": 405,
        },
    )
    assert variants == []


def test_rl_mutation_skips_auth_adaptive_hypothesis_for_non_auth_case(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )
    spec = agent._load_spec()
    agent._auth_required_ops = agent._build_auth_requirement_map(spec)
    op_meta = agent._build_operation_index(spec).get("GET /orders/{orderId}", {})

    scenario = ScenarioModel(
        name="test_get_orders_query_fuzz",
        description="non-auth security scenario",
        test_type=ScenarioType.SECURITY,
        endpoint="/orders/{orderId}",
        method="GET",
        expected_status=400,
        params={"q": "normal"},
    )

    variants = agent._build_rl_mutations_for_scenario(
        scenario,
        max_variants=8,
        operation_meta=op_meta,
        target_context={
            "failure_rate": 0.6,
            "op_failure_rate": 0.6,
            "uncertainty": 0.3,
            "op_attempts": 12,
            "dominant_actual_status": 401,
        },
    )
    strategies = _mutation_strategies(variants)
    assert "adaptive_status_hypothesis" not in strategies


def test_rl_mutation_skips_integration_adaptive_status_polarity_flip(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )
    spec = agent._load_spec()
    op_meta = agent._build_operation_index(spec).get("GET /orders/{orderId}", {})

    scenario = ScenarioModel(
        name="test_get_orders_dependency_unmet",
        description="integration unmet dependency",
        test_type=ScenarioType.INTEGRATION,
        endpoint="/orders/{orderId}",
        method="GET",
        expected_status=404,
        params={"orderId": "nonexistent_dependency_id"},
    )

    variants = agent._build_rl_mutations_for_scenario(
        scenario,
        max_variants=8,
        operation_meta=op_meta,
        target_context={
            "failure_rate": 0.4,
            "op_failure_rate": 0.4,
            "uncertainty": 0.4,
            "op_attempts": 12,
            "dominant_actual_status": 200,
        },
    )
    strategies = _mutation_strategies(variants)
    assert "adaptive_status_hypothesis" not in strategies


def test_path_type_fuzz_mutation_skips_string_path_params(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=6,
    )
    spec = agent._load_spec()
    op_meta = agent._build_operation_index(spec).get("GET /orders/{orderId}", {})

    scenario = ScenarioModel(
        name="test_get_order_happy",
        description="happy path",
        test_type=ScenarioType.HAPPY_PATH,
        endpoint="/orders/{orderId}",
        method="GET",
        expected_status=200,
    )

    variants = agent._build_rl_mutations_for_scenario(
        scenario,
        max_variants=8,
        operation_meta=op_meta,
        target_context={
            "failure_rate": 0.3,
            "op_failure_rate": 0.3,
            "uncertainty": 0.6,
            "op_attempts": 10,
        },
    )
    strategies = _mutation_strategies(variants)
    assert "path_type_fuzz" not in strategies


def test_query_fuzz_mutation_does_not_overwrite_path_params(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=6,
    )
    spec = agent._load_spec()
    op_meta = agent._build_operation_index(spec).get("GET /orders/{orderId}", {})

    scenario = ScenarioModel(
        name="test_get_order_integration",
        description="integration baseline",
        test_type=ScenarioType.INTEGRATION,
        endpoint="/orders/{orderId}",
        method="GET",
        expected_status=200,
        params={"orderId": "123"},
    )

    variants = agent._build_rl_mutations_for_scenario(
        scenario,
        max_variants=8,
        operation_meta=op_meta,
        target_context={
            "failure_rate": 0.2,
            "op_failure_rate": 0.2,
            "uncertainty": 0.4,
            "op_attempts": 8,
        },
    )

    query_variants = [
        item
        for item in variants
        if "rl_mutation:query_fuzz" in list(item.assertions or [])
    ]
    assert query_variants
    params = dict(query_variants[0].params or {})
    assert params.get("orderId") == "123"
    assert "q" in params


def test_selection_enforces_novelty_quota_after_warmup(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=8,
    )
    agent.learning_state["run_count"] = 5
    agent._operation_index = {}

    old_scenarios = [
        ScenarioModel(
            name=f"old_{idx}",
            description="historical scenario",
            test_type=ScenarioType.ERROR_HANDLING,
            endpoint=f"/old{idx}",
            method="GET",
            expected_status=400,
        )
        for idx in range(1, 7)
    ]
    new_scenarios = [
        ScenarioModel(
            name=f"new_{idx}",
            description="novel scenario",
            test_type=ScenarioType.ERROR_HANDLING,
            endpoint=f"/new{idx}",
            method="GET",
            expected_status=400,
        )
        for idx in range(1, 4)
    ]
    scenarios = old_scenarios + new_scenarios

    for item in old_scenarios:
        fp = agent._scenario_fingerprint(item)
        agent.adaptive_policy.scenario_stats[fp] = {
            "attempts": 12,
            "passes": 12,
            "failures": 0,
            "avg_reward": 0.95,
            "failure_rate": 0.0,
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
        base = 2.0 if endpoint.startswith("/old") else 0.2
        uncertainty = 0.2
        return {
            "score": base + novelty_bonus + legacy_weight_bonus + rl_risk,
            "expected_reward": base,
            "uncertainty": uncertainty,
            "exploration_bonus": 0.0,
            "failure_focus_bonus": 0.0,
            "historical_reward": base,
        }

    agent.adaptive_policy.score = fake_score  # type: ignore[assignment]
    selected = agent._select_scenarios_with_learning(scenarios)

    selected_new = [item for item in selected if item.endpoint.startswith("/new")]
    assert len(selected_new) >= 2
    assert int(agent._last_selection_summary.get("novelty_target", 0)) >= 2
    assert int(agent._last_selection_summary.get("novel_selected_count", 0)) >= 2


def test_script_sync_does_not_apply_corrected_expectation_status(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )

    scenario = ScenarioModel(
        name="test_path_type_fuzz",
        description="boundary scenario with corrected expectation",
        test_type=ScenarioType.BOUNDARY_TESTING,
        endpoint="/orders/{orderId}",
        method="GET",
        expected_status=400,
        params={"orderId": "not-a-valid-id"},
    )
    result = ScenarioExecutionResult(
        name=scenario.name,
        test_type=scenario.test_type.value,
        method=scenario.method,
        endpoint_template=scenario.endpoint,
        endpoint_resolved="/orders/not-a-valid-id",
        expected_status=400,
        actual_status=200,
        verdict="suspect",
        passed=True,
        duration_ms=0.4,
        error="",
        response_excerpt="{}",
        verification={
            "passed": True,
            "verdict": "suspect",
            "status_check": {
                "expected_status": 400,
                "actual_status": 200,
                "corrected_expected_status": 200,
                "corrected": True,
                "documented_statuses": [200, 401, 404],
            },
            "contract_check": {
                "checked": False,
                "schema_found": False,
                "valid": None,
                "issues": [],
            },
        },
    )

    synced = agent._sync_scenarios_with_corrected_expectations([scenario], [result])
    assert len(synced) == 1
    assert synced[0].name == scenario.name
    assert int(synced[0].expected_status) == 400


def test_corrected_expectation_is_marked_suspect_not_pass(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )
    agent._operation_index = {
        "GET /orders/{orderId}": {
            "response_statuses": [200, 401, 404],
            "response_schemas": {},
        }
    }
    scenario = ScenarioModel(
        name="test_path_type_fuzz",
        description="boundary scenario with corrected expectation",
        test_type=ScenarioType.BOUNDARY_TESTING,
        endpoint="/orders/{orderId}",
        method="GET",
        expected_status=400,
        params={"orderId": "not-a-valid-id"},
    )

    class _Resp:
        status_code = 200
        text = "{}"

        @staticmethod
        def json() -> dict:
            return {}

    verification = agent._verify_then_correct_result(
        scenario=scenario,
        actual_status=200,
        response=_Resp(),
        query_params={},
        body=None,
    )
    assert verification["passed"] is False
    assert verification["verdict"] == "suspect"
    assert verification["status_check"]["corrected"] is True


def test_operation_index_resolves_request_body_and_response_refs(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    spec = {
        "openapi": "3.0.3",
        "info": {"title": "Ref API", "version": "1.0.0"},
        "paths": {
            "/resources": {
                "get": {
                    "responses": {
                        "200": {"$ref": "#/components/responses/ListResourcesResponse"}
                    }
                },
                "post": {
                    "requestBody": {"$ref": "#/components/requestBodies/CreateResourceBody"},
                    "responses": {"201": {"description": "created"}},
                },
            }
        },
        "components": {
            "responses": {
                "ListResourcesResponse": {
                    "description": "ok",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {"items": {"type": "array", "items": {"type": "object"}}},
                            }
                        }
                    },
                }
            },
            "requestBodies": {
                "CreateResourceBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["name"],
                                "properties": {"name": {"type": "string"}},
                            }
                        }
                    },
                }
            },
        },
    }
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )

    index = agent._build_operation_index(spec)
    assert "200" in index["GET /resources"]["response_schemas"]
    assert index["POST /resources"]["required_fields"] == ["name"]


def test_summary_quality_gate_fails_when_llm_generation_is_degraded_in_strict_mode(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )
    agent._scenario_llm_mode = "on"
    agent._llm_generation_degraded = True
    agent._llm_generation_degraded_reason = "llm failed"

    scenario = ScenarioModel(
        name="test_ok",
        description="happy path",
        test_type=ScenarioType.HAPPY_PATH,
        endpoint="/orders/{orderId}",
        method="GET",
        expected_status=200,
        params={"orderId": "123"},
    )
    result = ScenarioExecutionResult(
        name=scenario.name,
        test_type=scenario.test_type.value,
        method=scenario.method,
        endpoint_template=scenario.endpoint,
        endpoint_resolved="/orders/123",
        expected_status=200,
        actual_status=200,
        verdict="pass",
        passed=True,
        duration_ms=0.2,
        verification={
            "passed": True,
            "verdict": "pass",
            "status_check": {"expected_status": 200, "actual_status": 200, "corrected": False},
            "contract_check": {"checked": False, "schema_found": False, "valid": None, "issues": []},
        },
    )

    summary = agent._build_summary({"paths": {}}, [scenario], [result])
    assert summary["pass_rate"] == 1.0
    assert summary["meets_quality_gate"] is False
    assert "llm_generation_degraded" in summary["quality_gate_fail_reasons"]
    assert summary["llm_degradation_policy"]["gate_blocking"] is True
    assert summary["llm_degradation_policy"]["llm_mode"] == "on"


def test_summary_quality_gate_allows_auto_mode_llm_fallback_when_coverage_is_good(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )
    agent._scenario_llm_mode = "auto"
    agent._llm_generation_degraded = True
    agent._llm_generation_degraded_reason = "llm failed"

    scenario = ScenarioModel(
        name="test_ok",
        description="happy path",
        test_type=ScenarioType.HAPPY_PATH,
        endpoint="/orders/{orderId}",
        method="GET",
        expected_status=200,
        params={"orderId": "123"},
    )
    result = ScenarioExecutionResult(
        name=scenario.name,
        test_type=scenario.test_type.value,
        method=scenario.method,
        endpoint_template=scenario.endpoint,
        endpoint_resolved="/orders/123",
        expected_status=200,
        actual_status=200,
        verdict="pass",
        passed=True,
        duration_ms=0.2,
        verification={
            "passed": True,
            "verdict": "pass",
            "status_check": {"expected_status": 200, "actual_status": 200, "corrected": False},
            "contract_check": {"checked": False, "schema_found": False, "valid": None, "issues": []},
        },
    )

    summary = agent._build_summary(agent._load_spec(), [scenario], [result])
    assert summary["pass_rate"] == 1.0
    assert summary["meets_quality_gate"] is True
    assert "llm_generation_degraded" not in summary["quality_gate_fail_reasons"]
    assert "llm_generation_degraded_auto_fallback" in summary["quality_gate_warnings"]
    assert summary["llm_degradation_policy"]["gate_blocking"] is False
    assert summary["llm_degradation_policy"]["fallback_quality_ok"] is True


def test_summary_emits_llm_degradation_policy_diagnostics(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )
    agent._scenario_llm_mode = "auto"
    agent._llm_generation_degraded = True
    agent._llm_generation_degraded_reason = "llm failed"

    scenario = ScenarioModel(
        name="test_not_enough_coverage",
        description="non-happy scenario only",
        test_type=ScenarioType.AUTHENTICATION,
        endpoint="/orders/{orderId}",
        method="GET",
        expected_status=401,
    )
    result = ScenarioExecutionResult(
        name=scenario.name,
        test_type=scenario.test_type.value,
        method=scenario.method,
        endpoint_template=scenario.endpoint,
        endpoint_resolved="/orders/123",
        expected_status=401,
        actual_status=401,
        verdict="pass",
        passed=True,
        duration_ms=0.2,
        verification={
            "passed": True,
            "verdict": "pass",
            "status_check": {"expected_status": 401, "actual_status": 401, "corrected": False},
            "contract_check": {"checked": False, "schema_found": False, "valid": None, "issues": []},
        },
    )

    summary = agent._build_summary(agent._load_spec(), [scenario], [result])
    policy = summary["llm_degradation_policy"]
    assert isinstance(policy, dict)
    assert policy["llm_mode"] == "auto"
    assert policy["degraded"] is True
    assert policy["required_operation_count"] >= 1
    assert policy["happy_path_coverage_complete"] is False
    assert policy["gate_blocking"] is True
    assert "llm_generation_degraded" in summary["quality_gate_fail_reasons"]


def test_gam_context_pack_requires_direct_weak_signal(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )

    pack = agent._build_gam_context_pack(
        memory_excerpts=[
            {
                "source": "trusted_docs_fallback",
                "title": "Agent Lightning RL Focus",
                "tags": ["trusted_docs", "agent_lightning", "rl"],
                "excerpt": (
                    "No persistent weak pattern detected yet; next action is exploration-focused "
                    "scenario diversification. Trend delta action: track run-over-run deltas."
                ),
            },
            {
                "source": "trusted_docs_fallback",
                "title": "Schemathesis Stateful Testing",
                "tags": ["trusted_docs", "schemathesis", "stateful"],
                "excerpt": "Spec risk action: prioritize stateful producer->consumer sequences.",
            },
            {
                "source": "trusted_docs_fallback",
                "title": "GAM Context Quality Contract",
                "tags": ["trusted_docs", "gam", "context_pack"],
                "excerpt": "Next action: enrich memory with one trend delta and one concrete mutation action.",
            },
        ],
        diagnostics={"quality_score": 1.0, "warnings": []},
        reflection="ok",
    )

    assert pack["status"] == "needs_retry"
    assert bool(pack["contract_checks"].get("has_weak_pattern")) is False
    assert str(pack["contract_checks"].get("weak_pattern_mode")) == "exploration_proxy"


def test_learned_numeric_underflow_mutation_keeps_required_fields(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    spec = {
        "openapi": "3.0.3",
        "info": {"title": "Mutation Body API", "version": "1.0.0"},
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
                                        "quantity": {"type": "integer", "minimum": 1},
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "201": {"description": "created"},
                        "400": {"description": "bad request"},
                        "401": {"description": "unauthorized"},
                    },
                    "security": [{"bearerAuth": []}],
                }
            }
        },
        "components": {
            "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer"}}
        },
    }
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=8,
    )
    loaded = agent._load_spec()
    agent._auth_required_ops = agent._build_auth_requirement_map(loaded)
    op_meta = agent._build_operation_index(loaded)["POST /orders"]

    base = ScenarioModel(
        name="test_post_orders_no_auth",
        description="base auth-negative scenario",
        test_type=ScenarioType.AUTHENTICATION,
        endpoint="/orders",
        method="POST",
        expected_status=401,
    )

    variants = agent._build_rl_mutations_for_scenario(
        base,
        max_variants=20,
        operation_meta=op_meta,
        target_context={
            "failure_rate": 0.7,
            "op_failure_rate": 0.7,
            "uncertainty": 0.4,
            "op_attempts": 10,
        },
    )

    underflow_variants = [
        item
        for item in variants
        if "rl_mutation:learned_numeric_underflow" in list(item.assertions or [])
    ]
    assert underflow_variants
    underflow_body = dict(underflow_variants[0].body or {})
    assert underflow_body.get("productId") not in {None, ""}
    assert underflow_body.get("quantity") is not None
    assert float(underflow_body["quantity"]) < 1.0


def test_build_curl_repro_command_masks_auth_and_quotes_payload(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)
    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )

    cmd = agent._build_curl_repro_command(
        method="POST",
        base_url_hint="http://testserver",
        endpoint_resolved="/products",
        headers={
            "Authorization": "Bearer valid_token",
            "Content-Type": "application/json",
        },
        query={"q": "x y", "page": 1},
        body={"name": "Product'; DROP TABLE products;--", "price": 10},
    )

    assert "Bearer ***" in cmd
    assert "valid_token" not in cmd
    assert "Content-Type: application/json" in cmd
    assert "q=x+y&page=1" in cmd
    assert "DROP TABLE products" in cmd


def test_verify_response_contract_uses_json_fallback_when_schema_missing(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)
    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=4,
    )

    scenario = ScenarioModel(
        name="test_get_orders_happy",
        description="happy path",
        test_type=ScenarioType.HAPPY_PATH,
        endpoint="/orders/{orderId}",
        method="GET",
        expected_status=200,
    )
    response = _FakeResponse({"ok": True})

    contract = agent._verify_response_contract(
        scenario=scenario,
        actual_status=200,
        response=response,
        query_params={},
        body=None,
        operation_meta={},
    )

    assert contract["checked"] is True
    assert contract["schema_found"] is False
    assert contract["valid"] is True
    assert "json_sanity_check" in " ".join(contract.get("issues", []))


def test_forced_weak_replay_respects_cadence(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)
    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=8,
    )
    loaded = agent._load_spec()
    agent._operation_index = agent._build_operation_index(loaded)
    agent._spec_paths = set((loaded.get("paths") or {}).keys())
    fingerprint = "GET|/orders/{orderId}|authentication|401|body=0|params=1"
    agent.learning_state["run_count"] = 10
    agent.adaptive_policy.scenario_stats = {
        fingerprint: {
            "attempts": 6,
            "failure_rate": 1.0,
            "avg_reward": -0.2,
            "method": "GET",
            "endpoint": "/orders/{orderId}",
            "test_type": "authentication",
            "expected_status": 401,
        }
    }
    agent.learning_state["replay_schedule"] = {fingerprint: 9}
    assert agent._build_forced_weak_replay_targets(limit=2) == []

    agent.learning_state["replay_schedule"] = {fingerprint: 7}
    targets = agent._build_forced_weak_replay_targets(limit=2)
    assert targets
    assert targets[0]["fingerprint"] == fingerprint


def test_extract_focus_points_avoids_recent_repetition(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)
    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=8,
    )
    repeated = "Weak RL pattern: GET /orders/{orderId} expected 404, failure_rate=1.0."
    agent.learning_state["gam_recent_focus_points"] = [repeated.lower()]
    points = agent._extract_focus_points_from_excerpts(
        [
            {"source": "memo", "title": "a", "excerpt": repeated},
            {"source": "memo", "title": "b", "excerpt": "Trend delta: run_reward improved by +0.10"},
        ],
        limit=2,
    )
    assert all("weak rl pattern: get /orders/{orderid}" not in p.lower() for p in points)


def test_selection_summary_contains_portfolio_policy(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    _write_spec(spec_path)
    agent = QASpecialistAgent(
        spec_path=str(spec_path),
        output_dir=str(tmp_path / "out"),
        max_scenarios=10,
    )
    scenarios = [
        ScenarioModel(
            name=f"s{i}",
            description="case",
            test_type=ScenarioType.AUTHENTICATION if i < 3 else ScenarioType.HAPPY_PATH,
            endpoint=f"/orders/{{orderId}}",
            method="GET",
            expected_status=401 if i < 3 else 200,
            params={"orderId": str(i)},
        )
        for i in range(10)
    ]
    selected = agent._select_scenarios_with_learning(scenarios)
    assert selected
    policy = agent._last_selection_summary.get("portfolio_policy", {})
    assert isinstance(policy, dict)
    assert policy.get("stable_ratio") == 0.7
    assert policy.get("focus_ratio") == 0.2
    assert policy.get("explore_ratio") == 0.1
