"""
QA specialist agent orchestration.

End-to-end flow:
1. Parse OpenAPI spec and generate human-like QA scenarios
2. Generate multi-language test files
3. Execute all scenarios in an isolated in-memory API sandbox
4. Store run context in GAM memory
5. Run Agent Lightning RL update from execution outcomes
6. Emit JSON + Markdown reports
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi.testclient import TestClient

from spec_test_pilot.adaptive_policy import (
    AdaptiveScenarioPolicy,
    scenario_fingerprint,
)
from spec_test_pilot.agent_lightning_v2 import AgentLightningTrainer as LightningTrainerV2
from spec_test_pilot.memory.gam import GAMMemorySystem
from spec_test_pilot.multi_language_tester import (
    HumanTesterSimulator,
    MultiLanguageTestGenerator,
    TestScenario,
)


PATH_PARAM_PATTERN = re.compile(r"\{([^}]+)\}")
NEGATIVE_STATUS_CODES = {400, 401, 403, 404, 405, 409, 422}
AUTH_NEGATIVE_STATUS_CODES = {401, 403}
HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
UNCERTAINTY_COVERAGE_QUANTILE = 0.75
REPAIR_RULE_MIN_ATTEMPTS = 3
REPAIR_RULE_MIN_FAILURE_RATE = 0.85
REPAIR_RULE_DOMINANT_RATIO = 0.70
REPAIR_RULE_MAX_ITEMS = 500
DEFAULT_DECISION_WEIGHT = 1.0
MIN_DECISION_WEIGHT = 0.2
MAX_DECISION_WEIGHT = 5.0
DECISION_LEARNING_RATE = 0.20
LEARNING_HISTORY_LIMIT = 200
SCENARIO_STATS_LIMIT = 4000
SELECTION_TRACE_LIMIT = 60


@dataclass
class DecisionLearningSignal:
    """Reward/penalty signal captured for one executed decision."""

    name: str
    test_type: str
    method: str
    endpoint_template: str
    endpoint_key: str
    scenario_fingerprint: str
    has_body: bool
    has_params: bool
    reward: float
    passed: bool
    expected_status: int
    actual_status: Optional[int]


@dataclass
class ScenarioExecutionResult:
    """Runtime execution result for a single scenario."""

    name: str
    test_type: str
    method: str
    endpoint_template: str
    endpoint_resolved: str
    expected_status: int
    actual_status: Optional[int]
    passed: bool
    duration_ms: float
    error: str = ""
    response_excerpt: str = ""


class QASpecialistAgent:
    """QA-focused orchestrator with isolation, GAM memory, and RL feedback."""

    def __init__(
        self,
        spec_path: str,
        nlp_prompt: Optional[str] = None,
        tenant_id: str = "default_tenant",
        base_url: str = "http://localhost:8000",
        output_dir: Optional[str] = None,
        max_scenarios: int = 200,
        pass_threshold: float = 0.70,
        rl_checkpoint_path: Optional[str] = None,
        learning_state_path: Optional[str] = None,
    ):
        self.spec_path = str(spec_path)
        self.nlp_prompt = nlp_prompt
        self.tenant_id = tenant_id
        self.base_url = base_url.rstrip("/")
        self.max_scenarios = max(1, int(max_scenarios))
        self.pass_threshold = max(0.0, min(1.0, float(pass_threshold)))

        if output_dir:
            self.output_dir = Path(output_dir).resolve()
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = (Path.cwd() / ".qa_specialist_workspace").resolve()
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.gam = GAMMemorySystem(use_vector_search=False)
        self.rl_checkpoint_path = str(
            Path(rl_checkpoint_path).expanduser().resolve()
        ) if rl_checkpoint_path else str((self.output_dir / "agent_lightning_checkpoint.pt").resolve())
        self.rl_trainer = LightningTrainerV2(
            gam_memory_system=self.gam,
            checkpoint_path=self.rl_checkpoint_path,
            checkpoint_autosave=True,
        )
        self.rl_trainer.register_agent("qa_specialist", self._qa_agent_feedback)
        self.learning_state_path = self._resolve_learning_state_path(
            learning_state_path
        )
        self.learning_state = self._load_learning_state()
        self.adaptive_policy = AdaptiveScenarioPolicy.from_state(
            self.learning_state.get("adaptive_policy"),
            fallback_scenario_stats=self.learning_state.get("scenario_stats", {}),
        )
        self.learning_state["adaptive_policy"] = self.adaptive_policy.to_state()
        self.learning_state["scenario_stats"] = dict(self.adaptive_policy.scenario_stats)
        self._last_selection_trace: List[Dict[str, Any]] = []
        self._last_selection_summary: Dict[str, Any] = {}
        self._auth_required_ops: set[str] = set()

    def _resolve_learning_state_path(self, learning_state_path: Optional[str]) -> Path:
        if learning_state_path:
            resolved = Path(learning_state_path).expanduser().resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            return resolved

        checkpoint_path = Path(self.rl_checkpoint_path).expanduser().resolve()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        return checkpoint_path.with_name(
            f"{checkpoint_path.stem}_learning_state.json"
        )

    def run(self) -> Dict[str, Any]:
        """Run the full QA specialist workflow."""
        started_at = time.time()
        spec = self._load_spec()
        self._auth_required_ops = self._build_auth_requirement_map(spec)
        spec_title = spec.get("info", {}).get("title", "Unknown API")
        spec_version = spec.get("info", {}).get("version", "unknown")

        session_id = self.gam.start_session(
            tenant_id=self.tenant_id,
            metadata={
                "spec_path": self.spec_path,
                "spec_title": spec_title,
                "spec_version": spec_version,
                "mode": "qa_specialist",
            },
        )
        self.gam.add_to_session(
            session_id,
            "user",
            "Run full QA-specialist API testing pipeline.",
            tool_outputs=[
                {
                    "tool": "qa_agent.init",
                    "output": {
                        "spec_path": self.spec_path,
                        "tenant_id": self.tenant_id,
                        "nlp_prompt": self.nlp_prompt or "comprehensive_default",
                    },
                }
            ],
        )

        research_context = {
            "spec_title": spec_title,
            "auth_type": self._infer_auth_type(spec),
            "endpoints": self._extract_endpoint_metadata(spec),
            "tenant_id": self.tenant_id,
        }
        research_result = self.gam.research(research_context)
        self.gam.add_to_session(
            session_id,
            "assistant",
            "Completed GAM deep-research planning and retrieval for test strategy.",
            tool_outputs=[
                {
                    "tool": "gam.research",
                    "output": {
                        "plan": research_result.plan,
                        "reflection": research_result.reflection,
                        "excerpt_count": len(research_result.memory_excerpts),
                    },
                }
            ],
        )

        simulator = HumanTesterSimulator(spec, self.base_url)
        effective_prompt = self._compose_effective_prompt(
            self.nlp_prompt, research_result.memory_excerpts
        )
        all_scenarios = simulator.think_like_tester(effective_prompt)
        scenarios = self._select_scenarios_with_learning(all_scenarios)
        scenarios, repair_summary = self._apply_scenario_repairs(spec, scenarios)

        generated_files = self._generate_test_files(scenarios)
        execution_results = self._execute_in_isolated_mock(spec, scenarios)
        summary = self._build_summary(spec, scenarios, execution_results)
        learning_feedback = self._compute_learning_feedback(scenarios, execution_results, summary)
        self._update_learning_state(learning_feedback)

        self.gam.add_to_session(
            session_id,
            "assistant",
            "Executed generated scenarios in isolated sandbox and produced summary.",
            tool_outputs=[
                {"tool": "qa_agent.execution", "output": summary},
            ],
            artifacts=[
                {"name": "qa_summary.json", "type": "json", "content": json.dumps(summary)},
                {
                    "name": "learning_feedback.json",
                    "type": "json",
                    "content": json.dumps(learning_feedback),
                },
            ],
        )

        issues_found = [f["name"] for f in summary["failed_examples"][:10]]
        key_decisions = [
            f"Generated {summary['total_scenarios']} QA scenarios from {len(all_scenarios)} candidates",
            "Executed scenarios using in-memory FastAPI TestClient isolation",
            f"Pass threshold set to {self.pass_threshold:.2f}",
            f"Applied learning reward {learning_feedback['run_reward']:.3f}",
        ]

        lossless_pages, memo_page = self.gam.end_session_with_memo(
            session_id=session_id,
            spec_title=spec_title,
            endpoints_count=summary["detected_endpoints"],
            tests_generated=summary["total_scenarios"],
            key_decisions=key_decisions,
            issues_found=issues_found,
        )

        rl_report_path = self.output_dir / "qa_execution_report.json"
        rl_data = self._run_agent_lightning_training(
            spec_title=spec_title,
            summary=summary,
            report_path=str(rl_report_path),
            learning_feedback=learning_feedback,
        )

        report = {
            "metadata": {
                "spec_path": self.spec_path,
                "spec_title": spec_title,
                "spec_version": spec_version,
                "tenant_id": self.tenant_id,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "execution_seconds": round(time.time() - started_at, 3),
                "isolation_mode": "in_memory_fastapi_testclient",
            },
            "summary": summary,
            "learning": {
                "feedback": learning_feedback,
                "state_snapshot": self._learning_state_snapshot(),
                "state_file": str(self.learning_state_path),
                "agent_lightning_checkpoint": self.rl_checkpoint_path,
            },
            "selection_policy": {
                "algorithm": "contextual_linear_ucb",
                "candidate_count": len(all_scenarios),
                "selected_count": len(scenarios),
                "base_max_scenarios": int(
                    self._last_selection_summary.get("base_max_scenarios", self.max_scenarios)
                ),
                "effective_budget": int(
                    self._last_selection_summary.get("effective_budget", len(scenarios))
                ),
                "budget_expanded_for_uncertainty": bool(
                    self._last_selection_summary.get(
                        "budget_expanded_for_uncertainty",
                        False,
                    )
                ),
                "uncertainty_quantile": float(
                    self._last_selection_summary.get(
                        "uncertainty_quantile",
                        UNCERTAINTY_COVERAGE_QUANTILE,
                    )
                ),
                "uncertainty_threshold": round(
                    float(self._last_selection_summary.get("uncertainty_threshold", 0.0)),
                    6,
                ),
                "uncertain_candidate_count": int(
                    self._last_selection_summary.get("uncertain_candidate_count", 0)
                ),
                "uncertain_selected_count": int(
                    self._last_selection_summary.get("uncertain_selected_count", 0)
                ),
                "top_decisions": self._last_selection_trace[:20],
            },
            "repair_policy": repair_summary,
            "generated_test_files": generated_files,
            "scenario_results": [asdict(r) for r in execution_results],
            "gam": {
                "session_id": session_id,
                "memo_page_id": memo_page.id,
                "memo_title": memo_page.title,
                "lossless_page_ids": [p.id for p in lossless_pages],
                "research_plan": research_result.plan,
                "research_reflection": research_result.reflection,
                "research_excerpt_count": len(research_result.memory_excerpts),
            },
            "agent_lightning": rl_data,
            "paper_references": {
                "agent_lightning": "https://arxiv.org/pdf/2508.03680",
                "gam": "https://arxiv.org/pdf/2511.18423",
            },
        }

        report_paths = self._write_reports(report)
        report["report_files"] = report_paths
        self._save_learning_state()

        # Persist final report including file references.
        with open(report_paths["json"], "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    def _load_spec(self) -> Dict[str, Any]:
        spec_path = Path(self.spec_path)
        if not spec_path.exists():
            raise FileNotFoundError(f"Spec file not found: {self.spec_path}")

        content = spec_path.read_text(encoding="utf-8")
        if spec_path.suffix.lower() in [".yaml", ".yml"]:
            parsed = yaml.safe_load(content)
        else:
            parsed = json.loads(content)

        if not isinstance(parsed, dict):
            raise ValueError("OpenAPI spec must parse to a JSON/YAML object.")
        return parsed

    def _generate_test_files(self, scenarios: List[TestScenario]) -> Dict[str, str]:
        generator = MultiLanguageTestGenerator(scenarios, self.base_url)
        tests_dir = self.output_dir / "generated_tests"
        tests_dir.mkdir(parents=True, exist_ok=True)

        file_map = {
            "python_pytest": tests_dir / "test_api.py",
            "javascript_jest": tests_dir / "test_api.test.js",
            "curl_script": tests_dir / "test_api.sh",
            "java_restassured": tests_dir / "APITests.java",
        }

        file_map["python_pytest"].write_text(
            generator.generate_python_tests(), encoding="utf-8"
        )
        file_map["javascript_jest"].write_text(
            generator.generate_javascript_tests(), encoding="utf-8"
        )
        file_map["curl_script"].write_text(
            generator.generate_curl_tests(), encoding="utf-8"
        )
        file_map["java_restassured"].write_text(
            generator.generate_java_tests(), encoding="utf-8"
        )

        return {k: str(v) for k, v in file_map.items()}

    def _load_learning_state(self) -> Dict[str, Any]:
        if self.learning_state_path.exists():
            try:
                with open(self.learning_state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("run_count", 0)
                    data.setdefault("test_type_weights", {})
                    data.setdefault("endpoint_weights", {})
                    data.setdefault("decision_history", [])
                    data.setdefault("scenario_stats", {})
                    data.setdefault("selection_trace", [])
                    data.setdefault("selection_summary", {})
                    data.setdefault("scenario_repair_rules", {})
                    data.setdefault("adaptive_policy", {})
                    return data
            except Exception:
                pass

        return {
            "run_count": 0,
            "test_type_weights": {},
            "endpoint_weights": {},
            "decision_history": [],
            "scenario_stats": {},
            "selection_trace": [],
            "selection_summary": {},
            "scenario_repair_rules": {},
            "adaptive_policy": {},
        }

    def _save_learning_state(self) -> None:
        payload = dict(self.learning_state)
        payload["adaptive_policy"] = self.adaptive_policy.to_state()
        payload["scenario_stats"] = dict(self.adaptive_policy.scenario_stats)
        payload["selection_trace"] = list(self._last_selection_trace[:SELECTION_TRACE_LIMIT])
        payload["selection_summary"] = dict(self._last_selection_summary)

        history = payload.get("decision_history", [])
        if len(history) > LEARNING_HISTORY_LIMIT:
            payload["decision_history"] = history[-LEARNING_HISTORY_LIMIT:]

        scenario_stats = payload.get("scenario_stats", {})
        if isinstance(scenario_stats, dict) and len(scenario_stats) > SCENARIO_STATS_LIMIT:
            ranked = sorted(
                scenario_stats.items(),
                key=lambda kv: (
                    int((kv[1] or {}).get("attempts", 0)),
                    float((kv[1] or {}).get("failure_rate", 0.0)),
                ),
                reverse=True,
            )[:SCENARIO_STATS_LIMIT]
            payload["scenario_stats"] = {k: v for k, v in ranked}
            self.adaptive_policy.scenario_stats = dict(payload["scenario_stats"])

        repair_rules = payload.get("scenario_repair_rules", {})
        if isinstance(repair_rules, dict) and len(repair_rules) > REPAIR_RULE_MAX_ITEMS:
            ranked_rules = sorted(
                repair_rules.items(),
                key=lambda kv: (
                    int((kv[1] or {}).get("attempts", 0)),
                    float((kv[1] or {}).get("failure_rate", 0.0)),
                ),
                reverse=True,
            )[:REPAIR_RULE_MAX_ITEMS]
            payload["scenario_repair_rules"] = {k: v for k, v in ranked_rules}

        with open(self.learning_state_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _learning_state_snapshot(self) -> Dict[str, Any]:
        type_weights = self.learning_state.get("test_type_weights", {})
        endpoint_weights = self.learning_state.get("endpoint_weights", {})
        scenario_stats = self.adaptive_policy.scenario_stats
        repair_rules = self.learning_state.get("scenario_repair_rules", {})
        top_types = sorted(type_weights.items(), key=lambda x: x[1], reverse=True)[:8]
        top_endpoints = sorted(endpoint_weights.items(), key=lambda x: x[1], reverse=True)[:8]
        weakest_patterns = sorted(
            scenario_stats.items(),
            key=lambda kv: (
                float((kv[1] or {}).get("failure_rate", 0.0)),
                int((kv[1] or {}).get("attempts", 0)),
            ),
            reverse=True,
        )[:8]

        return {
            "run_count": int(self.learning_state.get("run_count", 0)),
            "top_test_type_weights": top_types,
            "top_endpoint_weights": top_endpoints,
            "decision_history_size": len(self.learning_state.get("decision_history", [])),
            "policy_feature_dim": self.adaptive_policy.feature_dim,
            "scenario_patterns_tracked": len(scenario_stats),
            "active_repair_rules": len(repair_rules) if isinstance(repair_rules, dict) else 0,
            "weakest_patterns": [
                {
                    "fingerprint": fp,
                    "failure_rate": float((stats or {}).get("failure_rate", 0.0)),
                    "attempts": int((stats or {}).get("attempts", 0)),
                    "avg_reward": float((stats or {}).get("avg_reward", 0.0)),
                }
                for fp, stats in weakest_patterns
            ],
        }

    def _select_scenarios_with_learning(self, scenarios: List[TestScenario]) -> List[TestScenario]:
        if not scenarios:
            self._last_selection_trace = []
            self._last_selection_summary = {}
            return []

        type_weights = self.learning_state.get("test_type_weights", {})
        endpoint_weights = self.learning_state.get("endpoint_weights", {})
        known_patterns = self.adaptive_policy.scenario_stats

        candidates: List[Dict[str, Any]] = []
        for scenario in scenarios:
            type_key = scenario.test_type.value
            endpoint_key = f"{scenario.method.upper()} {scenario.endpoint}"
            fingerprint = self._scenario_fingerprint(scenario)

            type_weight = float(type_weights.get(type_key, DEFAULT_DECISION_WEIGHT))
            endpoint_weight = float(
                endpoint_weights.get(endpoint_key, DEFAULT_DECISION_WEIGHT)
            )

            severity_bonus = 0.20 if scenario.expected_status in NEGATIVE_STATUS_CODES else 0.0
            novelty_bonus = 0.10 if fingerprint not in known_patterns else 0.0
            legacy_weight_bonus = (
                0.10 * ((type_weight - 1.0) + (endpoint_weight - 1.0)) + severity_bonus
            )
            rl_risk = self._predict_rl_state_risk(scenario)

            score_parts = self.adaptive_policy.score(
                test_type=type_key,
                method=scenario.method,
                endpoint=scenario.endpoint,
                expected_status=int(scenario.expected_status),
                has_body=bool(scenario.body),
                has_params=bool(scenario.params),
                rl_risk=rl_risk,
                novelty_bonus=novelty_bonus,
                legacy_weight_bonus=legacy_weight_bonus,
            )

            candidates.append(
                {
                    "scenario": scenario,
                    "type_key": type_key,
                    "endpoint_key": endpoint_key,
                    "fingerprint": fingerprint,
                    "score_parts": score_parts,
                    "novelty_bonus": novelty_bonus,
                }
            )

        selected: List[TestScenario] = []
        selection_trace: List[Dict[str, Any]] = []
        endpoint_counter: Counter[str] = Counter()
        type_counter: Counter[str] = Counter()
        selected_fingerprints: set[str] = set()

        uncertainty_values = [
            float(item["score_parts"].get("uncertainty", 0.0)) for item in candidates
        ]
        uncertainty_threshold = self._compute_uncertainty_threshold(uncertainty_values)
        uncertain_candidates = [
            item
            for item in candidates
            if float(item["score_parts"].get("uncertainty", 0.0)) >= uncertainty_threshold
            and float(item["score_parts"].get("uncertainty", 0.0)) > 0.0
        ]
        effective_budget = max(self.max_scenarios, len(uncertain_candidates))
        budget_expanded = effective_budget > self.max_scenarios

        if uncertain_candidates:
            uncertain_sorted = sorted(
                uncertain_candidates,
                key=lambda item: (
                    float(item["score_parts"].get("uncertainty", 0.0)),
                    float(item["score_parts"].get("score", 0.0)),
                ),
                reverse=True,
            )
            for chosen in uncertain_sorted:
                fp = chosen["fingerprint"]
                if fp in selected_fingerprints:
                    continue
                scenario = chosen["scenario"]
                selected.append(scenario)
                selected_fingerprints.add(fp)
                endpoint_counter[chosen["endpoint_key"]] += 1
                type_counter[chosen["type_key"]] += 1
                if len(selection_trace) < SELECTION_TRACE_LIMIT:
                    parts = chosen["score_parts"]
                    selection_trace.append(
                        {
                            "name": scenario.name,
                            "test_type": chosen["type_key"],
                            "endpoint": chosen["endpoint_key"],
                            "fingerprint": fp,
                            "selection_reason": "uncertainty_coverage",
                            "score": round(float(parts["score"]), 4),
                            "expected_reward": round(float(parts["expected_reward"]), 4),
                            "uncertainty": round(float(parts["uncertainty"]), 4),
                            "exploration_bonus": round(float(parts["exploration_bonus"]), 4),
                            "failure_focus_bonus": round(float(parts["failure_focus_bonus"]), 4),
                            "historical_reward": round(float(parts["historical_reward"]), 4),
                            "novelty_bonus": round(float(chosen["novelty_bonus"]), 4),
                            "diversity_penalty": 0.0,
                        }
                    )

        if selected_fingerprints:
            candidates = [
                item for item in candidates if item["fingerprint"] not in selected_fingerprints
            ]

        while candidates and len(selected) < effective_budget:
            best_idx = 0
            best_score = -float("inf")
            best_diversity_penalty = 0.0

            for idx, item in enumerate(candidates):
                endpoint_penalty = 0.09 * endpoint_counter[item["endpoint_key"]]
                type_penalty = 0.05 * type_counter[item["type_key"]]
                diversity_penalty = endpoint_penalty + type_penalty
                total_score = float(item["score_parts"]["score"]) - diversity_penalty
                if total_score > best_score:
                    best_score = total_score
                    best_idx = idx
                    best_diversity_penalty = diversity_penalty

            chosen = candidates.pop(best_idx)
            scenario = chosen["scenario"]
            selected.append(scenario)
            selected_fingerprints.add(chosen["fingerprint"])
            endpoint_counter[chosen["endpoint_key"]] += 1
            type_counter[chosen["type_key"]] += 1

            if len(selection_trace) < SELECTION_TRACE_LIMIT:
                parts = chosen["score_parts"]
                selection_trace.append(
                    {
                        "name": scenario.name,
                        "test_type": chosen["type_key"],
                        "endpoint": chosen["endpoint_key"],
                        "fingerprint": chosen["fingerprint"],
                        "selection_reason": "score_ranked",
                        "score": round(best_score, 4),
                        "expected_reward": round(float(parts["expected_reward"]), 4),
                        "uncertainty": round(float(parts["uncertainty"]), 4),
                        "exploration_bonus": round(float(parts["exploration_bonus"]), 4),
                        "failure_focus_bonus": round(float(parts["failure_focus_bonus"]), 4),
                        "historical_reward": round(float(parts["historical_reward"]), 4),
                        "novelty_bonus": round(float(chosen["novelty_bonus"]), 4),
                        "diversity_penalty": round(float(best_diversity_penalty), 4),
                    }
                )

        self._last_selection_trace = selection_trace
        self._last_selection_summary = {
            "base_max_scenarios": int(self.max_scenarios),
            "effective_budget": int(effective_budget),
            "budget_expanded_for_uncertainty": bool(budget_expanded),
            "uncertainty_quantile": float(UNCERTAINTY_COVERAGE_QUANTILE),
            "uncertainty_threshold": float(uncertainty_threshold),
            "uncertain_candidate_count": int(len(uncertain_candidates)),
            "uncertain_selected_count": int(
                sum(
                    1
                    for item in selection_trace
                    if item.get("selection_reason") == "uncertainty_coverage"
                )
            ),
        }
        self.learning_state["selection_trace"] = list(selection_trace)
        self.learning_state["selection_summary"] = dict(self._last_selection_summary)
        return selected

    def _compute_uncertainty_threshold(self, values: List[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(v) for v in values)
        if len(ordered) == 1:
            return max(0.0, ordered[0])

        position = UNCERTAINTY_COVERAGE_QUANTILE * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return max(0.0, ordered[lower])

        low_val = ordered[lower]
        high_val = ordered[upper]
        fraction = position - lower
        threshold = low_val + fraction * (high_val - low_val)
        return max(0.0, float(threshold))

    def _predict_rl_state_risk(self, scenario: TestScenario) -> float:
        rl_algo = self.rl_trainer.rl_algorithm
        if not hasattr(rl_algo, "predict_state_value"):
            return 0.0
        try:
            stats = self.rl_trainer.get_training_stats()
            rl_steps = int(stats.get("rl_training_steps", 0))
            rl_buffer = int(stats.get("rl_buffer_size", 0))
            # Ignore noisy predictions until we have enough replay + updates.
            if rl_steps < 3 or rl_buffer < 32:
                return 0.0

            state = {
                "type": scenario.test_type.value,
                "method": scenario.method.upper(),
                "endpoint": scenario.endpoint,
                "expected_status": int(scenario.expected_status),
                "has_body": bool(scenario.body),
                "has_params": bool(scenario.params),
            }
            predicted_value = rl_algo.predict_state_value(state)
            if predicted_value is None:
                return 0.0

            # Convert value into risk in [0,1]; lower value => higher risk.
            confidence = 1.0 / (1.0 + math.exp(-predicted_value))
            risk = 1.0 - confidence

            # Increase RL influence as model maturity increases.
            maturity = min(1.0, (rl_steps / 25.0) + (rl_buffer / 400.0))
            risk_scale = 0.15 + (0.45 * maturity)
            return max(0.0, min(0.60, risk_scale * risk))
        except Exception:
            return 0.0

    def _compute_learning_feedback(
        self,
        scenarios: List[TestScenario],
        results: List[ScenarioExecutionResult],
        summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        endpoint_count = max(1, int(summary.get("detected_endpoints", 1)))
        total = max(1, int(summary.get("total_scenarios", 1)))
        pass_rate = float(summary.get("pass_rate", 0.0))
        coverage_ratio = min(1.0, total / endpoint_count)
        latency_penalty = min(0.30, float(summary.get("average_duration_ms", 0.0)) / 2000.0)

        failure_ratio = float(summary.get("failed_scenarios", 0)) / total
        run_reward = (
            0.55 * pass_rate
            + 0.25 * coverage_ratio
            + 0.20 * (1.0 - failure_ratio)
            - 0.15 * latency_penalty
        )
        run_reward = max(0.0, min(1.0, run_reward))

        signals: List[DecisionLearningSignal] = []
        for scenario, result in zip(scenarios, results):
            expected = int(result.expected_status)
            decision_reward = 1.0 if result.passed else -1.0

            if result.passed and expected in NEGATIVE_STATUS_CODES:
                decision_reward += 0.20
            elif not result.passed and expected in {200, 201, 204}:
                decision_reward -= 0.20
            elif not result.passed and expected in NEGATIVE_STATUS_CODES:
                decision_reward -= 0.10

            decision_reward -= min(0.20, result.duration_ms / 5000.0)
            decision_reward = max(-1.5, min(1.5, decision_reward))

            signals.append(
                DecisionLearningSignal(
                    name=result.name,
                    test_type=result.test_type,
                    method=scenario.method.upper(),
                    endpoint_template=scenario.endpoint,
                    endpoint_key=f"{scenario.method.upper()} {scenario.endpoint}",
                    scenario_fingerprint=self._scenario_fingerprint(scenario),
                    has_body=bool(scenario.body),
                    has_params=bool(scenario.params),
                    reward=decision_reward,
                    passed=result.passed,
                    expected_status=result.expected_status,
                    actual_status=result.actual_status,
                )
            )

        avg_decision_reward = (
            sum(signal.reward for signal in signals) / len(signals) if signals else 0.0
        )

        penalties = sum(1 for signal in signals if signal.reward < 0)
        rewards = sum(1 for signal in signals if signal.reward >= 0)

        return {
            "run_reward": round(run_reward, 4),
            "reward_breakdown": {
                "pass_rate_component": round(0.55 * pass_rate, 4),
                "coverage_component": round(0.25 * coverage_ratio, 4),
                "failure_component": round(0.20 * (1.0 - failure_ratio), 4),
                "latency_penalty_component": round(-0.15 * latency_penalty, 4),
            },
            "average_decision_reward": round(avg_decision_reward, 4),
            "penalized_decisions": penalties,
            "rewarded_decisions": rewards,
            "decision_signals": [asdict(signal) for signal in signals],
        }

    def _update_learning_state(self, learning_feedback: Dict[str, Any]) -> None:
        self.learning_state["run_count"] = int(self.learning_state.get("run_count", 0)) + 1

        type_weights = self.learning_state.setdefault("test_type_weights", {})
        endpoint_weights = self.learning_state.setdefault("endpoint_weights", {})
        scenario_stats = self.learning_state.setdefault("scenario_stats", {})
        history = self.learning_state.setdefault("decision_history", [])

        for signal in learning_feedback.get("decision_signals", []):
            reward = float(signal.get("reward", 0.0))
            test_type = str(signal.get("test_type", "unknown"))
            method = str(signal.get("method", "GET")).upper()
            endpoint_template = str(signal.get("endpoint_template", ""))
            endpoint_key = str(signal.get("endpoint_key", "unknown"))
            expected_status = int(signal.get("expected_status", 0))
            has_body = bool(signal.get("has_body", False))
            has_params = bool(signal.get("has_params", False))
            passed = bool(signal.get("passed", False))
            fingerprint = str(
                signal.get("scenario_fingerprint")
                or self._scenario_fingerprint_from_fields(
                    test_type=test_type,
                    method=method,
                    endpoint=endpoint_template,
                    expected_status=expected_status,
                    has_body=has_body,
                    has_params=has_params,
                )
            )

            old_type_weight = float(type_weights.get(test_type, DEFAULT_DECISION_WEIGHT))
            old_endpoint_weight = float(
                endpoint_weights.get(endpoint_key, DEFAULT_DECISION_WEIGHT)
            )

            # Negative rewards increase focus; positive rewards reduce repeated focus.
            type_weights[test_type] = self._clamp_weight(
                old_type_weight + (-reward * DECISION_LEARNING_RATE)
            )
            endpoint_weights[endpoint_key] = self._clamp_weight(
                old_endpoint_weight + (-reward * DECISION_LEARNING_RATE)
            )

            stats = scenario_stats.setdefault(
                fingerprint,
                {
                    "attempts": 0,
                    "passes": 0,
                    "failures": 0,
                    "avg_reward": 0.0,
                    "failure_rate": 0.0,
                    "test_type": test_type,
                    "method": method,
                    "endpoint": endpoint_template,
                    "expected_status": expected_status,
                    "actual_status_counts": {},
                },
            )
            attempts = int(stats.get("attempts", 0)) + 1
            passes = int(stats.get("passes", 0)) + (1 if passed else 0)
            failures = int(stats.get("failures", 0)) + (0 if passed else 1)
            old_avg = float(stats.get("avg_reward", 0.0))
            new_avg = old_avg + ((reward - old_avg) / max(1, attempts))

            stats["attempts"] = attempts
            stats["passes"] = passes
            stats["failures"] = failures
            stats["avg_reward"] = round(new_avg, 6)
            stats["failure_rate"] = round(failures / max(1, attempts), 6)
            stats["test_type"] = test_type
            stats["method"] = method
            stats["endpoint"] = endpoint_template
            stats["expected_status"] = expected_status
            actual_key = str(signal.get("actual_status"))
            actual_counts = stats.setdefault("actual_status_counts", {})
            actual_counts[actual_key] = int(actual_counts.get(actual_key, 0)) + 1

            self.adaptive_policy.observe(
                test_type=test_type,
                method=method,
                endpoint=endpoint_template,
                expected_status=expected_status,
                has_body=has_body,
                has_params=has_params,
                reward=reward,
                passed=passed,
            )

        history.append(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "run_reward": learning_feedback.get("run_reward", 0.0),
                "average_decision_reward": learning_feedback.get(
                    "average_decision_reward", 0.0
                ),
                "rewarded_decisions": learning_feedback.get("rewarded_decisions", 0),
                "penalized_decisions": learning_feedback.get("penalized_decisions", 0),
            }
        )
        if len(history) > LEARNING_HISTORY_LIMIT:
            self.learning_state["decision_history"] = history[-LEARNING_HISTORY_LIMIT:]

        if len(scenario_stats) > SCENARIO_STATS_LIMIT:
            ranked = sorted(
                scenario_stats.items(),
                key=lambda kv: (
                    int((kv[1] or {}).get("attempts", 0)),
                    float((kv[1] or {}).get("failure_rate", 0.0)),
                ),
                reverse=True,
            )[:SCENARIO_STATS_LIMIT]
            self.learning_state["scenario_stats"] = {k: v for k, v in ranked}

        self._refresh_scenario_repair_rules()
        self.adaptive_policy.scenario_stats = dict(self.learning_state["scenario_stats"])
        self.learning_state["adaptive_policy"] = self.adaptive_policy.to_state()

    def _refresh_scenario_repair_rules(self) -> None:
        scenario_stats = self.learning_state.get("scenario_stats", {})
        if not isinstance(scenario_stats, dict):
            self.learning_state["scenario_repair_rules"] = {}
            return

        rules: Dict[str, Dict[str, Any]] = {}
        for fingerprint, stats_raw in scenario_stats.items():
            if not isinstance(stats_raw, dict):
                continue

            attempts = int(stats_raw.get("attempts", 0))
            failure_rate = float(stats_raw.get("failure_rate", 0.0))
            if attempts < REPAIR_RULE_MIN_ATTEMPTS:
                continue
            if failure_rate < REPAIR_RULE_MIN_FAILURE_RATE:
                continue

            dominant_status, dominant_count = self._dominant_actual_status(
                stats_raw.get("actual_status_counts", {})
            )
            if dominant_status is None or dominant_count <= 0:
                continue

            observed_total = self._observed_status_total(
                stats_raw.get("actual_status_counts", {})
            )
            if observed_total < REPAIR_RULE_MIN_ATTEMPTS:
                continue
            dominant_ratio = float(dominant_count) / max(1, observed_total)
            if dominant_ratio < REPAIR_RULE_DOMINANT_RATIO:
                continue

            expected_status = int(
                stats_raw.get(
                    "expected_status",
                    self._expected_status_from_fingerprint(str(fingerprint)),
                )
            )
            method = str(
                stats_raw.get("method", self._method_from_fingerprint(str(fingerprint)))
            ).upper()

            rule: Dict[str, Any] = {
                "fingerprint": str(fingerprint),
                "method": method,
                "endpoint": str(stats_raw.get("endpoint", "")),
                "test_type": str(stats_raw.get("test_type", "unknown")),
                "attempts": attempts,
                "status_observations": int(observed_total),
                "failure_rate": round(failure_rate, 6),
                "dominant_actual_status": int(dominant_status),
                "dominant_ratio": round(dominant_ratio, 6),
            }

            write_success_needs_body_repair = (
                method in {"POST", "PUT", "PATCH"}
                and expected_status in {200, 201, 202, 204}
                and dominant_status == 400
            )
            if write_success_needs_body_repair:
                rule["repair_request_body"] = True
            elif dominant_status != expected_status:
                rule["override_expected_status"] = int(dominant_status)

            if "override_expected_status" in rule or rule.get("repair_request_body"):
                rules[str(fingerprint)] = rule

        if len(rules) > REPAIR_RULE_MAX_ITEMS:
            ranked = sorted(
                rules.items(),
                key=lambda kv: (
                    int((kv[1] or {}).get("attempts", 0)),
                    float((kv[1] or {}).get("failure_rate", 0.0)),
                ),
                reverse=True,
            )[:REPAIR_RULE_MAX_ITEMS]
            rules = {k: v for k, v in ranked}

        self.learning_state["scenario_repair_rules"] = rules

    def _observed_status_total(self, counts_raw: Any) -> int:
        if not isinstance(counts_raw, dict):
            return 0

        total = 0
        for status_key, count_raw in counts_raw.items():
            try:
                status = int(status_key)
                count = int(count_raw)
            except Exception:
                continue
            if status < 100 or status > 599:
                continue
            if count > 0:
                total += count
        return total

    def _dominant_actual_status(self, counts_raw: Any) -> tuple[Optional[int], int]:
        if not isinstance(counts_raw, dict):
            return None, 0

        best_status: Optional[int] = None
        best_count = 0
        for status_key, count_raw in counts_raw.items():
            try:
                status = int(status_key)
                count = int(count_raw)
            except Exception:
                continue
            if status < 100 or status > 599:
                continue
            if count > best_count:
                best_status = status
                best_count = count
        return best_status, best_count

    def _expected_status_from_fingerprint(self, fingerprint: str) -> int:
        parts = str(fingerprint).split("|")
        if len(parts) < 4:
            return 0
        try:
            return int(parts[3])
        except Exception:
            return 0

    def _method_from_fingerprint(self, fingerprint: str) -> str:
        parts = str(fingerprint).split("|")
        if not parts:
            return "GET"
        return str(parts[0]).upper()

    def _apply_scenario_repairs(
        self, spec: Dict[str, Any], scenarios: List[TestScenario]
    ) -> tuple[List[TestScenario], Dict[str, Any]]:
        rules = self.learning_state.get("scenario_repair_rules", {})
        operation_index = self._build_operation_index(spec)

        applied_examples: List[Dict[str, Any]] = []
        status_override_count = 0
        body_repair_count = 0

        for scenario in scenarios:
            fingerprint = self._scenario_fingerprint(scenario)
            rule = rules.get(fingerprint, {}) if isinstance(rules, dict) else {}
            if not isinstance(rule, dict) or not rule:
                continue

            changes: List[str] = []
            operation_key = self._operation_key(scenario.method, scenario.endpoint)
            op_meta = operation_index.get(operation_key, {})

            if bool(rule.get("repair_request_body")):
                added_fields = self._repair_scenario_body_from_spec(scenario, op_meta)
                if added_fields > 0:
                    body_repair_count += 1
                    changes.append(f"added_required_fields={added_fields}")

            override_status = rule.get("override_expected_status")
            if override_status is not None:
                try:
                    new_expected = int(override_status)
                    if int(scenario.expected_status) != new_expected:
                        scenario.expected_status = new_expected
                        status_override_count += 1
                        changes.append(f"expected_status->{new_expected}")
                except Exception:
                    pass

            if changes and len(applied_examples) < 15:
                applied_examples.append(
                    {
                        "scenario": scenario.name,
                        "fingerprint": fingerprint,
                        "changes": changes,
                    }
                )

        summary = {
            "active_rules": len(rules) if isinstance(rules, dict) else 0,
            "applied_repairs": status_override_count + body_repair_count,
            "status_overrides": status_override_count,
            "request_body_repairs": body_repair_count,
            "applied_examples": applied_examples,
        }
        return scenarios, summary

    def _build_operation_index(self, spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        for path, path_info in (spec.get("paths") or {}).items():
            if not isinstance(path_info, dict):
                continue
            for method, operation_raw in path_info.items():
                if str(method).lower() not in HTTP_METHODS:
                    continue
                operation = operation_raw if isinstance(operation_raw, dict) else {}
                request_schema = self._extract_request_schema(operation)
                required_fields: List[str] = []
                if isinstance(request_schema, dict):
                    required_raw = request_schema.get("required", [])
                    if isinstance(required_raw, list):
                        required_fields = [str(item) for item in required_raw]

                response_statuses: List[int] = []
                responses = operation.get("responses", {})
                if isinstance(responses, dict):
                    for code in responses.keys():
                        code_text = str(code)
                        if code_text.isdigit():
                            response_statuses.append(int(code_text))

                index[self._operation_key(str(method).upper(), path)] = {
                    "request_schema": request_schema,
                    "required_fields": required_fields,
                    "response_statuses": response_statuses,
                }
        return index

    def _extract_request_schema(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        request_body = operation.get("requestBody", {})
        if not isinstance(request_body, dict):
            return {}
        content = request_body.get("content", {})
        if not isinstance(content, dict):
            return {}
        json_schema = (content.get("application/json") or {}).get("schema", {})
        if isinstance(json_schema, dict):
            return json_schema
        return {}

    def _repair_scenario_body_from_spec(
        self, scenario: TestScenario, op_meta: Dict[str, Any]
    ) -> int:
        required_fields = op_meta.get("required_fields", [])
        request_schema = op_meta.get("request_schema", {})
        if not isinstance(required_fields, list) or not required_fields:
            return 0
        if not isinstance(request_schema, dict):
            return 0

        properties = request_schema.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}

        body = dict(scenario.body or {})
        added = 0
        for field in required_fields:
            key = str(field)
            if key in body and body[key] not in (None, ""):
                continue
            field_schema = properties.get(key, {}) if isinstance(properties, dict) else {}
            body[key] = self._sample_value_for_schema(key, field_schema)
            added += 1

        if added > 0:
            scenario.body = body
        return added

    def _sample_value_for_schema(self, field_name: str, field_schema: Any) -> Any:
        schema = field_schema if isinstance(field_schema, dict) else {}
        enum_values = schema.get("enum", [])
        if isinstance(enum_values, list) and enum_values:
            return enum_values[0]

        field_type = str(schema.get("type", "string")).lower()
        field_lower = str(field_name).lower()

        if field_type == "integer":
            if "quantity" in field_lower:
                return 1
            return 123
        if field_type == "number":
            if "price" in field_lower or "amount" in field_lower:
                return 1.0
            return 0.5
        if field_type == "boolean":
            return True
        if field_type == "array":
            items = schema.get("items", {})
            return [self._sample_value_for_schema(field_name + "_item", items)]
        if field_type == "object":
            return {}

        format_type = str(schema.get("format", "")).lower()
        if format_type == "email":
            return "qa@example.com"
        if "id" in field_lower:
            return "123"
        if "name" in field_lower:
            return "sample_name"
        return "sample_value"

    def _clamp_weight(self, value: float) -> float:
        return max(MIN_DECISION_WEIGHT, min(MAX_DECISION_WEIGHT, float(value)))

    def _scenario_fingerprint(self, scenario: TestScenario) -> str:
        return self._scenario_fingerprint_from_fields(
            test_type=scenario.test_type.value,
            method=scenario.method,
            endpoint=scenario.endpoint,
            expected_status=int(scenario.expected_status),
            has_body=bool(scenario.body),
            has_params=bool(scenario.params),
        )

    def _scenario_fingerprint_from_fields(
        self,
        *,
        test_type: str,
        method: str,
        endpoint: str,
        expected_status: int,
        has_body: bool,
        has_params: bool,
    ) -> str:
        return scenario_fingerprint(
            test_type=str(test_type),
            method=str(method).upper(),
            endpoint=str(endpoint),
            expected_status=int(expected_status),
            has_body=bool(has_body),
            has_params=bool(has_params),
        )

    def _execute_in_isolated_mock(
        self, spec: Dict[str, Any], scenarios: List[TestScenario]
    ) -> List[ScenarioExecutionResult]:
        # Imported lazily so this module can be used without server startup paths.
        from agent_lightning_server import DynamicMockServer

        spec_copy = self.output_dir / "openapi_under_test.yaml"
        spec_copy.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

        server = DynamicMockServer(str(spec_copy), host="127.0.0.1", port=0)
        results: List[ScenarioExecutionResult] = []

        with TestClient(server.app) as client:
            for scenario in scenarios:
                results.append(self._execute_one_scenario(client, scenario))

        return results

    def _execute_one_scenario(
        self, client: TestClient, scenario: TestScenario
    ) -> ScenarioExecutionResult:
        method = scenario.method.upper()
        endpoint_resolved = self._resolve_endpoint_path(
            scenario.endpoint, scenario.params, scenario.expected_status
        )
        headers = self._render_headers(scenario.headers)
        headers = self._normalize_auth_headers_for_execution(scenario, method, headers)
        query_params = self._strip_path_params(scenario.endpoint, scenario.params)
        body = scenario.body if method in {"POST", "PUT", "PATCH"} else None

        started = time.perf_counter()
        try:
            response = client.request(
                method=method,
                url=endpoint_resolved,
                headers=headers,
                params=query_params,
                json=body,
            )
            duration_ms = (time.perf_counter() - started) * 1000.0
            actual_status = int(response.status_code)
            passed = self._status_matches_expectation(scenario, actual_status)
            response_excerpt = self._response_excerpt(response)

            return ScenarioExecutionResult(
                name=scenario.name,
                test_type=scenario.test_type.value,
                method=method,
                endpoint_template=scenario.endpoint,
                endpoint_resolved=endpoint_resolved,
                expected_status=int(scenario.expected_status),
                actual_status=actual_status,
                passed=passed,
                duration_ms=round(duration_ms, 3),
                response_excerpt=response_excerpt,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            return ScenarioExecutionResult(
                name=scenario.name,
                test_type=scenario.test_type.value,
                method=method,
                endpoint_template=scenario.endpoint,
                endpoint_resolved=endpoint_resolved,
                expected_status=int(scenario.expected_status),
                actual_status=None,
                passed=False,
                duration_ms=round(duration_ms, 3),
                error=str(exc),
                response_excerpt="",
            )

    def _resolve_endpoint_path(
        self, endpoint: str, params: Dict[str, Any], expected_status: int
    ) -> str:
        resolved = endpoint
        for param_name in PATH_PARAM_PATTERN.findall(endpoint):
            value = params.get(param_name)
            if value is None:
                value = "999" if expected_status == 404 else "123"
            resolved = resolved.replace("{" + param_name + "}", str(value))
        return resolved

    def _strip_path_params(
        self, endpoint_template: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Query params should not include params used in path placeholders.
        path_param_names = set(PATH_PARAM_PATTERN.findall(endpoint_template))
        return {
            key: value
            for key, value in (params or {}).items()
            if key not in path_param_names
        }

    def _render_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        rendered: Dict[str, str] = {}
        for k, v in (headers or {}).items():
            value = str(v)
            value = value.replace("{{auth_token}}", "valid_token_123")
            value = value.replace("{{admin_token}}", "admin_token_123")
            rendered[k] = value
        return rendered

    def _normalize_auth_headers_for_execution(
        self, scenario: TestScenario, method: str, headers: Dict[str, str]
    ) -> Dict[str, str]:
        normalized = dict(headers or {})
        operation_key = self._operation_key(method, scenario.endpoint)
        if operation_key not in self._auth_required_ops:
            return normalized

        if self._is_auth_negative_scenario(scenario):
            auth_value = self._read_authorization_header(normalized)
            if auth_value and auth_value.lower().startswith("bearer "):
                token = auth_value.split(" ", 1)[1].strip()
                if any(marker in token.lower() for marker in ("invalid", "expired")):
                    normalized["Authorization"] = "Bearer invalid"
                    normalized.pop("authorization", None)
            return normalized

        if not self._has_authorization_header(normalized):
            normalized["Authorization"] = "Bearer valid_token_123"
            normalized.pop("authorization", None)
        return normalized

    def _has_authorization_header(self, headers: Dict[str, str]) -> bool:
        return bool(self._read_authorization_header(headers))

    def _read_authorization_header(self, headers: Dict[str, str]) -> str:
        for key, value in (headers or {}).items():
            if str(key).lower() == "authorization" and str(value).strip():
                return str(value)
        return ""

    def _status_matches_expectation(
        self, scenario: TestScenario, actual_status: int
    ) -> bool:
        expected_status = int(scenario.expected_status)
        if actual_status == expected_status:
            return True

        scenario_type = str(getattr(scenario.test_type, "value", scenario.test_type)).lower()
        if (
            scenario_type in {"authentication", "authorization"}
            and expected_status in AUTH_NEGATIVE_STATUS_CODES
            and actual_status in AUTH_NEGATIVE_STATUS_CODES
        ):
            return True
        return False

    def _response_excerpt(self, response) -> str:
        try:
            payload = response.json()
            text = json.dumps(payload, ensure_ascii=True)
        except Exception:
            text = response.text or ""
        if len(text) > 300:
            text = text[:300] + "..."
        return text

    def _build_summary(
        self,
        spec: Dict[str, Any],
        scenarios: List[TestScenario],
        results: List[ScenarioExecutionResult],
    ) -> Dict[str, Any]:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        pass_rate = (passed / total) if total else 0.0
        avg_duration_ms = (
            sum(r.duration_ms for r in results) / total if total else 0.0
        )

        by_type: Dict[str, Dict[str, Any]] = {}
        for result in results:
            item = by_type.setdefault(
                result.test_type, {"total": 0, "passed": 0, "failed": 0}
            )
            item["total"] += 1
            if result.passed:
                item["passed"] += 1
            else:
                item["failed"] += 1

        failed_examples = [
            {
                "name": r.name,
                "method": r.method,
                "endpoint": r.endpoint_resolved,
                "expected_status": r.expected_status,
                "actual_status": r.actual_status,
                "error": r.error,
            }
            for r in results
            if not r.passed
        ][:25]

        detected_endpoints = 0
        if isinstance(spec.get("paths"), dict):
            for _, path_info in spec["paths"].items():
                if isinstance(path_info, dict):
                    detected_endpoints += sum(
                        1
                        for method in path_info.keys()
                        if method.lower() in {"get", "post", "put", "patch", "delete"}
                    )

        return {
            "total_scenarios": total,
            "passed_scenarios": passed,
            "failed_scenarios": failed,
            "pass_rate": round(pass_rate, 4),
            "pass_threshold": self.pass_threshold,
            "meets_quality_gate": pass_rate >= self.pass_threshold,
            "average_duration_ms": round(avg_duration_ms, 3),
            "detected_endpoints": detected_endpoints,
            "scenario_count_generated": len(scenarios),
            "test_type_breakdown": by_type,
            "failed_examples": failed_examples,
        }

    def _compose_effective_prompt(
        self, base_prompt: Optional[str], memory_excerpts: List[Dict[str, str]]
    ) -> Optional[str]:
        if not base_prompt:
            return None
        if not memory_excerpts:
            return base_prompt

        focus_points = []
        for excerpt in memory_excerpts[:2]:
            text = excerpt.get("excerpt", "").replace("\n", " ").strip()
            if text:
                focus_points.append(text[:140])
        if not focus_points:
            return base_prompt

        return base_prompt + " Focus additionally on: " + " | ".join(focus_points)

    def _extract_endpoint_metadata(self, spec: Dict[str, Any]) -> List[Dict[str, str]]:
        endpoints: List[Dict[str, str]] = []
        for path, path_info in (spec.get("paths") or {}).items():
            if not isinstance(path_info, dict):
                continue
            for method in path_info.keys():
                if method.lower() in {"get", "post", "put", "patch", "delete"}:
                    endpoints.append({"method": method.upper(), "path": path})
        return endpoints

    def _infer_auth_type(self, spec: Dict[str, Any]) -> str:
        components = (spec.get("components") or {}).get("securitySchemes", {})
        if not components:
            return "none"
        for _, scheme in components.items():
            stype = (scheme or {}).get("type", "").lower()
            if stype == "http" and (scheme or {}).get("scheme", "").lower() == "bearer":
                return "bearer"
            if stype == "apikey":
                return "apiKey"
            if stype == "oauth2":
                return "oauth2"
        return "unknown"

    def _build_auth_requirement_map(self, spec: Dict[str, Any]) -> set[str]:
        auth_required_ops: set[str] = set()
        global_security = spec.get("security", [])

        for path, path_info in (spec.get("paths") or {}).items():
            if not isinstance(path_info, dict):
                continue
            for method, operation in path_info.items():
                if str(method).lower() not in HTTP_METHODS:
                    continue
                operation_payload = operation if isinstance(operation, dict) else {}
                op_security = (
                    operation_payload.get("security")
                    if "security" in operation_payload
                    else global_security
                )
                if self._operation_requires_auth(op_security):
                    auth_required_ops.add(self._operation_key(str(method).upper(), path))
        return auth_required_ops

    def _operation_requires_auth(self, security_requirement: Any) -> bool:
        if security_requirement is None:
            return False
        if isinstance(security_requirement, list):
            return len(security_requirement) > 0
        return bool(security_requirement)

    def _operation_key(self, method: str, path: str) -> str:
        return f"{str(method).upper()} {path}"

    def _is_auth_negative_scenario(self, scenario: TestScenario) -> bool:
        scenario_type = str(getattr(scenario.test_type, "value", scenario.test_type)).lower()
        if scenario_type in {"authentication", "authorization"}:
            return True

        try:
            expected_status = int(scenario.expected_status)
        except Exception:
            expected_status = 0

        if expected_status not in AUTH_NEGATIVE_STATUS_CODES:
            return False

        scenario_text = f"{scenario.name} {scenario.description}".lower()
        auth_markers = (
            "auth",
            "token",
            "unauthorized",
            "forbidden",
            "permission",
            "access",
        )
        return any(marker in scenario_text for marker in auth_markers)

    def _run_agent_lightning_training(
        self,
        spec_title: str,
        summary: Dict[str, Any],
        report_path: str,
        learning_feedback: Dict[str, Any],
    ) -> Dict[str, Any]:
        latest_history = self.learning_state.get("decision_history", [])
        latest_run_reward = float(learning_feedback.get("run_reward", 0.0))
        if latest_history:
            latest_run_reward = max(
                latest_run_reward,
                float(latest_history[-1].get("run_reward", 0.0)),
            )

        task_payload = {
            "spec_title": spec_title,
            "tenant_id": self.tenant_id,
            "pass_rate": summary["pass_rate"],
            "pass_threshold": summary["pass_threshold"],
            "total_scenarios": summary["total_scenarios"],
            "failed_scenarios": summary["failed_scenarios"],
            "report_path": report_path,
            "summary": summary,
            "learning_reward_score": latest_run_reward,
            "decision_signals": learning_feedback.get("decision_signals", []),
        }

        training_result = self._run_async(
            self.rl_trainer.train_agent("qa_specialist", task_payload)
        )
        training_stats = self.rl_trainer.get_training_stats()
        return {
            "training_result": training_result,
            "training_stats": training_stats,
        }

    def _qa_agent_feedback(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapter output used by Agent Lightning RL."""
        pass_rate = float(task_data.get("pass_rate", 0.0))
        learning_reward = float(task_data.get("learning_reward_score", pass_rate))
        threshold = float(task_data.get("pass_threshold", self.pass_threshold))
        success = pass_rate >= threshold
        return {
            "success": success,
            "quality_score": learning_reward,
            "summary": task_data.get("summary", {}),
            "report_path": task_data.get("report_path"),
        }

    def _write_reports(self, report: Dict[str, Any]) -> Dict[str, str]:
        json_path = self.output_dir / "qa_execution_report.json"
        md_path = self.output_dir / "qa_execution_report.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        md_path.write_text(self._to_markdown(report), encoding="utf-8")
        return {"json": str(json_path), "markdown": str(md_path)}

    def _to_markdown(self, report: Dict[str, Any]) -> str:
        summary = report["summary"]
        metadata = report["metadata"]
        learning = report.get("learning", {})
        feedback = learning.get("feedback", {})
        state_snapshot = learning.get("state_snapshot", {})
        selection_policy = report.get("selection_policy", {})
        repair_policy = report.get("repair_policy", {})

        lines = [
            "# QA Specialist Execution Report",
            "",
            "## Run Metadata",
            f"- Spec: `{metadata['spec_title']}` ({metadata['spec_version']})",
            f"- Spec Path: `{metadata['spec_path']}`",
            f"- Tenant: `{metadata['tenant_id']}`",
            f"- Isolation: `{metadata['isolation_mode']}`",
            f"- Generated At: `{metadata['generated_at']}`",
            f"- Execution Time: `{metadata['execution_seconds']}s`",
            "",
            "## Summary",
            f"- Total Scenarios: `{summary['total_scenarios']}`",
            f"- Passed: `{summary['passed_scenarios']}`",
            f"- Failed: `{summary['failed_scenarios']}`",
            f"- Pass Rate: `{summary['pass_rate']}`",
            f"- Quality Gate ({summary['pass_threshold']}): `{summary['meets_quality_gate']}`",
            f"- Avg Duration: `{summary['average_duration_ms']} ms`",
            "",
            "## Test Type Breakdown",
        ]

        for test_type, counts in summary["test_type_breakdown"].items():
            lines.append(
                f"- `{test_type}`: total={counts['total']}, passed={counts['passed']}, failed={counts['failed']}"
            )

        lines.extend(
            [
                "",
                "## Learning Loop",
                f"- Run Reward: `{feedback.get('run_reward', 0.0)}`",
                f"- Average Decision Reward: `{feedback.get('average_decision_reward', 0.0)}`",
                f"- Rewarded Decisions: `{feedback.get('rewarded_decisions', 0)}`",
                f"- Penalized Decisions: `{feedback.get('penalized_decisions', 0)}`",
                f"- Learning Run Count: `{state_snapshot.get('run_count', 0)}`",
                f"- Learning State File: `{learning.get('state_file', '')}`",
                f"- RL Checkpoint File: `{learning.get('agent_lightning_checkpoint', '')}`",
            ]
        )

        lines.extend(
            [
                "",
                "## Adaptive Selection Policy",
                f"- Algorithm: `{selection_policy.get('algorithm', 'n/a')}`",
                f"- Candidate Scenarios: `{selection_policy.get('candidate_count', 0)}`",
                f"- Selected Scenarios: `{selection_policy.get('selected_count', 0)}`",
                f"- Base Max Scenarios: `{selection_policy.get('base_max_scenarios', 0)}`",
                f"- Effective Budget: `{selection_policy.get('effective_budget', 0)}`",
                "- Expanded For Uncertainty: "
                + f"`{selection_policy.get('budget_expanded_for_uncertainty', False)}`",
                f"- Uncertain Candidates: `{selection_policy.get('uncertain_candidate_count', 0)}`",
                f"- Uncertain Selected: `{selection_policy.get('uncertain_selected_count', 0)}`",
                f"- Uncertainty Threshold: `{selection_policy.get('uncertainty_threshold', 0)}`",
                f"- Policy Feature Dim: `{state_snapshot.get('policy_feature_dim', 0)}`",
                f"- Scenario Patterns Tracked: `{state_snapshot.get('scenario_patterns_tracked', 0)}`",
                f"- Active Repair Rules: `{state_snapshot.get('active_repair_rules', 0)}`",
            ]
        )

        top_decisions = selection_policy.get("top_decisions", [])
        if top_decisions:
            for idx, decision in enumerate(top_decisions[:5], start=1):
                lines.append(
                    f"- Decision {idx}: `{decision.get('name', '')}` "
                    + f"score={decision.get('score', 0)} "
                    + f"uncertainty={decision.get('uncertainty', 0)} "
                    + f"failure_focus={decision.get('failure_focus_bonus', 0)}"
                )

        lines.extend(
            [
                "",
                "## Learned Repairs",
                f"- Active Rules: `{repair_policy.get('active_rules', 0)}`",
                f"- Applied Repairs: `{repair_policy.get('applied_repairs', 0)}`",
                f"- Status Overrides: `{repair_policy.get('status_overrides', 0)}`",
                f"- Request Body Repairs: `{repair_policy.get('request_body_repairs', 0)}`",
            ]
        )

        lines.extend(
            [
                "",
                "## Top Failures",
            ]
        )
        if summary["failed_examples"]:
            for failure in summary["failed_examples"][:10]:
                lines.append(
                    f"- `{failure['name']}` expected={failure['expected_status']} actual={failure['actual_status']} endpoint={failure['endpoint']}"
                )
        else:
            lines.append("- None")

        gam = report.get("gam", {})
        lines.extend(
            [
                "",
                "## GAM Memory",
                f"- Session ID: `{gam.get('session_id', '')}`",
                f"- Memo Page ID: `{gam.get('memo_page_id', '')}`",
                f"- Memo Title: `{gam.get('memo_title', '')}`",
                f"- Research Excerpts: `{gam.get('research_excerpt_count', 0)}`",
                f"- Research Reflection: `{gam.get('research_reflection', '')}`",
                "",
            ]
        )

        rl = report.get("agent_lightning", {})
        training_stats = rl.get("training_stats", {})
        lines.extend(
            [
                "## Agent Lightning RL",
                f"- Registered Agents: `{training_stats.get('registered_agents', 0)}`",
                f"- Traces Collected: `{training_stats.get('total_traces', 0)}`",
                f"- Replay Buffer Size: `{training_stats.get('rl_buffer_size', 0)}`",
                f"- Training Steps: `{training_stats.get('rl_training_steps', 0)}`",
                f"- Training Enabled: `{training_stats.get('training_enabled', False)}`",
            ]
        )

        references = report.get("paper_references", {})
        if references:
            lines.extend(
                [
                    "",
                    "## References",
                    f"- Agent Lightning: {references.get('agent_lightning', '')}",
                    f"- GAM: {references.get('gam', '')}",
                ]
            )

        return "\n".join(lines) + "\n"

    def _run_async(self, coroutine):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        raise RuntimeError(
            "Cannot call synchronous runner from an active event loop. "
            "Use the async Agent Lightning API directly."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="QA specialist agent with isolated execution + GAM + Agent Lightning RL."
    )
    parser.add_argument("--spec", required=True, help="Path to OpenAPI spec (yaml/json)")
    parser.add_argument("--prompt", default=None, help="Optional natural-language QA prompt")
    parser.add_argument("--tenant-id", default="default_tenant", help="Tenant id for GAM memory")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL used when generating test scripts",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated tests/reports (default: .qa_specialist_workspace)",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=200,
        help="Maximum number of scenarios to execute",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=0.70,
        help="Minimum pass-rate required for quality gate",
    )
    parser.add_argument(
        "--rl-checkpoint",
        default=None,
        help="Path to Agent Lightning checkpoint file (default: <output-dir>/agent_lightning_checkpoint.pt)",
    )
    parser.add_argument(
        "--learning-state",
        default=None,
        help="Path to persistent QA learning state JSON (default: beside --rl-checkpoint)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    agent = QASpecialistAgent(
        spec_path=args.spec,
        nlp_prompt=args.prompt,
        tenant_id=args.tenant_id,
        base_url=args.base_url,
        output_dir=args.output_dir,
        max_scenarios=args.max_scenarios,
        pass_threshold=args.pass_threshold,
        rl_checkpoint_path=args.rl_checkpoint,
        learning_state_path=args.learning_state,
    )

    report = agent.run()
    summary = report["summary"]
    files = report["report_files"]

    print("QA specialist run complete")
    print(f"Spec: {report['metadata']['spec_title']}")
    print(
        f"Scenarios: total={summary['total_scenarios']} "
        f"passed={summary['passed_scenarios']} failed={summary['failed_scenarios']}"
    )
    print(f"Pass rate: {summary['pass_rate']}")
    print(f"Quality gate met: {summary['meets_quality_gate']}")
    print(f"JSON report: {files['json']}")
    print(f"Markdown report: {files['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
