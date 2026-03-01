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
        self.learning_state_path = self.output_dir / "learning_state.json"
        self.learning_state = self._load_learning_state()
        self.adaptive_policy = AdaptiveScenarioPolicy.from_state(
            self.learning_state.get("adaptive_policy"),
            fallback_scenario_stats=self.learning_state.get("scenario_stats", {}),
        )
        self.learning_state["adaptive_policy"] = self.adaptive_policy.to_state()
        self.learning_state["scenario_stats"] = dict(self.adaptive_policy.scenario_stats)
        self._last_selection_trace: List[Dict[str, Any]] = []

    def run(self) -> Dict[str, Any]:
        """Run the full QA specialist workflow."""
        started_at = time.time()
        spec = self._load_spec()
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
                "top_decisions": self._last_selection_trace[:20],
            },
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
            "adaptive_policy": {},
        }

    def _save_learning_state(self) -> None:
        payload = dict(self.learning_state)
        payload["adaptive_policy"] = self.adaptive_policy.to_state()
        payload["scenario_stats"] = dict(self.adaptive_policy.scenario_stats)
        payload["selection_trace"] = list(self._last_selection_trace[:SELECTION_TRACE_LIMIT])

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

        with open(self.learning_state_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _learning_state_snapshot(self) -> Dict[str, Any]:
        type_weights = self.learning_state.get("test_type_weights", {})
        endpoint_weights = self.learning_state.get("endpoint_weights", {})
        scenario_stats = self.adaptive_policy.scenario_stats
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

        while candidates and len(selected) < self.max_scenarios:
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
        self.learning_state["selection_trace"] = list(selection_trace)
        return selected

    def _predict_rl_state_risk(self, scenario: TestScenario) -> float:
        rl_algo = self.rl_trainer.rl_algorithm
        if not hasattr(rl_algo, "value_net"):
            return 0.0
        try:
            import torch  # Local import to avoid hard dependency when disabled.

            state = {
                "type": scenario.test_type.value,
                "method": scenario.method.upper(),
                "endpoint": scenario.endpoint,
                "expected_status": int(scenario.expected_status),
                "has_body": bool(scenario.body),
                "has_params": bool(scenario.params),
            }
            encoded = rl_algo._encode_state(state)
            with torch.no_grad():
                predicted_value = float(
                    rl_algo.value_net(torch.FloatTensor([encoded])).squeeze().item()
                )

            # Convert value into risk in [0,1]; lower value => higher risk.
            confidence = 1.0 / (1.0 + math.exp(-predicted_value))
            risk = 1.0 - confidence
            return 0.30 * risk
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
                    "endpoint": endpoint_template,
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
            stats["endpoint"] = endpoint_template

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

        self.adaptive_policy.scenario_stats = dict(self.learning_state["scenario_stats"])
        self.learning_state["adaptive_policy"] = self.adaptive_policy.to_state()

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
            passed = actual_status == int(scenario.expected_status)
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

    def _run_agent_lightning_training(
        self, spec_title: str, summary: Dict[str, Any], report_path: str
    ) -> Dict[str, Any]:
        latest_history = self.learning_state.get("decision_history", [])
        latest_run_reward = 0.0
        if latest_history:
            latest_run_reward = float(latest_history[-1].get("run_reward", 0.0))

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
                f"- Policy Feature Dim: `{state_snapshot.get('policy_feature_dim', 0)}`",
                f"- Scenario Patterns Tracked: `{state_snapshot.get('scenario_patterns_tracked', 0)}`",
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
