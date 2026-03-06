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
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import time
import uuid
from copy import deepcopy
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse

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
    TestType,
    TestScenario,
)


PATH_PARAM_PATTERN = re.compile(r"\{([^}]+)\}")
MISSING_PATH_PARAM_SENTINEL = "__qa_missing_path_param__"
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
GAM_CONTEXT_MIN_QUALITY = 0.55
RL_MUTATION_TARGET_LIMIT = 20
RL_MUTATION_PER_TARGET_LIMIT = 2
RL_MUTATION_MAX_VARIANTS_PER_TARGET = 5
RL_MUTATION_MAX_NEW_SCENARIOS = 48
RL_MUTATION_MIN_PRIORITY = 0.08
RL_HISTORY_SEED_TARGET_LIMIT = 20
RL_HISTORY_SEED_MAX_NEW_SCENARIOS = 24
RL_HISTORY_SEED_MIN_ATTEMPTS = 1
RL_HISTORY_SEED_MIN_PRIORITY = 0.12
RL_WEAK_MIN_ATTEMPTS = 3
RL_WEAK_FAILURE_RATE_THRESHOLD = 0.20
PORTFOLIO_STABLE_RATIO = 0.70
PORTFOLIO_FOCUS_RATIO = 0.20
PORTFOLIO_EXPLORE_RATIO = 0.10
FORCED_REPLAY_CADENCE_RUNS = 2
GAM_RECENT_FOCUS_WINDOW = 3
GAM_RECENT_FOCUS_LIMIT = 40
FLAKY_RERUN_MAX_ATTEMPTS = 3
RUNTIME_REPAIR_SUGGESTION_LIMIT = 50
SUPPORTED_SCRIPT_KINDS = {
    "python_pytest",
    "javascript_jest",
    "curl_script",
    "java_restassured",
}
DEFAULT_SCRIPT_KIND = "python_pytest"
DEFAULT_ENVIRONMENT_PROFILE = "mock"
SUPPORTED_ENVIRONMENT_PROFILES = {"mock", "staging", "prod_safe"}


def _bootstrap_runtime_env() -> None:
    """Best-effort .env loading for direct CLI runs.

    Shell scripts already export env vars, but direct `python ...qa_specialist_agent.py`
    invocations may skip `.env` sourcing.
    """
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    candidate_dirs = [
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent.parent,
    ]
    seen: set[str] = set()
    for directory in candidate_dirs:
        env_path = (directory / ".env").resolve()
        key = str(env_path)
        if key in seen:
            continue
        seen.add(key)
        if env_path.is_file():
            # Do not override existing exported env vars.
            load_dotenv(dotenv_path=env_path, override=False)


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
    corrected_expectation: bool = False
    contract_valid: Optional[bool] = None
    display_name: str = ""
    name_raw: str = ""
    mutation_strategy: str = ""
    history_seeded: bool = False


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
    verdict: str
    passed: bool
    duration_ms: float
    error: str = ""
    response_excerpt: str = ""
    verification: Dict[str, Any] = field(default_factory=dict)
    display_name: str = ""
    name_raw: str = ""
    mutation_strategy: str = ""
    history_seeded: bool = False


class _ScriptResponseProxy:
    """Minimal requests.Response-like wrapper backed by TestClient responses."""

    def __init__(self, response: Any):
        self._response = response
        self.status_code = int(getattr(response, "status_code", 0))
        self.text = getattr(response, "text", "")
        self.headers = getattr(response, "headers", {})

    def json(self) -> Any:
        return self._response.json()


class _TestClientRequestsAdapter:
    """Adapter exposing requests-style APIs over FastAPI TestClient."""

    def __init__(self, client: TestClient):
        self.client = client

    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> _ScriptResponseProxy:
        parsed = urlparse(str(url))
        path = parsed.path or "/"

        merged_params: Dict[str, Any] = {}
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            merged_params[key] = value
        if isinstance(params, dict):
            merged_params.update(params)

        response = self.client.request(
            method=str(method).upper(),
            url=path,
            headers=headers or {},
            params=merged_params or None,
            json=json,
        )
        return _ScriptResponseProxy(response)

    def get(self, url: str, **kwargs: Any) -> _ScriptResponseProxy:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> _ScriptResponseProxy:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> _ScriptResponseProxy:
        return self.request("PUT", url, **kwargs)

    def patch(self, url: str, **kwargs: Any) -> _ScriptResponseProxy:
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> _ScriptResponseProxy:
        return self.request("DELETE", url, **kwargs)


class _LiveRequestsAdapter:
    """Adapter exposing requests-style APIs against a real base URL."""

    def __init__(self, base_url: str, timeout_sec: float = 12.0):
        self.base_url = str(base_url or "").rstrip("/")
        self.timeout_sec = max(1.0, float(timeout_sec))
        self._session = None

    def __enter__(self) -> "_LiveRequestsAdapter":
        import requests

        self._session = requests.Session()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
        self._session = None

    def _absolute_url(self, url: str) -> str:
        raw = str(url or "").strip()
        if raw.startswith("http://") or raw.startswith("https://"):
            return raw
        if raw.startswith("/"):
            return f"{self.base_url}{raw}"
        return f"{self.base_url}/{raw}"

    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        if self._session is None:
            raise RuntimeError("Live adapter session is not open")
        request_timeout = float(timeout) if timeout is not None else float(self.timeout_sec)
        return self._session.request(
            method=str(method).upper(),
            url=self._absolute_url(url),
            headers=headers or {},
            params=params or None,
            json=json,
            timeout=request_timeout,
            **kwargs,
        )

    def get(self, url: str, **kwargs: Any) -> Any:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> Any:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> Any:
        return self.request("PUT", url, **kwargs)

    def patch(self, url: str, **kwargs: Any) -> Any:
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> Any:
        return self.request("DELETE", url, **kwargs)


class QASpecialistAgent:
    """QA-focused orchestrator with isolation, GAM memory, and RL feedback."""

    def __init__(
        self,
        spec_path: str,
        nlp_prompt: Optional[str] = None,
        tenant_id: str = "default_tenant",
        workspace_id: Optional[str] = None,
        base_url: str = "http://localhost:8000",
        output_dir: Optional[str] = None,
        max_scenarios: int = 200,
        max_runtime_sec: Optional[int] = None,
        llm_token_cap: Optional[int] = None,
        environment_profile: str = DEFAULT_ENVIRONMENT_PROFILE,
        pass_threshold: float = 0.70,
        rl_checkpoint_path: Optional[str] = None,
        learning_state_path: Optional[str] = None,
        script_kind: str = DEFAULT_SCRIPT_KIND,
        rl_train_mode: str = "periodic",
    ):
        _bootstrap_runtime_env()
        self.spec_path = str(spec_path)
        self.nlp_prompt = nlp_prompt
        self.tenant_id = tenant_id
        normalized_workspace = str(workspace_id or tenant_id or "default_workspace").strip()
        self.workspace_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", normalized_workspace)[:80] or "default_workspace"
        self.base_url = base_url.rstrip("/")
        self.max_scenarios = max(1, int(max_scenarios))
        self.max_runtime_sec = (
            max(1, int(max_runtime_sec))
            if max_runtime_sec is not None and int(max_runtime_sec) > 0
            else None
        )
        self.llm_token_cap = (
            max(64, int(llm_token_cap))
            if llm_token_cap is not None and int(llm_token_cap) > 0
            else None
        )
        env_profile = str(environment_profile or DEFAULT_ENVIRONMENT_PROFILE).strip().lower()
        if env_profile not in SUPPORTED_ENVIRONMENT_PROFILES:
            env_profile = DEFAULT_ENVIRONMENT_PROFILE
        self.environment_profile = env_profile
        self.pass_threshold = max(0.0, min(1.0, float(pass_threshold)))
        normalized_script_kind = str(script_kind or DEFAULT_SCRIPT_KIND).strip().lower()
        if normalized_script_kind not in SUPPORTED_SCRIPT_KINDS:
            raise ValueError(
                f"Unsupported --script-kind '{script_kind}'. "
                + f"Supported: {', '.join(sorted(SUPPORTED_SCRIPT_KINDS))}"
            )
        self.script_kind = normalized_script_kind
        normalized_rl_train_mode = str(rl_train_mode or os.getenv("RL_TRAIN_MODE", "periodic")).strip().lower()
        self.rl_train_mode = "periodic" if normalized_rl_train_mode != "periodic" else normalized_rl_train_mode

        if output_dir:
            self.output_dir = Path(output_dir).resolve()
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = (Path.cwd() / ".qa_specialist_workspace").resolve()
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm_scenario_debug_log_path = str(
            (self.output_dir / "llm_scenario_debug.jsonl").resolve()
        )

        self.rl_checkpoint_path = str(
            Path(rl_checkpoint_path).expanduser().resolve()
        ) if rl_checkpoint_path else str((self.output_dir / "agent_lightning_checkpoint.pt").resolve())
        self.learning_state_path = self._resolve_learning_state_path(
            learning_state_path
        )
        self.gam_storage_path = self.learning_state_path.with_name("gam_memory_pages.json")
        self.gam = GAMMemorySystem(
            use_vector_search=False,
            storage_path=str(self.gam_storage_path),
            autosave=True,
        )
        self.rl_trainer = LightningTrainerV2(
            gam_memory_system=self.gam,
            checkpoint_path=self.rl_checkpoint_path,
            checkpoint_autosave=True,
            gam_writeback=False,
            train_mode=self.rl_train_mode,
        )
        self.rl_trainer.register_agent("qa_specialist", self._qa_agent_feedback)
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
        self._operation_index: Dict[str, Dict[str, Any]] = {}
        self._spec_paths: set[str] = set()
        self._spec_scope_key: str = ""
        self._spec_memory_tags: set[str] = set()
        self._runtime_cap_hit: bool = False
        self._runtime_skipped_count: int = 0
        self._last_gam_rejected_excerpts: List[Dict[str, Any]] = []
        self._execution_isolation_mode: str = "in_memory_fastapi_testclient"
        self._base_scenario_source: str = "llm_base"
        self._llm_generation_degraded: bool = False
        self._llm_generation_degraded_reason: str = ""
        self._scenario_llm_mode: str = (
            str(os.getenv("QA_SCENARIO_LLM_MODE", "auto")).strip().lower() or "auto"
        )

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
        run_id = uuid.uuid4().hex
        stage_metrics_ms: Dict[str, float] = {}
        stage_started = time.perf_counter()
        self._runtime_cap_hit = False
        self._runtime_skipped_count = 0
        self._llm_generation_degraded = False
        self._llm_generation_degraded_reason = ""
        spec = self._load_spec()
        self._auth_required_ops = self._build_auth_requirement_map(spec)
        self._operation_index = self._build_operation_index(spec)
        self._spec_paths = {
            str(path)
            for path in (spec.get("paths") or {}).keys()
            if isinstance(path, str)
        }
        self._spec_scope_key = self._compute_spec_scope_key(spec)
        self._apply_llm_token_cap()
        spec_intelligence = self._build_spec_intelligence(spec)
        oss_tooling = self._collect_oss_tooling_status(spec)
        oss_checks = self._run_optional_oss_checks(tooling_status=oss_tooling)
        stage_metrics_ms["stage_1_spec_intelligence"] = round(
            (time.perf_counter() - stage_started) * 1000.0,
            3,
        )
        spec_title = spec.get("info", {}).get("title", "Unknown API")
        spec_version = spec.get("info", {}).get("version", "unknown")
        self._spec_memory_tags = set(self._build_spec_memory_tags(spec_title))

        stage_started = time.perf_counter()
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
        learning_signal_page_id = self._persist_rl_learning_signal_page(spec_title)
        learning_trend_page_id = self._persist_rl_trend_signal_page(spec_title)

        seeded_page_ids = [
            page_id
            for page_id in [learning_signal_page_id, learning_trend_page_id]
            if isinstance(page_id, str) and page_id.strip()
        ]
        research_context = {
            "spec_title": spec_title,
            "auth_type": self._infer_auth_type(spec),
            "endpoints": self._extract_endpoint_metadata(spec),
            "tenant_id": self.tenant_id,
            "learning_weakness_hints": self._build_gam_learning_hints(limit=5),
            "learning_run_count": int(self.learning_state.get("run_count", 0)),
            "prior_page_ids": seeded_page_ids,
            "spec_memory_tags": sorted(self._spec_memory_tags),
        }
        spec_context_page_id = self._persist_gam_spec_context_page(
            spec_title=spec_title,
            spec=spec,
            learning_hints=research_context.get("learning_weakness_hints", []),
        )
        if isinstance(spec_context_page_id, str) and spec_context_page_id.strip():
            research_context["prior_page_ids"] = [
                *seeded_page_ids,
                spec_context_page_id,
            ]
        research_result = self.gam.research(research_context)
        research_result.memory_excerpts = self._filter_memory_excerpts_for_current_spec(
            research_result.memory_excerpts,
            spec_title=spec_title,
        )
        if not research_result.memory_excerpts:
            fallback_excerpt = self._build_gam_spec_context_fallback_excerpt(
                spec_context_page_id=spec_context_page_id
            )
            if fallback_excerpt:
                research_result.memory_excerpts = [fallback_excerpt]
        gam_diagnostics = self._build_gam_diagnostics(research_result.memory_excerpts)
        pre_fallback_pack = self._build_gam_context_pack(
            memory_excerpts=research_result.memory_excerpts,
            diagnostics=gam_diagnostics,
            reflection=research_result.reflection,
        )
        external_fallback_excerpts: List[Dict[str, Any]] = []
        if self._needs_external_context(pre_fallback_pack):
            external_fallback_excerpts = self._build_trusted_external_doc_excerpts(
                spec_title=spec_title,
                auth_type=self._infer_auth_type(spec),
                learning_hints=research_context.get("learning_weakness_hints", []),
            )
            if external_fallback_excerpts:
                research_result.memory_excerpts = (
                    list(research_result.memory_excerpts)
                    + list(external_fallback_excerpts)
                )
                gam_diagnostics = self._build_gam_diagnostics(research_result.memory_excerpts)
        stage_metrics_ms["stage_2_gam_memory_research"] = round(
            (time.perf_counter() - stage_started) * 1000.0,
            3,
        )
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
                        "diagnostics": gam_diagnostics,
                        "fallback_external_excerpt_count": len(external_fallback_excerpts),
                        "learning_signal_page_id": learning_signal_page_id,
                        "spec_context_page_id": spec_context_page_id,
                    },
                }
            ],
        )

        stage_started = time.perf_counter()
        simulator = HumanTesterSimulator(
            spec,
            self.base_url,
            llm_debug_log_path=self.llm_scenario_debug_log_path,
        )
        rl_prompt_focus_points = self._build_rl_prompt_focus_points(limit=3)
        effective_prompt = self._compose_effective_prompt(
            self.nlp_prompt,
            research_result.memory_excerpts,
            rl_focus_points=rl_prompt_focus_points,
        )
        prompt_trace = self._build_prompt_trace(
            self.nlp_prompt,
            research_result.memory_excerpts,
            effective_prompt,
            rl_focus_points=rl_prompt_focus_points,
        )
        base_scenarios = simulator.think_like_tester(effective_prompt)
        base_scenarios, happy_guardrail_summary = self._ensure_happy_path_coverage(
            base_scenarios
        )
        base_scenarios, workflow_guardrail_summary = self._inject_workflow_sequence_scenarios(
            base_scenarios,
            spec_intelligence=spec_intelligence,
            limit=6,
        )
        base_scenarios = self._dedupe_scenarios_by_fingerprint(base_scenarios)
        prompt_trace["scenario_generation"] = {
            "engine": str(getattr(simulator, "last_generation_engine", "unknown")),
            "llm_enabled": bool(getattr(simulator, "llm_enabled", False)),
            "llm_mode": str(getattr(simulator, "_llm_mode", "")),
            "llm_model": (
                str(getattr(simulator, "_llm_model", ""))
                if bool(getattr(simulator, "llm_enabled", False))
                else ""
            ),
            "llm_stats": dict(getattr(simulator, "llm_stats", {}) or {}),
            "llm_diagnostics": dict(
                getattr(simulator, "last_llm_generation_diagnostics", {}) or {}
            ),
            "base_scenario_count": int(len(base_scenarios)),
            "happy_path_guardrail": happy_guardrail_summary,
            "workflow_guardrail": workflow_guardrail_summary,
        }
        llm_stats = prompt_trace["scenario_generation"].get("llm_stats", {})
        llm_success = (
            int(llm_stats.get("scenario_success", 0) or 0)
            if isinstance(llm_stats, dict)
            else 0
        )
        self._base_scenario_source = "llm_base" if llm_success > 0 else "heuristic_base"
        prompt_trace["scenario_generation"]["base_source"] = self._base_scenario_source
        self._enforce_scenario_generation_quality(prompt_trace)
        prompt_trace["scenario_generation"]["degraded"] = bool(self._llm_generation_degraded)
        if self._llm_generation_degraded_reason:
            prompt_trace["scenario_generation"]["degraded_reason"] = str(
                self._llm_generation_degraded_reason
            )
        scenario_generation_trace = self._build_scenario_generation_trace(
            base_scenarios=base_scenarios,
            effective_prompt=effective_prompt,
        )
        stage_metrics_ms["stage_3_scenario_generation"] = round(
            (time.perf_counter() - stage_started) * 1000.0,
            3,
        )

        stage_started = time.perf_counter()
        all_scenarios, mutation_summary = self._augment_scenarios_with_rl_mutation(
            spec,
            base_scenarios,
        )
        all_scenarios = self._dedupe_scenarios_by_fingerprint(all_scenarios)
        scenarios = self._select_scenarios_with_learning(all_scenarios)
        scenarios, repair_summary = self._apply_scenario_repairs(spec, scenarios)
        scenarios = self._prepare_scenarios_for_execution_and_scripts(scenarios)
        stage_metrics_ms["stage_4_mutation_selection"] = round(
            (time.perf_counter() - stage_started) * 1000.0,
            3,
        )

        stage_started = time.perf_counter()
        execution_results = self._execute_scenarios(spec, scenarios)
        runtime_repair_ingest = self._ingest_runtime_repair_suggestions(
            scenarios=scenarios,
            results=execution_results,
        )
        script_scenarios = list(scenarios)
        generated_files = self._generate_test_files(script_scenarios)
        generated_script_execution = self._execute_generated_script(spec, generated_files)
        stage_metrics_ms["stage_6_execute_verify"] = round(
            (time.perf_counter() - stage_started) * 1000.0,
            3,
        )

        stage_started = time.perf_counter()
        summary = self._build_summary(spec, scenarios, execution_results)
        repro_artifacts = self._build_repro_artifacts(
            spec=spec,
            scenarios=scenarios,
            results=execution_results,
            limit=30,
        )
        learning_feedback = self._compute_learning_feedback(
            scenarios,
            execution_results,
            summary,
            repro_artifacts=repro_artifacts,
        )
        scenario_stats_before = dict(
            self._scenario_stats_for_current_spec(self.adaptive_policy.scenario_stats)
        )
        run_count_before = int(self.learning_state.get("run_count", 0))
        self._update_learning_state(learning_feedback)
        scenario_stats_after = dict(
            self._scenario_stats_for_current_spec(self.adaptive_policy.scenario_stats)
        )
        scenario_context = self._build_scenario_context(
            base_scenarios=base_scenarios,
            candidate_scenarios=all_scenarios,
            selected_scenarios=scenarios,
            execution_results=execution_results,
            scenario_stats_before=scenario_stats_before,
            run_count_before=run_count_before,
        )
        weak_pattern_deltas = self._build_weak_pattern_deltas(
            before=scenario_stats_before,
            after=scenario_stats_after,
            limit=40,
        )
        policy_movement = self._build_policy_movement_metrics(scenarios)
        learning_delta = self._compute_learning_delta_summary(
            weak_pattern_deltas=weak_pattern_deltas,
            policy_movement=policy_movement,
        )
        self._persist_policy_movement_state(scenarios)
        gam_context_pack = self._build_gam_context_pack(
            memory_excerpts=research_result.memory_excerpts,
            diagnostics=gam_diagnostics,
            reflection=research_result.reflection,
        )
        stage_metrics_ms["stage_7_reward_training"] = round(
            (time.perf_counter() - stage_started) * 1000.0,
            3,
        )

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

        issues_found = [
            (
                f"{str(item.get('name', 'unknown'))}: expected "
                f"{item.get('expected_status')} got {item.get('actual_status')} "
                f"at {str(item.get('endpoint', 'unknown'))}"
            )
            for item in summary.get("failed_examples", [])[:10]
            if isinstance(item, dict)
        ]
        weakness_decisions = self._build_current_run_weakness_decisions(
            learning_feedback=learning_feedback,
            limit=2,
        )
        key_decisions = [
            "Generated "
            + f"{summary['total_scenarios']} QA scenarios from {len(all_scenarios)} candidates "
            + f"(base={len(base_scenarios)}, rl_mutations={mutation_summary.get('mutated_candidates_added', 0)})",
            *weakness_decisions,
            "Executed scenarios using isolation mode "
            + f"`{self._execution_isolation_mode}` under profile `{self.environment_profile}`",
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

        stage_started = time.perf_counter()
        rl_report_path = self.output_dir / "qa_execution_report.json"
        rl_data = self._run_agent_lightning_training(
            spec_title=spec_title,
            summary=summary,
            report_path=str(rl_report_path),
            learning_feedback=learning_feedback,
        )
        stage_metrics_ms["stage_8_reporting_rl"] = round(
            (time.perf_counter() - stage_started) * 1000.0,
            3,
        )

        report = {
            "metadata": {
                "run_id": run_id,
                "spec_path": self.spec_path,
                "spec_title": spec_title,
                "spec_version": spec_version,
                "tenant_id": self.tenant_id,
                "workspace_id": self.workspace_id,
                "spec_key": self._spec_scope_key,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "execution_seconds": round(time.time() - started_at, 3),
                "isolation_mode": self._execution_isolation_mode,
                "environment_profile": self.environment_profile,
                "stage_metrics_ms": stage_metrics_ms,
                "runtime_caps": {
                    "max_scenarios": int(self.max_scenarios),
                    "max_runtime_sec": self.max_runtime_sec,
                    "llm_token_cap": self.llm_token_cap,
                },
                "script_kind": self.script_kind,
                "rl_train_mode": self.rl_train_mode,
                "llm_scenario_debug_log": self.llm_scenario_debug_log_path,
                "llm_generation_degraded": bool(self._llm_generation_degraded),
                "llm_generation_degraded_reason": str(
                    self._llm_generation_degraded_reason
                ),
            },
            "summary": summary,
            "learning": {
                "feedback": learning_feedback,
                "state_snapshot": self._learning_state_snapshot(),
                "state_file": str(self.learning_state_path),
                "agent_lightning_checkpoint": self.rl_checkpoint_path,
                "learning_delta_status": str(learning_delta.get("status", "no_learning_delta_detected")),
                "learning_delta_reason": str(learning_delta.get("reason", "")),
                "policy_movement": policy_movement,
            },
            "selection_policy": {
                "algorithm": "contextual_linear_ucb",
                "candidate_count": len(all_scenarios),
                "base_candidate_count": int(len(base_scenarios)),
                "mutated_candidate_count": int(
                    mutation_summary.get("mutated_candidates_added", 0)
                ),
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
                "novelty_target": int(
                    self._last_selection_summary.get("novelty_target", 0)
                ),
                "novel_selected_count": int(
                    self._last_selection_summary.get("novel_selected_count", 0)
                ),
                "top_decisions": self._last_selection_trace[:20],
            },
            "selection_decision_trace": self._last_selection_trace[:120],
            "mutation_policy": mutation_summary,
            "mutation_decision_trace": list(mutation_summary.get("applied_examples", []) or [])[:120],
            "repair_policy": {
                **repair_summary,
                "runtime_suggestions_applied": int(
                    runtime_repair_ingest.get("applied", 0)
                ),
                "runtime_suggestion_examples": runtime_repair_ingest.get(
                    "examples", []
                ),
            },
            "scenario_source_breakdown": scenario_context.get("source_breakdown", {}),
            "weak_pattern_deltas": weak_pattern_deltas,
            "repro_artifacts": repro_artifacts,
            "environment_tier_signal": self._environment_tier_signal(),
            "gam_context_pack": gam_context_pack,
            "prompt_trace": prompt_trace,
            "scenario_generation_trace": scenario_generation_trace,
            "scenario_context": scenario_context,
            "generated_test_files": generated_files,
            "generated_script_execution": generated_script_execution,
            "scenario_results": [asdict(r) for r in execution_results],
            "spec_intelligence": spec_intelligence,
            "oss_tooling": oss_tooling,
            "oss_checks": oss_checks,
            "gam": {
                "session_id": session_id,
                "memo_page_id": memo_page.id,
                "memo_title": memo_page.title,
                "lossless_page_ids": [p.id for p in lossless_pages],
                "memory_store_path": str(self.gam_storage_path),
                "research_plan": research_result.plan,
                "research_plan_count": len(research_result.plan),
                "research_iterations": int(research_result.iteration),
                "research_reflection": research_result.reflection,
                "research_info_checks": list(getattr(research_result, "info_checks", []) or []),
                "research_retrieval_trace": list(
                    getattr(research_result, "retrieval_trace", []) or []
                ),
                "research_engine": dict(getattr(research_result, "research_engine", {}) or {}),
                "research_excerpt_count": len(research_result.memory_excerpts),
                "research_excerpts": research_result.memory_excerpts,
                "fallback_external_excerpts": external_fallback_excerpts,
                "context_pack": gam_context_pack,
                "learning_signal_page_id": learning_signal_page_id,
                "learning_trend_page_id": learning_trend_page_id,
                "spec_context_page_id": spec_context_page_id,
                "excerpt_source_breakdown": gam_diagnostics.get("source_breakdown", {}),
                "excerpt_preview": gam_diagnostics.get("excerpt_preview", []),
                "diagnostics": gam_diagnostics,
            },
            "agent_lightning": rl_data,
            "paper_references": {
                "agent_lightning": "https://arxiv.org/pdf/2508.03680",
                "gam": "https://arxiv.org/pdf/2511.18423",
            },
        }

        report_paths = self._write_reports(report)
        report["report_files"] = {
            **report_paths,
            "llm_scenario_debug": self.llm_scenario_debug_log_path,
        }
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

        targets = {
            "python_pytest": {
                "path": tests_dir / "test_api.py",
                "render": generator.generate_python_tests,
            },
            "javascript_jest": {
                "path": tests_dir / "test_api.test.js",
                "render": generator.generate_javascript_tests,
            },
            "curl_script": {
                "path": tests_dir / "test_api.sh",
                "render": generator.generate_curl_tests,
            },
            "java_restassured": {
                "path": tests_dir / "APITests.java",
                "render": generator.generate_java_tests,
            },
        }

        selected = targets[self.script_kind]
        output_path = selected["path"]
        output_path.write_text(selected["render"](), encoding="utf-8")
        return {self.script_kind: str(output_path)}

    def _execute_generated_script(
        self, spec: Dict[str, Any], generated_files: Dict[str, str]
    ) -> Dict[str, Any]:
        script_path_raw = generated_files.get(self.script_kind)
        if not script_path_raw:
            return {
                "kind": self.script_kind,
                "status": "missing",
                "executed": False,
                "error": "Generated script path missing",
            }

        script_path = Path(script_path_raw).resolve()
        if not script_path.exists():
            return {
                "kind": self.script_kind,
                "status": "missing",
                "executed": False,
                "script_path": str(script_path),
                "error": "Generated script file not found",
            }

        if self.script_kind != "python_pytest":
            return {
                "kind": self.script_kind,
                "status": "skipped",
                "executed": False,
                "script_path": str(script_path),
                "reason": "Automatic script execution currently supports python_pytest only",
            }

        profile = str(self.environment_profile or DEFAULT_ENVIRONMENT_PROFILE).lower()
        if profile == "prod_safe":
            return {
                "kind": self.script_kind,
                "status": "skipped",
                "executed": False,
                "script_path": str(script_path),
                "reason": "Generated script execution is disabled for prod_safe profile",
            }
        if profile == "mock":
            return self._execute_python_script_in_isolated_mock(spec, script_path)
        return self._execute_python_script_live(script_path)

    def _execute_python_script_in_isolated_mock(
        self, spec: Dict[str, Any], script_path: Path
    ) -> Dict[str, Any]:
        # Imported lazily so this module can be used without server startup paths.
        from dynamic_mock_server import DynamicMockServer

        spec_copy = self.output_dir / "openapi_under_test_script_exec.yaml"
        spec_copy.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

        started = time.perf_counter()
        try:
            with TestClient(DynamicMockServer(str(spec_copy), host="127.0.0.1", port=0).app) as client:
                module_name = f"generated_python_tests_{int(time.time() * 1000)}"
                module_spec = importlib.util.spec_from_file_location(module_name, str(script_path))
                if module_spec is None or module_spec.loader is None:
                    raise RuntimeError("Unable to load generated python test module")

                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)
                setattr(module, "requests", _TestClientRequestsAdapter(client))
                setattr(module, "BASE_URL", "http://testserver")

                test_class = getattr(module, "TestAPI", None)
                if test_class is None:
                    raise RuntimeError("Generated python script missing TestAPI class")

                test_instance = test_class()
                test_methods = sorted(
                    name
                    for name in dir(test_instance)
                    if name.startswith("test_") and callable(getattr(test_instance, name))
                )

                method_results: List[Dict[str, Any]] = []
                passed_count = 0
                for method_name in test_methods:
                    fn = getattr(test_instance, method_name)
                    method_started = time.perf_counter()
                    try:
                        fn()
                        passed = True
                        error_msg = ""
                    except AssertionError as exc:
                        passed = False
                        error_msg = str(exc) or "AssertionError"
                    except Exception as exc:
                        passed = False
                        error_msg = f"{type(exc).__name__}: {exc}"

                    duration_ms = (time.perf_counter() - method_started) * 1000.0
                    if passed:
                        passed_count += 1
                    method_results.append(
                        {
                            "name": method_name,
                            "passed": passed,
                            "error": error_msg,
                            "duration_ms": round(duration_ms, 3),
                        }
                    )

                total_count = len(method_results)
                failed_count = total_count - passed_count
                elapsed_ms = (time.perf_counter() - started) * 1000.0

                return {
                    "kind": self.script_kind,
                    "status": "executed",
                    "executed": True,
                    "script_path": str(script_path),
                    "total_tests": total_count,
                    "passed_tests": passed_count,
                    "failed_tests": failed_count,
                    "pass_rate": round((passed_count / total_count), 4) if total_count else 0.0,
                    "execution_ms": round(elapsed_ms, 3),
                    "results": method_results,
                }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return {
                "kind": self.script_kind,
                "status": "error",
                "executed": False,
                "script_path": str(script_path),
                "execution_ms": round(elapsed_ms, 3),
                "error": str(exc),
            }

    def _execute_python_script_live(self, script_path: Path) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            with _LiveRequestsAdapter(self.base_url, timeout_sec=12.0) as client:
                module_name = f"generated_python_tests_live_{int(time.time() * 1000)}"
                module_spec = importlib.util.spec_from_file_location(module_name, str(script_path))
                if module_spec is None or module_spec.loader is None:
                    raise RuntimeError("Unable to load generated python test module")

                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)
                setattr(module, "requests", client)
                setattr(module, "BASE_URL", str(self.base_url))

                test_class = getattr(module, "TestAPI", None)
                if test_class is None:
                    raise RuntimeError("Generated python script missing TestAPI class")

                test_instance = test_class()
                test_methods = sorted(
                    name
                    for name in dir(test_instance)
                    if name.startswith("test_") and callable(getattr(test_instance, name))
                )

                method_results: List[Dict[str, Any]] = []
                passed_count = 0
                for method_name in test_methods:
                    fn = getattr(test_instance, method_name)
                    method_started = time.perf_counter()
                    try:
                        fn()
                        passed = True
                        error_msg = ""
                    except AssertionError as exc:
                        passed = False
                        error_msg = str(exc) or "AssertionError"
                    except Exception as exc:
                        passed = False
                        error_msg = f"{type(exc).__name__}: {exc}"

                    duration_ms = (time.perf_counter() - method_started) * 1000.0
                    if passed:
                        passed_count += 1
                    method_results.append(
                        {
                            "name": method_name,
                            "passed": passed,
                            "error": error_msg,
                            "duration_ms": round(duration_ms, 3),
                        }
                    )

                total_count = len(method_results)
                failed_count = total_count - passed_count
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                return {
                    "kind": self.script_kind,
                    "status": "executed",
                    "executed": True,
                    "script_path": str(script_path),
                    "total_tests": total_count,
                    "passed_tests": passed_count,
                    "failed_tests": failed_count,
                    "pass_rate": round((passed_count / total_count), 4) if total_count else 0.0,
                    "execution_ms": round(elapsed_ms, 3),
                    "results": method_results,
                }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return {
                "kind": self.script_kind,
                "status": "error",
                "executed": False,
                "script_path": str(script_path),
                "execution_ms": round(elapsed_ms, 3),
                "error": str(exc),
            }

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
                    data.setdefault("gam_recent_focus_points", [])
                    data.setdefault("replay_schedule", {})
                    data["scenario_stats"] = self._sanitize_loaded_scenario_stats(
                        data.get("scenario_stats", {})
                    )
                    adaptive_policy_state = data.get("adaptive_policy", {})
                    if isinstance(adaptive_policy_state, dict):
                        adaptive_policy_state = dict(adaptive_policy_state)
                        adaptive_policy_state["scenario_stats"] = (
                            self._sanitize_loaded_scenario_stats(
                                adaptive_policy_state.get("scenario_stats", {})
                            )
                        )
                        data["adaptive_policy"] = adaptive_policy_state
                    # Repair rules reference specific fingerprints; rebuild from sanitized stats.
                    data["scenario_repair_rules"] = {}
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
            "gam_recent_focus_points": [],
            "replay_schedule": {},
        }

    def _sanitize_loaded_scenario_stats(
        self, scenario_stats_raw: Any
    ) -> Dict[str, Dict[str, Any]]:
        if not isinstance(scenario_stats_raw, dict):
            return {}

        def _safe_int(value: Any, default: int = 0) -> int:
            try:
                return int(value)
            except Exception:
                return int(default)

        def _safe_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except Exception:
                return float(default)

        normalized: Dict[str, Dict[str, Any]] = {}
        reward_weighted_sums: Dict[str, float] = {}

        for fingerprint, stats_raw in scenario_stats_raw.items():
            if not isinstance(stats_raw, dict):
                continue
            parsed = self._parse_scenario_fingerprint(str(fingerprint))
            if not parsed:
                # Keep unknown legacy entry as-is.
                normalized[str(fingerprint)] = dict(stats_raw)
                continue

            method = str(parsed.get("method", "GET")).upper()
            endpoint = str(parsed.get("endpoint", ""))
            test_type = self._normalize_learning_test_type(
                str(parsed.get("test_type", "error_handling"))
            )
            normalized_expected = self._normalize_expected_status_for_test_type(
                test_type=test_type,
                expected_status=int(parsed.get("expected_status", 400) or 400),
            )
            has_body = bool(parsed.get("has_body", False))
            has_params = bool(parsed.get("has_params", False))
            normalized_fp = self._scenario_fingerprint_from_fields(
                method=method,
                endpoint=endpoint,
                test_type=test_type,
                expected_status=normalized_expected,
                has_body=has_body,
                has_params=has_params,
            )

            attempts = max(0, _safe_int(stats_raw.get("attempts", 0), 0))
            failures = max(0, _safe_int(stats_raw.get("failures", 0), 0))
            if attempts > 0 and failures <= 0:
                failures = int(
                    round(_safe_float(stats_raw.get("failure_rate", 0.0), 0.0) * float(attempts))
                )
            failures = max(0, min(attempts, failures))
            avg_reward = _safe_float(stats_raw.get("avg_reward", 0.0), 0.0)
            actual_status_counts = (
                stats_raw.get("actual_status_counts", {})
                if isinstance(stats_raw.get("actual_status_counts"), dict)
                else {}
            )

            entry = normalized.setdefault(
                normalized_fp,
                {
                    "attempts": 0,
                    "failures": 0,
                    "failure_rate": 0.0,
                    "avg_reward": 0.0,
                    "method": method,
                    "endpoint": endpoint,
                    "test_type": test_type,
                    "expected_status": int(normalized_expected),
                    "actual_status_counts": {},
                    "spec_scope": str(stats_raw.get("spec_scope", "")).strip() or None,
                },
            )
            entry["attempts"] = int(entry.get("attempts", 0)) + attempts
            entry["failures"] = int(entry.get("failures", 0)) + failures
            reward_weighted_sums[normalized_fp] = (
                float(reward_weighted_sums.get(normalized_fp, 0.0))
                + (avg_reward * float(max(1, attempts)))
            )

            merged_counts = (
                entry.get("actual_status_counts", {})
                if isinstance(entry.get("actual_status_counts"), dict)
                else {}
            )
            for status_raw, count_raw in actual_status_counts.items():
                try:
                    status = int(status_raw)
                    count = int(count_raw)
                except Exception:
                    continue
                if status < 100 or status > 599 or count <= 0:
                    continue
                key = str(status)
                merged_counts[key] = int(merged_counts.get(key, 0)) + count
            entry["actual_status_counts"] = merged_counts

        for fp, entry in normalized.items():
            attempts = max(0, _safe_int(entry.get("attempts", 0), 0))
            failures = max(0, min(attempts, _safe_int(entry.get("failures", 0), 0)))
            entry["attempts"] = attempts
            entry["failures"] = failures
            entry["failure_rate"] = (float(failures) / float(attempts)) if attempts > 0 else 0.0
            weighted_sum = float(reward_weighted_sums.get(fp, 0.0))
            entry["avg_reward"] = (
                weighted_sum / float(max(1, attempts)) if attempts > 0 else 0.0
            )

        return normalized

    def _normalize_learning_test_type(self, raw_test_type: str) -> str:
        normalized = re.sub(r"[\s\-]+", "_", str(raw_test_type or "").strip().lower())
        aliases = {
            "auth": "authentication",
            "authn": "authentication",
            "authz": "authorization",
            "security": "authentication",
            "validation": "input_validation",
            "boundary": "boundary_testing",
            "error": "error_handling",
            "edge_cases": "boundary_testing",
        }
        return aliases.get(normalized, normalized or "error_handling")

    def _canonicalize_learning_pattern(
        self,
        *,
        method: str,
        endpoint: str,
        test_type: str,
        expected_status: int,
    ) -> Optional[Dict[str, Any]]:
        normalized_method = str(method or "GET").upper()
        normalized_endpoint = str(endpoint or "").strip()
        if not normalized_endpoint:
            return None

        normalized_test_type = self._normalize_learning_test_type(test_type)
        normalized_expected = self._normalize_expected_status_for_test_type(
            test_type=normalized_test_type,
            expected_status=int(expected_status),
        )
        operation_key = self._operation_key(normalized_method, normalized_endpoint)
        operation_exists = operation_key in self._operation_index

        if not operation_exists:
            normalized_test_type = "error_handling"
            normalized_expected = 405

        if (
            normalized_test_type == "error_handling"
            and normalized_expected in AUTH_NEGATIVE_STATUS_CODES
            and operation_key in self._auth_required_ops
        ):
            normalized_test_type = "authentication"
            normalized_expected = 401 if normalized_expected == 401 else 403

        normalized_expected = self._normalize_expected_status_for_test_type(
            test_type=normalized_test_type,
            expected_status=normalized_expected,
        )
        return {
            "method": normalized_method,
            "endpoint": normalized_endpoint,
            "test_type": normalized_test_type,
            "expected_status": int(normalized_expected),
            "operation_exists": bool(operation_exists),
        }

    def _compute_spec_scope_key(self, spec: Dict[str, Any]) -> str:
        info = spec.get("info", {}) if isinstance(spec.get("info"), dict) else {}
        title = str(info.get("title", "unknown")).strip().lower()
        version = str(info.get("version", "unknown")).strip().lower()
        paths = sorted(
            str(path).strip()
            for path in (spec.get("paths") or {}).keys()
            if isinstance(path, str)
        )
        raw = "|".join([title, version, *paths])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        return f"spec_{digest[:16]}"

    def _build_spec_memory_tags(self, spec_title: str) -> List[str]:
        tags: set[str] = set()
        title_slug = re.sub(r"[^a-z0-9]+", "_", str(spec_title).lower()).strip("_")
        if title_slug:
            tags.add(title_slug)
        scope_key = str(self._spec_scope_key or "").strip().lower()
        if scope_key:
            tags.add(scope_key)
        return sorted(tag for tag in tags if tag)

    def _build_spec_intelligence(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        paths = spec.get("paths", {}) if isinstance(spec.get("paths"), dict) else {}
        scenario_stats = self._scenario_stats_for_current_spec(
            self.adaptive_policy.scenario_stats
        )
        op_failure: Dict[str, Dict[str, float]] = {}
        for fingerprint, stats_raw in scenario_stats.items():
            if not isinstance(stats_raw, dict):
                continue
            parsed = self._parse_scenario_fingerprint(str(fingerprint))
            if not parsed:
                continue
            op_key = self._operation_key(parsed["method"], parsed["endpoint"])
            op_failure[op_key] = {
                "attempts": float(stats_raw.get("attempts", 0)),
                "failure_rate": float(stats_raw.get("failure_rate", 0.0)),
            }

        operation_risks: List[Dict[str, Any]] = []
        for op_key, meta in self._operation_index.items():
            method = str(op_key.split(" ", 1)[0]).upper()
            endpoint = str(op_key.split(" ", 1)[1]) if " " in op_key else ""
            auth_required = bool(op_key in self._auth_required_ops)
            write_method = method in {"POST", "PUT", "PATCH", "DELETE"}
            schema_complexity = self._estimate_schema_complexity(meta.get("request_schema", {}))
            history = op_failure.get(op_key, {})
            historical_failure_rate = float(history.get("failure_rate", 0.0))
            historical_attempts = int(history.get("attempts", 0))
            risk_score = (
                0.25 * (1.0 if auth_required else 0.0)
                + 0.20 * (1.0 if write_method else 0.0)
                + 0.30 * min(1.0, schema_complexity / 8.0)
                + 0.25 * min(1.0, historical_failure_rate)
            )
            operation_risks.append(
                {
                    "operation_key": op_key,
                    "method": method,
                    "endpoint": endpoint,
                    "auth_required": auth_required,
                    "write_operation": write_method,
                    "schema_complexity": round(float(schema_complexity), 4),
                    "historical_attempts": historical_attempts,
                    "historical_failure_rate": round(historical_failure_rate, 4),
                    "risk_score": round(risk_score, 4),
                    "required_fields": list(meta.get("required_fields", []) or []),
                    "path_params": list(meta.get("path_param_names", []) or []),
                    "query_params": list(meta.get("query_param_names", []) or []),
                }
            )

        operation_risks.sort(
            key=lambda item: (
                float(item.get("risk_score", 0.0)),
                float(item.get("historical_failure_rate", 0.0)),
                int(item.get("historical_attempts", 0)),
            ),
            reverse=True,
        )
        dependency_edges = self._infer_dependency_edges(paths, self._operation_index)
        workflow_candidates = self._extract_workflow_candidates(dependency_edges, limit=20)
        return {
            "operations_total": len(self._operation_index),
            "dependency_graph": {
                "edge_count": len(dependency_edges),
                "edges": dependency_edges[:120],
            },
            "workflow_candidates": workflow_candidates,
            "risk_map": {
                "top_risky_operations": operation_risks[:25],
            },
        }

    def _estimate_schema_complexity(self, schema: Any) -> float:
        if not isinstance(schema, dict) or not schema:
            return 0.0
        score = 0.0
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if isinstance(properties, dict):
            score += min(5.0, float(len(properties)) * 0.35)
            for prop_schema in properties.values():
                if isinstance(prop_schema, dict):
                    prop_type = str(prop_schema.get("type", "")).lower()
                    if prop_type in {"array", "object"}:
                        score += 0.35
                    if "enum" in prop_schema:
                        score += 0.20
                    if any(key in prop_schema for key in ("minimum", "maximum", "minLength", "maxLength")):
                        score += 0.15
        if isinstance(required, list):
            score += min(2.0, float(len(required)) * 0.20)
        return float(score)

    def _infer_dependency_edges(
        self,
        paths: Dict[str, Any],
        operation_index: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        edges: List[Dict[str, Any]] = []
        for consumer_key, meta in operation_index.items():
            consumer_method = str(consumer_key.split(" ", 1)[0]).upper()
            consumer_path = str(consumer_key.split(" ", 1)[1]) if " " in consumer_key else ""
            path_params = [str(p) for p in (meta.get("path_param_names", []) or [])]
            request_schema = meta.get("request_schema", {})
            request_props = (
                request_schema.get("properties", {})
                if isinstance(request_schema, dict) and isinstance(request_schema.get("properties"), dict)
                else {}
            )
            candidate_resources = set()
            for pname in path_params:
                if pname.lower().endswith("id") and len(pname) > 2:
                    candidate_resources.add(pname[:-2].lower())
            for prop in request_props.keys():
                pname = str(prop)
                if pname.lower().endswith("id") and len(pname) > 2:
                    candidate_resources.add(pname[:-2].lower())

            for producer_key in operation_index.keys():
                producer_method = str(producer_key.split(" ", 1)[0]).upper()
                producer_path = str(producer_key.split(" ", 1)[1]) if " " in producer_key else ""
                if producer_key == consumer_key:
                    continue
                if producer_method not in {"POST", "PUT"}:
                    continue
                producer_resource = producer_path.strip("/").split("/", 1)[0].lower()
                if producer_resource in candidate_resources:
                    edges.append(
                        {
                            "producer": producer_key,
                            "consumer": consumer_key,
                            "reason": "resource_id_dependency",
                            "resource": producer_resource,
                        }
                    )
                    continue
                if consumer_method in {"GET", "PATCH", "DELETE"} and "{" in consumer_path:
                    producer_base = producer_path.rstrip("/").lower()
                    consumer_base = consumer_path.split("/{", 1)[0].rstrip("/").lower()
                    if producer_base and producer_base == consumer_base:
                        edges.append(
                            {
                                "producer": producer_key,
                                "consumer": consumer_key,
                                "reason": "path_parameter_dependency",
                                "resource": producer_base.strip("/"),
                            }
                        )
        deduped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for edge in edges:
            key = (
                str(edge.get("producer", "")),
                str(edge.get("consumer", "")),
                str(edge.get("reason", "")),
            )
            if key not in deduped:
                deduped[key] = edge
        return list(deduped.values())

    def _extract_workflow_candidates(
        self,
        edges: List[Dict[str, Any]],
        *,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        if not edges:
            return []
        by_consumer: Dict[str, List[Dict[str, Any]]] = {}
        for edge in edges:
            consumer = str(edge.get("consumer", ""))
            by_consumer.setdefault(consumer, []).append(edge)
        candidates: List[Dict[str, Any]] = []
        for consumer, refs in by_consumer.items():
            for edge in refs:
                candidates.append(
                    {
                        "sequence": [str(edge.get("producer", "")), consumer],
                        "reason": str(edge.get("reason", "")),
                        "resource": str(edge.get("resource", "")),
                    }
                )
        candidates.sort(key=lambda item: (item.get("resource", ""), item.get("reason", "")))
        return candidates[: max(1, int(limit))]

    def _collect_oss_tooling_status(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        python_pkg_checks = {
            "schemathesis": "schemathesis",
            "hypothesis": "hypothesis",
            "openapi_core": "openapi_core",
            "openapi_spec_validator": "openapi_spec_validator",
            "pact": "pact",
            "locust": "locust",
            "testcontainers": "testcontainers",
        }
        cli_checks = {
            "restler": "restler",
            "evomaster": "evomaster",
            "zap_cli": "zap-baseline.py",
            "k6": "k6",
        }
        packages = {
            name: bool(importlib.util.find_spec(module))
            for name, module in python_pkg_checks.items()
        }
        binaries = {
            name: bool(shutil.which(binary))
            for name, binary in cli_checks.items()
        }
        spec_validation = self._validate_openapi_contract(spec, packages)
        return {
            "packages": packages,
            "binaries": binaries,
            "spec_validation": spec_validation,
        }

    def _run_optional_oss_checks(
        self,
        *,
        tooling_status: Dict[str, Any],
    ) -> Dict[str, Any]:
        packages = tooling_status.get("packages", {}) if isinstance(tooling_status.get("packages", {}), dict) else {}
        binaries = tooling_status.get("binaries", {}) if isinstance(tooling_status.get("binaries", {}), dict) else {}
        checks: Dict[str, Dict[str, Any]] = {}

        checks["openapi_spec_validation"] = dict(
            tooling_status.get("spec_validation", {}) or {}
        )

        def _probe_command(cmd: List[str]) -> Dict[str, Any]:
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=8,
                    check=False,
                )
                return {
                    "ok": bool(proc.returncode == 0),
                    "exit_code": int(proc.returncode),
                    "stdout": str(proc.stdout or "").strip()[:200],
                    "stderr": str(proc.stderr or "").strip()[:200],
                }
            except Exception as exc:
                return {"ok": False, "exit_code": -1, "stdout": "", "stderr": str(exc)}

        checks["hypothesis_shrinker"] = {
            "status": "active",
            "mode": "in_agent_greedy_shrinker",
            "notes": "Failed payload minimization runs in-agent for reproducible failures.",
        }
        schemathesis_available = bool(packages.get("schemathesis", False))
        schemathesis_probe = (
            _probe_command(["schemathesis", "--version"])
            if bool(shutil.which("schemathesis"))
            else {"ok": schemathesis_available, "exit_code": 0 if schemathesis_available else -1}
        )
        checks["schemathesis_stateful_smoke"] = {
            "available": schemathesis_available,
            "status": (
                "skipped"
                if not schemathesis_available
                else ("active" if bool(schemathesis_probe.get("ok", False)) else "misconfigured")
            ),
            "reason": (
                "schemathesis_not_installed"
                if not schemathesis_available
                else (
                    "cli_probe_failed"
                    if not bool(schemathesis_probe.get("ok", False))
                    else "stateful_smoke_cli_ready"
                )
            ),
            "probe": schemathesis_probe,
        }
        restler_available = bool(binaries.get("restler", False))
        restler_probe = _probe_command(["restler", "--help"]) if restler_available else {}
        checks["restler_sequence_seed"] = {
            "available": restler_available,
            "status": (
                "skipped"
                if not restler_available
                else ("active" if bool(restler_probe.get("ok", False)) else "misconfigured")
            ),
            "reason": (
                "restler_cli_not_found"
                if not restler_available
                else (
                    "restler_probe_failed"
                    if not bool(restler_probe.get("ok", False))
                    else "sequence_seed_cli_ready"
                )
            ),
            "probe": restler_probe,
        }
        evomaster_available = bool(binaries.get("evomaster", False))
        evomaster_probe = _probe_command(["evomaster", "--help"]) if evomaster_available else {}
        checks["evomaster_search_based"] = {
            "available": evomaster_available,
            "status": (
                "skipped"
                if not evomaster_available
                else ("active" if bool(evomaster_probe.get("ok", False)) else "misconfigured")
            ),
            "reason": (
                "evomaster_cli_not_found"
                if not evomaster_available
                else (
                    "evomaster_probe_failed"
                    if not bool(evomaster_probe.get("ok", False))
                    else "search_based_cli_ready"
                )
            ),
            "probe": evomaster_probe,
        }
        checks["pact_contract_check"] = {
            "available": bool(packages.get("pact", False)),
            "status": "skipped" if not packages.get("pact", False) else "active",
            "reason": (
                "pact_not_installed"
                if not packages.get("pact", False)
                else "consumer_provider_contract_stage_enabled"
            ),
        }
        zap_available = bool(binaries.get("zap_cli", False))
        zap_probe = _probe_command(["zap-baseline.py", "-h"]) if zap_available else {}
        checks["zap_api_scan"] = {
            "available": zap_available,
            "status": (
                "skipped"
                if not zap_available
                else ("active" if bool(zap_probe.get("ok", False)) else "misconfigured")
            ),
            "reason": (
                "zap_cli_not_found"
                if not zap_available
                else (
                    "zap_cli_probe_failed"
                    if not bool(zap_probe.get("ok", False))
                    else "security_scan_stage_enabled"
                )
            ),
            "probe": zap_probe,
        }
        load_available = bool(binaries.get("k6", False) or packages.get("locust", False))
        k6_probe = _probe_command(["k6", "version"]) if bool(binaries.get("k6", False)) else {}
        checks["k6_load_probe"] = {
            "available": load_available,
            "status": (
                "skipped"
                if not load_available
                else ("active" if (not k6_probe or bool(k6_probe.get("ok", False))) else "misconfigured")
            ),
            "reason": (
                "k6_and_locust_not_available"
                if not load_available
                else (
                    "k6_probe_failed"
                    if k6_probe and not bool(k6_probe.get("ok", False))
                    else "perf_probe_stage_enabled"
                )
            ),
            "probe": k6_probe,
        }
        checks["testcontainers_isolation"] = {
            "available": bool(packages.get("testcontainers", False)),
            "status": "skipped" if not packages.get("testcontainers", False) else "active",
            "reason": (
                "testcontainers_not_installed"
                if not packages.get("testcontainers", False)
                else "real_dependency_isolation_enabled"
            ),
        }
        return checks

    def _validate_openapi_contract(
        self,
        spec: Dict[str, Any],
        package_status: Dict[str, bool],
    ) -> Dict[str, Any]:
        result = {
            "validator": "openapi_spec_validator",
            "available": bool(package_status.get("openapi_spec_validator", False)),
            "valid": None,
            "error": "",
        }
        if not result["available"]:
            return result
        try:
            from openapi_spec_validator import validate_spec
            validate_spec(spec)
            result["valid"] = True
        except Exception as exc:
            result["valid"] = False
            result["error"] = str(exc)
        return result

    def _needs_external_context(self, gam_context_pack: Dict[str, Any]) -> bool:
        if not isinstance(gam_context_pack, dict):
            return True
        checks = gam_context_pack.get("contract_checks", {})
        if not isinstance(checks, dict):
            return True
        quality = float(gam_context_pack.get("quality_score", 0.0) or 0.0)
        required = ["has_weak_pattern_or_proxy", "has_trend_delta", "has_spec_risk", "has_next_action"]
        missing = [key for key in required if not bool(checks.get(key, False))]
        if quality < GAM_CONTEXT_MIN_QUALITY:
            return True
        return bool(missing)

    def _build_trusted_external_doc_excerpts(
        self,
        *,
        spec_title: str,
        auth_type: str,
        learning_hints: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        # Keep this concise and trusted-doc anchored; no generic static conventions.
        hints_input = [
            item for item in (learning_hints or [])
            if isinstance(item, dict)
        ]
        weak_hint_text = ""
        for hint in hints_input:
            failure_rate = float(hint.get("failure_rate", 0.0) or 0.0)
            avg_reward = float(hint.get("avg_reward", 0.0) or 0.0)
            if failure_rate >= RL_WEAK_FAILURE_RATE_THRESHOLD or avg_reward < 0.0:
                weak_hint_text = (
                    f"Weak RL pattern: {hint.get('method', 'GET')} {hint.get('endpoint', '/')} "
                    f"({hint.get('test_type', 'error_handling')}) expected "
                    f"{hint.get('expected_status', 400)}, failure_rate={round(failure_rate, 4)}."
                )
                break
        if not weak_hint_text:
            weak_hint_text = (
                "No persistent weak pattern detected yet; next action is exploration-focused "
                "scenario diversification across auth, validation, and dependency-order paths."
            )

        hints = [
            {
                "title": "Schemathesis Stateful Testing",
                "source": "trusted_docs_fallback",
                "tags": ["trusted_docs", "schemathesis", "stateful"],
                "excerpt": (
                    f"Spec risk action ({spec_title}): prioritize stateful sequences "
                    "for producer->consumer operations and verify dependent IDs."
                ),
                "url": "https://schemathesis.readthedocs.io/en/stable/",
            },
            {
                "title": "Agent Lightning RL Focus",
                "source": "trusted_docs_fallback",
                "tags": ["trusted_docs", "agent_lightning", "rl"],
                "excerpt": (
                    f"{weak_hint_text} Trend delta action: track run-over-run "
                    "failure-rate deltas with checkpointed policy updates."
                ),
                "url": "https://arxiv.org/abs/2508.03680",
            },
            {
                "title": "GAM Context Quality Contract",
                "source": "trusted_docs_fallback",
                "tags": ["trusted_docs", "gam", "context_pack"],
                "excerpt": (
                    f"Next action: enrich memory with non-generic signals for auth={auth_type} "
                    "including one trend delta, one spec risk, and one concrete mutation action."
                ),
                "url": "https://arxiv.org/abs/2511.18423",
            },
        ]
        return hints

    def _filter_memory_excerpts_for_current_spec(
        self,
        memory_excerpts: List[Dict[str, Any]],
        *,
        spec_title: str,
    ) -> List[Dict[str, Any]]:
        self._last_gam_rejected_excerpts = []
        if not isinstance(memory_excerpts, list) or not memory_excerpts:
            return []

        spec_title_lower = str(spec_title or "").strip().lower()
        required_tags = {str(tag).lower() for tag in self._spec_memory_tags if str(tag).strip()}
        filtered: List[Dict[str, Any]] = []

        for excerpt in memory_excerpts:
            if not isinstance(excerpt, dict):
                continue
            source = str(excerpt.get("source", "")).strip().lower()
            if source != "memo":
                filtered.append(excerpt)
                continue

            tags_raw = excerpt.get("tags", [])
            tags = (
                {str(item).strip().lower() for item in tags_raw if str(item).strip()}
                if isinstance(tags_raw, list)
                else set()
            )
            title_text = str(excerpt.get("title", "")).strip().lower()
            body_text = str(excerpt.get("excerpt", "")).strip().lower()
            matches_tag = bool(required_tags and tags.intersection(required_tags))
            matches_title = bool(spec_title_lower and spec_title_lower in title_text)
            matches_excerpt = bool(spec_title_lower and spec_title_lower in body_text)
            if matches_tag or matches_title or matches_excerpt:
                filtered.append(excerpt)
                continue
            self._last_gam_rejected_excerpts.append(
                {
                    "source": source or "memo",
                    "title": str(excerpt.get("title", "")),
                    "tags": list(tags_raw) if isinstance(tags_raw, list) else [],
                    "excerpt": self._smart_trim_text(str(excerpt.get("excerpt", "")), max_chars=220),
                    "reason": "spec_scope_mismatch",
                }
            )

        if filtered:
            return filtered[: len(memory_excerpts)]
        non_memo = [
            item for item in memory_excerpts
            if str(item.get("source", "")).strip().lower() != "memo"
        ][: len(memory_excerpts)]
        if not non_memo:
            return []
        for item in memory_excerpts:
            if str(item.get("source", "")).strip().lower() == "memo":
                self._last_gam_rejected_excerpts.append(
                    {
                        "source": "memo",
                        "title": str(item.get("title", "")),
                        "tags": list(item.get("tags", [])) if isinstance(item.get("tags"), list) else [],
                        "excerpt": self._smart_trim_text(str(item.get("excerpt", "")), max_chars=220),
                        "reason": "fallback_non_memo_only",
                    }
                )
        return non_memo

    def _build_gam_spec_context_fallback_excerpt(
        self,
        spec_context_page_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        page_id = str(spec_context_page_id or "").strip()
        if not page_id:
            return None
        page = self.gam.page_store.get_page(page_id)
        if not page:
            return None

        raw_excerpt = str(getattr(page, "content", "")).strip()
        if not raw_excerpt:
            return None

        normalized = self._smart_trim_text(
            re.sub(r"\s+", " ", raw_excerpt),
            max_chars=220,
        )
        if not normalized:
            return None

        return {
            "source": str(getattr(page, "source", "memo") or "memo"),
            "title": str(getattr(page, "title", "")),
            "tags": list(getattr(page, "tags", []) or []),
            "similarity": 1.0,
            "excerpt": normalized,
        }

    def _scenario_belongs_to_current_spec(self, method: str, endpoint: str) -> bool:
        normalized_endpoint = str(endpoint or "").strip()
        if not normalized_endpoint:
            return False
        if normalized_endpoint in self._spec_paths:
            return True
        op_key = self._operation_key(str(method or "GET").upper(), normalized_endpoint)
        if op_key in self._operation_index:
            return True

        for spec_path in self._spec_paths:
            if not self._path_matches_spec_template(spec_path, normalized_endpoint):
                continue
            spec_op_key = self._operation_key(str(method or "GET").upper(), spec_path)
            if spec_op_key in self._operation_index:
                return True
        return False

    def _path_matches_spec_template(self, spec_path: str, endpoint: str) -> bool:
        spec_value = str(spec_path or "").strip()
        endpoint_value = str(endpoint or "").strip()
        if not spec_value or not endpoint_value:
            return False
        if spec_value == endpoint_value:
            return True

        spec_parts = [part for part in spec_value.strip("/").split("/") if part]
        endpoint_parts = [part for part in endpoint_value.strip("/").split("/") if part]
        if len(spec_parts) != len(endpoint_parts):
            return False

        for spec_seg, endpoint_seg in zip(spec_parts, endpoint_parts):
            if spec_seg.startswith("{") and spec_seg.endswith("}"):
                if not endpoint_seg:
                    return False
                continue
            if spec_seg != endpoint_seg:
                return False
        return True

    def _scenario_stats_for_current_spec(self, scenario_stats_raw: Any) -> Dict[str, Any]:
        if not isinstance(scenario_stats_raw, dict):
            return {}

        current_scope = str(self._spec_scope_key or "").strip()
        has_matching_scoped_entries = False
        if current_scope:
            for stats_raw in scenario_stats_raw.values():
                if not isinstance(stats_raw, dict):
                    continue
                if str(stats_raw.get("spec_scope", "")).strip() == current_scope:
                    has_matching_scoped_entries = True
                    break

        filtered: Dict[str, Any] = {}
        for fingerprint, stats_raw in scenario_stats_raw.items():
            if not isinstance(stats_raw, dict):
                continue
            stats_scope = str(stats_raw.get("spec_scope", "")).strip()
            if current_scope and stats_scope:
                if stats_scope != current_scope:
                    continue
            elif current_scope and has_matching_scoped_entries and not stats_scope:
                # When scoped data for this spec exists, do not blend old unscoped legacy entries.
                continue

            parsed = self._parse_scenario_fingerprint(str(fingerprint))
            method = str((parsed or {}).get("method") or stats_raw.get("method", "GET")).upper()
            endpoint = str((parsed or {}).get("endpoint") or stats_raw.get("endpoint", "")).strip()
            if not self._scenario_belongs_to_current_spec(method, endpoint):
                continue
            filtered[str(fingerprint)] = stats_raw

        return filtered

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
        scenario_stats = self._scenario_stats_for_current_spec(
            self.adaptive_policy.scenario_stats
        )
        repair_rules = self.learning_state.get("scenario_repair_rules", {})
        decision_history_raw = self.learning_state.get("decision_history", [])
        decision_history = (
            decision_history_raw if isinstance(decision_history_raw, list) else []
        )
        history_tail = [item for item in decision_history[-10:] if isinstance(item, dict)]
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

        latest_run = history_tail[-1] if history_tail else {}
        previous_run = history_tail[-2] if len(history_tail) > 1 else {}
        latest_reward = float(latest_run.get("run_reward", 0.0)) if latest_run else 0.0
        previous_reward = float(previous_run.get("run_reward", 0.0)) if previous_run else 0.0
        latest_avg_decision = (
            float(latest_run.get("average_decision_reward", 0.0)) if latest_run else 0.0
        )
        previous_avg_decision = (
            float(previous_run.get("average_decision_reward", 0.0))
            if previous_run
            else 0.0
        )
        latest_penalized = (
            int(latest_run.get("penalized_decisions", 0)) if latest_run else 0
        )
        previous_penalized = (
            int(previous_run.get("penalized_decisions", 0)) if previous_run else 0
        )
        latest_rewarded = (
            int(latest_run.get("rewarded_decisions", 0)) if latest_run else 0
        )
        previous_rewarded = (
            int(previous_run.get("rewarded_decisions", 0)) if previous_run else 0
        )

        return {
            "run_count": int(self.learning_state.get("run_count", 0)),
            "spec_scope_key": self._spec_scope_key,
            "top_test_type_weights": top_types,
            "top_endpoint_weights": top_endpoints,
            "decision_history_size": len(decision_history),
            "decision_history_tail": history_tail,
            "latest_run_metrics": latest_run,
            "previous_run_metrics": previous_run,
            "improvement_deltas": {
                "run_reward_delta": round(latest_reward - previous_reward, 6),
                "avg_decision_reward_delta": round(
                    latest_avg_decision - previous_avg_decision, 6
                ),
                "penalized_decisions_delta": int(
                    latest_penalized - previous_penalized
                ),
                "rewarded_decisions_delta": int(latest_rewarded - previous_rewarded),
            },
            "policy_feature_dim": self.adaptive_policy.feature_dim,
            "scenario_patterns_tracked": len(scenario_stats),
            "scenario_patterns_tracked_total": (
                len(self.adaptive_policy.scenario_stats)
                if isinstance(self.adaptive_policy.scenario_stats, dict)
                else 0
            ),
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

    def _augment_scenarios_with_rl_mutation(
        self, spec: Dict[str, Any], scenarios: List[TestScenario]
    ) -> tuple[List[TestScenario], Dict[str, Any]]:
        """
        Expand candidate pool with RL-driven mutations before selection.

        This stage targets historically weak/failure-prone patterns and high-uncertainty
        states, then synthesizes bounded scenario variants to reduce static behavior.
        """
        self._operation_index = self._build_operation_index(spec)
        self._spec_paths = {
            str(path)
            for path in (spec.get("paths") or {}).keys()
            if isinstance(path, str)
        }
        self._spec_scope_key = self._compute_spec_scope_key(spec)

        base_count = len(scenarios)
        scenario_stats_raw = self.learning_state.get("scenario_stats", {})
        scenario_stats = self._scenario_stats_for_current_spec(scenario_stats_raw)
        operation_profiles = self._build_operation_failure_profiles(scenario_stats)
        operation_index = dict(self._operation_index)

        ranked_targets: List[Dict[str, Any]] = []
        for scenario in scenarios:
            fingerprint = self._scenario_fingerprint(scenario)
            stats = scenario_stats.get(fingerprint, {})
            attempts = int((stats or {}).get("attempts", 0))
            failure_rate = float((stats or {}).get("failure_rate", 0.0))
            avg_reward = float((stats or {}).get("avg_reward", 0.0))
            rl_risk = float(self._predict_rl_state_risk(scenario))
            operation_key = self._operation_key(scenario.method, scenario.endpoint)
            op_profile = operation_profiles.get(operation_key, {})
            op_attempts = int(op_profile.get("attempts", 0))
            op_failure_rate = float(op_profile.get("failure_rate", 0.0))
            op_avg_reward = float(op_profile.get("avg_reward", 0.0))
            dominant_actual_status_raw = op_profile.get("dominant_actual_status")
            try:
                dominant_actual_status = (
                    int(dominant_actual_status_raw)
                    if dominant_actual_status_raw is not None
                    else None
                )
            except Exception:
                dominant_actual_status = None

            score_parts = self.adaptive_policy.score(
                test_type=scenario.test_type.value,
                method=scenario.method,
                endpoint=scenario.endpoint,
                expected_status=int(scenario.expected_status),
                has_body=bool(scenario.body),
                has_params=bool(scenario.params),
                rl_risk=0.0,
                novelty_bonus=0.0,
                legacy_weight_bonus=0.0,
            )
            uncertainty = float(score_parts.get("uncertainty", 0.0))
            has_direct_signal = (
                failure_rate >= RL_WEAK_FAILURE_RATE_THRESHOLD or avg_reward < 0.0
            )
            has_operation_signal = (
                op_failure_rate >= RL_WEAK_FAILURE_RATE_THRESHOLD or op_avg_reward < 0.0
            )
            has_challenging_signal = bool(has_direct_signal or has_operation_signal)

            if attempts > 0:
                weakness = (
                    0.72 * failure_rate
                    + 0.22 * max(0.0, -avg_reward)
                )
                if has_direct_signal:
                    weakness += 0.06 * min(1.0, attempts / 6.0)
            else:
                # Bootstrap exploration when no historical outcomes exist yet.
                weakness = 0.12 + (0.18 * uncertainty)

            # Inject operation-level memory so new/base fingerprints still benefit from endpoint history.
            if has_operation_signal:
                weakness += (
                    0.18 * op_failure_rate
                    + 0.10 * max(0.0, -op_avg_reward)
                )
            else:
                # Keep mild exploration without letting stable easy-pass patterns dominate.
                weakness += 0.03 * uncertainty
            if (
                dominant_actual_status is not None
                and int(scenario.expected_status) != int(dominant_actual_status)
                and op_attempts >= 2
            ):
                weakness += 0.10
                has_challenging_signal = True

            if has_challenging_signal:
                priority = weakness + rl_risk
            else:
                # Stable patterns get much lower target priority.
                priority = min(0.08, weakness) + (0.20 * uncertainty) + (0.20 * rl_risk)
            ranked_targets.append(
                {
                    "scenario": scenario,
                    "operation_key": operation_key,
                    "fingerprint": fingerprint,
                    "priority": float(priority),
                    "attempts": attempts,
                    "failure_rate": failure_rate,
                    "avg_reward": avg_reward,
                    "uncertainty": uncertainty,
                    "rl_risk": rl_risk,
                    "op_attempts": op_attempts,
                    "op_failure_rate": op_failure_rate,
                    "op_avg_reward": op_avg_reward,
                    "dominant_actual_status": dominant_actual_status,
                    "has_challenging_signal": has_challenging_signal,
                }
            )

        ranked_targets.sort(
            key=lambda item: (
                float(item["priority"]),
                float(item["failure_rate"]),
                float(item["uncertainty"]),
            ),
            reverse=True,
        )

        max_target_count = min(RL_MUTATION_TARGET_LIMIT, len(ranked_targets))
        selected_targets = [
            item
            for item in ranked_targets
            if (
                float(item["priority"]) >= RL_MUTATION_MIN_PRIORITY
                and bool(item.get("has_challenging_signal", False))
            )
        ][:max_target_count]
        if not selected_targets:
            exploratory_pool = [
                item
                for item in ranked_targets
                if float(item["priority"]) >= RL_MUTATION_MIN_PRIORITY
            ]
            fallback_count = max(2, min(max_target_count, len(ranked_targets) // 6 or 1))
            selected_targets = (
                exploratory_pool[:fallback_count]
                if exploratory_pool
                else ranked_targets[:fallback_count]
            )

        existing = list(scenarios)
        existing_fingerprints = {self._scenario_fingerprint(item) for item in existing}
        existing_names = {str(item.name) for item in existing}

        mutated: List[TestScenario] = []
        applied_examples: List[Dict[str, Any]] = []
        for target in selected_targets:
            mutation_budget = self._compute_target_mutation_budget(target)
            op_meta = operation_index.get(str(target["operation_key"]), {})
            variants = self._build_rl_mutations_for_scenario(
                target["scenario"],
                max_variants=mutation_budget,
                operation_meta=op_meta,
                target_context=target,
            )
            for variant in variants:
                if len(mutated) >= RL_MUTATION_MAX_NEW_SCENARIOS:
                    break
                fingerprint = self._scenario_fingerprint(variant)
                if fingerprint in existing_fingerprints:
                    continue

                variant.name = self._dedupe_mutation_name(str(variant.name), existing_names)
                existing_names.add(variant.name)
                existing_fingerprints.add(fingerprint)
                mutated.append(variant)

                if len(applied_examples) < 20:
                    applied_examples.append(
                        {
                            "from": target["scenario"].name,
                            "to": variant.name,
                            "fingerprint": fingerprint,
                            "priority": round(float(target["priority"]), 4),
                            "rl_risk": round(float(target["rl_risk"]), 4),
                            "failure_rate": round(float(target["failure_rate"]), 4),
                            "uncertainty": round(float(target["uncertainty"]), 4),
                            "expected_status": int(variant.expected_status),
                            "test_type": variant.test_type.value,
                            "operation_failure_rate": round(
                                float(target.get("op_failure_rate", 0.0)),
                                4,
                            ),
                            "mutation_budget": int(mutation_budget),
                            "strategy": self._extract_rl_mutation_strategy(variant),
                        }
                    )
            if len(mutated) >= RL_MUTATION_MAX_NEW_SCENARIOS:
                break

        history_seeded, history_seed_examples, history_seed_targets = (
            self._build_history_seed_scenarios(
                spec,
                existing_fingerprints=existing_fingerprints,
                existing_names=existing_names,
                max_new=RL_HISTORY_SEED_MAX_NEW_SCENARIOS,
            )
        )

        final_candidates = existing + mutated + history_seeded
        strategy_breakdown = dict(
            Counter(
                str(item.get("strategy", "unknown"))
                for item in (applied_examples + history_seed_examples)
            )
        )
        summary = {
            "enabled": True,
            "base_candidate_count": int(base_count),
            "targeted_candidates": int(len(selected_targets)),
            "mutated_candidates_added": int(len(mutated) + len(history_seeded)),
            "direct_mutation_candidates_added": int(len(mutated)),
            "history_seed_candidates_added": int(len(history_seeded)),
            "final_candidate_count": int(len(final_candidates)),
            "priority_threshold": float(RL_MUTATION_MIN_PRIORITY),
            "max_new_candidates": int(
                RL_MUTATION_MAX_NEW_SCENARIOS + RL_HISTORY_SEED_MAX_NEW_SCENARIOS
            ),
            "max_variants_per_target": int(RL_MUTATION_MAX_VARIANTS_PER_TARGET),
            "history_seed_priority_threshold": float(RL_HISTORY_SEED_MIN_PRIORITY),
            "operation_profiles_tracked": int(len(operation_profiles)),
            "mutation_strategy_breakdown": strategy_breakdown,
            "top_targets": [
                {
                    "name": item["scenario"].name,
                    "fingerprint": item["fingerprint"],
                    "priority": round(float(item["priority"]), 4),
                    "attempts": int(item["attempts"]),
                    "failure_rate": round(float(item["failure_rate"]), 4),
                    "avg_reward": round(float(item["avg_reward"]), 4),
                    "uncertainty": round(float(item["uncertainty"]), 4),
                    "rl_risk": round(float(item["rl_risk"]), 4),
                    "operation_failure_rate": round(float(item["op_failure_rate"]), 4),
                    "operation_attempts": int(item["op_attempts"]),
                    "has_challenging_signal": bool(item.get("has_challenging_signal", False)),
                }
                for item in selected_targets[:15]
            ],
            "history_seed_targets": history_seed_targets[:15],
            "applied_examples": (applied_examples + history_seed_examples)[:20],
        }
        return final_candidates, summary

    def _build_operation_failure_profiles(
        self, scenario_stats: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        profiles: Dict[str, Dict[str, Any]] = {}
        if not isinstance(scenario_stats, dict):
            return profiles

        for fingerprint, stats_raw in scenario_stats.items():
            if not isinstance(stats_raw, dict):
                continue
            parsed = self._parse_scenario_fingerprint(str(fingerprint))
            if not parsed:
                continue

            op_key = self._operation_key(parsed["method"], parsed["endpoint"])
            attempts = max(0, int(stats_raw.get("attempts", 0)))
            failures = max(0, int(stats_raw.get("failures", 0)))
            if attempts > 0 and failures <= 0:
                failures = int(
                    round(float(stats_raw.get("failure_rate", 0.0)) * float(attempts))
                )
            failures = max(0, min(attempts, failures))
            avg_reward = float(stats_raw.get("avg_reward", 0.0))

            profile = profiles.setdefault(
                op_key,
                {
                    "attempts": 0,
                    "failures": 0,
                    "reward_weighted_sum": 0.0,
                    "actual_status_counts": {},
                },
            )
            profile["attempts"] += attempts
            profile["failures"] += failures
            profile["reward_weighted_sum"] += avg_reward * float(max(1, attempts))

            counts = stats_raw.get("actual_status_counts", {})
            if isinstance(counts, dict):
                bucket = profile.setdefault("actual_status_counts", {})
                for status_raw, count_raw in counts.items():
                    try:
                        status = int(status_raw)
                        count = int(count_raw)
                    except Exception:
                        continue
                    if status < 100 or status > 599 or count <= 0:
                        continue
                    key = str(status)
                    bucket[key] = int(bucket.get(key, 0)) + count

        for op_key, profile in profiles.items():
            attempts = max(0, int(profile.get("attempts", 0)))
            failures = max(0, int(profile.get("failures", 0)))
            weighted_sum = float(profile.get("reward_weighted_sum", 0.0))
            profile["failure_rate"] = (
                float(failures) / float(attempts) if attempts > 0 else 0.0
            )
            profile["avg_reward"] = (
                weighted_sum / float(max(1, attempts)) if attempts > 0 else 0.0
            )
            dominant_status, _ = self._dominant_actual_status(
                profile.get("actual_status_counts", {})
            )
            profile["dominant_actual_status"] = dominant_status
            profile.pop("reward_weighted_sum", None)
            profiles[op_key] = profile

        return profiles

    def _compute_target_mutation_budget(self, target: Dict[str, Any]) -> int:
        budget = int(RL_MUTATION_PER_TARGET_LIMIT)
        if float(target.get("failure_rate", 0.0)) >= 0.5:
            budget += 1
        if float(target.get("op_failure_rate", 0.0)) >= 0.5:
            budget += 1
        if float(target.get("uncertainty", 0.0)) >= 0.2:
            budget += 1
        if float(target.get("priority", 0.0)) >= 0.9:
            budget += 1
        return max(1, min(RL_MUTATION_MAX_VARIANTS_PER_TARGET, int(budget)))

    def _extract_rl_mutation_strategy(self, scenario: TestScenario) -> str:
        saw_history_seed = False
        for item in list(scenario.assertions or []):
            value = str(item)
            if value.startswith("rl_mutation:"):
                strategy = value.split(":", 1)[1].strip()
                if strategy:
                    return strategy
            if value == "rl_history_seed":
                saw_history_seed = True
        return "history_seed" if saw_history_seed else "unknown"

    def _is_history_seeded_scenario(self, scenario: TestScenario) -> bool:
        for item in list(scenario.assertions or []):
            if str(item).strip() == "rl_history_seed":
                return True
        return False

    def _humanize_mutation_strategy(self, strategy: str) -> str:
        normalized = re.sub(r"[^a-z0-9_]+", "_", str(strategy or "").strip().lower())
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        if not normalized:
            return "scenario check"
        if normalized == "missing_auth":
            return "missing auth token"
        if normalized == "invalid_auth":
            return "invalid auth token"
        if normalized == "query_fuzz":
            return "unexpected query parameter payload"
        if normalized == "path_not_found":
            return "path not found"
        if normalized == "adaptive_status_hypothesis":
            return "nonexistent resource id check"
        if normalized.startswith("missing_required_"):
            field_name = normalized[len("missing_required_") :].replace("_", " ").strip()
            return f"missing required field: {field_name or 'unknown'}"
        if normalized.startswith("learned_missing_required_"):
            field_name = normalized[len("learned_missing_required_") :].replace("_", " ").strip()
            return f"missing required field: {field_name or 'unknown'}"
        if normalized.startswith("learned_missing_"):
            field_name = normalized[len("learned_missing_") :].replace("_", " ").strip()
            return f"missing field: {field_name or 'unknown'}"
        if normalized.startswith("missing_"):
            field_name = normalized[len("missing_") :].replace("_", " ").strip()
            return f"missing field: {field_name or 'unknown'}"
        if normalized.startswith("learned_below_min_"):
            field_name = normalized[len("learned_below_min_") :].replace("_", " ").strip()
            return f"{field_name or 'value'} below minimum"
        if normalized.startswith("below_min_"):
            field_name = normalized[len("below_min_") :].replace("_", " ").strip()
            return f"{field_name or 'value'} below minimum"
        return normalized.replace("_", " ")

    def _scenario_display_name(self, scenario: TestScenario) -> str:
        method = str(getattr(scenario, "method", "") or "").upper().strip()
        endpoint = str(getattr(scenario, "endpoint", "") or "").strip()
        description = str(getattr(scenario, "description", "") or "").strip()
        if description:
            description = re.sub(r"\s*\[RL mutation:[^\]]+\]\s*$", "", description, flags=re.IGNORECASE).strip()
        strategy = self._extract_rl_mutation_strategy(scenario)
        history_seeded = self._is_history_seeded_scenario(scenario)

        if strategy not in {"unknown", "history_seed"}:
            intent = self._humanize_mutation_strategy(strategy)
        elif description:
            intent = description
        elif history_seeded:
            intent = "history seeded regression probe"
        else:
            intent = str(getattr(getattr(scenario, "test_type", None), "value", getattr(scenario, "test_type", "scenario"))).replace("_", " ")

        intent = " ".join(intent.split()).strip() or "scenario check"
        if intent and intent[0].islower():
            intent = intent[0].upper() + intent[1:]

        prefix = " ".join(part for part in [method, endpoint] if part).strip()
        if prefix:
            return f"{prefix} - {intent}"
        return intent

    def _build_history_seed_scenarios(
        self,
        spec: Dict[str, Any],
        *,
        existing_fingerprints: set[str],
        existing_names: set[str],
        max_new: int = RL_HISTORY_SEED_MAX_NEW_SCENARIOS,
    ) -> tuple[List[TestScenario], List[Dict[str, Any]], List[Dict[str, Any]]]:
        if max_new <= 0:
            return [], [], []

        scenario_stats_raw = self.learning_state.get("scenario_stats", {})
        scenario_stats = self._scenario_stats_for_current_spec(scenario_stats_raw)
        if not scenario_stats:
            return [], [], []

        operation_index = self._build_operation_index(spec)
        ranked_targets: List[Dict[str, Any]] = []
        for fingerprint, stats_raw in scenario_stats.items():
            if not isinstance(stats_raw, dict):
                continue
            parsed = self._parse_scenario_fingerprint(str(fingerprint))
            if not parsed:
                continue
            attempts = max(
                int(stats_raw.get("attempts", 0)),
                int(parsed.get("attempts_hint", 0)),
            )
            if attempts < RL_HISTORY_SEED_MIN_ATTEMPTS:
                continue

            failure_rate = float(stats_raw.get("failure_rate", 0.0))
            avg_reward = float(stats_raw.get("avg_reward", 0.0))
            priority = (
                0.70 * failure_rate
                + 0.20 * max(0.0, -avg_reward)
                + 0.10 * min(1.0, attempts / 5.0)
            )
            if priority < RL_HISTORY_SEED_MIN_PRIORITY:
                continue

            op_key = self._operation_key(parsed["method"], parsed["endpoint"])
            op_meta = operation_index.get(op_key)
            if not isinstance(op_meta, dict):
                continue

            ranked_targets.append(
                {
                    "fingerprint": str(fingerprint),
                    "operation_key": op_key,
                    "stats": stats_raw,
                    "attempts": attempts,
                    "failure_rate": failure_rate,
                    "avg_reward": avg_reward,
                    "priority": float(priority),
                    "method": parsed["method"],
                    "endpoint": parsed["endpoint"],
                    "test_type": parsed["test_type"],
                    "expected_status": self._normalize_expected_status_for_test_type(
                        test_type=str(parsed["test_type"]),
                        expected_status=int(parsed["expected_status"]),
                    ),
                    "has_body": bool(parsed["has_body"]),
                    "has_params": bool(parsed["has_params"]),
                    "op_meta": op_meta,
                }
            )

        ranked_targets.sort(
            key=lambda item: (
                float(item["priority"]),
                float(item["failure_rate"]),
                int(item["attempts"]),
            ),
            reverse=True,
        )
        if not ranked_targets:
            return [], [], []

        selected_targets = ranked_targets[:RL_HISTORY_SEED_TARGET_LIMIT]
        generated: List[TestScenario] = []
        applied_examples: List[Dict[str, Any]] = []
        target_summary: List[Dict[str, Any]] = []
        for target in selected_targets:
            if len(generated) >= max_new:
                break

            if len(target_summary) < 30:
                target_summary.append(
                    {
                        "fingerprint": target["fingerprint"],
                        "priority": round(float(target["priority"]), 4),
                        "attempts": int(target["attempts"]),
                        "failure_rate": round(float(target["failure_rate"]), 4),
                        "avg_reward": round(float(target["avg_reward"]), 4),
                    }
                )

            base = self._build_history_seed_scenario(target)
            candidate_variants: List[TestScenario] = [base]

            if float(target["failure_rate"]) > 0.0:
                candidate_variants.extend(
                    self._build_rl_mutations_for_scenario(
                        base,
                        max_variants=1,
                        operation_meta=target.get("op_meta", {}),
                        target_context=target,
                    )
                )

            dominant_status = self._dominant_actual_status_from_stats(target["stats"])
            normalized_dominant_status = (
                self._normalize_expected_status_for_test_type(
                    test_type=base.test_type.value,
                    expected_status=int(dominant_status),
                )
                if dominant_status is not None
                else None
            )
            if (
                normalized_dominant_status is not None
                and self._allow_adaptive_status_hypothesis(
                    base,
                    int(normalized_dominant_status),
                )
            ):
                candidate_variants.append(
                    self._make_mutation_scenario(
                        base,
                        suffix=f"rl_adaptive_expect_{int(normalized_dominant_status)}",
                        strategy="adaptive_status_hypothesis",
                        expected_status=int(normalized_dominant_status),
                    )
                )

            for variant in candidate_variants:
                if len(generated) >= max_new:
                    break
                fingerprint = self._scenario_fingerprint(variant)
                if fingerprint in existing_fingerprints:
                    continue

                variant.name = self._dedupe_mutation_name(str(variant.name), existing_names)
                existing_names.add(variant.name)
                existing_fingerprints.add(fingerprint)
                generated.append(variant)

                if len(applied_examples) < 20:
                    applied_examples.append(
                        {
                            "from_fingerprint": target["fingerprint"],
                            "to": variant.name,
                            "fingerprint": fingerprint,
                            "priority": round(float(target["priority"]), 4),
                            "failure_rate": round(float(target["failure_rate"]), 4),
                            "expected_status": int(variant.expected_status),
                            "test_type": variant.test_type.value,
                            "source": "history_seed",
                            "strategy": self._extract_rl_mutation_strategy(variant),
                        }
                    )

        return generated, applied_examples, target_summary

    def _build_history_seed_scenario(self, target: Dict[str, Any]) -> TestScenario:
        endpoint = str(target.get("endpoint", "/"))
        method = str(target.get("method", "GET")).upper()
        test_type = self._coerce_test_type(str(target.get("test_type", "error_handling")))
        expected_status = self._normalize_expected_status_for_test_type(
            test_type=test_type.value,
            expected_status=int(target.get("expected_status", 400)),
        )
        op_meta = target.get("op_meta", {})
        if not isinstance(op_meta, dict):
            op_meta = {}

        response_statuses = op_meta.get("response_statuses", [])
        is_auth_seed = test_type in {TestType.AUTHENTICATION, TestType.AUTHORIZATION}
        if isinstance(response_statuses, list) and response_statuses:
            if is_auth_seed and int(expected_status) in AUTH_NEGATIVE_STATUS_CODES:
                # Preserve auth-negative intent even when specs omit explicit 401/403
                # so history-seeded auth checks don't drift to 2xx.
                expected_status = 401 if int(expected_status) == 401 else 403
            elif expected_status not in response_statuses:
                preferred = [s for s in response_statuses if int(s) in NEGATIVE_STATUS_CODES]
                if preferred and expected_status in NEGATIVE_STATUS_CODES:
                    expected_status = int(preferred[0])
                else:
                    expected_status = int(response_statuses[0])

        expected_status = self._normalize_expected_status_for_test_type(
            test_type=test_type.value,
            expected_status=int(expected_status),
        )

        # Do not keep auth-negative expectations on non-auth history-seeded probes.
        # Those variants frequently carry boundary/input intent and should target
        # validation/not-found behavior instead of token semantics.
        if (
            test_type in {TestType.BOUNDARY_TESTING, TestType.INPUT_VALIDATION}
            and int(expected_status) in AUTH_NEGATIVE_STATUS_CODES
        ):
            fallback_status = 404
            if isinstance(response_statuses, list):
                non_auth_negative = [
                    int(status)
                    for status in response_statuses
                    if int(status) in NEGATIVE_STATUS_CODES
                    and int(status) not in AUTH_NEGATIVE_STATUS_CODES
                ]
                if non_auth_negative:
                    fallback_status = int(non_auth_negative[0])
                elif 404 not in response_statuses:
                    fallback_status = 400
            expected_status = self._normalize_expected_status_for_test_type(
                test_type=test_type.value,
                expected_status=int(fallback_status),
            )

        has_body = bool(target.get("has_body", False))
        has_params = bool(target.get("has_params", False))
        params: Dict[str, Any] = {}
        for path_name in op_meta.get("path_param_names", []) or []:
            params[str(path_name)] = "999999" if expected_status == 404 else "123"

        query_names = op_meta.get("query_param_names", []) or []
        if has_params:
            if query_names:
                for query_name in query_names[:2]:
                    params[str(query_name)] = self._sample_query_value(
                        str(query_name),
                        expected_status,
                    )
            elif not params:
                params["q"] = self._sample_query_value("q", expected_status)

        request_schema = op_meta.get("request_schema", {})
        if (
            method in {"POST", "PUT", "PATCH"}
            and not has_body
            and 200 <= int(expected_status) < 300
            and isinstance(request_schema, dict)
            and bool(request_schema)
        ):
            has_body = True

        body: Optional[Dict[str, Any]] = None
        if method in {"POST", "PUT", "PATCH"} and has_body:
            body = self._build_seed_body_from_operation_meta(
                op_meta,
                expect_negative=expected_status in NEGATIVE_STATUS_CODES,
            )

        operation_key = self._operation_key(method, endpoint)
        headers: Dict[str, str] = {}
        if operation_key in self._auth_required_ops:
            if expected_status in AUTH_NEGATIVE_STATUS_CODES or test_type in {
                TestType.AUTHENTICATION,
                TestType.AUTHORIZATION,
            }:
                if expected_status == 403:
                    headers["Authorization"] = "Bearer invalid"
            else:
                headers["Authorization"] = "Bearer valid_token_123"

        slug = re.sub(r"[^a-zA-Z0-9]+", "_", endpoint).strip("_") or "root"
        name = (
            f"test_{method.lower()}_{slug}_rl_history_seed_"
            + f"{test_type.value}_{int(expected_status)}"
        )
        attempts = int(target.get("attempts", 0))
        failure_rate = float(target.get("failure_rate", 0.0))
        description = (
            f"Learning-seeded scenario from historical outcomes "
            + f"(attempts={attempts}, failure_rate={failure_rate:.2f})"
        )
        assertions = [
            "rl_history_seed",
            f"history_attempts:{attempts}",
            f"history_failure_rate:{round(failure_rate, 4)}",
        ]

        return TestScenario(
            name=name,
            description=description,
            test_type=test_type,
            endpoint=endpoint,
            method=method,
            headers=headers,
            params=params,
            body=body,
            expected_status=int(expected_status),
            expected_response_fields=[],
            assertions=assertions,
        )

    def _build_seed_body_from_operation_meta(
        self,
        op_meta: Dict[str, Any],
        *,
        expect_negative: bool,
    ) -> Dict[str, Any]:
        request_schema = op_meta.get("request_schema", {})
        required_fields = op_meta.get("required_fields", [])
        if not isinstance(request_schema, dict):
            request_schema = {}
        if not isinstance(required_fields, list):
            required_fields = []

        properties = request_schema.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}

        if expect_negative:
            if required_fields:
                # Purposefully omit required fields to trigger validation failures.
                return {}
            if properties:
                first_key = str(next(iter(properties.keys())))
                sample = self._sample_value_for_schema(first_key, properties.get(first_key, {}))
                if isinstance(sample, str):
                    bad_value = 12345
                elif isinstance(sample, (int, float, bool)):
                    bad_value = "invalid_type"
                else:
                    bad_value = "invalid_payload"
                return {first_key: bad_value}
            return {"invalid": "payload"}

        body: Dict[str, Any] = {}
        fields = [str(item) for item in required_fields] if required_fields else [
            str(key) for key in list(properties.keys())[:2]
        ]
        for key in fields:
            body[key] = self._sample_value_for_schema(key, properties.get(key, {}))
        if body:
            return body
        return {"name": "sample_value"}

    def _sample_query_value(self, query_name: str, expected_status: int) -> Any:
        qname = str(query_name).lower()
        if expected_status in NEGATIVE_STATUS_CODES:
            if "page" in qname or "offset" in qname:
                return -1
            if "limit" in qname or "size" in qname:
                return 999999
            return "'; DROP TABLE users; --"
        if "page" in qname:
            return 1
        if "limit" in qname or "size" in qname:
            return 10
        return "sample"

    def _coerce_test_type(self, raw_test_type: str) -> TestType:
        value = str(raw_test_type).strip().lower()
        aliases = {
            "auth": TestType.AUTHENTICATION,
            "authn": TestType.AUTHENTICATION,
            "authz": TestType.AUTHORIZATION,
            "validation": TestType.INPUT_VALIDATION,
            "boundary": TestType.BOUNDARY_TESTING,
            "error": TestType.ERROR_HANDLING,
        }
        if value in aliases:
            return aliases[value]
        for enum_item in TestType:
            if enum_item.value == value:
                return enum_item
        return TestType.ERROR_HANDLING

    def _parse_scenario_fingerprint(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        parts = str(fingerprint).split("|")
        if len(parts) < 6:
            return None
        method, endpoint, test_type, expected_raw, body_raw, params_raw = parts[:6]
        try:
            expected_status = int(expected_raw)
        except Exception:
            return None
        has_body = str(body_raw).split("=", 1)[-1].strip() in {"1", "true", "True"}
        has_params = str(params_raw).split("=", 1)[-1].strip() in {"1", "true", "True"}
        return {
            "method": str(method).upper(),
            "endpoint": str(endpoint),
            "test_type": str(test_type),
            "expected_status": expected_status,
            "has_body": has_body,
            "has_params": has_params,
            "attempts_hint": 0,
        }

    def _dominant_actual_status_from_stats(self, stats: Dict[str, Any]) -> Optional[int]:
        counts = stats.get("actual_status_counts", {})
        if not isinstance(counts, dict) or not counts:
            return None
        best_status: Optional[int] = None
        best_count = -1
        for key, value in counts.items():
            try:
                status = int(key)
                count = int(value)
            except Exception:
                continue
            if count > best_count:
                best_status = status
                best_count = count
        return best_status

    def _allow_adaptive_status_hypothesis(
        self, scenario: TestScenario, candidate_status: int
    ) -> bool:
        try:
            base_status = int(getattr(scenario, "expected_status", 0) or 0)
            target_status = int(candidate_status)
        except Exception:
            return False

        if target_status < 100 or target_status > 599:
            return False
        if target_status == base_status:
            return False
        if self._is_auth_negative_scenario(scenario):
            return target_status in AUTH_NEGATIVE_STATUS_CODES
        if base_status == 405:
            return target_status == 405

        base_is_negative = base_status in NEGATIVE_STATUS_CODES
        target_is_negative = target_status in NEGATIVE_STATUS_CODES
        # Keep adaptive hypotheses within the same polarity to avoid
        # contradictory variants (e.g. happy-path 201 -> adaptive 400).
        if base_is_negative != target_is_negative:
            return False

        return True

    def _build_rl_mutations_for_scenario(
        self,
        scenario: TestScenario,
        *,
        max_variants: int = RL_MUTATION_PER_TARGET_LIMIT,
        operation_meta: Optional[Dict[str, Any]] = None,
        target_context: Optional[Dict[str, Any]] = None,
    ) -> List[TestScenario]:
        op_meta = operation_meta if isinstance(operation_meta, dict) else {}
        ctx = target_context if isinstance(target_context, dict) else {}

        weighted_variants: List[tuple[float, TestScenario]] = []
        method = str(scenario.method).upper()
        endpoint = str(scenario.endpoint)
        placeholders = PATH_PARAM_PATTERN.findall(endpoint)
        path_param_schemas = (
            op_meta.get("path_param_schemas", {})
            if isinstance(op_meta.get("path_param_schemas", {}), dict)
            else {}
        )
        query_names = [
            str(item)
            for item in (op_meta.get("query_param_names", []) or [])
            if str(item).strip()
        ]
        required_fields = [
            str(item)
            for item in (op_meta.get("required_fields", []) or [])
            if str(item).strip()
        ]
        path_safe_params = self._coerce_valid_path_params(
            endpoint=endpoint,
            params=dict(scenario.params or {}),
            op_meta=op_meta,
        )
        op_failure_rate = float(ctx.get("op_failure_rate", 0.0))
        failure_rate = float(ctx.get("failure_rate", 0.0))
        uncertainty = float(ctx.get("uncertainty", 0.0))
        op_attempts = int(ctx.get("op_attempts", 0))
        dominant_status_raw = ctx.get("dominant_actual_status")
        try:
            dominant_status = int(dominant_status_raw) if dominant_status_raw is not None else None
        except Exception:
            dominant_status = None
        risk_boost = min(0.40, (0.30 * failure_rate) + (0.20 * op_failure_rate) + (0.25 * uncertainty))
        base_expected_status = int(getattr(scenario, "expected_status", 0))
        base_is_auth_negative = self._is_auth_negative_scenario(scenario)
        base_is_method_not_allowed = base_expected_status == 405

        def enqueue(weight: float, variant: TestScenario) -> None:
            weighted_variants.append((float(weight), variant))

        if (
            placeholders
            and method in {"GET", "DELETE", "PUT", "PATCH"}
            and not base_is_auth_negative
            and not base_is_method_not_allowed
        ):
            params = dict(scenario.params or {})
            for name in placeholders:
                params[name] = "999999"
            enqueue(
                0.95 + risk_boost,
                self._make_mutation_scenario(
                    scenario,
                    suffix="rl_path_not_found",
                    strategy="path_not_found",
                    test_type=TestType.ERROR_HANDLING,
                    params=params,
                    expected_status=404,
                )
            )
            # Only emit path-type fuzzing when OpenAPI path param types support
            # a meaningful type violation. String path params often accept any
            # segment and lead to noisy corrected expectations.
            type_fuzz_params = dict(scenario.params or {})
            can_path_type_fuzz = False
            for name in placeholders:
                schema = (
                    path_param_schemas.get(name, {})
                    if isinstance(path_param_schemas.get(name, {}), dict)
                    else {}
                )
                field_type = str(schema.get("type", "")).strip().lower()
                if field_type in {"integer", "number"}:
                    type_fuzz_params[name] = "not-a-number"
                    can_path_type_fuzz = True
                elif field_type == "boolean":
                    type_fuzz_params[name] = "not-a-bool"
                    can_path_type_fuzz = True
            if can_path_type_fuzz:
                enqueue(
                    0.86 + risk_boost,
                    self._make_mutation_scenario(
                        scenario,
                        suffix="rl_path_type_fuzz",
                        strategy="path_type_fuzz",
                        test_type=TestType.BOUNDARY_TESTING,
                        params=type_fuzz_params,
                        expected_status=400,
                    ),
                )

        if method in {"POST", "PUT", "PATCH"} and not base_is_auth_negative:
            empty_body_params = dict(path_safe_params)
            enqueue(
                0.90 + risk_boost,
                self._make_mutation_scenario(
                    scenario,
                    suffix="rl_empty_body",
                    strategy="empty_body",
                    test_type=TestType.INPUT_VALIDATION,
                    params=empty_body_params,
                    body={},
                    expected_status=400,
                )
            )
            type_fuzz_params = dict(path_safe_params)
            enqueue(
                0.88 + risk_boost,
                self._make_mutation_scenario(
                    scenario,
                    suffix="rl_type_fuzz",
                    strategy="type_fuzz",
                    test_type=TestType.INPUT_VALIDATION,
                    params=type_fuzz_params,
                    body=self._build_type_fuzz_body(scenario),
                    expected_status=400,
                )
            )
            if required_fields:
                body_seed = (
                    deepcopy(scenario.body) if isinstance(scenario.body, dict) else {}
                )
                if not body_seed:
                    body_seed = self._build_seed_body_from_operation_meta(
                        op_meta,
                        expect_negative=False,
                    )
                missing_field = required_fields[0]
                body_seed.pop(str(missing_field), None)
                req_params = dict(path_safe_params)
                field_slug = re.sub(r"[^a-z0-9]+", "_", str(missing_field).lower()).strip("_")
                if not field_slug:
                    field_slug = "field"
                enqueue(
                    1.02 + risk_boost,
                    self._make_mutation_scenario(
                        scenario,
                        suffix=f"rl_missing_required_{field_slug}",
                        strategy="missing_required_field",
                        test_type=TestType.ERROR_HANDLING,
                        params=req_params,
                        body=body_seed,
                        expected_status=400,
                    ),
                )

        if method == "GET" and not base_is_auth_negative and not base_is_method_not_allowed:
            params = dict(path_safe_params)
            query_key_candidates = [
                str(name)
                for name in query_names
                if str(name) in params
            ]
            if query_key_candidates:
                key = sorted(query_key_candidates)[0]
                params[key] = self._fuzz_query_value(params.get(key))
            else:
                params["q"] = "'; DROP TABLE users; --"
            enqueue(
                0.84 + risk_boost,
                self._make_mutation_scenario(
                    scenario,
                    suffix="rl_query_fuzz",
                    strategy="query_fuzz",
                    test_type=TestType.SECURITY,
                    params=params,
                    expected_status=400,
                )
            )
            boundary_terms = ("limit", "page", "size", "offset")
            boundary_fields = [name for name in query_names if any(term in name.lower() for term in boundary_terms)]
            if boundary_fields:
                boundary_params = dict(scenario.params or {})
                for name in boundary_fields[:2]:
                    boundary_params[name] = self._sample_query_value(name, 400)
                enqueue(
                    0.98 + risk_boost,
                    self._make_mutation_scenario(
                        scenario,
                        suffix="rl_query_boundary_extreme",
                        strategy="query_boundary_extreme",
                        test_type=TestType.BOUNDARY_TESTING,
                        params=boundary_params,
                        expected_status=400,
                    ),
                )

        operation_key = self._operation_key(method, endpoint)
        if operation_key in self._auth_required_ops:
            if base_is_auth_negative:
                if self._has_authorization_header(scenario.headers or {}):
                    missing_auth_headers = dict(scenario.headers or {})
                    missing_auth_headers.pop("Authorization", None)
                    missing_auth_headers.pop("authorization", None)
                    missing_auth_params = dict(path_safe_params)
                    enqueue(
                        0.93 + risk_boost,
                        self._make_mutation_scenario(
                            scenario,
                            suffix="rl_missing_auth",
                            strategy="missing_auth",
                            test_type=TestType.AUTHENTICATION,
                            headers=missing_auth_headers,
                            params=missing_auth_params,
                            expected_status=401,
                        ),
                    )
                else:
                    invalid_headers = dict(scenario.headers or {})
                    invalid_headers["Authorization"] = "Bearer invalid"
                    invalid_headers.pop("authorization", None)
                    invalid_params = dict(path_safe_params)
                    enqueue(
                        0.91 + risk_boost,
                        self._make_mutation_scenario(
                            scenario,
                            suffix="rl_invalid_auth",
                            strategy="invalid_auth",
                            test_type=TestType.AUTHENTICATION,
                            headers=invalid_headers,
                            params=invalid_params,
                            expected_status=401,
                        ),
                    )
            else:
                headers = dict(scenario.headers or {})
                headers["Authorization"] = "Bearer invalid"
                headers.pop("authorization", None)
                auth_params = dict(path_safe_params)
                enqueue(
                    0.87 + risk_boost,
                    self._make_mutation_scenario(
                        scenario,
                        suffix="rl_invalid_auth",
                        strategy="invalid_auth",
                        test_type=TestType.AUTHENTICATION,
                        headers=headers,
                        params=auth_params,
                        expected_status=401,
                    )
                )
                if self._has_authorization_header(scenario.headers or {}):
                    missing_auth_headers = dict(scenario.headers or {})
                    missing_auth_headers.pop("Authorization", None)
                    missing_auth_headers.pop("authorization", None)
                    missing_auth_params = dict(path_safe_params)
                    enqueue(
                        0.92 + risk_boost,
                        self._make_mutation_scenario(
                            scenario,
                            suffix="rl_missing_auth",
                            strategy="missing_auth",
                            test_type=TestType.AUTHENTICATION,
                            headers=missing_auth_headers,
                            params=missing_auth_params,
                            expected_status=401,
                        ),
                    )

        if (
            dominant_status is not None
            and 100 <= int(dominant_status) <= 599
            and int(dominant_status) != int(scenario.expected_status)
            and op_attempts >= 3
        ):
            if (
                int(dominant_status) in AUTH_NEGATIVE_STATUS_CODES
                and not base_is_auth_negative
            ):
                dominant_status = None
            elif (
                base_is_auth_negative
                and int(dominant_status) not in AUTH_NEGATIVE_STATUS_CODES
            ):
                dominant_status = None
            elif base_is_method_not_allowed and int(dominant_status) != 405:
                dominant_status = None

        normalized_adaptive_status = (
            self._normalize_expected_status_for_test_type(
                test_type=scenario.test_type.value,
                expected_status=int(dominant_status),
            )
            if dominant_status is not None
            else None
        )
        if (
            normalized_adaptive_status is not None
            and self._allow_adaptive_status_hypothesis(
                scenario,
                int(normalized_adaptive_status),
            )
        ):
            adaptive_params = dict(scenario.params or {})
            enqueue(
                0.89 + risk_boost,
                self._make_mutation_scenario(
                    scenario,
                    suffix=f"rl_adaptive_expect_{int(normalized_adaptive_status)}",
                    strategy="adaptive_status_hypothesis",
                    params=adaptive_params,
                    expected_status=int(normalized_adaptive_status),
                ),
            )

        learned_proposals = self._propose_learned_schema_mutations(
            scenario=scenario,
            operation_meta=op_meta,
            target_context=ctx,
        )
        for proposal in learned_proposals:
            enqueue(
                float(proposal.get("weight", 0.75)) + risk_boost,
                self._make_mutation_scenario(
                    scenario,
                    suffix=str(proposal.get("suffix", "rl_learned_mutation")),
                    strategy=str(proposal.get("strategy", "learned_schema_mutation")),
                    test_type=self._coerce_test_type(
                        str(proposal.get("test_type", scenario.test_type.value))
                    ),
                    headers=proposal.get("headers")
                    if isinstance(proposal.get("headers"), dict)
                    else None,
                    params=proposal.get("params")
                    if isinstance(proposal.get("params"), dict)
                    else None,
                    body=proposal.get("body")
                    if isinstance(proposal.get("body"), dict)
                    else None,
                    expected_status=int(
                        proposal.get("expected_status", scenario.expected_status)
                    ),
                ),
            )

        if not weighted_variants:
            return []

        weighted_variants.sort(
            key=lambda item: (
                float(item[0]),
                str(item[1].name),
            ),
            reverse=True,
        )
        selected: List[TestScenario] = []
        seen: set[str] = set()
        budget = max(1, int(max_variants))
        for _, variant in weighted_variants:
            fingerprint = self._scenario_fingerprint(variant)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            selected.append(variant)
            if len(selected) >= budget:
                break
        return selected

    def _propose_learned_schema_mutations(
        self,
        *,
        scenario: TestScenario,
        operation_meta: Dict[str, Any],
        target_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        request_schema = operation_meta.get("request_schema", {})
        properties = (
            request_schema.get("properties", {})
            if isinstance(request_schema, dict) and isinstance(request_schema.get("properties"), dict)
            else {}
        )
        required_fields = [
            str(item) for item in (operation_meta.get("required_fields", []) or []) if str(item).strip()
        ]
        if not properties and not required_fields:
            return []

        base_params = self._coerce_valid_path_params(
            endpoint=str(scenario.endpoint),
            params=dict(scenario.params or {}),
            op_meta=operation_meta if isinstance(operation_meta, dict) else {},
        )
        proposals: List[Dict[str, Any]] = []
        failure_rate = float(target_context.get("failure_rate", 0.0))
        uncertainty = float(target_context.get("uncertainty", 0.0))
        context_boost = min(0.25, 0.15 * failure_rate + 0.10 * uncertainty)

        def add_proposal(
            *,
            suffix: str,
            strategy: str,
            test_type: str,
            expected_status: int,
            weight: float,
            body: Optional[Dict[str, Any]] = None,
            params: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, str]] = None,
        ) -> None:
            payload: Dict[str, Any] = {
                "suffix": suffix,
                "strategy": strategy,
                "test_type": test_type,
                "expected_status": int(expected_status),
                "weight": float(weight),
            }
            if isinstance(body, dict):
                payload["body"] = deepcopy(body)
            if isinstance(params, dict):
                payload["params"] = deepcopy(params)
            if isinstance(headers, dict):
                payload["headers"] = deepcopy(headers)
            proposals.append(payload)

        if required_fields:
            body = deepcopy(scenario.body) if isinstance(scenario.body, dict) else {}
            missing = required_fields[0]
            body.pop(missing, None)
            params = dict(base_params)
            add_proposal(
                suffix=f"rl_learned_missing_{re.sub(r'[^a-z0-9]+', '_', missing.lower()).strip('_') or 'field'}",
                strategy="learned_missing_required_field",
                test_type="input_validation",
                body=body,
                params=params,
                expected_status=400,
                weight=0.88 + context_boost,
            )

        for field, field_schema in list(properties.items())[:4]:
            schema = field_schema if isinstance(field_schema, dict) else {}
            field_type = str(schema.get("type", "string")).lower()
            params = dict(base_params)
            body = deepcopy(scenario.body) if isinstance(scenario.body, dict) else {}
            if not body:
                body = self._build_seed_body_from_operation_meta(
                    operation_meta,
                    expect_negative=False,
                )

            if "enum" in schema and isinstance(schema.get("enum"), list) and schema.get("enum"):
                body[str(field)] = "__invalid_enum_value__"
                add_proposal(
                    suffix=f"rl_learned_enum_{re.sub(r'[^a-z0-9]+', '_', str(field).lower()).strip('_') or 'field'}",
                    strategy="learned_enum_violation",
                    test_type="input_validation",
                    body=body,
                    params=params,
                    expected_status=400,
                    weight=0.84 + context_boost,
                )

            if field_type == "string":
                max_length = schema.get("maxLength")
                min_length = schema.get("minLength")
                if isinstance(max_length, int) and max_length > 0:
                    body[str(field)] = "x" * (max_length + 8)
                    add_proposal(
                        suffix=f"rl_learned_maxlen_{re.sub(r'[^a-z0-9]+', '_', str(field).lower()).strip('_') or 'field'}",
                        strategy="learned_string_length_overflow",
                        test_type="boundary_testing",
                        body=body,
                        params=params,
                        expected_status=400,
                        weight=0.82 + context_boost,
                    )
                if isinstance(min_length, int) and min_length > 0:
                    body[str(field)] = ""
                    add_proposal(
                        suffix=f"rl_learned_minlen_{re.sub(r'[^a-z0-9]+', '_', str(field).lower()).strip('_') or 'field'}",
                        strategy="learned_string_length_underflow",
                        test_type="boundary_testing",
                        body=body,
                        params=params,
                        expected_status=400,
                        weight=0.80 + context_boost,
                    )

            if field_type in {"integer", "number"}:
                minimum = schema.get("minimum")
                maximum = schema.get("maximum")
                if isinstance(minimum, (int, float)):
                    body[str(field)] = float(minimum) - 1.0
                    add_proposal(
                        suffix=f"rl_learned_below_min_{re.sub(r'[^a-z0-9]+', '_', str(field).lower()).strip('_') or 'field'}",
                        strategy="learned_numeric_underflow",
                        test_type="boundary_testing",
                        body=body,
                        params=params,
                        expected_status=400,
                        weight=0.81 + context_boost,
                    )
                if isinstance(maximum, (int, float)):
                    body[str(field)] = float(maximum) + 1.0
                    add_proposal(
                        suffix=f"rl_learned_above_max_{re.sub(r'[^a-z0-9]+', '_', str(field).lower()).strip('_') or 'field'}",
                        strategy="learned_numeric_overflow",
                        test_type="boundary_testing",
                        body=body,
                        params=params,
                        expected_status=400,
                        weight=0.83 + context_boost,
                    )

            if field_type in {"integer", "number", "string", "boolean"} and str(field) in required_fields:
                body[str(field)] = None
                add_proposal(
                    suffix=f"rl_learned_null_{re.sub(r'[^a-z0-9]+', '_', str(field).lower()).strip('_') or 'field'}",
                    strategy="learned_required_null",
                    test_type="input_validation",
                    body=body,
                    params=params,
                    expected_status=400,
                    weight=0.79 + context_boost,
                )

        path_params = [str(p) for p in (operation_meta.get("path_param_names", []) or []) if str(p).strip()]
        if path_params and scenario.method.upper() in {"GET", "PATCH", "PUT", "DELETE"}:
            params = dict(base_params)
            for pname in path_params:
                params[pname] = "nonexistent_dependency_id"
            add_proposal(
                suffix="rl_learned_dependency_order_unmet",
                strategy="learned_dependency_order_unmet",
                test_type="error_handling",
                params=params,
                expected_status=404,
                weight=0.86 + context_boost,
            )

        dedup: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for item in proposals:
            key = (str(item.get("strategy", "")), str(item.get("suffix", "")))
            if key not in dedup:
                dedup[key] = item
        output = list(dedup.values())
        output.sort(key=lambda item: float(item.get("weight", 0.0)), reverse=True)
        return output[:6]

    def _coerce_valid_path_params(
        self,
        *,
        endpoint: str,
        params: Dict[str, Any],
        op_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized = dict(params or {})
        placeholders = [
            str(item).strip()
            for item in PATH_PARAM_PATTERN.findall(str(endpoint or ""))
            if str(item).strip()
        ]
        if not placeholders:
            return normalized
        path_param_schemas = (
            op_meta.get("path_param_schemas", {})
            if isinstance(op_meta.get("path_param_schemas", {}), dict)
            else {}
        )
        for name in placeholders:
            raw_value = normalized.get(name)
            raw_text = str(raw_value).strip() if raw_value is not None else ""
            lower_value = raw_text.lower()
            invalid_hint = (
                raw_value is None
                or raw_text == ""
                or raw_text == MISSING_PATH_PARAM_SENTINEL
                or any(token in lower_value for token in ("nonexistent", "notfound", "missing", "invalid"))
                or bool(re.fullmatch(r"9{3,}", raw_text))
            )
            if invalid_hint:
                normalized[name] = self._default_path_param_value(
                    name,
                    path_param_schemas.get(name, {}),
                )
        return normalized

    def _make_mutation_scenario(
        self,
        base: TestScenario,
        *,
        suffix: str,
        strategy: str,
        test_type: Optional[TestType] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        expected_status: Optional[int] = None,
    ) -> TestScenario:
        assertions = list(base.assertions or [])
        assertions.append(f"rl_mutation:{strategy}")
        return TestScenario(
            name=f"{base.name}_{suffix}",
            description=f"{base.description} [RL mutation: {strategy}]",
            test_type=test_type or base.test_type,
            endpoint=str(base.endpoint),
            method=str(base.method).upper(),
            headers=deepcopy(headers) if headers is not None else deepcopy(base.headers or {}),
            params=deepcopy(params) if params is not None else deepcopy(base.params or {}),
            body=deepcopy(body) if body is not None else deepcopy(base.body),
            expected_status=int(
                expected_status if expected_status is not None else int(base.expected_status)
            ),
            expected_response_fields=list(base.expected_response_fields or []),
            assertions=assertions,
        )

    def _build_type_fuzz_body(self, scenario: TestScenario) -> Dict[str, Any]:
        body = deepcopy(scenario.body) if isinstance(scenario.body, dict) else {}
        if not body:
            return {
                "id": "not_an_int",
                "count": "invalid_integer",
                "enabled": "not_a_bool",
            }

        for key in sorted(body.keys()):
            value = body.get(key)
            if isinstance(value, bool):
                body[key] = "not_a_bool"
                return body
            if isinstance(value, (int, float)):
                body[key] = "not_a_number"
                return body
            if isinstance(value, str):
                body[key] = value + "_x" * 32
                return body
            if isinstance(value, list):
                body[key] = "not_a_list"
                return body
            if isinstance(value, dict):
                body[key] = "not_an_object"
                return body

        body["rl_mutation"] = "'; DROP TABLE users; --"
        return body

    def _fuzz_query_value(self, value: Any) -> Any:
        if isinstance(value, bool):
            return "not_a_bool"
        if isinstance(value, int):
            return -999999 if value >= 0 else 999999
        if isinstance(value, float):
            return "not_a_float"
        if isinstance(value, str):
            return "'; DROP TABLE users; --" if value else "x" * 2048
        return "'; DROP TABLE users; --"

    def _dedupe_mutation_name(self, base_name: str, used_names: set[str]) -> str:
        name = str(base_name).strip() or "rl_mutation_case"
        if name not in used_names:
            return name
        idx = 2
        while f"{name}_{idx}" in used_names:
            idx += 1
        return f"{name}_{idx}"

    def _dedupe_scenarios_by_fingerprint(
        self, scenarios: List[TestScenario]
    ) -> List[TestScenario]:
        if not scenarios:
            return []
        deduped: List[TestScenario] = []
        seen_fingerprints: set[str] = set()
        seen_names: set[str] = set()
        for scenario in scenarios:
            fingerprint = self._scenario_fingerprint(scenario)
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            scenario.name = self._dedupe_mutation_name(str(scenario.name), seen_names)
            seen_names.add(str(scenario.name))
            deduped.append(scenario)
        return deduped

    def _portfolio_target_counts(self, budget: int) -> Dict[str, int]:
        total = max(0, int(budget))
        if total <= 0:
            return {"stable": 0, "focus": 0, "explore": 0}
        stable = int(math.floor(total * PORTFOLIO_STABLE_RATIO))
        focus = int(math.floor(total * PORTFOLIO_FOCUS_RATIO))
        explore = int(math.floor(total * PORTFOLIO_EXPLORE_RATIO))
        assigned = stable + focus + explore
        remainder = total - assigned
        # Prefer extra slots for stable first, then focus.
        for bucket in ("stable", "focus", "explore"):
            if remainder <= 0:
                break
            if bucket == "stable":
                stable += 1
            elif bucket == "focus":
                focus += 1
            else:
                explore += 1
            remainder -= 1
        return {"stable": stable, "focus": focus, "explore": explore}

    def _candidate_portfolio_bucket(
        self,
        *,
        candidate: Dict[str, Any],
        known_patterns: Dict[str, Any],
        uncertainty_threshold: float,
    ) -> str:
        fingerprint = str(candidate.get("fingerprint", ""))
        attempts = int(candidate.get("historical_attempts", 0))
        failure_rate = float(candidate.get("historical_failure_rate", 0.0))
        avg_reward = float(candidate.get("historical_avg_reward", 0.0))
        uncertainty = float(candidate.get("score_parts", {}).get("uncertainty", 0.0))
        novelty_bonus = float(candidate.get("novelty_bonus", 0.0))

        if novelty_bonus > 0.0 or uncertainty >= max(0.0001, float(uncertainty_threshold)):
            return "explore"
        if (
            fingerprint in known_patterns
            and attempts >= RL_WEAK_MIN_ATTEMPTS
            and (failure_rate >= RL_WEAK_FAILURE_RATE_THRESHOLD or avg_reward < 0.25)
        ):
            return "focus"
        return "stable"

    def _normalize_focus_key(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "").strip().lower())
        return cleaned[:240]

    def _get_recent_gam_focus_keys(self) -> List[str]:
        raw = self.learning_state.get("gam_recent_focus_points", [])
        if not isinstance(raw, list):
            return []
        keys = [self._normalize_focus_key(str(item)) for item in raw if str(item).strip()]
        if len(keys) > GAM_RECENT_FOCUS_LIMIT:
            keys = keys[-GAM_RECENT_FOCUS_LIMIT:]
        return keys

    def _remember_gam_focus_points(self, focus_points: List[str]) -> None:
        prior = self._get_recent_gam_focus_keys()
        additions = [self._normalize_focus_key(item) for item in (focus_points or []) if str(item).strip()]
        combined = (prior + additions)[-GAM_RECENT_FOCUS_LIMIT:]
        self.learning_state["gam_recent_focus_points"] = combined

    def _build_spec_signature_focus_point(self) -> str:
        op_index = self._operation_index if isinstance(self._operation_index, dict) else {}
        if not op_index:
            return ""
        operations = len(op_index)
        auth_required = sum(1 for meta in op_index.values() if bool((meta or {}).get("auth_required", False)))
        with_path_params = sum(
            1 for meta in op_index.values() if bool((meta or {}).get("path_param_names", []))
        )
        write_ops = sum(
            1
            for key in op_index.keys()
            if str(key).split(" ", 1)[0] in {"POST", "PUT", "PATCH", "DELETE"}
        )
        return (
            "Spec signature: "
            f"ops={operations}, auth_required={auth_required}, write_ops={write_ops}, "
            f"path_param_ops={with_path_params}. Prioritize scenarios reflecting this shape."
        )

    def _select_scenarios_with_learning(self, scenarios: List[TestScenario]) -> List[TestScenario]:
        if not scenarios:
            self._last_selection_trace = []
            self._last_selection_summary = {}
            return []

        type_weights = self.learning_state.get("test_type_weights", {})
        endpoint_weights = self.learning_state.get("endpoint_weights", {})
        known_patterns = self._scenario_stats_for_current_spec(
            self.adaptive_policy.scenario_stats
        )

        candidates: List[Dict[str, Any]] = []
        for scenario in scenarios:
            type_key = scenario.test_type.value
            endpoint_key = f"{scenario.method.upper()} {scenario.endpoint}"
            fingerprint = self._scenario_fingerprint(scenario)
            historical = known_patterns.get(fingerprint, {}) if isinstance(known_patterns, dict) else {}

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
                    "historical_attempts": int(
                        historical.get("attempts", 0) if isinstance(historical, dict) else 0
                    ),
                    "historical_failure_rate": float(
                        historical.get("failure_rate", 0.0) if isinstance(historical, dict) else 0.0
                    ),
                    "historical_avg_reward": float(
                        historical.get("avg_reward", 0.0) if isinstance(historical, dict) else 0.0
                    ),
                }
            )

        selected: List[TestScenario] = []
        selection_trace: List[Dict[str, Any]] = []
        endpoint_counter: Counter[str] = Counter()
        type_counter: Counter[str] = Counter()
        selected_fingerprints: set[str] = set()

        # Coverage guardrail: reserve one happy-path candidate per operation when available.
        mandatory_happy_candidates: List[Dict[str, Any]] = []
        if isinstance(self._operation_index, dict) and self._operation_index:
            for operation_key in sorted(self._operation_index.keys()):
                if len(mandatory_happy_candidates) >= int(self.max_scenarios):
                    break
                happy_options = [
                    item
                    for item in candidates
                    if (
                        str(item["endpoint_key"]) == str(operation_key)
                        and str(item["type_key"]) == str(TestType.HAPPY_PATH.value)
                        and int(getattr(item["scenario"], "expected_status", 0)) >= 200
                        and int(getattr(item["scenario"], "expected_status", 0)) < 300
                    )
                ]
                if not happy_options:
                    continue
                chosen = max(
                    happy_options,
                    key=lambda item: float(item["score_parts"].get("score", 0.0)),
                )
                fp = str(chosen["fingerprint"])
                if fp in {str(item["fingerprint"]) for item in mandatory_happy_candidates}:
                    continue
                mandatory_happy_candidates.append(chosen)

        if mandatory_happy_candidates:
            for chosen in mandatory_happy_candidates:
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
                            "selection_reason": "happy_path_coverage",
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
            candidates = [
                item for item in candidates if item["fingerprint"] not in selected_fingerprints
            ]

        # Diversity guardrail: keep at least one candidate from core QA categories
        # when available, so policy doesn't collapse into one class of tests.
        core_types = [
            TestType.AUTHENTICATION.value,
            TestType.INPUT_VALIDATION.value,
            TestType.ERROR_HANDLING.value,
            TestType.BOUNDARY_TESTING.value,
        ]
        for core_type in core_types:
            if len(selected) >= int(self.max_scenarios):
                break
            matching = [
                item
                for item in candidates
                if str(item.get("type_key", "")) == str(core_type)
                and item["fingerprint"] not in selected_fingerprints
            ]
            if not matching:
                continue
            chosen = max(
                matching,
                key=lambda item: float(item["score_parts"].get("score", 0.0)),
            )
            fp = chosen["fingerprint"]
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
                        "selection_reason": "core_type_coverage",
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
            candidates = [item for item in candidates if item["fingerprint"] != fp]

        forced_targets = self._build_forced_weak_replay_targets(limit=2)
        if forced_targets:
            for target in forced_targets:
                if len(selected) >= int(self.max_scenarios):
                    break
                matching: List[Dict[str, Any]] = [
                    item
                    for item in candidates
                    if self._scenario_matches_replay_target(item["scenario"], target)
                    and item["fingerprint"] not in selected_fingerprints
                ]
                if not matching:
                    continue
                chosen = max(
                    matching,
                    key=lambda item: float(item["score_parts"].get("score", 0.0)),
                )
                fp = chosen["fingerprint"]
                scenario = chosen["scenario"]
                selected.append(scenario)
                selected_fingerprints.add(fp)
                replay_schedule = self.learning_state.setdefault("replay_schedule", {})
                if isinstance(replay_schedule, dict):
                    replay_schedule[str(target.get("fingerprint", fp))] = int(
                        self.learning_state.get("run_count", 0)
                    )
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
                            "selection_reason": "forced_weak_replay",
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
                candidates = [item for item in candidates if item["fingerprint"] != fp]

        uncertainty_values = [
            float(item["score_parts"].get("uncertainty", 0.0)) for item in candidates
        ]
        uncertainty_threshold = self._compute_uncertainty_threshold(uncertainty_values)
        uncertain_candidates = [
            item
            for item in candidates
            if float(item["score_parts"].get("uncertainty", 0.0)) >= uncertainty_threshold
            and float(item["score_parts"].get("uncertainty", 0.0)) > 0.0
            and (
                float(item.get("novelty_bonus", 0.0)) > 0.0
                or int(item.get("historical_attempts", 0)) < 3
                or float(item.get("historical_failure_rate", 0.0)) >= 0.05
                or float(item.get("historical_avg_reward", 0.0)) < 0.55
            )
        ]
        effective_budget = max(int(self.max_scenarios), len(uncertain_candidates), len(selected))
        budget_expanded = effective_budget > int(self.max_scenarios)

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
                if len(selected) >= int(effective_budget):
                    break
                fp = str(chosen["fingerprint"])
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

        remaining_budget = max(0, int(effective_budget - len(selected)))
        portfolio_targets = self._portfolio_target_counts(remaining_budget)
        bucket_counts = {"stable": 0, "focus": 0, "explore": 0}

        def _select_from_bucket(bucket: str, selection_reason: str) -> None:
            nonlocal candidates
            target = int(portfolio_targets.get(bucket, 0))
            if target <= 0:
                return
            while bucket_counts[bucket] < target and len(selected) < int(effective_budget):
                pool = [
                    item
                    for item in candidates
                    if item["fingerprint"] not in selected_fingerprints
                    and self._candidate_portfolio_bucket(
                        candidate=item,
                        known_patterns=known_patterns,
                        uncertainty_threshold=uncertainty_threshold,
                    )
                    == bucket
                ]
                if not pool:
                    break
                chosen = max(
                    pool,
                    key=lambda item: (
                        float(item["score_parts"].get("score", 0.0)),
                        float(item["score_parts"].get("uncertainty", 0.0)),
                    ),
                )
                fp = str(chosen["fingerprint"])
                scenario = chosen["scenario"]
                selected.append(scenario)
                selected_fingerprints.add(fp)
                bucket_counts[bucket] += 1
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
                            "selection_reason": selection_reason,
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
                candidates = [item for item in candidates if item["fingerprint"] != fp]

        _select_from_bucket("focus", "focus_quota")
        _select_from_bucket("explore", "explore_quota")
        _select_from_bucket("stable", "stable_quota")

        # Fill any leftover slots with score-ranked diversity-aware selection.
        while candidates and len(selected) < int(effective_budget):
            best_idx = 0
            best_score = -float("inf")
            best_diversity_penalty = 0.0
            best_bucket = "stable"
            for idx, item in enumerate(candidates):
                endpoint_penalty = 0.09 * endpoint_counter[item["endpoint_key"]]
                type_penalty = 0.05 * type_counter[item["type_key"]]
                history_penalty = (
                    0.07 if str(item.get("fingerprint", "")) in known_patterns else 0.0
                )
                diversity_penalty = endpoint_penalty + type_penalty + history_penalty
                total_score = float(item["score_parts"]["score"]) - diversity_penalty
                if total_score > best_score:
                    best_score = total_score
                    best_idx = idx
                    best_diversity_penalty = diversity_penalty
                    best_bucket = self._candidate_portfolio_bucket(
                        candidate=item,
                        known_patterns=known_patterns,
                        uncertainty_threshold=uncertainty_threshold,
                    )

            chosen = candidates.pop(best_idx)
            scenario = chosen["scenario"]
            selected.append(scenario)
            selected_fingerprints.add(chosen["fingerprint"])
            bucket_counts[best_bucket] = int(bucket_counts.get(best_bucket, 0)) + 1
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

        run_count = int(self.learning_state.get("run_count", 0))
        novelty_target = 0
        if run_count >= 3 and self.max_scenarios >= 8:
            novelty_target = max(1, min(4, int(math.ceil(float(self.max_scenarios) * 0.25))))

        final_novel_selected_count = int(
            sum(1 for fp in selected_fingerprints if fp not in known_patterns)
        )
        scenario_by_name = {
            str(getattr(item, "name", "") or ""): item for item in scenarios
        }
        for trace_item in selection_trace:
            if not isinstance(trace_item, dict):
                continue
            name_key = str(trace_item.get("name", "") or "")
            scenario = scenario_by_name.get(name_key)
            if scenario is None:
                continue
            trace_item["display_name"] = self._scenario_display_name(scenario)
            trace_item["name_raw"] = str(getattr(scenario, "name", name_key))
            trace_item["mutation_strategy"] = self._extract_rl_mutation_strategy(scenario)
            trace_item["history_seeded"] = bool(self._is_history_seeded_scenario(scenario))
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
            "novelty_target": int(novelty_target),
            "novel_selected_count": int(final_novel_selected_count),
            "portfolio_policy": {
                "stable_ratio": PORTFOLIO_STABLE_RATIO,
                "focus_ratio": PORTFOLIO_FOCUS_RATIO,
                "explore_ratio": PORTFOLIO_EXPLORE_RATIO,
                "targets": {k: int(v) for k, v in portfolio_targets.items()},
                "selected": {k: int(v) for k, v in bucket_counts.items()},
            },
        }
        self.learning_state["selection_trace"] = list(selection_trace)
        self.learning_state["selection_summary"] = dict(self._last_selection_summary)
        return selected

    def _build_forced_weak_replay_targets(
        self, limit: int = 2
    ) -> List[Dict[str, Any]]:
        scenario_stats = self._scenario_stats_for_current_spec(
            self.adaptive_policy.scenario_stats
        )
        if not scenario_stats:
            return []

        replay_schedule = self.learning_state.get("replay_schedule", {})
        if not isinstance(replay_schedule, dict):
            replay_schedule = {}
        current_run = int(self.learning_state.get("run_count", 0))

        ranked: List[Dict[str, Any]] = []
        for fingerprint, stats_raw in scenario_stats.items():
            if not isinstance(stats_raw, dict):
                continue
            attempts = int(stats_raw.get("attempts", 0))
            failure_rate = float(stats_raw.get("failure_rate", 0.0))
            avg_reward = float(stats_raw.get("avg_reward", 0.0))
            if attempts < 3 or failure_rate < 0.75:
                continue
            last_replay_run = int(replay_schedule.get(str(fingerprint), -FORCED_REPLAY_CADENCE_RUNS))
            if (current_run - last_replay_run) < FORCED_REPLAY_CADENCE_RUNS:
                continue

            parsed = self._parse_scenario_fingerprint(str(fingerprint))
            method = str((parsed or {}).get("method") or stats_raw.get("method", "GET")).upper()
            endpoint = str((parsed or {}).get("endpoint") or stats_raw.get("endpoint", ""))
            test_type = str(
                (parsed or {}).get("test_type") or stats_raw.get("test_type", "error_handling")
            )
            raw_expected_status = int(
                (parsed or {}).get("expected_status") or stats_raw.get("expected_status", 400)
            )
            canonical = self._canonicalize_learning_pattern(
                method=method,
                endpoint=endpoint,
                test_type=test_type,
                expected_status=raw_expected_status,
            )
            if not canonical:
                continue
            canonical_method = str(canonical.get("method", method))
            canonical_endpoint = str(canonical.get("endpoint", endpoint))
            if not self._scenario_belongs_to_current_spec(canonical_method, canonical_endpoint):
                continue

            ranked.append(
                {
                    "fingerprint": str(fingerprint),
                    "method": canonical_method,
                    "endpoint": canonical_endpoint,
                    "test_type": str(canonical.get("test_type", test_type)),
                    "expected_status": int(canonical.get("expected_status", raw_expected_status)),
                    "priority": (
                        0.70 * failure_rate
                        + 0.20 * max(0.0, -avg_reward)
                        + 0.10 * min(1.0, attempts / 10.0)
                    ),
                    "attempts": attempts,
                    "failure_rate": failure_rate,
                }
            )

        ranked.sort(
            key=lambda item: (
                float(item.get("priority", 0.0)),
                float(item.get("failure_rate", 0.0)),
                int(item.get("attempts", 0)),
            ),
            reverse=True,
        )
        targets: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for item in ranked:
            key = (
                str(item.get("method", "")),
                str(item.get("endpoint", "")),
                str(item.get("test_type", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            targets.append(item)
            if len(targets) >= max(0, int(limit)):
                break
        return targets

    def _scenario_matches_replay_target(
        self, scenario: TestScenario, target: Dict[str, Any]
    ) -> bool:
        method = str(scenario.method).upper()
        endpoint = str(scenario.endpoint)
        test_type = str(scenario.test_type.value)
        expected_status = int(scenario.expected_status)

        return (
            method == str(target.get("method", ""))
            and endpoint == str(target.get("endpoint", ""))
            and test_type == str(target.get("test_type", ""))
            and expected_status == int(target.get("expected_status", expected_status))
        )

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
        repro_artifacts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        endpoint_count = max(1, int(summary.get("detected_endpoints", 1)))
        total = max(1, int(summary.get("total_scenarios", 1)))
        pass_rate = float(summary.get("pass_rate", 0.0))
        coverage_ratio = min(1.0, total / endpoint_count)
        latency_penalty = min(0.30, float(summary.get("average_duration_ms", 0.0)) / 2000.0)
        historical_patterns = set(
            self._scenario_stats_for_current_spec(self.adaptive_policy.scenario_stats).keys()
        )

        signals: List[DecisionLearningSignal] = []
        informative_failure_count = 0
        novel_positive_count = 0
        novel_count = 0
        redundant_easy_pass_count = 0
        unsafe_action_blocked_count = 0
        for scenario, result in zip(scenarios, results):
            expected = int(result.expected_status)
            fingerprint = self._scenario_fingerprint(scenario)
            seen_before = fingerprint in historical_patterns
            runtime_skipped = str(result.error or "").startswith("runtime_cap_exceeded")
            unsafe_action_blocked = str(result.error or "").startswith("unsafe_action_blocked")
            verification = result.verification if isinstance(result.verification, dict) else {}
            status_check = verification.get("status_check", {}) if isinstance(verification.get("status_check", {}), dict) else {}
            contract_check = verification.get("contract_check", {}) if isinstance(verification.get("contract_check", {}), dict) else {}
            verdict = str(
                verification.get(
                    "verdict",
                    getattr(result, "verdict", "pass" if result.passed else "fail"),
                )
                or "fail"
            ).strip().lower()
            corrected_expectation = bool(status_check.get("corrected", False))
            corrected_expected_raw = status_check.get("corrected_expected_status")
            try:
                corrected_expected = (
                    int(corrected_expected_raw)
                    if corrected_expected_raw is not None
                    else expected
                )
            except Exception:
                corrected_expected = expected
            effective_expected = corrected_expected if corrected_expectation else expected
            expected_negative = effective_expected in NEGATIVE_STATUS_CODES
            contract_valid_raw = contract_check.get("valid")
            contract_valid = (
                bool(contract_valid_raw)
                if contract_valid_raw is not None
                else None
            )
            if not seen_before:
                novel_count += 1

            if runtime_skipped:
                decision_reward = -0.35
            elif unsafe_action_blocked:
                unsafe_action_blocked_count += 1
                decision_reward = -1.20
            elif verdict == "suspect":
                decision_reward = -0.25
            elif verdict == "pass":
                # Passing already-known patterns should be low reward to avoid
                # policy collapse into safe repetitive tests.
                if expected_negative:
                    decision_reward = 0.65 if not seen_before else 0.22
                else:
                    decision_reward = 0.50 if not seen_before else 0.20
                if seen_before:
                    redundant_easy_pass_count += 1
                    decision_reward -= 0.08
            else:
                if expected_negative:
                    # Treat failing negative assertions as informative bug signal.
                    informative_failure_count += 1
                    decision_reward = 0.95
                else:
                    decision_reward = -1.10

            if corrected_expectation:
                # Corrected expectations indicate expectation drift; keep as suspect.
                decision_reward -= 0.15
            if contract_valid is False:
                decision_reward -= 0.35

            if not seen_before and decision_reward > 0:
                decision_reward += 0.15
                novel_positive_count += 1

            decision_reward -= min(0.20, result.duration_ms / 5000.0)
            decision_reward = max(-1.5, min(1.5, decision_reward))

            signals.append(
                DecisionLearningSignal(
                    name=result.name,
                    test_type=result.test_type,
                    method=scenario.method.upper(),
                    endpoint_template=scenario.endpoint,
                    endpoint_key=f"{scenario.method.upper()} {scenario.endpoint}",
                    scenario_fingerprint=fingerprint,
                    has_body=bool(scenario.body),
                    has_params=bool(scenario.params),
                    reward=decision_reward,
                    passed=bool(verdict == "pass"),
                    expected_status=result.expected_status,
                    actual_status=result.actual_status,
                    corrected_expectation=corrected_expectation,
                    contract_valid=contract_valid,
                    display_name=self._scenario_display_name(scenario),
                    name_raw=str(scenario.name),
                    mutation_strategy=self._extract_rl_mutation_strategy(scenario),
                    history_seeded=self._is_history_seeded_scenario(scenario),
                )
            )

        avg_decision_reward = (
            sum(signal.reward for signal in signals) / len(signals) if signals else 0.0
        )
        decision_quality_component = max(0.0, min(1.0, (avg_decision_reward + 1.5) / 3.0))
        informative_failure_ratio = informative_failure_count / total
        novelty_ratio = novel_count / total
        novelty_success_ratio = novel_positive_count / max(1, novel_count)
        redundancy_ratio = redundant_easy_pass_count / total
        unsafe_action_ratio = unsafe_action_blocked_count / total
        repro_list = [
            item for item in (repro_artifacts or [])
            if isinstance(item, dict)
        ]
        reproducible_failure_ratio = (
            sum(
                1
                for item in repro_list
                if item.get("actual_status") is not None
                and item.get("minimized_request", {}).get("status") == item.get("actual_status")
            )
            / float(max(1, len(repro_list)))
            if repro_list
            else 0.0
        )
        env_signal = self._environment_tier_signal()
        env_weight = float(env_signal.get("reward_weight", 0.90))

        run_reward = (
            0.08 * pass_rate
            + 0.10 * coverage_ratio
            + 0.26 * decision_quality_component
            + 0.30 * informative_failure_ratio
            + 0.18 * novelty_success_ratio
            + 0.14 * reproducible_failure_ratio
            - 0.10 * latency_penalty
            - 0.20 * redundancy_ratio
            - 0.20 * unsafe_action_ratio
        )
        run_reward = max(0.0, min(1.0, run_reward * env_weight))

        penalties = sum(1 for signal in signals if signal.reward < 0)
        rewards = sum(1 for signal in signals if signal.reward >= 0)

        return {
            "run_reward": round(run_reward, 4),
            "reward_breakdown": {
                "pass_rate_component": round(0.08 * pass_rate, 4),
                "coverage_component": round(0.10 * coverage_ratio, 4),
                "decision_quality_component": round(0.26 * decision_quality_component, 4),
                "informative_failure_component": round(0.30 * informative_failure_ratio, 4),
                "novelty_component": round(0.18 * novelty_success_ratio, 4),
                "reproducible_failure_component": round(0.14 * reproducible_failure_ratio, 4),
                "latency_penalty_component": round(-0.10 * latency_penalty, 4),
                "redundancy_penalty_component": round(-0.20 * redundancy_ratio, 4),
                "unsafe_action_penalty_component": round(-0.20 * unsafe_action_ratio, 4),
                "environment_weight_multiplier": round(env_weight, 4),
            },
            "average_decision_reward": round(avg_decision_reward, 4),
            "decision_quality_score": round(decision_quality_component, 4),
            "informative_failure_ratio": round(informative_failure_ratio, 4),
            "novelty_ratio": round(novelty_ratio, 4),
            "redundancy_ratio": round(redundancy_ratio, 4),
            "reproducible_failure_ratio": round(reproducible_failure_ratio, 4),
            "unsafe_action_ratio": round(unsafe_action_ratio, 4),
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
                    "spec_scope": str(self._spec_scope_key or "").strip() or None,
                    "maturity_level": "L1",
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
            if self._spec_scope_key:
                stats["spec_scope"] = str(self._spec_scope_key)
            actual_key = str(signal.get("actual_status"))
            actual_counts = stats.setdefault("actual_status_counts", {})
            actual_counts[actual_key] = int(actual_counts.get(actual_key, 0)) + 1
            # Mimic gradual junior-engineer progression by promoting maturity
            # only after enough repetitions with improving reliability.
            maturity = "L1"
            if attempts >= 12 and failures <= max(1, int(attempts * 0.15)):
                maturity = "L3"
            elif attempts >= 6 and failures <= max(1, int(attempts * 0.30)):
                maturity = "L2"
            stats["maturity_level"] = maturity

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

        replay_schedule = self.learning_state.setdefault("replay_schedule", {})
        if isinstance(replay_schedule, dict):
            valid_keys = set(str(k) for k in self.learning_state["scenario_stats"].keys())
            stale = [str(k) for k in replay_schedule.keys() if str(k) not in valid_keys]
            for key in stale:
                replay_schedule.pop(key, None)

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
            test_type = str(stats_raw.get("test_type", "unknown"))
            normalized_expected_status = self._normalize_expected_status_for_test_type(
                test_type=test_type,
                expected_status=expected_status,
            )
            normalized_dominant_status = self._normalize_expected_status_for_test_type(
                test_type=test_type,
                expected_status=int(dominant_status),
            )
            method = str(
                stats_raw.get("method", self._method_from_fingerprint(str(fingerprint)))
            ).upper()

            rule: Dict[str, Any] = {
                "fingerprint": str(fingerprint),
                "method": method,
                "endpoint": str(stats_raw.get("endpoint", "")),
                "test_type": test_type,
                "attempts": attempts,
                "status_observations": int(observed_total),
                "failure_rate": round(failure_rate, 6),
                "dominant_actual_status": int(normalized_dominant_status),
                "dominant_ratio": round(dominant_ratio, 6),
            }

            write_success_needs_body_repair = (
                method in {"POST", "PUT", "PATCH"}
                and normalized_expected_status in {200, 201, 202, 204}
                and normalized_dominant_status == 400
            )
            if write_success_needs_body_repair:
                rule["repair_request_body"] = True
            elif normalized_dominant_status != normalized_expected_status:
                rule["override_expected_status"] = int(normalized_dominant_status)

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

    def _resolve_local_ref(self, spec: Dict[str, Any], ref: str) -> Any:
        ref_text = str(ref or "").strip()
        if not ref_text.startswith("#/"):
            return None
        node: Any = spec
        for token in ref_text[2:].split("/"):
            key = token.replace("~1", "/").replace("~0", "~")
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return deepcopy(node)

    def _resolve_refs(self, spec: Dict[str, Any], node: Any, max_depth: int = 12) -> Any:
        if max_depth <= 0:
            return deepcopy(node)
        if isinstance(node, dict):
            if "$ref" in node and isinstance(node.get("$ref"), str):
                resolved = self._resolve_local_ref(spec, str(node.get("$ref")))
                if isinstance(resolved, dict):
                    # Local overrides alongside $ref are allowed in OpenAPI.
                    overlay = {k: deepcopy(v) for k, v in node.items() if k != "$ref"}
                    resolved.update(overlay)
                    return self._resolve_refs(spec, resolved, max_depth=max_depth - 1)
            return {
                str(key): self._resolve_refs(spec, value, max_depth=max_depth - 1)
                for key, value in node.items()
            }
        if isinstance(node, list):
            return [self._resolve_refs(spec, item, max_depth=max_depth - 1) for item in node]
        return deepcopy(node)

    def _build_operation_index(self, spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        for path, path_info in (spec.get("paths") or {}).items():
            if not isinstance(path_info, dict):
                continue
            path_parameters = self._resolve_refs(spec, path_info.get("parameters", []))
            if not isinstance(path_parameters, list):
                path_parameters = []

            for method, operation_raw in path_info.items():
                if str(method).lower() not in HTTP_METHODS:
                    continue
                operation = self._resolve_refs(
                    spec,
                    operation_raw if isinstance(operation_raw, dict) else {},
                )
                if not isinstance(operation, dict):
                    operation = {}
                request_schema = self._extract_request_schema(spec, operation)
                required_fields: List[str] = []
                if isinstance(request_schema, dict):
                    required_raw = request_schema.get("required", [])
                    if isinstance(required_raw, list):
                        required_fields = [str(item) for item in required_raw]

                response_statuses: List[int] = []
                response_schemas: Dict[str, Dict[str, Any]] = {}
                responses = operation.get("responses", {})
                if isinstance(responses, dict):
                    for code, response_meta in responses.items():
                        code_text = str(code)
                        if code_text.isdigit():
                            response_statuses.append(int(code_text))
                        response_payload = self._resolve_refs(spec, response_meta)
                        if isinstance(response_payload, dict):
                            extracted_schema = self._extract_response_schema_from_response_payload(
                                spec=spec,
                                response_payload=response_payload,
                            )
                            if isinstance(extracted_schema, dict) and extracted_schema:
                                response_schemas[code_text] = extracted_schema

                raw_operation_params = self._resolve_refs(
                    spec,
                    operation.get("parameters", []),
                )
                operation_params = (
                    raw_operation_params if isinstance(raw_operation_params, list) else []
                )
                merged_parameters = path_parameters + operation_params
                path_param_names = set(PATH_PARAM_PATTERN.findall(path))
                path_param_schemas: Dict[str, Dict[str, Any]] = {}
                query_param_names: List[str] = []
                for param in merged_parameters:
                    if not isinstance(param, dict):
                        continue
                    name = str(param.get("name", "")).strip()
                    if not name:
                        continue
                    location = str(param.get("in", "")).strip().lower()
                    if location == "path":
                        path_param_names.add(name)
                        schema = self._resolve_refs(spec, param.get("schema", {}))
                        if isinstance(schema, dict):
                            path_param_schemas[name] = deepcopy(schema)
                    elif location == "query":
                        query_param_names.append(name)

                index[self._operation_key(str(method).upper(), path)] = {
                    "request_schema": request_schema,
                    "required_fields": required_fields,
                    "response_statuses": response_statuses,
                    "response_schemas": response_schemas,
                    "path_param_names": sorted(path_param_names),
                    "path_param_schemas": path_param_schemas,
                    "query_param_names": sorted(dict.fromkeys(query_param_names)),
                    "auth_required": bool(self._operation_key(str(method).upper(), path) in self._auth_required_ops),
                }
        return index

    def _extract_response_schema_from_response_payload(
        self, *, spec: Dict[str, Any], response_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        content = response_payload.get("content", {})
        if not isinstance(content, dict):
            return {}

        for media_type, media_value in content.items():
            media_key = str(media_type).lower()
            if "json" not in media_key:
                continue
            media_obj = self._resolve_refs(
                spec,
                media_value if isinstance(media_value, dict) else {},
            )
            if not isinstance(media_obj, dict):
                continue

            schema = self._resolve_refs(spec, media_obj.get("schema", {}))
            if isinstance(schema, dict) and schema:
                return schema

            example = media_obj.get("example")
            if example is not None:
                return self._derive_schema_from_example(example)
            examples = media_obj.get("examples", {})
            if isinstance(examples, dict) and examples:
                for item in examples.values():
                    example_obj = self._resolve_refs(
                        spec,
                        item if isinstance(item, dict) else {},
                    )
                    if not isinstance(example_obj, dict):
                        continue
                    if "value" in example_obj:
                        return self._derive_schema_from_example(example_obj.get("value"))
        return {}

    def _derive_schema_from_example(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, bool):
            return {"type": "boolean"}
        if isinstance(value, int) and not isinstance(value, bool):
            return {"type": "integer"}
        if isinstance(value, float):
            return {"type": "number"}
        if isinstance(value, str):
            return {"type": "string"}
        if isinstance(value, list):
            if value:
                return {"type": "array", "items": self._derive_schema_from_example(value[0])}
            return {"type": "array"}
        if isinstance(value, dict):
            properties: Dict[str, Any] = {}
            required: List[str] = []
            for key, item in value.items():
                key_text = str(key)
                required.append(key_text)
                properties[key_text] = self._derive_schema_from_example(item)
            result: Dict[str, Any] = {"type": "object", "properties": properties}
            if required:
                result["required"] = required
            return result
        return {"type": "string"}

    def _extract_request_schema(self, spec: Dict[str, Any], operation: Dict[str, Any]) -> Dict[str, Any]:
        request_body = self._resolve_refs(spec, operation.get("requestBody", {}))
        if not isinstance(request_body, dict):
            return {}
        content = request_body.get("content", {})
        if not isinstance(content, dict):
            return {}
        json_schema = self._resolve_refs(
            spec,
            (content.get("application/json") or {}).get("schema", {}),
        )
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
        base = self._scenario_fingerprint_from_fields(
            test_type=scenario.test_type.value,
            method=scenario.method,
            endpoint=scenario.endpoint,
            expected_status=int(scenario.expected_status),
            has_body=bool(scenario.body),
            has_params=bool(scenario.params),
        )
        mutation_strategy = self._extract_rl_mutation_strategy(scenario)
        if mutation_strategy and mutation_strategy != "unknown":
            return f"{base}|mut={mutation_strategy}"
        return base

    def _pick_happy_status_for_operation(self, method: str, op_meta: Dict[str, Any]) -> int:
        response_statuses = op_meta.get("response_statuses", [])
        happy_codes = []
        if isinstance(response_statuses, list):
            for raw in response_statuses:
                try:
                    code = int(raw)
                except Exception:
                    continue
                if 200 <= code < 300:
                    happy_codes.append(code)
        if happy_codes:
            for preferred in (200, 201, 202, 204):
                if preferred in happy_codes:
                    return preferred
            return int(sorted(happy_codes)[0])
        return 201 if str(method).upper() == "POST" else 200

    def _build_happy_path_guardrail_scenario(
        self,
        *,
        operation_key: str,
        op_meta: Dict[str, Any],
    ) -> Optional[TestScenario]:
        op_text = str(operation_key or "").strip()
        if " " not in op_text:
            return None
        method, endpoint = op_text.split(" ", 1)
        method = str(method).upper().strip()
        endpoint = str(endpoint).strip()
        if not method or not endpoint:
            return None

        expected_status = self._pick_happy_status_for_operation(method, op_meta)
        params: Dict[str, Any] = {}
        for path_name in list(op_meta.get("path_param_names", []) or []):
            pname = str(path_name).strip()
            if pname:
                params[pname] = "123"
        for query_name in list(op_meta.get("query_param_names", []) or [])[:2]:
            qname = str(query_name).strip()
            if qname:
                params[qname] = self._sample_query_value(qname, expected_status)

        body: Optional[Dict[str, Any]] = None
        if method in {"POST", "PUT", "PATCH"}:
            request_schema = op_meta.get("request_schema", {})
            required_fields = op_meta.get("required_fields", [])
            if (
                (isinstance(request_schema, dict) and bool(request_schema))
                or (isinstance(required_fields, list) and bool(required_fields))
            ):
                body = self._build_seed_body_from_operation_meta(
                    op_meta,
                    expect_negative=False,
                )

        headers: Dict[str, str] = {}
        if bool(op_meta.get("auth_required", False)):
            headers["Authorization"] = "Bearer valid_token_123"

        slug = re.sub(r"[^a-zA-Z0-9]+", "_", endpoint).strip("_") or "root"
        name = f"test_{method.lower()}_{slug}_happy_path_guardrail"
        description = f"Coverage guardrail happy path for {method} {endpoint}"
        return TestScenario(
            name=name,
            description=description,
            test_type=TestType.HAPPY_PATH,
            endpoint=endpoint,
            method=method,
            headers=headers,
            params=params,
            body=body,
            expected_status=int(expected_status),
            expected_response_fields=[],
            assertions=["coverage_guardrail:happy_path"],
        )

    def _ensure_happy_path_coverage(
        self, scenarios: List[TestScenario]
    ) -> tuple[List[TestScenario], Dict[str, Any]]:
        base = list(scenarios or [])
        if not base or not isinstance(self._operation_index, dict) or not self._operation_index:
            return base, {
                "required_operations": int(len(self._operation_index or {})),
                "covered_operations_before": 0,
                "added": 0,
                "missing_after": int(len(self._operation_index or {})),
            }

        existing_fingerprints = {self._scenario_fingerprint(item) for item in base}
        existing_names = {str(item.name).strip() for item in base}
        covered_before: set[str] = set()
        for scenario in base:
            scenario_type = str(
                getattr(getattr(scenario, "test_type", ""), "value", getattr(scenario, "test_type", ""))
            ).lower()
            try:
                expected = int(getattr(scenario, "expected_status", 0))
            except Exception:
                expected = 0
            if (
                scenario_type == str(TestType.HAPPY_PATH.value)
                and 200 <= expected < 300
            ):
                covered_before.add(
                    self._operation_key(
                        str(getattr(scenario, "method", "GET")),
                        str(getattr(scenario, "endpoint", "/")),
                    )
                )

        added = 0
        required_operations = sorted(self._operation_index.keys())
        for operation_key in required_operations:
            if operation_key in covered_before:
                continue
            op_meta = self._operation_index.get(operation_key, {})
            if not isinstance(op_meta, dict):
                continue
            guardrail = self._build_happy_path_guardrail_scenario(
                operation_key=operation_key,
                op_meta=op_meta,
            )
            if guardrail is None:
                continue
            fp = self._scenario_fingerprint(guardrail)
            if fp in existing_fingerprints:
                continue
            guardrail.name = self._dedupe_mutation_name(str(guardrail.name), existing_names)
            existing_names.add(str(guardrail.name))
            existing_fingerprints.add(fp)
            base.append(guardrail)
            added += 1

        covered_after: set[str] = set()
        for scenario in base:
            scenario_type = str(
                getattr(getattr(scenario, "test_type", ""), "value", getattr(scenario, "test_type", ""))
            ).lower()
            try:
                expected = int(getattr(scenario, "expected_status", 0))
            except Exception:
                expected = 0
            if (
                scenario_type == str(TestType.HAPPY_PATH.value)
                and 200 <= expected < 300
            ):
                covered_after.add(
                    self._operation_key(
                        str(getattr(scenario, "method", "GET")),
                        str(getattr(scenario, "endpoint", "/")),
                    )
                )

        summary = {
            "required_operations": int(len(required_operations)),
            "covered_operations_before": int(len(covered_before)),
            "added": int(added),
            "missing_after": int(max(0, len(required_operations) - len(covered_after))),
        }
        return base, summary

    def _inject_workflow_sequence_scenarios(
        self,
        scenarios: List[TestScenario],
        spec_intelligence: Dict[str, Any],
        limit: int = 6,
    ) -> tuple[List[TestScenario], Dict[str, Any]]:
        base = list(scenarios or [])
        workflows = (
            list((spec_intelligence or {}).get("workflow_candidates", []) or [])
            if isinstance(spec_intelligence, dict)
            else []
        )
        if not workflows:
            return base, {
                "workflow_candidates": 0,
                "added_sequence_probes": 0,
                "added_order_unmet": 0,
            }

        existing_fingerprints = {self._scenario_fingerprint(item) for item in base}
        existing_names = {str(item.name).strip() for item in base}
        added_sequence = 0
        added_order_unmet = 0
        added_total = 0

        for workflow in workflows:
            if added_total >= max(0, int(limit)):
                break
            if not isinstance(workflow, dict):
                continue
            sequence = workflow.get("sequence", [])
            if not isinstance(sequence, list) or len(sequence) < 2:
                continue
            producer = str(sequence[0]).strip()
            consumer = str(sequence[1]).strip()
            if " " not in consumer:
                continue
            method, endpoint = consumer.split(" ", 1)
            method = str(method).upper().strip()
            endpoint = str(endpoint).strip()
            op_key = self._operation_key(method, endpoint)
            op_meta = self._operation_index.get(op_key, {})
            if not isinstance(op_meta, dict):
                continue

            expected_status = self._pick_happy_status_for_operation(method, op_meta)
            params: Dict[str, Any] = {}
            for path_name in list(op_meta.get("path_param_names", []) or []):
                pname = str(path_name).strip()
                if pname:
                    params[pname] = "123"
            headers: Dict[str, str] = {}
            if bool(op_meta.get("auth_required", False)):
                headers["Authorization"] = "Bearer valid_token_123"

            slug = re.sub(r"[^a-zA-Z0-9]+", "_", endpoint).strip("_") or "root"
            sequence_probe = TestScenario(
                name=f"test_{method.lower()}_{slug}_integration_sequence_probe",
                description=(
                    f"Stateful workflow probe: run `{producer}` before `{consumer}` "
                    "and validate dependent access path."
                ),
                test_type=TestType.INTEGRATION,
                endpoint=endpoint,
                method=method,
                headers=headers,
                params=params,
                expected_status=int(expected_status),
                assertions=[
                    "workflow_sequence_probe",
                    f"workflow_dependency:{producer}->{consumer}",
                ],
            )
            fp = self._scenario_fingerprint(sequence_probe)
            if fp not in existing_fingerprints:
                sequence_probe.name = self._dedupe_mutation_name(
                    str(sequence_probe.name),
                    existing_names,
                )
                existing_names.add(sequence_probe.name)
                existing_fingerprints.add(fp)
                base.append(sequence_probe)
                added_sequence += 1
                added_total += 1
                if added_total >= max(0, int(limit)):
                    break

            path_params = [str(p).strip() for p in (op_meta.get("path_param_names", []) or []) if str(p).strip()]
            if not path_params or method not in {"GET", "PUT", "PATCH", "DELETE"}:
                continue
            order_params = dict(params)
            for pname in path_params:
                order_params[pname] = "nonexistent_dependency_id"
            order_unmet = TestScenario(
                name=f"test_{method.lower()}_{slug}_integration_order_unmet",
                description=(
                    f"Dependency-order negative probe for `{consumer}` before `{producer}`."
                ),
                test_type=TestType.INTEGRATION,
                endpoint=endpoint,
                method=method,
                headers=headers,
                params=order_params,
                expected_status=404,
                assertions=[
                    "workflow_dependency_order_unmet",
                    f"workflow_dependency:{producer}->{consumer}",
                ],
            )
            fp_order = self._scenario_fingerprint(order_unmet)
            if fp_order in existing_fingerprints:
                continue
            order_unmet.name = self._dedupe_mutation_name(
                str(order_unmet.name),
                existing_names,
            )
            existing_names.add(order_unmet.name)
            existing_fingerprints.add(fp_order)
            base.append(order_unmet)
            added_order_unmet += 1
            added_total += 1

        return base, {
            "workflow_candidates": int(len(workflows)),
            "added_sequence_probes": int(added_sequence),
            "added_order_unmet": int(added_order_unmet),
        }

    def _scenario_origin(self, scenario: TestScenario) -> Dict[str, str]:
        strategy = self._extract_rl_mutation_strategy(scenario)
        assertions = [str(item) for item in list(scenario.assertions or [])]
        if "rl_history_seed" in assertions or strategy == "history_seed":
            return {"source": "rl_history_seed", "strategy": "history_seed"}
        if strategy and strategy != "unknown":
            return {"source": "rl_mutation", "strategy": strategy}
        return {"source": str(self._base_scenario_source or "llm_base"), "strategy": ""}

    def _build_scenario_context(
        self,
        *,
        base_scenarios: List[TestScenario],
        candidate_scenarios: List[TestScenario],
        selected_scenarios: List[TestScenario],
        execution_results: List[ScenarioExecutionResult],
        scenario_stats_before: Dict[str, Any],
        run_count_before: int,
    ) -> Dict[str, Any]:
        base_fingerprints = {self._scenario_fingerprint(item) for item in base_scenarios}
        result_by_name = {item.name: item for item in execution_results}
        decision_by_fingerprint: Dict[str, Dict[str, Any]] = {}
        for item in self._last_selection_trace:
            if not isinstance(item, dict):
                continue
            fingerprint = str(item.get("fingerprint", "")).strip()
            if not fingerprint or fingerprint in decision_by_fingerprint:
                continue
            decision_by_fingerprint[fingerprint] = item

        def _source_breakdown(scenarios: List[TestScenario]) -> Dict[str, int]:
            counts = {
                "llm_base": 0,
                "heuristic_base": 0,
                "rl_mutation": 0,
                "rl_history_seed": 0,
            }
            for scenario in scenarios:
                source = str(self._scenario_origin(scenario).get("source", "llm_base"))
                counts[source] = int(counts.get(source, 0)) + 1
            return counts

        selected_rows: List[Dict[str, Any]] = []
        historical_patterns_before = set(str(key) for key in scenario_stats_before.keys())
        weak_selected_count = 0
        for scenario in selected_scenarios:
            fingerprint = self._scenario_fingerprint(scenario)
            source_info = self._scenario_origin(scenario)
            decision = decision_by_fingerprint.get(fingerprint, {})
            result = result_by_name.get(scenario.name)
            historical = scenario_stats_before.get(fingerprint, {})
            if not isinstance(historical, dict):
                historical = {}
            attempts = int(historical.get("attempts", 0))
            failure_rate = float(historical.get("failure_rate", 0.0))
            if (
                attempts >= RL_WEAK_MIN_ATTEMPTS
                and failure_rate >= RL_WEAK_FAILURE_RATE_THRESHOLD
            ):
                weak_selected_count += 1

            selected_rows.append(
                {
                    "name": scenario.name,
                    "display_name": self._scenario_display_name(scenario),
                    "name_raw": str(scenario.name),
                    "fingerprint": fingerprint,
                    "source": str(source_info.get("source", "llm_base")),
                    "mutation_strategy": str(
                        source_info.get("strategy", "") or self._extract_rl_mutation_strategy(scenario)
                    ),
                    "history_seeded": bool(self._is_history_seeded_scenario(scenario)),
                    "method": str(scenario.method).upper(),
                    "endpoint": str(scenario.endpoint),
                    "test_type": str(scenario.test_type.value),
                    "expected_status": int(scenario.expected_status),
                    "selection_reason": str(decision.get("selection_reason", "n/a")),
                    "selection_score": decision.get("score"),
                    "selection_uncertainty": decision.get("uncertainty"),
                    "historical_seen_before": bool(fingerprint in historical_patterns_before),
                    "historical_attempts_before": attempts,
                    "historical_failure_rate_before": round(failure_rate, 4),
                    "passed": bool(result.passed) if result is not None else None,
                    "verdict": str(getattr(result, "verdict", "n/a")) if result is not None else "n/a",
                    "actual_status": int(result.actual_status) if (result is not None and result.actual_status is not None) else None,
                }
            )

        selected_rows.sort(
            key=lambda item: (
                str(item.get("source", "")),
                str(item.get("selection_reason", "")),
                str(item.get("name", "")),
            )
        )

        selected_new_count = sum(
            1 for item in selected_rows if not bool(item.get("historical_seen_before", False))
        )
        selected_source_breakdown = _source_breakdown(selected_scenarios)
        candidate_source_breakdown = _source_breakdown(candidate_scenarios)
        base_source_breakdown = _source_breakdown(base_scenarios)

        return {
            "run_count_before": int(run_count_before),
            "run_count_after": int(self.learning_state.get("run_count", run_count_before)),
            "counts": {
                "base_generated": int(len(base_scenarios)),
                "candidate_total": int(len(candidate_scenarios)),
                "selected_total": int(len(selected_scenarios)),
                "selected_new_vs_history": int(selected_new_count),
                "selected_historical_weak_patterns": int(weak_selected_count),
            },
            "source_breakdown": {
                "base_generated": base_source_breakdown,
                "candidate_pool": candidate_source_breakdown,
                "selected": selected_source_breakdown,
            },
            "selected_scenarios": selected_rows[:200],
            "notes": {
                "llm_base": "Scenarios directly generated from prompt + spec.",
                "heuristic_base": "Scenarios generated by deterministic fallback when LLM generation fails or is disabled.",
                "rl_mutation": "Variants added by RL mutation strategies.",
                "rl_history_seed": "Scenarios synthesized from historical failure patterns.",
            },
        }

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

    def _runtime_cap_result(self, scenario: TestScenario) -> ScenarioExecutionResult:
        display_name = self._scenario_display_name(scenario)
        mutation_strategy = self._extract_rl_mutation_strategy(scenario)
        history_seeded = self._is_history_seeded_scenario(scenario)
        return ScenarioExecutionResult(
            name=str(scenario.name),
            test_type=str(scenario.test_type.value),
            method=str(scenario.method).upper(),
            endpoint_template=str(scenario.endpoint),
            endpoint_resolved=self._resolve_endpoint_path(
                scenario.endpoint,
                scenario.params,
                scenario.expected_status,
            ),
            expected_status=int(scenario.expected_status),
            actual_status=None,
            verdict="blocked",
            passed=False,
            duration_ms=0.0,
            error=f"runtime_cap_exceeded_{self.max_runtime_sec}s",
            response_excerpt="",
            verification={
                "passed": False,
                "verdict": "blocked",
                "status_check": {
                    "expected_status": int(scenario.expected_status),
                    "actual_status": None,
                    "corrected_expected_status": None,
                    "corrected": False,
                    "matched": False,
                },
                "contract_check": {
                    "checked": False,
                    "schema_found": False,
                    "valid": None,
                    "issues": [f"runtime_cap_exceeded_{self.max_runtime_sec}s"],
                },
            },
            display_name=display_name,
            name_raw=str(scenario.name),
            mutation_strategy=str(mutation_strategy),
            history_seeded=bool(history_seeded),
        )

    def _unsafe_action_result(self, scenario: TestScenario) -> ScenarioExecutionResult:
        display_name = self._scenario_display_name(scenario)
        mutation_strategy = self._extract_rl_mutation_strategy(scenario)
        history_seeded = self._is_history_seeded_scenario(scenario)
        return ScenarioExecutionResult(
            name=str(scenario.name),
            test_type=str(scenario.test_type.value),
            method=str(scenario.method).upper(),
            endpoint_template=str(scenario.endpoint),
            endpoint_resolved=self._resolve_endpoint_path(
                scenario.endpoint,
                scenario.params,
                scenario.expected_status,
            ),
            expected_status=int(scenario.expected_status),
            actual_status=None,
            verdict="blocked",
            passed=False,
            duration_ms=0.0,
            error=f"unsafe_action_blocked_{self.environment_profile}",
            response_excerpt="",
            verification={
                "passed": False,
                "verdict": "blocked",
                "status_check": {
                    "expected_status": int(scenario.expected_status),
                    "actual_status": None,
                    "corrected_expected_status": None,
                    "corrected": False,
                    "matched": False,
                },
                "contract_check": {
                    "checked": False,
                    "schema_found": False,
                    "valid": None,
                    "issues": [f"unsafe_action_blocked_{self.environment_profile}"],
                },
            },
            display_name=display_name,
            name_raw=str(scenario.name),
            mutation_strategy=str(mutation_strategy),
            history_seeded=bool(history_seeded),
        )

    def _is_safe_method_for_profile(self, method: str) -> bool:
        profile = str(self.environment_profile or DEFAULT_ENVIRONMENT_PROFILE).lower()
        normalized_method = str(method or "").upper()
        if profile != "prod_safe":
            return True
        return normalized_method in {"GET", "HEAD", "OPTIONS"}

    def _execute_scenarios(
        self, spec: Dict[str, Any], scenarios: List[TestScenario]
    ) -> List[ScenarioExecutionResult]:
        profile = str(self.environment_profile or DEFAULT_ENVIRONMENT_PROFILE).lower()
        if profile == "mock":
            self._execution_isolation_mode = "in_memory_fastapi_testclient"
            return self._execute_in_isolated_mock(spec, scenarios)
        self._execution_isolation_mode = "live_http_base_url"
        return self._execute_against_live_api(scenarios)

    def _sync_scenarios_with_corrected_expectations(
        self,
        scenarios: List[TestScenario],
        results: List[ScenarioExecutionResult],
    ) -> List[TestScenario]:
        # Corrected expectations are now treated as suspect outcomes and must not
        # rewrite scenario truth for downstream generated scripts or RL learning.
        return list(scenarios or [])

    def _execute_in_isolated_mock(
        self, spec: Dict[str, Any], scenarios: List[TestScenario]
    ) -> List[ScenarioExecutionResult]:
        # Imported lazily so this module can be used without server startup paths.
        from dynamic_mock_server import DynamicMockServer

        spec_copy = self.output_dir / "openapi_under_test.yaml"
        spec_copy.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

        server = DynamicMockServer(str(spec_copy), host="127.0.0.1", port=0)
        results: List[ScenarioExecutionResult] = []

        run_started = time.perf_counter()
        with TestClient(server.app) as client:
            adapter = _TestClientRequestsAdapter(client)
            for scenario in scenarios:
                if self.max_runtime_sec is not None:
                    elapsed = time.perf_counter() - run_started
                    if elapsed >= float(self.max_runtime_sec):
                        self._runtime_cap_hit = True
                        self._runtime_skipped_count += 1
                        results.append(self._runtime_cap_result(scenario))
                        continue
                result = self._execute_one_scenario(adapter, scenario)
                result = self._apply_failure_triage_and_rerun(
                    client=adapter,
                    scenario=scenario,
                    result=result,
                )
                results.append(result)

        return results

    def _execute_against_live_api(
        self, scenarios: List[TestScenario]
    ) -> List[ScenarioExecutionResult]:
        results: List[ScenarioExecutionResult] = []
        run_started = time.perf_counter()
        with _LiveRequestsAdapter(self.base_url, timeout_sec=12.0) as client:
            for scenario in scenarios:
                if self.max_runtime_sec is not None:
                    elapsed = time.perf_counter() - run_started
                    if elapsed >= float(self.max_runtime_sec):
                        self._runtime_cap_hit = True
                        self._runtime_skipped_count += 1
                        results.append(self._runtime_cap_result(scenario))
                        continue
                if not self._is_safe_method_for_profile(str(scenario.method)):
                    results.append(self._unsafe_action_result(scenario))
                    continue
                result = self._execute_one_scenario(client, scenario)
                result = self._apply_failure_triage_and_rerun(
                    client=client,
                    scenario=scenario,
                    result=result,
                )
                results.append(result)
        return results

    def _apply_failure_triage_and_rerun(
        self,
        *,
        client: Any,
        scenario: TestScenario,
        result: ScenarioExecutionResult,
    ) -> ScenarioExecutionResult:
        verification = (
            dict(result.verification) if isinstance(result.verification, dict) else {}
        )
        taxonomy = self._classify_failure_taxonomy(scenario=scenario, result=result)
        verification["failure_taxonomy"] = taxonomy

        if str(result.verdict).lower() == "pass":
            result.verification = verification
            return result
        if str(result.verdict).lower() == "blocked":
            result.verification = verification
            return result

        rerun_attempts = max(1, int(FLAKY_RERUN_MAX_ATTEMPTS)) - 1
        rerun_results: List[ScenarioExecutionResult] = []
        for _ in range(rerun_attempts):
            rerun_results.append(self._execute_one_scenario(client, scenario))

        pass_count = sum(1 for item in rerun_results if str(item.verdict).lower() == "pass")
        fail_count = len(rerun_results) - pass_count
        all_statuses = [result.actual_status] + [item.actual_status for item in rerun_results]
        all_verdicts = [str(result.verdict).lower()] + [
            str(item.verdict).lower() for item in rerun_results
        ]
        distinct_statuses = {
            int(item) for item in all_statuses if item is not None and str(item).isdigit()
        }
        distinct_verdicts = {item for item in all_verdicts if item}
        flaky = bool(
            rerun_results
            and (
                len(distinct_verdicts) > 1
                or len(distinct_statuses) > 1
                or (pass_count > 0 and fail_count > 0)
                or (pass_count > 0 and str(result.verdict).lower() != "pass")
            )
        )

        verification["flaky_check"] = {
            "reruns": int(len(rerun_results)),
            "pass_count": int(pass_count),
            "fail_count": int(fail_count),
            "statuses": [item for item in all_statuses],
            "verdicts": all_verdicts,
            "flaky": bool(flaky),
        }
        verification["repair_suggestion"] = self._derive_runtime_repair_suggestion(
            scenario=scenario,
            result=result,
            taxonomy=taxonomy,
        )
        if flaky:
            verification["flaky_check"]["stabilized_verdict"] = "suspect"
            verification["passed"] = False
            verification["verdict"] = "suspect"
            result.verdict = "suspect"
            result.passed = False
        result.verification = verification
        return result

    def _classify_failure_taxonomy(
        self,
        *,
        scenario: TestScenario,
        result: ScenarioExecutionResult,
    ) -> Dict[str, Any]:
        verdict = str(getattr(result, "verdict", "fail") or "fail").strip().lower()
        method = str(getattr(scenario, "method", "") or "").upper()
        operation_key = self._operation_key(method, str(getattr(scenario, "endpoint", "") or ""))
        op_meta = self._operation_index.get(operation_key, {})
        documented = set(op_meta.get("response_statuses", []) if isinstance(op_meta, dict) else [])
        actual = result.actual_status
        expected = int(getattr(result, "expected_status", 0) or 0)
        response_excerpt = str(getattr(result, "response_excerpt", "") or "").lower()
        error_text = str(getattr(result, "error", "") or "").lower()
        endpoint_resolved = str(getattr(result, "endpoint_resolved", "") or "")

        if verdict == "pass":
            return {"category": "none", "kind": "none", "reason": "passed", "confidence": 1.0}
        if verdict == "blocked":
            return {"category": "safety_block", "kind": "policy", "reason": "unsafe_action_blocked", "confidence": 1.0}
        if "timeout" in error_text or "connection" in error_text:
            return {"category": "environment_error", "kind": "env_issue", "reason": "transport_failure", "confidence": 0.95}
        if actual in {500, 502, 503, 504}:
            return {"category": "server_error", "kind": "service_issue", "reason": f"http_{actual}", "confidence": 0.95}
        if actual == 405 and endpoint_resolved.endswith("/"):
            return {"category": "path_param_empty_segment", "kind": "agent_issue", "reason": "resolved_path_trailing_slash", "confidence": 0.95}
        if actual is not None and actual in documented and expected not in documented:
            return {"category": "expectation_mismatch_documented", "kind": "agent_issue", "reason": "expected_not_in_documented_statuses", "confidence": 0.9}
        if "missing required field" in response_excerpt:
            return {"category": "request_body_missing_fields", "kind": "agent_issue", "reason": "payload_missing_required", "confidence": 0.85}
        return {"category": "behavioral_regression", "kind": "service_or_spec_issue", "reason": "assertion_failed", "confidence": 0.7}

    def _derive_runtime_repair_suggestion(
        self,
        *,
        scenario: TestScenario,
        result: ScenarioExecutionResult,
        taxonomy: Dict[str, Any],
    ) -> Dict[str, Any]:
        category = str(taxonomy.get("category", "") or "")
        method = str(getattr(scenario, "method", "") or "").upper()
        operation_key = self._operation_key(method, str(getattr(scenario, "endpoint", "") or ""))
        op_meta = self._operation_index.get(operation_key, {})
        documented = set(op_meta.get("response_statuses", []) if isinstance(op_meta, dict) else [])
        actual = result.actual_status

        if category in {"expectation_mismatch_documented", "path_param_empty_segment"}:
            if actual is not None and int(actual) in {int(item) for item in documented if str(item).isdigit()}:
                return {
                    "type": "override_expected_status",
                    "expected_status": int(actual),
                    "confidence": 0.9,
                    "reason": category,
                }
        if category == "request_body_missing_fields" and method in {"POST", "PUT", "PATCH"}:
            return {
                "type": "repair_request_body",
                "repair_request_body": True,
                "confidence": 0.85,
                "reason": category,
            }
        return {}

    def _ingest_runtime_repair_suggestions(
        self,
        *,
        scenarios: List[TestScenario],
        results: List[ScenarioExecutionResult],
    ) -> Dict[str, Any]:
        rules = self.learning_state.setdefault("scenario_repair_rules", {})
        if not isinstance(rules, dict):
            rules = {}
            self.learning_state["scenario_repair_rules"] = rules

        applied = 0
        examples: List[Dict[str, Any]] = []
        for scenario, result in zip(scenarios, results):
            verification = result.verification if isinstance(result.verification, dict) else {}
            suggestion = verification.get("repair_suggestion", {})
            if not isinstance(suggestion, dict) or not suggestion:
                continue
            fingerprint = self._scenario_fingerprint(scenario)
            existing = rules.get(fingerprint, {})
            rule = dict(existing) if isinstance(existing, dict) else {}
            changed = False
            if suggestion.get("type") == "override_expected_status":
                try:
                    proposed = int(suggestion.get("expected_status"))
                    if int(rule.get("override_expected_status", -1)) != proposed:
                        rule["override_expected_status"] = proposed
                        changed = True
                except Exception:
                    pass
            if bool(suggestion.get("repair_request_body")) and not bool(
                rule.get("repair_request_body")
            ):
                rule["repair_request_body"] = True
                changed = True
            if not changed:
                continue
            rule["runtime_suggested"] = True
            rule["runtime_reason"] = str(suggestion.get("reason", ""))
            rules[fingerprint] = rule
            applied += 1
            if len(examples) < RUNTIME_REPAIR_SUGGESTION_LIMIT:
                examples.append(
                    {
                        "scenario": result.name,
                        "fingerprint": fingerprint,
                        "suggestion": suggestion,
                    }
                )
        self.learning_state["scenario_repair_rules"] = rules
        return {"applied": int(applied), "examples": examples}

    def _prepare_scenarios_for_execution_and_scripts(
        self, scenarios: List[TestScenario]
    ) -> List[TestScenario]:
        prepared: List[TestScenario] = []
        for scenario in scenarios:
            method = str(scenario.method).upper()
            operation_key = self._operation_key(method, scenario.endpoint)
            op_meta = self._operation_index.get(operation_key, {})
            if isinstance(op_meta, dict):
                self._apply_intent_aware_execution_normalization(
                    scenario,
                    method=method,
                    op_meta=op_meta,
                )
                scenario.params = self._normalize_path_params_for_execution(
                    scenario,
                    op_meta=op_meta,
                )
                scenario.expected_status = self._normalize_expected_status_for_execution(
                    scenario,
                    method=method,
                    op_meta=op_meta,
                )
            rendered_headers = self._render_headers(scenario.headers)
            scenario.headers = self._normalize_auth_headers_for_execution(
                scenario, method, rendered_headers
            )
            prepared.append(scenario)
        return prepared

    def _apply_intent_aware_execution_normalization(
        self,
        scenario: TestScenario,
        *,
        method: str,
        op_meta: Dict[str, Any],
    ) -> None:
        intent = self._infer_scenario_execution_intent(
            scenario=scenario,
            method=method,
            op_meta=op_meta,
        )
        self._annotate_scenario_with_intent(scenario, intent)
        primary_intent = str(intent.get("primary", "generic") or "generic")
        operation_key = self._operation_key(method, str(scenario.endpoint))
        auth_required = operation_key in self._auth_required_ops
        placeholders = [
            str(item).strip()
            for item in PATH_PARAM_PATTERN.findall(str(scenario.endpoint or ""))
            if str(item).strip()
        ]
        path_safe_params = self._coerce_valid_path_params(
            endpoint=str(scenario.endpoint),
            params=dict(scenario.params or {}),
            op_meta=op_meta,
        )
        required_fields = [
            str(item) for item in list(op_meta.get("required_fields", []) or []) if str(item).strip()
        ]

        if primary_intent == "auth_negative":
            scenario.params = path_safe_params
            if method in {"POST", "PUT", "PATCH"}:
                self._repair_scenario_body_from_spec(scenario, op_meta)
            if auth_required:
                scenario.expected_status = 401
                scenario.headers = self._build_auth_negative_headers(scenario)
            return

        # For all non-auth intents, force a valid bearer token when auth is required.
        if auth_required:
            scenario.headers = self._build_valid_auth_headers(scenario)

        if primary_intent == "path_missing":
            params = dict(path_safe_params)
            for name in placeholders:
                raw = params.get(name)
                if raw is None or not str(raw).strip():
                    params[name] = MISSING_PATH_PARAM_SENTINEL
            scenario.params = params
            if method in {"POST", "PUT", "PATCH"}:
                self._repair_scenario_body_from_spec(scenario, op_meta)
            scenario.expected_status = self._normalize_missing_path_expected_status(op_meta=op_meta)
            return

        if primary_intent == "path_lookup_not_found":
            params = dict(path_safe_params)
            for name in placeholders:
                params[name] = "nonexistent_dependency_id"
            scenario.params = params
            if method in {"POST", "PUT", "PATCH"}:
                self._repair_scenario_body_from_spec(scenario, op_meta)
            scenario.expected_status = 404
            return

        if primary_intent == "body_missing_required":
            scenario.params = path_safe_params
            body = deepcopy(scenario.body) if isinstance(scenario.body, dict) else {}
            if not body:
                body = self._build_seed_body_from_operation_meta(op_meta, expect_negative=False)
            missing_field = required_fields[0] if required_fields else ""
            if missing_field:
                body.pop(missing_field, None)
            scenario.body = body
            scenario.expected_status = 400
            return

        if primary_intent == "query_validation":
            params = dict(path_safe_params)
            has_non_path_query = any(str(k) not in set(placeholders) for k in params.keys())
            if not has_non_path_query:
                params["q"] = "'; DROP TABLE users; --"
            scenario.params = params
            if method in {"POST", "PUT", "PATCH"}:
                self._repair_scenario_body_from_spec(scenario, op_meta)
            scenario.expected_status = 400
            return

        if primary_intent in {"happy_path", "integration"}:
            scenario.params = path_safe_params
            if method in {"POST", "PUT", "PATCH"}:
                self._repair_scenario_body_from_spec(scenario, op_meta)
            if int(getattr(scenario, "expected_status", 0) or 0) >= 400:
                scenario.expected_status = self._pick_happy_status_for_operation(method, op_meta)
            return

        # Generic fallback: prefer orthogonal negatives (single-fault execution).
        if bool(intent.get("path_conflict", False)):
            scenario.params = path_safe_params
        if method in {"POST", "PUT", "PATCH"} and bool(intent.get("body_conflict", False)):
            self._repair_scenario_body_from_spec(scenario, op_meta)

    def _infer_scenario_execution_intent(
        self,
        *,
        scenario: TestScenario,
        method: str,
        op_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        scenario_type = str(getattr(scenario.test_type, "value", scenario.test_type)).lower()
        strategy = str(self._extract_rl_mutation_strategy(scenario))
        text = " ".join(
            [
                str(getattr(scenario, "name", "") or ""),
                str(getattr(scenario, "description", "") or ""),
                " ".join(str(item) for item in list(getattr(scenario, "assertions", []) or [])),
            ]
        ).lower()
        placeholders = [
            str(item).strip()
            for item in PATH_PARAM_PATTERN.findall(str(scenario.endpoint or ""))
            if str(item).strip()
        ]
        params = dict(scenario.params or {})
        path_param_schemas = (
            op_meta.get("path_param_schemas", {})
            if isinstance(op_meta.get("path_param_schemas", {}), dict)
            else {}
        )
        query_names = set(str(item) for item in list(op_meta.get("query_param_names", []) or []))
        required_fields = [
            str(item) for item in list(op_meta.get("required_fields", []) or []) if str(item).strip()
        ]
        body = scenario.body if isinstance(scenario.body, dict) else {}
        operation_key = self._operation_key(method, str(scenario.endpoint))
        auth_required = operation_key in self._auth_required_ops

        missing_path_names: List[str] = []
        invalid_path_values: List[str] = []
        for name in placeholders:
            raw = params.get(name)
            raw_text = str(raw).strip() if raw is not None else ""
            if raw is None or raw_text == "":
                missing_path_names.append(name)
                continue
            lower_value = raw_text.lower()
            if (
                raw_text == MISSING_PATH_PARAM_SENTINEL
                or any(token in lower_value for token in ("nonexistent", "notfound", "missing", "invalid"))
                or bool(re.fullmatch(r"9{3,}", raw_text))
            ):
                invalid_path_values.append(name)
                continue
            schema = path_param_schemas.get(name, {}) if isinstance(path_param_schemas, dict) else {}
            field_type = str((schema or {}).get("type", "")).strip().lower()
            if field_type in {"integer", "number"} and not raw_text.lstrip("-").isdigit():
                invalid_path_values.append(name)

        missing_required_body = [
            field
            for field in required_fields
            if field not in body or body.get(field) in (None, "")
        ]
        unknown_query = [
            str(key)
            for key in params.keys()
            if str(key) not in set(placeholders) and str(key) not in query_names
        ]

        auth_intent = bool(
            auth_required
            and (
                self._is_auth_negative_scenario(scenario)
                or strategy in {"missing_auth", "invalid_auth"}
                or "unauthorized" in text
            )
        )
        path_missing_intent = bool(
            missing_path_names
            and self._scenario_targets_missing_path_parameter(
                scenario=scenario,
                missing_placeholders=missing_path_names,
            )
        )
        path_lookup_intent = bool(
            strategy == "path_not_found"
            or (invalid_path_values and not path_missing_intent)
            or (
                placeholders
                and int(getattr(scenario, "expected_status", 0) or 0) == 404
                and scenario_type in {"input_validation", "error_handling", "edge_cases", "boundary_testing"}
            )
        )
        body_missing_intent = bool(
            method in {"POST", "PUT", "PATCH"}
            and (
                strategy in {"missing_required_field", "learned_missing_required_field", "learned_required_null"}
                or (
                    scenario_type in {"input_validation", "error_handling", "boundary_testing", "security"}
                    and bool(missing_required_body)
                )
            )
        )
        query_validation_intent = bool(
            strategy in {"query_fuzz", "query_boundary_extreme"}
            or (
                method == "GET"
                and scenario_type in {"input_validation", "error_handling", "boundary_testing", "security"}
                and bool(unknown_query)
            )
        )

        primary = "generic_negative"
        if auth_intent:
            primary = "auth_negative"
        elif path_missing_intent:
            primary = "path_missing"
        elif path_lookup_intent:
            primary = "path_lookup_not_found"
        elif body_missing_intent:
            primary = "body_missing_required"
        elif query_validation_intent:
            primary = "query_validation"
        elif scenario_type in {"happy_path", "performance"}:
            primary = "happy_path"
        elif scenario_type == "integration":
            primary = "integration"

        active_axes = int(auth_intent) + int(path_missing_intent or path_lookup_intent) + int(body_missing_intent) + int(query_validation_intent)
        return {
            "primary": primary,
            "strategy": strategy,
            "missing_path_names": missing_path_names,
            "invalid_path_values": invalid_path_values,
            "missing_required_body": missing_required_body,
            "unknown_query": unknown_query,
            "active_axes": int(active_axes),
            "path_conflict": bool(primary not in {"path_missing", "path_lookup_not_found"} and (missing_path_names or invalid_path_values)),
            "body_conflict": bool(primary != "body_missing_required" and bool(missing_required_body)),
        }

    def _annotate_scenario_with_intent(self, scenario: TestScenario, intent: Dict[str, Any]) -> None:
        assertions = list(scenario.assertions or [])
        filtered = [
            item
            for item in assertions
            if not str(item).startswith("intent_primary:") and not str(item).startswith("intent_axes:")
        ]
        primary = str(intent.get("primary", "generic_negative"))
        axes = int(intent.get("active_axes", 0) or 0)
        filtered.append(f"intent_primary:{primary}")
        filtered.append(f"intent_axes:{axes}")
        if bool(intent.get("path_conflict", False)):
            filtered.append("intent_path_conflict")
        if bool(intent.get("body_conflict", False)):
            filtered.append("intent_body_conflict")
        scenario.assertions = filtered

    def _build_valid_auth_headers(self, scenario: TestScenario) -> Dict[str, str]:
        headers = dict(self._render_headers(scenario.headers))
        headers["Authorization"] = "Bearer valid_token_123"
        headers.pop("authorization", None)
        return headers

    def _build_auth_negative_headers(self, scenario: TestScenario) -> Dict[str, str]:
        headers = dict(self._render_headers(scenario.headers))
        strategy = str(self._extract_rl_mutation_strategy(scenario))
        scenario_text = " ".join(
            [
                str(getattr(scenario, "name", "") or ""),
                str(getattr(scenario, "description", "") or ""),
            ]
        ).lower()
        missing_hint = strategy == "missing_auth" or "missing auth" in scenario_text or "without auth" in scenario_text
        if missing_hint:
            headers.pop("Authorization", None)
            headers.pop("authorization", None)
            return headers
        headers["Authorization"] = "Bearer invalid"
        headers.pop("authorization", None)
        return headers

    def _normalize_expected_status_for_execution(
        self,
        scenario: TestScenario,
        *,
        method: str,
        op_meta: Dict[str, Any],
    ) -> int:
        test_type_value = str(getattr(scenario.test_type, "value", scenario.test_type)).lower()
        normalized_status = self._normalize_expected_status_for_test_type(
            test_type=test_type_value,
            expected_status=int(scenario.expected_status),
        )
        response_statuses = set(op_meta.get("response_statuses", []) or [])
        operation_key = self._operation_key(method, scenario.endpoint)
        auth_required = operation_key in self._auth_required_ops
        if (
            auth_required
            and self._is_auth_negative_scenario(scenario)
            and int(normalized_status) in AUTH_NEGATIVE_STATUS_CODES
        ):
            # Keep auth-negative expectations stable even if undocumented in spec.
            return int(normalized_status)

        # Boundary cases that still satisfy the schema should assert the happy
        # path status instead of forcing a 4xx expectation.
        if (
            scenario.test_type == TestType.BOUNDARY_TESTING
            and int(normalized_status) in NEGATIVE_STATUS_CODES
            and method in {"POST", "PUT", "PATCH"}
            and isinstance(scenario.body, dict)
            and self._request_body_satisfies_schema(
                body=dict(scenario.body or {}),
                request_schema=op_meta.get("request_schema", {}),
            )
        ):
            return int(self._pick_happy_status_for_operation(method, op_meta))

        # Some APIs allow unknown/extra JSON fields (additionalProperties=true).
        # If payload is otherwise schema-valid, treat these as happy-path writes.
        if (
            int(normalized_status) in NEGATIVE_STATUS_CODES
            and method in {"POST", "PUT", "PATCH"}
            and isinstance(scenario.body, dict)
            and self._payload_has_only_extra_fields(
                body=dict(scenario.body or {}),
                request_schema=op_meta.get("request_schema", {}),
            )
            and self._request_body_satisfies_schema(
                body=dict(scenario.body or {}),
                request_schema=op_meta.get("request_schema", {}),
            )
        ):
            return int(self._pick_happy_status_for_operation(method, op_meta))

        # Unconstrained string path params often map to resource lookup misses
        # (404), not format-validation errors (400).
        if (
            scenario.test_type in {TestType.INPUT_VALIDATION, TestType.BOUNDARY_TESTING}
            and int(normalized_status) == 400
            and method in {"GET", "DELETE", "PATCH", "PUT"}
        ):
            path_schemas = op_meta.get("path_param_schemas", {})
            if (
                isinstance(path_schemas, dict)
                and path_schemas
                and 404 in response_statuses
                and self._path_params_are_unconstrained_strings(path_schemas)
            ):
                return 404

        # Edge-case probes on unconstrained string IDs should not expect 400
        # when the operation only documents 2xx/404 semantics.
        if (
            test_type_value == "edge_cases"
            and int(normalized_status) == 400
            and method in {"GET", "DELETE", "PATCH", "PUT"}
        ):
            path_schemas = op_meta.get("path_param_schemas", {})
            if (
                isinstance(path_schemas, dict)
                and path_schemas
                and self._path_params_are_unconstrained_strings(path_schemas)
                and 400 not in response_statuses
                and any(200 <= int(code) < 300 for code in response_statuses)
            ):
                return int(self._pick_happy_status_for_operation(method, op_meta))

        # Reconcile expectations with documented statuses for the selected
        # operation to reduce false negatives from mismatched assumptions.
        if response_statuses and int(normalized_status) not in response_statuses:
            # If operation exists for this method/path, 405 should not be
            # expected unless explicitly documented.
            if int(normalized_status) == 405:
                return int(self._pick_happy_status_for_operation(method, op_meta))

            negative_preference = [400, 401, 403, 404, 409, 422]
            if int(normalized_status) in NEGATIVE_STATUS_CODES:
                for code in negative_preference:
                    if code in response_statuses:
                        return int(code)
            return int(self._pick_happy_status_for_operation(method, op_meta))

        return int(normalized_status)

    def _normalize_query_params_for_execution(
        self, scenario: TestScenario, *, op_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        params = dict(scenario.params or {})
        placeholders = set(PATH_PARAM_PATTERN.findall(str(scenario.endpoint or "")))
        query_names = set(str(item) for item in (op_meta.get("query_param_names", []) or []))

        path_only = {k: v for k, v in params.items() if str(k) in placeholders}
        query_only = {k: v for k, v in params.items() if str(k) not in placeholders}

        expected_status = int(getattr(scenario, "expected_status", 0) or 0)
        scenario_type = str(getattr(scenario.test_type, "value", scenario.test_type)).lower()
        positive_scenario = 200 <= expected_status < 300

        if positive_scenario:
            # Keep happy/integration/performance scenarios deterministic by
            # removing unknown query params and coercing known params to safe values.
            filtered: Dict[str, Any] = {}
            for key, value in query_only.items():
                if str(key) in query_names:
                    filtered[str(key)] = value
            for qname in sorted(query_names):
                if qname in filtered:
                    raw_value = filtered[qname]
                    qlower = str(qname).lower()
                    if ("limit" in qlower or "size" in qlower) and str(raw_value).isdigit():
                        parsed = int(str(raw_value))
                        if parsed <= 0 or parsed > 1000:
                            filtered[qname] = 10
                    elif ("page" in qlower or "offset" in qlower) and str(raw_value).lstrip("-").isdigit():
                        parsed = int(str(raw_value))
                        if parsed < 0:
                            filtered[qname] = 0 if "offset" in qlower else 1
            return {**path_only, **filtered}

        # For negative scenarios, keep generated params intact to preserve
        # fuzzing power for validation/security/error checks.
        if scenario_type in {"security", "input_validation", "boundary_testing", "error_handling"}:
            return params
        return params

    def _normalize_path_params_for_execution(
        self, scenario: TestScenario, *, op_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        params = dict(scenario.params or {})
        placeholders = [
            str(item).strip()
            for item in PATH_PARAM_PATTERN.findall(str(scenario.endpoint or ""))
            if str(item).strip()
        ]
        if not placeholders:
            return params

        scenario_type = str(getattr(scenario.test_type, "value", scenario.test_type)).lower()
        expected_status = int(getattr(scenario, "expected_status", 0) or 0)
        path_param_schemas = (
            op_meta.get("path_param_schemas", {})
            if isinstance(op_meta.get("path_param_schemas", {}), dict)
            else {}
        )
        missing_placeholders: List[str] = []
        for name in placeholders:
            value = params.get(name)
            if value is None:
                missing_placeholders.append(name)
                continue
            if isinstance(value, str) and not value.strip():
                missing_placeholders.append(name)

        if missing_placeholders:
            if scenario_type in {"authentication", "authorization"} or expected_status in AUTH_NEGATIVE_STATUS_CODES:
                for name in missing_placeholders:
                    params[name] = self._default_path_param_value(
                        name,
                        path_param_schemas.get(name, {}),
                    )
            elif self._scenario_targets_missing_path_parameter(
                scenario=scenario,
                missing_placeholders=missing_placeholders,
            ):
                for name in missing_placeholders:
                    params[name] = MISSING_PATH_PARAM_SENTINEL
                scenario.expected_status = self._normalize_missing_path_expected_status(op_meta=op_meta)
                assertions = list(scenario.assertions or [])
                if "path_param_missing_intent" not in assertions:
                    assertions.append("path_param_missing_intent")
                scenario.assertions = assertions
                expected_status = int(getattr(scenario, "expected_status", expected_status) or expected_status)
            else:
                # Missing path params are usually generation artifacts for unrelated
                # scenarios; default them so intent stays aligned.
                for name in missing_placeholders:
                    params[name] = self._default_path_param_value(
                        name,
                        path_param_schemas.get(name, {}),
                    )

        if expected_status != 404:
            return params

        for name in placeholders:
            raw_value = str(params.get(name, "")).strip()
            if raw_value == MISSING_PATH_PARAM_SENTINEL:
                continue
            lower_value = raw_value.lower()
            has_not_found_hint = any(
                token in lower_value for token in ("nonexistent", "notfound", "missing", "invalid")
            ) or bool(re.fullmatch(r"9{3,}", raw_value))
            if not has_not_found_hint:
                params[name] = "nonexistent_dependency_id"
        return params

    def _default_path_param_value(self, name: str, schema: Optional[Dict[str, Any]] = None) -> str:
        raw_schema = schema if isinstance(schema, dict) else {}
        field_type = str(raw_schema.get("type", "")).strip().lower()
        if field_type in {"integer", "number"}:
            return "123"
        if field_type == "boolean":
            return "true"
        lowered = str(name or "").strip().lower()
        if lowered.endswith("id") or lowered == "id":
            return "123"
        return "sample"

    def _scenario_targets_missing_path_parameter(
        self,
        *,
        scenario: TestScenario,
        missing_placeholders: List[str],
    ) -> bool:
        if not missing_placeholders:
            return False
        scenario_type = str(getattr(scenario.test_type, "value", scenario.test_type)).lower()
        if scenario_type in {"authentication", "authorization"}:
            return False
        try:
            expected_status = int(getattr(scenario, "expected_status", 0) or 0)
        except Exception:
            expected_status = 0
        if expected_status in {404, 405}:
            return True

        name = str(getattr(scenario, "name", "") or "")
        description = str(getattr(scenario, "description", "") or "")
        assertions = " ".join(str(item) for item in list(getattr(scenario, "assertions", []) or []))
        haystack = " ".join([name, description, assertions]).lower()
        haystack = re.sub(r"\s+", " ", haystack).strip()
        phrase_patterns = (
            r"\bmissing\b.{0,40}\bpath\b.{0,20}\bparam",
            r"\bwithout\b.{0,40}\bpath\b.{0,20}\bparam",
            r"\bempty\b.{0,40}\bpath\b.{0,20}\bparam",
            r"\bpath\b.{0,25}\bparameter\b",
        )
        if any(re.search(pattern, haystack) for pattern in phrase_patterns):
            return True

        for pname in missing_placeholders:
            pname_l = str(pname).strip().lower()
            if not pname_l:
                continue
            if re.search(rf"\bmissing\b.{0,30}\b{re.escape(pname_l)}\b", haystack):
                return True
            if re.search(rf"\bwithout\b.{0,30}\b{re.escape(pname_l)}\b", haystack):
                return True
        return False

    def _normalize_missing_path_expected_status(self, *, op_meta: Dict[str, Any]) -> int:
        response_statuses = {
            int(item)
            for item in list((op_meta or {}).get("response_statuses", []) or [])
            if str(item).isdigit()
        }
        if 405 in response_statuses:
            return 405
        if 404 in response_statuses:
            return 404
        return 404

    def _path_params_are_unconstrained_strings(
        self, path_param_schemas: Dict[str, Any]
    ) -> bool:
        if not isinstance(path_param_schemas, dict) or not path_param_schemas:
            return False
        for schema_raw in path_param_schemas.values():
            schema = schema_raw if isinstance(schema_raw, dict) else {}
            field_type = str(schema.get("type", "string")).strip().lower() or "string"
            if field_type != "string":
                return False
            if any(
                key in schema
                for key in (
                    "enum",
                    "pattern",
                    "format",
                    "minLength",
                    "maxLength",
                    "minimum",
                    "maximum",
                    "exclusiveMinimum",
                    "exclusiveMaximum",
                )
            ):
                return False
        return True

    def _payload_has_only_extra_fields(self, *, body: Dict[str, Any], request_schema: Any) -> bool:
        if not isinstance(body, dict):
            return False
        schema = request_schema if isinstance(request_schema, dict) else {}
        properties = schema.get("properties", {}) if isinstance(schema.get("properties", {}), dict) else {}
        if not properties:
            return False
        required_fields = [str(item) for item in (schema.get("required", []) or []) if str(item).strip()]
        if not required_fields:
            return False
        if any(field not in body or body.get(field) in (None, "") for field in required_fields):
            return False
        extra_fields = [key for key in body.keys() if str(key) not in properties]
        if not extra_fields:
            return False
        known_fields_valid = True
        for field_name, value in body.items():
            field_schema = properties.get(str(field_name))
            if not isinstance(field_schema, dict):
                continue
            if not self._value_satisfies_schema(value, field_schema):
                known_fields_valid = False
                break
        return bool(known_fields_valid)

    def _request_body_satisfies_schema(
        self, *, body: Dict[str, Any], request_schema: Any
    ) -> bool:
        if not isinstance(body, dict):
            return False
        schema = request_schema if isinstance(request_schema, dict) else {}
        if not schema:
            return False

        properties = schema.get("properties", {})
        if not isinstance(properties, dict) or not properties:
            return False

        required_fields = [
            str(item) for item in (schema.get("required", []) or []) if str(item).strip()
        ]
        for field in required_fields:
            if field not in body or body.get(field) is None:
                return False

        additional_properties = schema.get("additionalProperties", True)
        for field_name, value in body.items():
            field_schema = properties.get(str(field_name))
            if not isinstance(field_schema, dict):
                if additional_properties is False:
                    return False
                continue
            if not self._value_satisfies_schema(value, field_schema):
                return False
        return True

    def _value_satisfies_schema(self, value: Any, schema: Dict[str, Any]) -> bool:
        if not isinstance(schema, dict):
            return True
        if value is None:
            return bool(schema.get("nullable", False))

        enum_values = schema.get("enum", [])
        if isinstance(enum_values, list) and enum_values and value not in enum_values:
            return False

        field_type = str(schema.get("type", "")).strip().lower()
        if field_type == "integer":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            numeric = float(value)
            if int(numeric) != numeric:
                return False
            return self._numeric_value_satisfies_schema(numeric, schema)

        if field_type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            return self._numeric_value_satisfies_schema(float(value), schema)

        if field_type == "string":
            if not isinstance(value, str):
                return False
            min_length = schema.get("minLength")
            max_length = schema.get("maxLength")
            if isinstance(min_length, int) and len(value) < min_length:
                return False
            if isinstance(max_length, int) and len(value) > max_length:
                return False
            pattern = schema.get("pattern")
            if isinstance(pattern, str):
                try:
                    if re.fullmatch(pattern, value) is None:
                        return False
                except re.error:
                    pass
            return True

        if field_type == "boolean":
            return isinstance(value, bool)

        if field_type == "array":
            if not isinstance(value, list):
                return False
            min_items = schema.get("minItems")
            max_items = schema.get("maxItems")
            if isinstance(min_items, int) and len(value) < min_items:
                return False
            if isinstance(max_items, int) and len(value) > max_items:
                return False
            item_schema = schema.get("items", {})
            if isinstance(item_schema, dict):
                return all(self._value_satisfies_schema(item, item_schema) for item in value)
            return True

        if field_type == "object":
            return isinstance(value, dict)

        return True

    def _numeric_value_satisfies_schema(self, value: float, schema: Dict[str, Any]) -> bool:
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        exclusive_maximum = schema.get("exclusiveMaximum")

        if minimum is not None and value < float(minimum):
            return False
        if maximum is not None and value > float(maximum):
            return False

        if isinstance(exclusive_minimum, (int, float)):
            if value <= float(exclusive_minimum):
                return False
        elif exclusive_minimum is True and minimum is not None and value <= float(minimum):
            return False

        if isinstance(exclusive_maximum, (int, float)):
            if value >= float(exclusive_maximum):
                return False
        elif exclusive_maximum is True and maximum is not None and value >= float(maximum):
            return False

        return True

    def _execute_one_scenario(
        self, client: Any, scenario: TestScenario
    ) -> ScenarioExecutionResult:
        method = scenario.method.upper()
        display_name = self._scenario_display_name(scenario)
        mutation_strategy = self._extract_rl_mutation_strategy(scenario)
        history_seeded = self._is_history_seeded_scenario(scenario)
        endpoint_resolved = self._resolve_endpoint_path(
            scenario.endpoint, scenario.params, scenario.expected_status
        )
        headers = self._render_headers(scenario.headers)
        headers = self._normalize_auth_headers_for_execution(scenario, method, headers)
        method_key = self._operation_key(method, scenario.endpoint)
        op_meta = self._operation_index.get(method_key, {})
        if not isinstance(op_meta, dict):
            op_meta = {}
        normalized_params = self._normalize_query_params_for_execution(
            scenario,
            op_meta=op_meta,
        )
        query_params = self._strip_path_params(scenario.endpoint, normalized_params)
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
            verification = self._verify_then_correct_result(
                scenario=scenario,
                actual_status=actual_status,
                response=response,
                query_params=query_params,
                body=body,
            )
            intent_meta = self._scenario_intent_metadata(scenario)
            if intent_meta:
                verification["intent"] = intent_meta
            verdict = str(verification.get("verdict", "fail"))
            passed = bool(verdict == "pass")
            response_excerpt = self._response_excerpt(response)

            return ScenarioExecutionResult(
                name=scenario.name,
                test_type=scenario.test_type.value,
                method=method,
                endpoint_template=scenario.endpoint,
                endpoint_resolved=endpoint_resolved,
                expected_status=int(scenario.expected_status),
                actual_status=actual_status,
                verdict=verdict,
                passed=passed,
                duration_ms=round(duration_ms, 3),
                response_excerpt=response_excerpt,
                verification=verification,
                display_name=display_name,
                name_raw=str(scenario.name),
                mutation_strategy=str(mutation_strategy),
                history_seeded=bool(history_seeded),
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
                verdict="fail",
                passed=False,
                duration_ms=round(duration_ms, 3),
                error=str(exc),
                response_excerpt="",
                verification={
                    "passed": False,
                    "verdict": "fail",
                    "status_check": {
                        "expected_status": int(scenario.expected_status),
                        "actual_status": None,
                        "corrected_expected_status": None,
                        "corrected": False,
                        "matched": False,
                    },
                    "contract_check": {
                        "checked": False,
                        "schema_found": False,
                        "valid": None,
                        "issues": [str(exc)],
                    },
                },
                display_name=display_name,
                name_raw=str(scenario.name),
                mutation_strategy=str(mutation_strategy),
                history_seeded=bool(history_seeded),
            )

    def _scenario_intent_metadata(self, scenario: TestScenario) -> Dict[str, Any]:
        primary = ""
        axes = 0
        path_conflict = False
        body_conflict = False
        for item in list(scenario.assertions or []):
            raw = str(item or "").strip()
            if raw.startswith("intent_primary:"):
                primary = raw.split(":", 1)[1].strip()
            elif raw.startswith("intent_axes:"):
                value = raw.split(":", 1)[1].strip()
                if str(value).isdigit():
                    axes = int(value)
            elif raw == "intent_path_conflict":
                path_conflict = True
            elif raw == "intent_body_conflict":
                body_conflict = True
        if not primary and axes <= 0 and not path_conflict and not body_conflict:
            return {}
        return {
            "primary": primary or "unknown",
            "axes": int(axes),
            "path_conflict": bool(path_conflict),
            "body_conflict": bool(body_conflict),
        }

    def _resolve_endpoint_path(
        self, endpoint: str, params: Dict[str, Any], expected_status: int
    ) -> str:
        resolved = endpoint
        for param_name in PATH_PARAM_PATTERN.findall(endpoint):
            value = params.get(param_name)
            if value is None:
                value = "999" if expected_status == 404 else "123"
            placeholder = "{" + param_name + "}"
            if str(value).strip() == MISSING_PATH_PARAM_SENTINEL:
                resolved = resolved.replace("/" + placeholder, "")
                resolved = resolved.replace(placeholder + "/", "")
                resolved = resolved.replace(placeholder, "")
                continue
            resolved = resolved.replace(placeholder, str(value))
        resolved = re.sub(r"/{2,}", "/", str(resolved or ""))
        if not resolved.startswith("/"):
            resolved = "/" + resolved
        if not resolved:
            resolved = "/"
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
                token_lower = token.lower()
                token_is_invalid_hint = (
                    token_lower in {"invalid", "expired", ""}
                    or any(marker in token_lower for marker in ("invalid", "expired"))
                )
                try:
                    expected_status = int(getattr(scenario, "expected_status", 0) or 0)
                except Exception:
                    expected_status = 0
                # For expected 401 auth-negative checks, force an invalid token
                # whenever a bearer token is present to avoid false-positive 2xx.
                if token_is_invalid_hint or expected_status == 401:
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
        return int(actual_status) == int(scenario.expected_status)

    def _response_excerpt(self, response) -> str:
        try:
            payload = response.json()
            text = json.dumps(payload, ensure_ascii=True)
        except Exception:
            text = response.text or ""
        if len(text) > 300:
            text = text[:300] + "..."
        return text

    def _verify_then_correct_result(
        self,
        *,
        scenario: TestScenario,
        actual_status: int,
        response: Any,
        query_params: Dict[str, Any],
        body: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        expected_status = int(scenario.expected_status)
        status_match = self._status_matches_expectation(scenario, actual_status)
        corrected_expected_status: Optional[int] = None
        corrected = False

        operation_key = self._operation_key(scenario.method.upper(), scenario.endpoint)
        op_meta = self._operation_index.get(operation_key, {})
        documented_statuses = {
            int(item)
            for item in (op_meta.get("response_statuses", []) or [])
            if isinstance(item, int) or str(item).isdigit()
        }
        if (
            not status_match
            and documented_statuses
            and expected_status not in documented_statuses
            and actual_status in documented_statuses
        ):
            corrected_expected_status = int(actual_status)
            corrected = True

        contract_check = self._verify_response_contract(
            scenario=scenario,
            actual_status=actual_status,
            response=response,
            query_params=query_params,
            body=body,
            operation_meta=op_meta if isinstance(op_meta, dict) else {},
        )
        contract_valid = contract_check.get("valid")
        if corrected:
            verdict = "suspect"
            suspect_reason = "status_auto_correction"
        elif not status_match:
            verdict = "fail"
            suspect_reason = ""
        elif contract_valid is False:
            verdict = "fail"
            suspect_reason = ""
        else:
            verdict = "pass"
            suspect_reason = ""

        return {
            "passed": bool(verdict == "pass"),
            "verdict": verdict,
            "status_check": {
                "expected_status": expected_status,
                "actual_status": int(actual_status),
                "corrected_expected_status": corrected_expected_status,
                "corrected": bool(corrected),
                "matched": bool(status_match),
                "suspect_reason": suspect_reason,
                "documented_statuses": sorted(documented_statuses),
            },
            "contract_check": contract_check,
        }

    def _verify_response_contract(
        self,
        *,
        scenario: TestScenario,
        actual_status: int,
        response: Any,
        query_params: Dict[str, Any],
        body: Optional[Dict[str, Any]],
        operation_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        response_schemas = (
            operation_meta.get("response_schemas", {})
            if isinstance(operation_meta.get("response_schemas", {}), dict)
            else {}
        )
        schema = response_schemas.get(str(actual_status))
        if not isinstance(schema, dict):
            schema = response_schemas.get("default")

        result = {
            "checked": False,
            "schema_found": bool(isinstance(schema, dict) and schema),
            "valid": None,
            "issues": [],
        }
        if not result["schema_found"]:
            # Fallback sanity check so contract metrics are still informative
            # when specs omit response schemas.
            result["checked"] = True
            if int(actual_status) == 204:
                body_text = str(getattr(response, "text", "") or "").strip()
                result["valid"] = body_text == ""
                if body_text:
                    result["issues"].append(
                        "no_response_schema_defined_expected_empty_body_for_204"
                    )
                else:
                    result["issues"].append(
                        "no_response_schema_defined_used_empty_body_sanity_check"
                    )
                return result
            try:
                payload = response.json()
                result["valid"] = isinstance(payload, (dict, list))
                if result["valid"]:
                    result["issues"].append(
                        "no_response_schema_defined_used_json_sanity_check"
                    )
                else:
                    result["issues"].append(
                        "no_response_schema_defined_non_container_json_payload"
                    )
            except Exception:
                result["valid"] = False
                result["issues"].append(
                    "no_response_schema_defined_response_not_json"
                )
            return result

        payload: Any = None
        try:
            payload = response.json()
        except Exception:
            payload = None

        try:
            from jsonschema import validate as jsonschema_validate
            from jsonschema import ValidationError as JSONSchemaValidationError
        except Exception:
            result["issues"].append("jsonschema_not_available")
            return result

        result["checked"] = True
        try:
            jsonschema_validate(instance=payload, schema=schema)
            result["valid"] = True
        except JSONSchemaValidationError as exc:
            result["valid"] = False
            result["issues"].append(
                f"response_schema_validation_failed: {str(exc.message)}"
            )
        except Exception as exc:
            result["valid"] = False
            result["issues"].append(f"response_schema_validation_error: {str(exc)}")
        return result

    def _build_summary(
        self,
        spec: Dict[str, Any],
        scenarios: List[TestScenario],
        results: List[ScenarioExecutionResult],
    ) -> Dict[str, Any]:
        total = len(results)
        verdict_counter: Counter[str] = Counter(
            str(getattr(r, "verdict", "") or "fail").strip().lower() for r in results
        )
        passed = int(verdict_counter.get("pass", 0))
        suspect = int(verdict_counter.get("suspect", 0))
        blocked = int(verdict_counter.get("blocked", 0))
        hard_failed = max(0, total - passed - suspect - blocked)
        failed = max(0, total - passed)
        pass_rate = (passed / total) if total else 0.0
        avg_duration_ms = (
            sum(r.duration_ms for r in results) / total if total else 0.0
        )

        by_type: Dict[str, Dict[str, Any]] = {}
        contract_checked = 0
        contract_failed = 0
        corrected_expectations = 0
        unsafe_actions_blocked = 0
        flaky_count = 0
        failure_taxonomy_breakdown: Dict[str, int] = {}
        for result in results:
            item = by_type.setdefault(
                result.test_type,
                {"total": 0, "passed": 0, "failed": 0, "suspect": 0, "blocked": 0},
            )
            item["total"] += 1
            verdict = str(getattr(result, "verdict", "fail") or "fail").strip().lower()
            if verdict == "pass":
                item["passed"] += 1
            elif verdict == "suspect":
                item["suspect"] += 1
            elif verdict == "blocked":
                item["blocked"] += 1
            else:
                item["failed"] += 1
            verification = result.verification if isinstance(result.verification, dict) else {}
            contract = verification.get("contract_check", {}) if isinstance(verification.get("contract_check", {}), dict) else {}
            status_check = verification.get("status_check", {}) if isinstance(verification.get("status_check", {}), dict) else {}
            if bool(contract.get("checked", False)):
                contract_checked += 1
                if contract.get("valid") is False:
                    contract_failed += 1
            if bool(status_check.get("corrected", False)):
                corrected_expectations += 1
            if str(result.error or "").startswith("unsafe_action_blocked"):
                unsafe_actions_blocked += 1
            flaky_check = verification.get("flaky_check", {}) if isinstance(verification.get("flaky_check", {}), dict) else {}
            if bool(flaky_check.get("flaky", False)):
                flaky_count += 1
            taxonomy = verification.get("failure_taxonomy", {}) if isinstance(verification.get("failure_taxonomy", {}), dict) else {}
            category = str(taxonomy.get("category", "") or "").strip().lower()
            if category and category != "none":
                failure_taxonomy_breakdown[category] = int(
                    failure_taxonomy_breakdown.get(category, 0) + 1
                )

        detected_endpoints = 0
        if isinstance(spec.get("paths"), dict):
            for _, path_info in spec["paths"].items():
                if isinstance(path_info, dict):
                    detected_endpoints += sum(
                        1
                        for method in path_info.keys()
                        if method.lower() in {"get", "post", "put", "patch", "delete"}
                    )

        quality_gate_fail_reasons: List[str] = []
        quality_gate_warnings: List[str] = []
        if pass_rate < self.pass_threshold:
            quality_gate_fail_reasons.append("pass_rate_below_threshold")
        if self._llm_generation_degraded:
            llm_gate_blocking, llm_degradation_policy = self._llm_degradation_quality_gate_policy(
                spec=spec,
                scenarios=scenarios,
                pass_rate=float(pass_rate),
            )
            if llm_gate_blocking:
                quality_gate_fail_reasons.append("llm_generation_degraded")
            else:
                quality_gate_warnings.append("llm_generation_degraded_auto_fallback")
        else:
            llm_degradation_policy = self._llm_degradation_quality_gate_policy(
                spec=spec,
                scenarios=scenarios,
                pass_rate=float(pass_rate),
            )[1]
        meets_quality_gate = len(quality_gate_fail_reasons) == 0

        failed_examples = [
            {
                "name": r.name,
                "display_name": str(getattr(r, "display_name", "") or ""),
                "name_raw": str(getattr(r, "name_raw", r.name) or r.name),
                "method": r.method,
                "endpoint": r.endpoint_resolved,
                "expected_status": r.expected_status,
                "actual_status": r.actual_status,
                "verdict": str(getattr(r, "verdict", "fail") or "fail"),
                "error": r.error,
                "mutation_strategy": str(getattr(r, "mutation_strategy", "") or ""),
                "history_seeded": bool(getattr(r, "history_seeded", False)),
            }
            for r in results
            if str(getattr(r, "verdict", "fail") or "fail") != "pass"
        ][:25]

        return {
            "total_scenarios": total,
            "passed_scenarios": passed,
            "failed_scenarios": failed,
            "hard_failed_scenarios": int(hard_failed),
            "suspect_scenarios": int(suspect),
            "blocked_scenarios": int(blocked),
            "pass_rate": round(pass_rate, 4),
            "true_pass_rate": round(pass_rate, 4),
            "pass_threshold": self.pass_threshold,
            "meets_quality_gate": bool(meets_quality_gate),
            "quality_gate_fail_reasons": quality_gate_fail_reasons,
            "quality_gate_warnings": quality_gate_warnings,
            "llm_degradation_policy": llm_degradation_policy,
            "average_duration_ms": round(avg_duration_ms, 3),
            "contract_checks_run": int(contract_checked),
            "contract_check_failures": int(contract_failed),
            "corrected_expectations": int(corrected_expectations),
            "detected_endpoints": detected_endpoints,
            "scenario_count_generated": len(scenarios),
            "runtime_cap_hit": bool(self._runtime_cap_hit),
            "runtime_skipped_scenarios": int(self._runtime_skipped_count),
            "unsafe_actions_blocked": int(unsafe_actions_blocked),
            "flaky_scenarios": int(flaky_count),
            "flaky_ratio": round((flaky_count / total) if total else 0.0, 4),
            "failure_taxonomy_breakdown": {
                key: int(value)
                for key, value in sorted(
                    failure_taxonomy_breakdown.items(),
                    key=lambda item: item[0],
                )
            },
            "max_runtime_sec": self.max_runtime_sec,
            "environment_profile": self.environment_profile,
            "test_type_breakdown": by_type,
            "failed_examples": failed_examples,
        }

    def _spec_operation_keys_for_summary(self, spec: Dict[str, Any]) -> set[str]:
        operation_keys: set[str] = set()
        paths = spec.get("paths", {}) if isinstance(spec, dict) else {}
        if not isinstance(paths, dict):
            return operation_keys
        for path, path_info in paths.items():
            if not isinstance(path, str) or not isinstance(path_info, dict):
                continue
            for method in path_info.keys():
                method_upper = str(method).upper()
                if method_upper in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                    operation_keys.add(f"{method_upper} {path}")
        return operation_keys

    def _happy_path_operation_keys_for_summary(
        self,
        scenarios: List[TestScenario],
    ) -> set[str]:
        operation_keys: set[str] = set()
        for scenario in scenarios:
            method = str(getattr(scenario, "method", "") or "").upper()
            endpoint = str(getattr(scenario, "endpoint", "") or "")
            if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"} or not endpoint:
                continue
            test_type = getattr(scenario, "test_type", None)
            if isinstance(test_type, TestType):
                is_happy = test_type == TestType.HAPPY_PATH
            else:
                is_happy = str(test_type or "").strip().lower() == TestType.HAPPY_PATH.value
            expected_status = int(getattr(scenario, "expected_status", 0) or 0)
            if is_happy and 200 <= expected_status < 300:
                operation_keys.add(f"{method} {endpoint}")
        return operation_keys

    def _llm_degradation_quality_gate_policy(
        self,
        *,
        spec: Dict[str, Any],
        scenarios: List[TestScenario],
        pass_rate: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        llm_mode = str(self._scenario_llm_mode or "auto").strip().lower() or "auto"
        required_ops = self._spec_operation_keys_for_summary(spec)
        happy_ops = self._happy_path_operation_keys_for_summary(scenarios)
        coverage_complete = bool(required_ops.issubset(happy_ops)) if required_ops else bool(scenarios)
        pass_ok = float(pass_rate) >= float(self.pass_threshold)
        fallback_quality_ok = bool(pass_ok and coverage_complete and len(scenarios) > 0)

        gate_blocking = bool(self._llm_generation_degraded)
        gate_reason = "none"
        if self._llm_generation_degraded:
            if llm_mode == "auto" and fallback_quality_ok:
                gate_blocking = False
                gate_reason = "auto_mode_fallback_quality_ok"
            else:
                gate_blocking = True
                gate_reason = "strict_mode_or_fallback_quality_poor"

        policy = {
            "llm_mode": llm_mode,
            "degraded": bool(self._llm_generation_degraded),
            "gate_blocking": bool(gate_blocking),
            "gate_reason": gate_reason,
            "fallback_quality_ok": bool(fallback_quality_ok),
            "pass_rate_ok": bool(pass_ok),
            "happy_path_coverage_complete": bool(coverage_complete),
            "required_operation_count": int(len(required_ops)),
            "happy_path_operation_count": int(len(happy_ops)),
            "scenario_count": int(len(scenarios)),
        }
        return gate_blocking, policy

    def _compose_effective_prompt(
        self,
        base_prompt: Optional[str],
        memory_excerpts: List[Dict[str, str]],
        rl_focus_points: Optional[List[str]] = None,
    ) -> Optional[str]:
        gam_focus_points = self._build_gam_prompt_focus_points(
            memory_excerpts, limit=2
        )
        rl_focus = [str(item).strip() for item in (rl_focus_points or []) if str(item).strip()]

        prompt_seed = base_prompt
        if not prompt_seed and (gam_focus_points or rl_focus):
            prompt_seed = (
                "Generate comprehensive QA API tests for authentication, validation, "
                "error handling, and boundary conditions."
            )
        if not prompt_seed:
            return None

        sections: List[str] = []
        if gam_focus_points:
            sections.append(
                "Focus additionally on (GAM):\n- "
                + "\n- ".join(gam_focus_points)
            )
        if rl_focus:
            sections.append("Prioritize historically weak patterns (RL):\n- " + "\n- ".join(rl_focus))
        if not sections:
            return prompt_seed
        return prompt_seed + "\n\n" + "\n\n".join(sections)

    def _build_gam_delta_focus_point(self) -> str:
        decision_history = self.learning_state.get("decision_history", [])
        if not isinstance(decision_history, list):
            return ""
        if len(decision_history) < 2:
            run_count = int(self.learning_state.get("run_count", 0) or 0)
            if run_count <= 0:
                return ""
            return (
                f"Trend delta (run_count={run_count}): warm-start exploration for this run; "
                "prioritize one auth mutation and one dependency-order mutation."
            )

        latest = decision_history[-1] if isinstance(decision_history[-1], dict) else {}
        previous = decision_history[-2] if isinstance(decision_history[-2], dict) else {}
        latest_reward = float(latest.get("run_reward", 0.0) or 0.0)
        previous_reward = float(previous.get("run_reward", 0.0) or 0.0)
        latest_avg = float(latest.get("average_decision_reward", 0.0) or 0.0)
        previous_avg = float(previous.get("average_decision_reward", 0.0) or 0.0)
        reward_delta = latest_reward - previous_reward
        avg_delta = latest_avg - previous_avg

        if reward_delta > 0.0005:
            trend_word = "improved"
            next_action = (
                "preserve one successful mutation and add one novel auth/boundary variant "
                "to validate stability."
            )
        elif reward_delta < -0.0005:
            trend_word = "degraded"
            next_action = (
                "increase mutation diversity on auth and dependency-order paths to recover "
                "decision quality."
            )
        else:
            trend_word = "flat"
            next_action = (
                "inject one new high-uncertainty mutation to avoid policy stagnation."
            )

        return (
            f"Trend delta: run_reward {trend_word} by {reward_delta:+.4f} "
            f"(avg_decision_delta {avg_delta:+.4f}); next action: {next_action}"
        )

    def _build_gam_prompt_focus_points(
        self, memory_excerpts: List[Dict[str, str]], limit: int = 2
    ) -> List[str]:
        max_points = max(0, int(limit))
        if max_points <= 0:
            return []

        excerpt_points = self._extract_focus_points_from_excerpts(
            memory_excerpts, limit=max(1, max_points)
        )
        delta_point = self._sanitize_focus_text(self._build_gam_delta_focus_point())
        spec_signature_point = self._sanitize_focus_text(self._build_spec_signature_focus_point())

        combined: List[str] = []
        if delta_point:
            combined.append(self._smart_trim_text(delta_point, max_chars=220))
        if spec_signature_point and len(combined) < max_points:
            combined.append(self._smart_trim_text(spec_signature_point, max_chars=220))
        for point in excerpt_points:
            cleaned = self._sanitize_focus_text(str(point or ""))
            if not cleaned:
                continue
            if cleaned in combined:
                continue
            combined.append(cleaned)
            if len(combined) >= max_points:
                break

        final_points = combined[:max_points]
        self._remember_gam_focus_points(final_points)
        return final_points

    def _build_rl_prompt_focus_points(self, limit: int = 3) -> List[str]:
        scenario_stats = self._scenario_stats_for_current_spec(
            self.adaptive_policy.scenario_stats
        )
        if not scenario_stats:
            return []

        ranked: List[Dict[str, Any]] = []
        for fingerprint, stats_raw in scenario_stats.items():
            if not isinstance(stats_raw, dict):
                continue
            attempts = int(stats_raw.get("attempts", 0))
            if attempts < RL_WEAK_MIN_ATTEMPTS:
                continue
            failure_rate = float(stats_raw.get("failure_rate", 0.0))
            avg_reward = float(stats_raw.get("avg_reward", 0.0))
            # Avoid surfacing nearly-stable patterns as "historically weak".
            if (
                failure_rate < RL_WEAK_FAILURE_RATE_THRESHOLD
                and avg_reward >= 0.0
            ):
                continue

            parsed = self._parse_scenario_fingerprint(str(fingerprint))
            method = str((parsed or {}).get("method") or stats_raw.get("method", "GET")).upper()
            endpoint = str((parsed or {}).get("endpoint") or stats_raw.get("endpoint", ""))
            test_type = str(
                (parsed or {}).get("test_type") or stats_raw.get("test_type", "error_handling")
            )
            raw_expected_status = int(
                (parsed or {}).get("expected_status") or stats_raw.get("expected_status", 400)
            )
            canonical = self._canonicalize_learning_pattern(
                method=method,
                endpoint=endpoint,
                test_type=test_type,
                expected_status=raw_expected_status,
            )
            if not canonical:
                continue
            method = str(canonical.get("method", method))
            endpoint = str(canonical.get("endpoint", endpoint))
            test_type = str(canonical.get("test_type", test_type))
            expected_status = int(canonical.get("expected_status", raw_expected_status))
            if not self._scenario_belongs_to_current_spec(method, endpoint):
                continue
            priority = (
                0.70 * failure_rate
                + 0.20 * max(0.0, -avg_reward)
                + 0.10 * min(1.0, attempts / 8.0)
            )
            ranked.append(
                {
                    "method": method,
                    "endpoint": endpoint,
                    "test_type": test_type,
                    "expected_status": expected_status,
                    "attempts": attempts,
                    "failure_rate": failure_rate,
                    "priority": float(priority),
                }
            )

        ranked.sort(
            key=lambda item: (
                float(item.get("priority", 0.0)),
                int(item.get("attempts", 0)),
                float(item.get("failure_rate", 0.0)),
            ),
            reverse=True,
        )

        points: List[str] = []
        seen: set[tuple[str, str, str]] = set()
        for item in ranked:
            key = (
                str(item.get("method", "")),
                str(item.get("endpoint", "")),
                str(item.get("test_type", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            point = (
                f"{item['method']} {item['endpoint']}: reinforce {item['test_type']} "
                f"checks targeting status {item['expected_status']} (historical failure "
                f"rate {item['failure_rate']:.2f} over {item['attempts']} attempts)."
            )
            points.append(point[:220])
            if len(points) >= max(0, int(limit)):
                break
        return points

    def _extract_focus_points_from_excerpts(
        self, memory_excerpts: List[Dict[str, str]], limit: int = 2
    ) -> List[str]:
        def _category(text: str) -> str:
            lowered = str(text or "").strip().lower()
            if lowered.startswith("weak rl pattern:"):
                return "weak_rl"
            if lowered.startswith("spec:"):
                return "spec_context"
            if "trend" in lowered and ("failure_rate" in lowered or "reward" in lowered):
                return "trend"
            if lowered.startswith("weakness observed:"):
                return "runtime_weakness"
            return "other"

        candidates: List[Dict[str, Any]] = []
        for excerpt in memory_excerpts:
            if not isinstance(excerpt, dict):
                continue
            source = str(excerpt.get("source", "unknown"))
            title = str(excerpt.get("title", ""))
            text = self._sanitize_focus_text(str(excerpt.get("excerpt", "")))
            if self._is_machine_like_focus_text(text):
                continue
            if text:
                if self._is_stable_rl_trend_excerpt(text, source=source, title=title):
                    continue
                formatted_text = self._format_focus_point_text(
                    text=text,
                    source=source,
                    max_chars=190,
                )
                if self._is_stable_rl_trend_excerpt(
                    formatted_text, source=source, title=title
                ):
                    continue
                candidates.append(
                    {
                        "score": float(self._focus_signal_score(formatted_text)),
                        "text": formatted_text,
                        "source": source,
                        "title": title,
                    }
                )

        if not candidates or limit <= 0:
            return []

        candidates.sort(
            key=lambda item: (float(item.get("score", 0.0)), len(str(item.get("text", "")))),
            reverse=True,
        )
        focus_points: List[str] = []
        seen_texts: set[str] = set()
        seen_categories: set[str] = set()
        recent_focus_keys = self._get_recent_gam_focus_keys()
        recent_window = set(recent_focus_keys[-GAM_RECENT_FOCUS_WINDOW:])

        filtered: List[Dict[str, Any]] = []
        seen_title_keys: set[str] = set()
        for candidate in candidates:
            candidate_text = str(candidate.get("text", "")).strip()
            if not candidate_text:
                continue
            if float(candidate.get("score", 0.0)) <= -0.25:
                continue
            key = candidate_text.lower()
            if key in seen_texts:
                continue
            if self._normalize_focus_key(candidate_text) in recent_window:
                continue
            source_key = str(candidate.get("source", "")).strip().lower()
            title_key = str(candidate.get("title", "")).strip().lower()
            # Prevent repeated pages with the same memo title from dominating
            # focus points (e.g., many RL Trend Signals pages across runs).
            if source_key == "memo" and title_key:
                if title_key in seen_title_keys:
                    continue
                seen_title_keys.add(title_key)
            seen_texts.add(key)
            candidate["category"] = _category(candidate_text)
            filtered.append(candidate)

        if not filtered:
            # If anti-repetition filtered everything, relax once.
            relaxed: List[Dict[str, Any]] = []
            seen_texts = set()
            for candidate in candidates:
                candidate_text = str(candidate.get("text", "")).strip()
                if not candidate_text:
                    continue
                if float(candidate.get("score", 0.0)) <= -0.25:
                    continue
                key = candidate_text.lower()
                if key in seen_texts:
                    continue
                seen_texts.add(key)
                candidate["category"] = _category(candidate_text)
                relaxed.append(candidate)
            filtered = relaxed
        if not filtered:
            return []

        # Avoid generic/static convention snippets when dynamic memo/learning
        # signals are present in the same run.
        non_convention_filtered = [
            item
            for item in filtered
            if str(item.get("source", "")).strip().lower() != "convention"
        ]
        if non_convention_filtered:
            filtered = non_convention_filtered

        def _take(predicate) -> None:
            if len(focus_points) >= int(limit):
                return
            for item in filtered:
                candidate_text = str(item.get("text", "")).strip()
                category = str(item.get("category", "other"))
                if not candidate_text or candidate_text in focus_points:
                    continue
                if not predicate(item):
                    continue
                focus_points.append(candidate_text)
                seen_categories.add(category)
                break

        # 1) Prefer one weak RL signal first when available.
        _take(lambda item: str(item.get("category", "")) == "weak_rl")

        # 2) Force diversity: prefer one non-weak non-convention focus.
        _take(
            lambda item: (
                str(item.get("category", "")) != "weak_rl"
                and str(item.get("category", "")) != "spec_context"
                and str(item.get("source", "")) != "convention"
            )
        )

        # 3) If still room, prefer unseen categories before duplicates.
        while len(focus_points) < int(limit):
            added = False
            for item in filtered:
                candidate_text = str(item.get("text", "")).strip()
                category = str(item.get("category", "other"))
                if not candidate_text or candidate_text in focus_points:
                    continue
                if category in seen_categories and len(seen_categories) < len(
                    {str(c.get("category", "other")) for c in filtered}
                ):
                    continue
                focus_points.append(candidate_text)
                seen_categories.add(category)
                added = True
                break
            if not added:
                break

        return focus_points[: int(limit)]

    def _format_focus_point_text(self, text: str, source: str, max_chars: int = 180) -> str:
        cleaned = str(text or "").replace("\n", " ").strip()
        if not cleaned:
            return ""
        while cleaned.startswith("- "):
            cleaned = cleaned[2:].strip()
        lower = cleaned.lower()
        source_key = str(source or "").strip().lower()

        if source_key == "convention":
            numbered_items = re.findall(r"\d+\.\s*[^.]+(?:\.)", cleaned)
            if numbered_items:
                top_items = numbered_items[:2]
                convention_text = " ".join(item.strip() for item in top_items)
                return self._smart_trim_text(convention_text, max_chars=max_chars)

        if source_key == "memo" and "top weak patterns from rl learning history" in lower:
            pattern = re.search(
                r"-\s*(GET|POST|PUT|PATCH|DELETE)\s+([^|]+)\|\s*([^|]+)\|\s*expect\s*([0-9]{3}|unknown)?\s*\|\s*failure_rate=([0-9.]+)",
                cleaned,
                flags=re.IGNORECASE,
            )
            if pattern:
                method = str(pattern.group(1)).upper()
                endpoint = str(pattern.group(2)).strip()
                test_type = str(pattern.group(3)).strip().replace("_", " ")
                raw_status = str(pattern.group(4) or "").strip().lower()
                expected_status = int(raw_status) if raw_status.isdigit() else 400
                expected_status = self._normalize_expected_status_for_test_type(
                    test_type=test_type,
                    expected_status=expected_status,
                )
                failure_rate = str(pattern.group(5)).strip()
                try:
                    failure_rate_value = float(failure_rate)
                except Exception:
                    failure_rate_value = 0.0
                if failure_rate_value >= RL_WEAK_FAILURE_RATE_THRESHOLD:
                    return (
                        f"Weak RL pattern: {method} {endpoint} ({test_type}) "
                        f"expected {expected_status}, failure_rate={failure_rate}."
                    )
                return (
                    f"RL pattern improving: {method} {endpoint} ({test_type}) "
                    f"expected {expected_status}, failure_rate={failure_rate}."
                )

        if source_key == "memo" and "recent learning trend:" in lower:
            run_count_match = re.search(
                r"historical run_count:\s*([0-9]+)",
                cleaned,
                flags=re.IGNORECASE,
            )
            run_count = run_count_match.group(1) if run_count_match else "n/a"
            pattern = re.search(
                r"-\s*(GET|POST|PUT|PATCH|DELETE)\s+([^|]+)\|\s*([^|]+)\|\s*expected=([0-9]{3})\s*\|\s*attempts=([0-9]+)\s*\|\s*failure_rate=([0-9.]+)\s*\|\s*avg_reward=([0-9.\-]+)",
                cleaned,
                flags=re.IGNORECASE,
            )
            if pattern:
                method = str(pattern.group(1)).upper()
                endpoint = str(pattern.group(2)).strip()
                test_type = str(pattern.group(3)).strip().replace("_", " ")
                expected_status = str(pattern.group(4)).strip()
                attempts = str(pattern.group(5)).strip()
                failure_rate = str(pattern.group(6)).strip()
                avg_reward = str(pattern.group(7)).strip()
                return (
                    f"RL trend (run {run_count}): {method} {endpoint} ({test_type}) "
                    f"expected {expected_status}, attempts={attempts}, "
                    f"failure_rate={failure_rate}, avg_reward={avg_reward}."
                )
            reward_match = re.search(
                r"last_run_reward=([0-9.\-]+)",
                cleaned,
                flags=re.IGNORECASE,
            )
            delta_match = re.search(
                r"reward_delta_vs_prev=([+\-]?[0-9.]+)",
                cleaned,
                flags=re.IGNORECASE,
            )
            if reward_match:
                reward = str(reward_match.group(1)).strip()
                delta = str(delta_match.group(1)).strip() if delta_match else "n/a"
                return f"RL trend (run {run_count}): last_run_reward={reward}, reward_delta_vs_prev={delta}."

        if source_key == "memo":
            pattern = re.search(
                r"(GET|POST|PUT|PATCH|DELETE)\s+([^|]+)\|\s*([^|]+)\|\s*expected=([0-9]{3})\s*\|\s*attempts=([0-9]+)\s*\|\s*failure_rate=([0-9.]+)\s*\|\s*avg_reward=([0-9.\-]+)",
                cleaned,
                flags=re.IGNORECASE,
            )
            if pattern:
                method = str(pattern.group(1)).upper()
                endpoint = str(pattern.group(2)).strip()
                test_type = str(pattern.group(3)).strip().replace("_", " ")
                expected_status = str(pattern.group(4)).strip()
                attempts = str(pattern.group(5)).strip()
                failure_rate = str(pattern.group(6)).strip()
                avg_reward = str(pattern.group(7)).strip()
                try:
                    failure_rate_value = float(failure_rate)
                except Exception:
                    failure_rate_value = 0.0
                label = (
                    "Weak RL pattern"
                    if failure_rate_value >= RL_WEAK_FAILURE_RATE_THRESHOLD
                    else "RL pattern trend"
                )
                return (
                    f"{label}: {method} {endpoint} ({test_type}) expected {expected_status}, "
                    f"attempts={attempts}, failure_rate={failure_rate}, avg_reward={avg_reward}."
                )

        if source_key == "memo" and cleaned.lower().startswith("spec:"):
            preview = self._smart_trim_text(cleaned, max_chars=max_chars)
            if "Spec-specific risk hints:" in preview:
                preview = preview.replace("Spec-specific risk hints:", "Risk hints:")
            return preview

        compact = self._smart_trim_text(cleaned, max_chars=max_chars)
        return compact

    def _smart_trim_text(self, text: str, max_chars: int = 180) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        if len(value) <= max_chars:
            return value

        sentence_endings = [value.rfind(".", 0, max_chars), value.rfind(";", 0, max_chars)]
        sentence_cut = max(sentence_endings)
        if sentence_cut >= max(60, int(max_chars * 0.55)):
            return value[: sentence_cut + 1]

        space_cut = value.rfind(" ", 0, max_chars)
        if space_cut >= max(40, int(max_chars * 0.45)):
            return value[:space_cut].rstrip() + "..."
        return value[:max_chars].rstrip() + "..."

    def _normalize_expected_status_for_test_type(
        self,
        test_type: str,
        expected_status: int,
    ) -> int:
        ttype = re.sub(r"[\s\-]+", "_", str(test_type or "").strip().lower())
        try:
            status = int(expected_status)
        except Exception:
            status = 400

        if ttype in {"authentication", "authorization"} and status not in {401, 403}:
            return 401
        if ttype == "error_handling" and status < 400:
            return 400
        if ttype in {"input_validation", "boundary_testing"} and status < 400:
            return 400
        return status

    def _sanitize_focus_text(self, text: str) -> str:
        cleaned = str(text or "").replace("\n", " ").strip()
        if not cleaned:
            return ""
        # Strip page references and opaque session pointer noise from memo text.
        cleaned = re.sub(r"Full session data:\s*page_id:[^|]+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"Full session data:\s*[^\n]+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        lowered = cleaned.lower()
        # Legacy low-value RL memo text from older runs should never shape prompts.
        if "none above threshold" in lowered and "top weak patterns from rl learning history" in lowered:
            return ""
        if "training executed" in lowered:
            return ""
        if lowered.startswith("context:") and not any(
            token in lowered
            for token in (
                "auth",
                "validation",
                "error",
                "boundary",
                "status",
                "pagination",
                "schema",
                "failing",
                "failure",
                "expected",
                "invalid",
                "missing",
            )
        ):
            return ""
        return cleaned

    def _extract_metric_value(self, text: str, metric: str) -> Optional[float]:
        raw = str(text or "")
        if not raw:
            return None
        match = re.search(
            rf"{re.escape(metric)}\s*[=:]\s*([+\-]?[0-9]*\.?[0-9]+)",
            raw,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        try:
            return float(match.group(1))
        except Exception:
            return None

    def _excerpt_has_weak_pattern_signal(self, text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        if "weak rl pattern:" in lowered or "weakness observed:" in lowered:
            return True
        failure_rate = self._extract_metric_value(lowered, "failure_rate")
        if failure_rate is not None and failure_rate >= RL_WEAK_FAILURE_RATE_THRESHOLD:
            return True
        avg_reward = self._extract_metric_value(lowered, "avg_reward")
        if avg_reward is not None and avg_reward < 0.0:
            return True
        return False

    def _is_stable_rl_trend_excerpt(self, text: str, source: str, title: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        source_key = str(source or "").strip().lower()
        title_lower = str(title or "").strip().lower()
        is_trend = (
            "trend" in lowered
            or "trend" in title_lower
            or "last_run_reward" in lowered
            or "avg_reward" in lowered
            or "failure_rate" in lowered
        )
        if source_key != "memo" or not is_trend:
            return False
        if self._excerpt_has_weak_pattern_signal(lowered):
            return False
        failure_rate = self._extract_metric_value(lowered, "failure_rate")
        avg_reward = self._extract_metric_value(lowered, "avg_reward")
        reward_delta = self._extract_metric_value(lowered, "reward_delta_vs_prev")
        if failure_rate is None:
            return False
        if failure_rate >= RL_WEAK_FAILURE_RATE_THRESHOLD:
            return False
        avg_non_negative = avg_reward is None or avg_reward >= 0.0
        delta_not_worse = reward_delta is None or reward_delta >= 0.0
        return avg_non_negative and delta_not_worse

    def _focus_signal_score(self, text: str) -> float:
        raw = str(text or "").strip()
        if not raw:
            return -100.0
        lower = raw.lower()
        if "none above threshold" in lower and "top weak patterns from rl learning history" in lower:
            return -100.0
        score = 0.0
        positive_keywords = [
            "test ",
            "auth",
            "validation",
            "boundary",
            "error",
            "status",
            "pagination",
            "missing",
            "invalid",
            "schema",
            "endpoint",
            "idempot",
            "token",
            "required",
            "failure",
            "weak",
            "expected",
            "reward",
            "failure_rate",
            "avg_reward",
        ]
        for kw in positive_keywords:
            if kw in lower:
                score += 1.0

        if re.search(r"\b(get|post|put|patch|delete)\s+\/", lower):
            score += 1.2

        if lower.startswith("context:"):
            score -= 0.8
        if (
            "spec:" in lower
            and "test " not in lower
            and all(token not in lower for token in ("failure", "weak", "expected", "reward"))
        ):
            score -= 0.8
        if "endpoints:" in lower and "test " not in lower:
            score -= 0.6
        if "decisions:" in lower and "test " not in lower:
            score -= 0.6
        if "agent qa_specialist training executed" in lower:
            score -= 1.5
        if "full session data" in lower:
            score -= 2.0
        failure_rate = self._extract_metric_value(lower, "failure_rate")
        avg_reward = self._extract_metric_value(lower, "avg_reward")
        if failure_rate is not None and failure_rate < RL_WEAK_FAILURE_RATE_THRESHOLD:
            score -= 1.0
            if failure_rate == 0.0 and (avg_reward is None or avg_reward >= 0.0):
                score -= 1.0
        return score

    def _is_machine_like_focus_text(self, text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return True
        lower = raw.lower()
        if raw.startswith("{") or raw.startswith("["):
            return True
        if '"total_scenarios"' in raw and '"pass_rate"' in raw:
            return True
        if "session id:" in lower or "=== transcript" in lower or "tool outputs" in lower:
            return True
        # Heuristic: very dense key/value payloads are usually log artifacts, not guidance.
        if (
            raw.count(":") >= 6
            and raw.count(",") >= 5
            and ("{" in raw or "}" in raw)
            and ('"' in raw or "'" in raw)
        ):
            return True
        return False

    def _build_gam_diagnostics(self, memory_excerpts: List[Dict[str, Any]]) -> Dict[str, Any]:
        excerpt_items = [item for item in memory_excerpts if isinstance(item, dict)]
        total = len(excerpt_items)
        source_counts = Counter(str(item.get("source", "unknown")) for item in excerpt_items)
        source_breakdown = dict(sorted(source_counts.items(), key=lambda kv: kv[1], reverse=True))
        convention_count = int(source_counts.get("convention", 0))
        non_convention_count = max(0, total - convention_count)

        machine_like_count = 0
        actionable_count = 0
        low_signal_count = 0
        preview: List[Dict[str, Any]] = []
        for item in excerpt_items[:8]:
            text = self._sanitize_focus_text(str(item.get("excerpt", "")))
            if not text:
                low_signal_count += 1
                continue
            if self._is_machine_like_focus_text(text):
                machine_like_count += 1
                continue
            score = self._focus_signal_score(text)
            if score >= 1.2:
                actionable_count += 1
            preview.append(
                {
                    "source": str(item.get("source", "unknown")),
                    "title": str(item.get("title", "")),
                    "tags": list(item.get("tags", [])) if isinstance(item.get("tags"), list) else [],
                    "similarity": item.get("similarity"),
                    "excerpt": text[:220],
                    "focus_score": round(float(score), 4),
                }
            )

        warnings: List[str] = []
        run_count = int(self.learning_state.get("run_count", 0))
        if total == 0:
            warnings.append("no_excerpts_retrieved")
        if total > 0 and convention_count / max(1, total) >= 0.80:
            warnings.append("convention_dominant_retrieval")
        if run_count > 0 and non_convention_count == 0:
            warnings.append("no_cross_run_memory_signal")
        if machine_like_count > 0:
            warnings.append("machine_like_excerpts_detected")
        if low_signal_count > 0:
            warnings.append("low_signal_excerpts_filtered")
        if total > 0 and actionable_count == 0:
            warnings.append("low_actionable_excerpt_quality")

        quality_score = 1.0
        quality_score -= 0.25 if "convention_dominant_retrieval" in warnings else 0.0
        quality_score -= 0.20 if "no_cross_run_memory_signal" in warnings else 0.0
        quality_score -= 0.20 if "machine_like_excerpts_detected" in warnings else 0.0
        quality_score -= 0.20 if "low_actionable_excerpt_quality" in warnings else 0.0
        quality_score = max(0.0, min(1.0, quality_score))

        return {
            "total_excerpts": total,
            "convention_excerpts": convention_count,
            "non_convention_excerpts": non_convention_count,
            "source_breakdown": source_breakdown,
            "rejected_excerpt_count": int(len(self._last_gam_rejected_excerpts or [])),
            "machine_like_excerpt_count": machine_like_count,
            "low_signal_excerpt_count": low_signal_count,
            "actionable_excerpt_count": actionable_count,
            "warnings": warnings,
            "quality_score": round(quality_score, 4),
            "excerpt_preview": preview,
        }

    def _build_prompt_trace(
        self,
        base_prompt: Optional[str],
        memory_excerpts: List[Dict[str, str]],
        effective_prompt: Optional[str],
        rl_focus_points: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        gam_focus_points_candidate = self._build_gam_prompt_focus_points(
            memory_excerpts, limit=2
        )
        rl_focus_points_used = [
            str(item).strip() for item in (rl_focus_points or []) if str(item).strip()
        ]
        prompt_seeded_from_default = bool((not base_prompt) and effective_prompt)
        gam_focus_points_used = (
            gam_focus_points_candidate
            if (base_prompt or prompt_seeded_from_default)
            else []
        )
        focus_points_used = gam_focus_points_used + rl_focus_points_used
        excerpt_preview: List[Dict[str, Any]] = []
        for excerpt in memory_excerpts[:8]:
            if not isinstance(excerpt, dict):
                continue
            source = str(excerpt.get("source", ""))
            title = str(excerpt.get("title", ""))
            text = self._sanitize_focus_text(str(excerpt.get("excerpt", "")))
            if not text:
                continue
            if self._is_stable_rl_trend_excerpt(text, source=source, title=title):
                continue
            excerpt_preview.append(
                {
                    "source": source,
                    "similarity": excerpt.get("similarity"),
                    "excerpt": text[:220],
                }
            )

        final_effective_prompt = effective_prompt or base_prompt
        return {
            "base_prompt": base_prompt,
            "effective_prompt": final_effective_prompt,
            "base_prompt_supplied_by_user": bool(base_prompt),
            "prompt_seeded_from_default": prompt_seeded_from_default,
            "focus_points_used": focus_points_used,
            "focus_points_used_count": len(focus_points_used),
            "gam_focus_points_used": gam_focus_points_used,
            "gam_focus_points_used_count": len(gam_focus_points_used),
            "rl_focus_points_used": rl_focus_points_used,
            "rl_focus_points_used_count": len(rl_focus_points_used),
            "memory_excerpts_available_count": len(memory_excerpts),
            "memory_excerpt_preview": excerpt_preview,
            "prompt_was_enriched_by_gam": bool(
                bool(gam_focus_points_used)
            ),
            "prompt_was_enriched_by_rl": bool(rl_focus_points_used),
            "notes": {
                "gam_excerpt_stability": (
                    "GAM focus points are filtered to suppress stable zero-failure trend lines; "
                    "when weak signals are absent, prompt enrichment favors risk/action context "
                    "plus a run-to-run trend delta focus point."
                ),
                "rl_influence": (
                    "RL influences scenario mutation, scenario selection, and training updates. "
                    "GAM controls memory retrieval; both can influence the final prompt."
                ),
            },
        }

    def _enforce_scenario_generation_quality(self, prompt_trace: Dict[str, Any]) -> None:
        if not isinstance(prompt_trace, dict):
            return
        generation = prompt_trace.get("scenario_generation", {})
        if not isinstance(generation, dict):
            return

        llm_enabled = bool(generation.get("llm_enabled", False))
        llm_stats = generation.get("llm_stats", {})
        if not isinstance(llm_stats, dict):
            llm_stats = {}
        llm_calls = int(llm_stats.get("scenario_calls", 0) or 0)
        llm_success = int(llm_stats.get("scenario_success", 0) or 0)
        llm_errors = int(llm_stats.get("scenario_errors", 0) or 0)
        base_count = int(generation.get("base_scenario_count", 0) or 0)

        if llm_enabled and llm_calls > 0 and llm_success == 0:
            warning = (
                f"LLM scenario generation had {llm_errors} error(s) and 0 successful calls; "
                "heuristic fallback was used."
            )
            generation["llm_reliability_warning"] = warning
            self._llm_generation_degraded = True
            self._llm_generation_degraded_reason = warning

        happy_guard = generation.get("happy_path_guardrail", {})
        if isinstance(happy_guard, dict):
            missing_after = int(happy_guard.get("missing_after", 0) or 0)
            if missing_after > 0:
                generation["coverage_warning"] = (
                    f"Happy-path guardrail still missing coverage for {missing_after} operation(s)."
                )

        if base_count <= 0:
            raise RuntimeError("Scenario generation produced zero candidates.")

    def _build_scenario_generation_trace(
        self,
        *,
        base_scenarios: List[TestScenario],
        effective_prompt: Optional[str],
    ) -> Dict[str, Any]:
        base_source = str(self._base_scenario_source or "llm_base")
        traces: List[Dict[str, Any]] = []
        for idx, scenario in enumerate(base_scenarios[:60]):
            expected = int(scenario.expected_status)
            traces.append(
                {
                    "step": int(idx + 1),
                    "scenario_name": str(scenario.name),
                    "source": base_source,
                    "thought": (
                        f"Target {scenario.test_type.value} for "
                        f"{scenario.method.upper()} {scenario.endpoint}"
                    ),
                    "action": (
                        f"Generate test with expected_status={expected}, "
                        f"params={len(scenario.params or {})}, body={1 if scenario.body else 0}"
                    ),
                    "observation_goal": (
                        "negative_path"
                        if expected in NEGATIVE_STATUS_CODES
                        else "positive_path"
                    ),
                }
            )
        return {
            "effective_prompt_present": bool(str(effective_prompt or "").strip()),
            "base_scenario_count": int(len(base_scenarios)),
            "trace_steps": traces,
        }

    def _extract_endpoint_metadata(self, spec: Dict[str, Any]) -> List[Dict[str, str]]:
        endpoints: List[Dict[str, str]] = []
        for path, path_info in (spec.get("paths") or {}).items():
            if not isinstance(path_info, dict):
                continue
            for method in path_info.keys():
                if method.lower() in {"get", "post", "put", "patch", "delete"}:
                    endpoints.append({"method": method.upper(), "path": path})
        return endpoints

    def _persist_gam_spec_context_page(
        self,
        spec_title: str,
        spec: Dict[str, Any],
        learning_hints: List[Dict[str, Any]],
    ) -> Optional[str]:
        endpoints = self._extract_endpoint_metadata(spec)
        if not endpoints:
            return None

        auth_type = self._infer_auth_type(spec)
        run_count = int(self.learning_state.get("run_count", 0))
        endpoint_preview = ", ".join(
            f"{item.get('method', '')} {item.get('path', '')}".strip()
            for item in endpoints[:8]
        )
        risk_lines: List[str] = []
        for hint in (learning_hints or [])[:4]:
            if not isinstance(hint, dict):
                continue
            method = str(hint.get("method", "")).upper()
            endpoint = str(hint.get("endpoint", ""))
            test_type = str(hint.get("test_type", "")).replace("_", " ")
            expected_status = self._normalize_expected_status_for_test_type(
                test_type=test_type,
                expected_status=int(hint.get("expected_status", 400) or 400),
            )
            failure_rate = hint.get("failure_rate")
            risk_lines.append(
                f"- {method} {endpoint} | {test_type} | expected={expected_status} | failure_rate={failure_rate}"
            )

        lines = [
            f"Spec: {spec_title}",
            f"Auth type: {auth_type}",
            f"Historical run_count: {run_count}",
            f"Operation count: {len(endpoints)}",
            f"Operations preview: {endpoint_preview}",
            "Spec-specific risk hints:",
        ]
        if risk_lines:
            lines.extend(risk_lines)
        else:
            lines.append(
                "- No weak RL pattern above threshold yet for this spec; prioritize operation-specific "
                f"negative/boundary checks across: {endpoint_preview}."
            )

        content = "\n".join(lines).strip()
        title = f"Spec Context Signals: {spec_title}"
        recent_pages = self.gam.page_store.pages[-120:]
        for page in reversed(recent_pages):
            if getattr(page, "tenant_id", None) != self.tenant_id:
                continue
            if str(getattr(page, "title", "")) != title:
                continue
            if str(getattr(page, "content", "")) == content:
                return str(getattr(page, "id", ""))

        spec_slug = re.sub(r"[^a-z0-9]+", "_", str(spec_title).lower()).strip("_")
        spec_tags = sorted(self._spec_memory_tags) if self._spec_memory_tags else [spec_slug or "spec"]
        page = self.gam.add_page(
            title=title,
            tags=["memo", "spec_context", "dynamic", "run_aware", *spec_tags],
            content=content,
            source="memo",
            tenant_id=self.tenant_id,
        )
        return str(page.id)

    def _persist_rl_learning_signal_page(self, spec_title: str) -> Optional[str]:
        hints = self._build_gam_learning_hints(limit=6)
        if not hints:
            # Do not persist a synthetic "none above threshold" page.
            # It creates repetitive static prompt snippets with no actionable value.
            return None

        run_count = int(self.learning_state.get("run_count", 0))
        lines = [
            f"Spec: {spec_title}",
            f"Historical run_count: {run_count}",
            "Top weak patterns from RL learning history:",
        ]
        for hint in hints:
            if not isinstance(hint, dict):
                continue
            method = str(hint.get("method", "")).upper()
            endpoint = str(hint.get("endpoint", ""))
            test_type = str(hint.get("test_type", "")).replace("_", " ")
            expected_status = hint.get("expected_status")
            attempts = hint.get("attempts")
            failure_rate = hint.get("failure_rate")
            avg_reward = hint.get("avg_reward")
            lines.append(
                f"- {method} {endpoint} | {test_type} | expect {expected_status} | "
                f"failure_rate={failure_rate} | avg_reward={avg_reward} | attempts={attempts}"
            )

        content = "\n".join(lines).strip()
        if not content:
            return None

        title = f"RL Learning Signals: {spec_title}"
        recent_pages = self.gam.page_store.pages[-80:]
        for page in reversed(recent_pages):
            if getattr(page, "tenant_id", None) != self.tenant_id:
                continue
            if str(getattr(page, "title", "")) != title:
                continue
            if str(getattr(page, "content", "")) == content:
                return str(getattr(page, "id", ""))

        spec_slug = re.sub(r"[^a-z0-9]+", "_", str(spec_title).lower()).strip("_")
        spec_tags = sorted(self._spec_memory_tags) if self._spec_memory_tags else [spec_slug or "spec"]
        page = self.gam.add_page(
            title=title,
            tags=["memo", "learning", "rl_signal", "weakness", *spec_tags],
            content=content,
            source="memo",
            tenant_id=self.tenant_id,
        )
        return str(page.id)

    def _persist_rl_trend_signal_page(self, spec_title: str) -> Optional[str]:
        scenario_stats = self._scenario_stats_for_current_spec(
            self.adaptive_policy.scenario_stats
        )
        history = self.learning_state.get("decision_history", [])
        if not isinstance(history, list):
            history = []

        if not scenario_stats and not history:
            return None

        latest = history[-1] if history else {}
        previous = history[-2] if len(history) >= 2 else {}
        latest_reward = float((latest or {}).get("run_reward", 0.0))
        previous_reward = float((previous or {}).get("run_reward", latest_reward))
        latest_avg_decision = float((latest or {}).get("average_decision_reward", 0.0))
        latest_rewarded = int((latest or {}).get("rewarded_decisions", 0))
        latest_penalized = int((latest or {}).get("penalized_decisions", 0))
        reward_delta = (
            latest_reward - previous_reward if len(history) >= 2 else 0.0
        )

        ranked_patterns: List[Dict[str, Any]] = []
        for fingerprint, stats_raw in scenario_stats.items():
            if not isinstance(stats_raw, dict):
                continue
            attempts = int(stats_raw.get("attempts", 0))
            if attempts <= 0:
                continue
            failure_rate = float(stats_raw.get("failure_rate", 0.0))
            avg_reward = float(stats_raw.get("avg_reward", 0.0))

            parsed = self._parse_scenario_fingerprint(str(fingerprint))
            method = str((parsed or {}).get("method") or stats_raw.get("method", "GET")).upper()
            endpoint = str((parsed or {}).get("endpoint") or stats_raw.get("endpoint", ""))
            test_type = str(
                (parsed or {}).get("test_type") or stats_raw.get("test_type", "error_handling")
            )
            raw_expected_status = int(
                (parsed or {}).get("expected_status") or stats_raw.get("expected_status", 400)
            )
            canonical = self._canonicalize_learning_pattern(
                method=method,
                endpoint=endpoint,
                test_type=test_type,
                expected_status=raw_expected_status,
            )
            if not canonical:
                continue
            method = str(canonical.get("method", method))
            endpoint = str(canonical.get("endpoint", endpoint))
            test_type = str(canonical.get("test_type", test_type))
            expected_status = int(canonical.get("expected_status", raw_expected_status))
            if not self._scenario_belongs_to_current_spec(method, endpoint):
                continue

            priority = (
                0.55 * failure_rate
                + 0.25 * max(0.0, -avg_reward)
                + 0.20 * min(1.0, attempts / 10.0)
            )
            ranked_patterns.append(
                {
                    "method": method,
                    "endpoint": endpoint,
                    "test_type": test_type,
                    "expected_status": expected_status,
                    "attempts": attempts,
                    "failure_rate": round(failure_rate, 4),
                    "avg_reward": round(avg_reward, 4),
                    "priority": round(priority, 4),
                }
            )

        ranked_patterns.sort(
            key=lambda item: (
                float(item.get("priority", 0.0)),
                int(item.get("attempts", 0)),
                float(item.get("failure_rate", 0.0)),
            ),
            reverse=True,
        )

        run_count = int(self.learning_state.get("run_count", 0))
        lines = [
            f"Trend: RL learning trend for {spec_title}",
            f"Spec: {spec_title}",
            f"Historical run_count: {run_count}",
            "Recent learning trend:",
            f"- last_run_reward={latest_reward:.4f}",
            f"- reward_delta_vs_prev={reward_delta:+.4f}",
            f"- avg_decision_reward={latest_avg_decision:.4f}",
            f"- rewarded={latest_rewarded}, penalized={latest_penalized}",
            "Top evolving patterns:",
        ]
        if ranked_patterns:
            for item in ranked_patterns[:3]:
                lines.append(
                    "- "
                    + f"{item['method']} {item['endpoint']} | {item['test_type']} | "
                    + f"expected={item['expected_status']} | attempts={item['attempts']} | "
                    + f"failure_rate={item['failure_rate']} | avg_reward={item['avg_reward']}"
                )
        else:
            lines.append("- No scenario history yet for this spec scope.")

        content = "\n".join(lines).strip()
        if not content:
            return None

        title = f"RL Trend Signals: {spec_title}"
        recent_pages = self.gam.page_store.pages[-80:]
        for page in reversed(recent_pages):
            if getattr(page, "tenant_id", None) != self.tenant_id:
                continue
            if str(getattr(page, "title", "")) != title:
                continue
            if str(getattr(page, "content", "")) == content:
                return str(getattr(page, "id", ""))

        spec_slug = re.sub(r"[^a-z0-9]+", "_", str(spec_title).lower()).strip("_")
        spec_tags = sorted(self._spec_memory_tags) if self._spec_memory_tags else [spec_slug or "spec"]
        page = self.gam.add_page(
            title=title,
            tags=["memo", "learning", "trend", "run_aware", *spec_tags],
            content=content,
            source="memo",
            tenant_id=self.tenant_id,
        )
        return str(page.id)

    def _build_current_run_weakness_decisions(
        self,
        learning_feedback: Dict[str, Any],
        limit: int = 2,
    ) -> List[str]:
        decision_signals = learning_feedback.get("decision_signals", [])
        if not isinstance(decision_signals, list) or not decision_signals:
            return []

        failures: List[Dict[str, Any]] = [
            item
            for item in decision_signals
            if isinstance(item, dict) and not bool(item.get("passed", True))
        ]
        if not failures:
            return []

        failures.sort(key=lambda item: float(item.get("reward", 0.0)))
        insights: List[str] = []
        for item in failures[: max(0, int(limit))]:
            method = str(item.get("method", "")).upper()
            endpoint = str(item.get("endpoint_template", item.get("endpoint_key", "")))
            test_type = str(item.get("test_type", "")).replace("_", " ")
            expected_status = item.get("expected_status")
            actual_status = item.get("actual_status")
            reward = float(item.get("reward", 0.0))
            insights.append(
                f"Weakness observed: {method} {endpoint} ({test_type}) "
                f"expected {expected_status}, got {actual_status}, reward {reward:.3f}"
            )
        return insights

    def _build_gam_learning_hints(self, limit: int = 5) -> List[Dict[str, Any]]:
        scenario_stats = self._scenario_stats_for_current_spec(
            self.adaptive_policy.scenario_stats
        )
        if not scenario_stats:
            return []

        ranked: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        for fingerprint, stats_raw in scenario_stats.items():
            if not isinstance(stats_raw, dict):
                continue
            attempts = int(stats_raw.get("attempts", 0))
            if attempts < RL_WEAK_MIN_ATTEMPTS:
                continue
            failure_rate = float(stats_raw.get("failure_rate", 0.0))
            avg_reward = float(stats_raw.get("avg_reward", 0.0))
            if (
                failure_rate < RL_WEAK_FAILURE_RATE_THRESHOLD
                and avg_reward >= 0.0
            ):
                continue

            parsed = self._parse_scenario_fingerprint(str(fingerprint))
            method = str((parsed or {}).get("method") or stats_raw.get("method", "GET")).upper()
            endpoint = str((parsed or {}).get("endpoint") or stats_raw.get("endpoint", ""))
            test_type = str(
                (parsed or {}).get("test_type") or stats_raw.get("test_type", "error_handling")
            )
            raw_expected_status = int(
                (parsed or {}).get("expected_status") or stats_raw.get("expected_status", 400)
            )
            canonical = self._canonicalize_learning_pattern(
                method=method,
                endpoint=endpoint,
                test_type=test_type,
                expected_status=raw_expected_status,
            )
            if not canonical:
                continue
            method = str(canonical.get("method", method))
            endpoint = str(canonical.get("endpoint", endpoint))
            test_type = str(canonical.get("test_type", test_type))
            expected_status = int(canonical.get("expected_status", raw_expected_status))
            if not self._scenario_belongs_to_current_spec(method, endpoint):
                continue

            priority = (
                0.70 * failure_rate
                + 0.20 * max(0.0, -avg_reward)
                + 0.10 * min(1.0, attempts / 10.0)
            )
            ranked.append(
                (
                    str(fingerprint),
                    stats_raw,
                    {
                        "method": method,
                        "endpoint": endpoint,
                        "test_type": test_type,
                        "expected_status": expected_status,
                        "attempts": attempts,
                        "failure_rate": round(failure_rate, 4),
                        "avg_reward": round(avg_reward, 4),
                        "priority": round(priority, 4),
                    },
                )
            )

        ranked.sort(
            key=lambda item: (
                float(item[2].get("priority", 0.0)),
                int(item[2].get("attempts", 0)),
                float(item[2].get("failure_rate", 0.0)),
            ),
            reverse=True,
        )
        unique_hints: List[Dict[str, Any]] = []
        seen_keys: set[tuple[str, str, str]] = set()
        for _, _, hint in ranked:
            key = (
                str(hint.get("method", "")),
                str(hint.get("endpoint", "")),
                str(hint.get("test_type", "")),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_hints.append(hint)
            if len(unique_hints) >= max(0, int(limit)):
                break
        return unique_hints

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

        return expected_status in AUTH_NEGATIVE_STATUS_CODES

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
            "workspace_id": self.workspace_id,
            "spec_key": self._spec_scope_key,
            "environment_profile": self.environment_profile,
            "rl_train_mode": self.rl_train_mode,
            "pass_rate": summary["pass_rate"],
            "pass_threshold": summary["pass_threshold"],
            "meets_quality_gate": bool(summary.get("meets_quality_gate", False)),
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
        quality_gate = bool(task_data.get("meets_quality_gate", pass_rate >= threshold))
        success = (pass_rate >= threshold) and quality_gate
        return {
            "success": success,
            "quality_score": learning_reward,
            "summary": task_data.get("summary", {}),
            "report_path": task_data.get("report_path"),
        }

    def _apply_llm_token_cap(self) -> None:
        if self.llm_token_cap is None:
            return
        cap = int(self.llm_token_cap)
        os.environ["QA_SCENARIO_LLM_MAX_TOKENS"] = str(cap)
        os.environ["GAM_LLM_MAX_TOKENS"] = str(cap)
        os.environ["GAM_MEMO_LLM_MAX_TOKENS"] = str(cap)
        # Keep active objects in sync for current process lifetime.
        try:
            if hasattr(self.gam, "researcher") and hasattr(self.gam.researcher, "_llm_max_tokens"):
                self.gam.researcher._llm_max_tokens = min(
                    int(getattr(self.gam.researcher, "_llm_max_tokens", cap) or cap),
                    cap,
                )
            if hasattr(self.gam, "memorizer") and hasattr(self.gam.memorizer, "_llm_max_tokens"):
                self.gam.memorizer._llm_max_tokens = min(
                    int(getattr(self.gam.memorizer, "_llm_max_tokens", cap) or cap),
                    cap,
                )
        except Exception:
            # Token cap is best-effort hardening; failures should not abort run.
            return

    def _environment_tier_signal(self) -> Dict[str, Any]:
        weights = {"mock": 0.90, "staging": 1.00, "prod_safe": 1.05}
        profile = str(self.environment_profile or DEFAULT_ENVIRONMENT_PROFILE).lower()
        return {
            "profile": profile,
            "reward_weight": float(weights.get(profile, 0.90)),
            "isolation_mode": str(self._execution_isolation_mode),
        }

    def _build_gam_context_pack(
        self,
        *,
        memory_excerpts: List[Dict[str, Any]],
        diagnostics: Dict[str, Any],
        reflection: str,
    ) -> Dict[str, Any]:
        selected: List[Dict[str, Any]] = []
        dynamic_rejected: List[Dict[str, Any]] = []
        for item in memory_excerpts:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", ""))
            title = str(item.get("title", ""))
            excerpt_text = str(item.get("excerpt", ""))
            if self._is_stable_rl_trend_excerpt(
                excerpt_text,
                source=source,
                title=title,
            ):
                dynamic_rejected.append(
                    {
                        "source": source,
                        "title": title,
                        "tags": list(item.get("tags", [])) if isinstance(item.get("tags"), list) else [],
                        "excerpt": excerpt_text,
                        "reason": "stable_trend_no_signal",
                    }
                )
                continue
            selected.append(
                {
                    "page_id": str(item.get("page_id", "")),
                    "title": title,
                    "source": source,
                    "tags": list(item.get("tags", [])) if isinstance(item.get("tags"), list) else [],
                    "similarity": item.get("similarity"),
                    "excerpt": excerpt_text,
                }
            )

        rejected_pages = [
            {
                "source": str(item.get("source", "")),
                "title": str(item.get("title", "")),
                "tags": list(item.get("tags", [])) if isinstance(item.get("tags"), list) else [],
                "excerpt": str(item.get("excerpt", "")),
                "reason": str(item.get("reason", "")),
            }
            for item in list(self._last_gam_rejected_excerpts or [])[:20]
            if isinstance(item, dict)
        ] + dynamic_rejected[:20]
        rejection_reasons: List[Dict[str, Any]] = []
        for warning in list(diagnostics.get("warnings", []) or []):
            rejection_reasons.append(
                {
                    "reason": str(warning),
                    "severity": "warning",
                }
            )
        quality_score = float(diagnostics.get("quality_score", 0.0) or 0.0)
        has_direct_weak_pattern = any(
            self._excerpt_has_weak_pattern_signal(str(item.get("excerpt", "")))
            for item in selected
        )
        has_exploration_proxy = any(
            token in str(item.get("excerpt", "")).lower()
            for item in selected
            for token in (
                "no persistent weak pattern detected yet",
                "exploration-focused",
                "exploration focused",
            )
        )
        endpoint_mentions: set[str] = set()
        test_type_mentions: set[str] = set()
        for item in selected:
            excerpt_text = str(item.get("excerpt", "") or "")
            for match in re.findall(r"(GET|POST|PUT|PATCH|DELETE)\s+(/[A-Za-z0-9_/{}/\-]+)", excerpt_text):
                endpoint_mentions.add(f"{str(match[0]).upper()} {str(match[1])}")
            for token in (
                "authentication",
                "authorization",
                "input_validation",
                "error_handling",
                "boundary_testing",
                "security",
                "integration",
                "edge_cases",
                "performance",
                "happy_path",
            ):
                if token in excerpt_text.lower():
                    test_type_mentions.add(token)
        contract_checks = {
            "has_weak_pattern": bool(has_direct_weak_pattern),
            "has_weak_pattern_or_proxy": bool(has_direct_weak_pattern or has_exploration_proxy),
            "has_trend_delta": any(
                "trend" in str(item.get("excerpt", "")).lower()
                or "delta" in str(item.get("excerpt", "")).lower()
                or "last_run_reward" in str(item.get("excerpt", "")).lower()
                for item in selected
            ),
            "has_spec_risk": any(
                "spec:" in str(item.get("excerpt", "")).lower()
                or "risk" in str(item.get("excerpt", "")).lower()
                or "auth type" in str(item.get("excerpt", "")).lower()
                for item in selected
            ),
            "has_next_action": any(
                "prioritize" in str(item.get("excerpt", "")).lower()
                or "reinforce" in str(item.get("excerpt", "")).lower()
                or "next action" in str(item.get("excerpt", "")).lower()
                or "action:" in str(item.get("excerpt", "")).lower()
                for item in selected
            ),
            "weak_pattern_mode": (
                "direct"
                if has_direct_weak_pattern
                else ("exploration_proxy" if has_exploration_proxy else "missing")
            ),
            "has_exploration_proxy": bool(has_exploration_proxy),
            "has_diverse_endpoint_signal": len(endpoint_mentions) >= (2 if len(selected) >= 4 else 1),
            "has_diverse_test_type_signal": len(test_type_mentions) >= (2 if len(selected) >= 4 else 1),
        }
        required_contract_keys = [
            "has_weak_pattern_or_proxy",
            "has_trend_delta",
            "has_spec_risk",
            "has_next_action",
            "has_diverse_endpoint_signal",
            "has_diverse_test_type_signal",
        ]
        missing_contract = [
            key for key in required_contract_keys if not bool(contract_checks.get(key))
        ]
        if missing_contract:
            rejection_reasons.append(
                {
                    "reason": "missing_contract_items:" + ",".join(missing_contract),
                    "severity": "warning",
                }
            )
        status = (
            "accepted"
            if quality_score >= GAM_CONTEXT_MIN_QUALITY and not missing_contract
            else "needs_retry"
        )

        return {
            "status": status,
            "quality_score": round(quality_score, 4),
            "selected_pages": selected[:12],
            "selected_count": len(selected),
            "rejected_pages": rejected_pages,
            "rejected_count": len(rejected_pages),
            "rejection_reasons": rejection_reasons,
            "reflection": str(reflection or ""),
            "contract_checks": contract_checks,
        }

    def _build_weak_pattern_deltas(
        self,
        *,
        before: Dict[str, Any],
        after: Dict[str, Any],
        limit: int = 40,
    ) -> Dict[str, Any]:
        before = before if isinstance(before, dict) else {}
        after = after if isinstance(after, dict) else {}
        keys = set(before.keys()) | set(after.keys())
        deltas: List[Dict[str, Any]] = []
        for fingerprint in keys:
            prev = before.get(fingerprint, {}) if isinstance(before.get(fingerprint), dict) else {}
            curr = after.get(fingerprint, {}) if isinstance(after.get(fingerprint), dict) else {}
            prev_failure = float(prev.get("failure_rate", 0.0))
            curr_failure = float(curr.get("failure_rate", 0.0))
            prev_attempts = int(prev.get("attempts", 0))
            curr_attempts = int(curr.get("attempts", 0))
            delta = curr_failure - prev_failure
            if abs(delta) < 1e-9 and prev_attempts == curr_attempts:
                continue
            if delta < -0.02:
                status = "improved"
            elif delta > 0.02:
                status = "regressed"
            else:
                status = "unchanged"
            deltas.append(
                {
                    "fingerprint": str(fingerprint),
                    "from_failure_rate": round(prev_failure, 4),
                    "to_failure_rate": round(curr_failure, 4),
                    "delta_failure_rate": round(delta, 4),
                    "from_attempts": prev_attempts,
                    "to_attempts": curr_attempts,
                    "status": status,
                }
            )

        deltas.sort(
            key=lambda item: (
                abs(float(item.get("delta_failure_rate", 0.0))),
                int(item.get("to_attempts", 0)),
            ),
            reverse=True,
        )
        improved = sum(1 for item in deltas if str(item.get("status")) == "improved")
        regressed = sum(1 for item in deltas if str(item.get("status")) == "regressed")
        unchanged = sum(1 for item in deltas if str(item.get("status")) == "unchanged")
        return {
            "total_changed_patterns": len(deltas),
            "improved_count": improved,
            "regressed_count": regressed,
            "unchanged_count": unchanged,
            "items": deltas[: max(1, int(limit))],
        }

    def _compute_learning_delta_status(self, weak_pattern_deltas: Dict[str, Any]) -> str:
        summary = self._compute_learning_delta_summary(
            weak_pattern_deltas=weak_pattern_deltas,
            policy_movement={},
        )
        return str(summary.get("status", "no_learning_delta_detected"))

    def _build_policy_movement_metrics(
        self, selected_scenarios: List[TestScenario]
    ) -> Dict[str, Any]:
        previous = set(
            str(item).strip()
            for item in (self.learning_state.get("last_selected_fingerprints", []) or [])
            if str(item).strip()
        )
        current = set(self._scenario_fingerprint(item) for item in selected_scenarios)
        added = sorted(current - previous)
        removed = sorted(previous - current)
        retained = sorted(current & previous)
        universe = len(current | previous)
        jaccard = (len(retained) / float(max(1, universe))) if universe > 0 else 1.0
        turnover = ((len(added) + len(removed)) / float(max(1, universe))) if universe > 0 else 0.0

        if not previous:
            status = "cold_start"
        elif added or removed:
            status = "shifted"
        else:
            status = "stable"

        return {
            "status": status,
            "previous_count": int(len(previous)),
            "current_count": int(len(current)),
            "retained_count": int(len(retained)),
            "added_count": int(len(added)),
            "removed_count": int(len(removed)),
            "jaccard_similarity": round(float(jaccard), 4),
            "turnover_ratio": round(float(turnover), 4),
            "added_preview": added[:20],
            "removed_preview": removed[:20],
        }

    def _persist_policy_movement_state(self, selected_scenarios: List[TestScenario]) -> None:
        current = sorted(
            self._scenario_fingerprint(item) for item in (selected_scenarios or [])
        )
        self.learning_state["last_selected_fingerprints"] = current[:1000]

    def _compute_learning_delta_summary(
        self,
        *,
        weak_pattern_deltas: Dict[str, Any],
        policy_movement: Dict[str, Any],
    ) -> Dict[str, Any]:
        deltas = weak_pattern_deltas if isinstance(weak_pattern_deltas, dict) else {}
        improved = int(deltas.get("improved_count", 0))
        regressed = int(deltas.get("regressed_count", 0))
        total_changed = int(deltas.get("total_changed_patterns", 0))
        movement = policy_movement if isinstance(policy_movement, dict) else {}
        turnover = float(movement.get("turnover_ratio", 0.0) or 0.0)
        history = self.learning_state.get("decision_history", [])
        if not isinstance(history, list) or len(history) < 2:
            return {
                "status": "no_learning_delta_detected",
                "reason": "insufficient_history_for_delta",
                "reward_delta": 0.0,
            }
        latest = history[-1] if isinstance(history, list) and history else {}
        previous = history[-2] if isinstance(history, list) and len(history) > 1 else {}
        latest_reward = float((latest or {}).get("run_reward", 0.0))
        previous_reward = float((previous or {}).get("run_reward", latest_reward))
        reward_delta = latest_reward - previous_reward

        if improved > max(0, regressed):
            return {
                "status": "improved",
                "reason": "weak_pattern_failure_rate_improved",
                "reward_delta": round(float(reward_delta), 4),
            }
        if regressed > max(0, improved):
            return {
                "status": "regressed",
                "reason": "weak_pattern_failure_rate_regressed",
                "reward_delta": round(float(reward_delta), 4),
            }
        if total_changed == 0 and abs(reward_delta) < 0.01 and turnover < 0.05:
            return {
                "status": "no_learning_delta_detected",
                "reason": "no_pattern_change_and_policy_stable",
                "reward_delta": round(float(reward_delta), 4),
            }
        if total_changed == 0 and turnover >= 0.05:
            return {
                "status": "unchanged",
                "reason": "policy_shifted_without_measurable_pattern_delta_yet",
                "reward_delta": round(float(reward_delta), 4),
            }
        return {
            "status": "unchanged",
            "reason": "minor_or_mixed_changes",
            "reward_delta": round(float(reward_delta), 4),
        }

    def _build_repro_artifacts(
        self,
        spec: Dict[str, Any],
        scenarios: List[TestScenario],
        results: List[ScenarioExecutionResult],
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        scenario_by_name = {str(item.name): item for item in scenarios}
        seen: set[str] = set()
        artifacts: List[Dict[str, Any]] = []
        profile = str(self.environment_profile or DEFAULT_ENVIRONMENT_PROFILE).lower()
        if profile == "mock":
            # Imported lazily so this module can be used without server startup paths.
            from dynamic_mock_server import DynamicMockServer

            spec_copy = self.output_dir / "openapi_under_test_repro.yaml"
            spec_copy.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
            server = DynamicMockServer(str(spec_copy), host="127.0.0.1", port=0)
            with TestClient(server.app) as raw_client:
                client = _TestClientRequestsAdapter(raw_client)
                self._collect_repro_artifacts_from_client(
                    client=client,
                    scenario_by_name=scenario_by_name,
                    results=results,
                    seen=seen,
                    artifacts=artifacts,
                    limit=limit,
                    base_url_hint="http://testserver",
                )
        else:
            with _LiveRequestsAdapter(self.base_url, timeout_sec=12.0) as client:
                self._collect_repro_artifacts_from_client(
                    client=client,
                    scenario_by_name=scenario_by_name,
                    results=results,
                    seen=seen,
                    artifacts=artifacts,
                    limit=limit,
                    base_url_hint=str(self.base_url),
                )
        return artifacts

    def _collect_repro_artifacts_from_client(
        self,
        *,
        client: Any,
        scenario_by_name: Dict[str, TestScenario],
        results: List[ScenarioExecutionResult],
        seen: set[str],
        artifacts: List[Dict[str, Any]],
        limit: int,
        base_url_hint: str,
    ) -> None:
        for result in results:
            if result.passed:
                continue
            scenario = scenario_by_name.get(str(result.name))
            if scenario is None:
                continue
            fingerprint = self._scenario_fingerprint(scenario)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            headers = self._render_headers(scenario.headers)
            payload = scenario.body if isinstance(scenario.body, dict) else None
            query = self._strip_path_params(scenario.endpoint, scenario.params)
            if str(result.error or "").startswith("unsafe_action_blocked"):
                minimized = {
                    "query": query,
                    "body": payload,
                    "status": None,
                }
            else:
                minimized = self._shrink_failed_input_for_repro(
                    client=client,
                    scenario=scenario,
                    baseline_status=result.actual_status,
                )
            minimized_query = minimized.get("query", query)
            minimized_body = minimized.get("body", payload)

            curl_command = self._build_curl_repro_command(
                method=scenario.method.upper(),
                base_url_hint=base_url_hint,
                endpoint_resolved=str(result.endpoint_resolved or ""),
                headers=headers,
                query=minimized_query if isinstance(minimized_query, dict) else {},
                body=minimized_body if isinstance(minimized_body, dict) else None,
            )

            artifacts.append(
                {
                    "fingerprint": fingerprint,
                    "name": result.name,
                    "display_name": str(getattr(result, "display_name", "") or ""),
                    "name_raw": str(getattr(result, "name_raw", result.name) or result.name),
                    "method": scenario.method.upper(),
                    "endpoint": result.endpoint_resolved,
                    "expected_status": int(result.expected_status),
                    "actual_status": result.actual_status,
                    "error": str(result.error or ""),
                    "response_excerpt": str(result.response_excerpt or ""),
                    "mutation_strategy": str(getattr(result, "mutation_strategy", "") or ""),
                    "history_seeded": bool(getattr(result, "history_seeded", False)),
                    "request": {
                        "headers": headers,
                        "query": query,
                        "body": payload,
                    },
                    "minimized_request": {
                        "query": minimized_query,
                        "body": minimized_body,
                        "status": minimized.get("status"),
                    },
                    "curl_repro": curl_command,
                }
            )
            if len(artifacts) >= max(1, int(limit)):
                break

    def _build_curl_repro_command(
        self,
        *,
        method: str,
        base_url_hint: str,
        endpoint_resolved: str,
        headers: Dict[str, Any],
        query: Dict[str, Any],
        body: Optional[Dict[str, Any]],
    ) -> str:
        normalized_base = str(base_url_hint or "").rstrip("/")
        endpoint_path = str(endpoint_resolved or "")
        if endpoint_path.startswith("http://") or endpoint_path.startswith("https://"):
            repro_url = endpoint_path
        elif endpoint_path.startswith("/"):
            repro_url = f"{normalized_base}{endpoint_path}"
        else:
            repro_url = f"{normalized_base}/{endpoint_path}"

        query_items: List[Tuple[str, Any]] = []
        for key, value in (query or {}).items():
            if value is None:
                continue
            if isinstance(value, list):
                for item in value:
                    query_items.append((str(key), item))
            else:
                query_items.append((str(key), value))
        if query_items:
            query_text = urlencode(query_items, doseq=True)
            separator = "&" if "?" in repro_url else "?"
            repro_url = f"{repro_url}{separator}{query_text}"

        curl_parts = ["curl", "-X", str(method).upper(), shlex.quote(repro_url)]
        rendered_header_keys: set[str] = set()
        for key, value in (headers or {}).items():
            key_text = str(key)
            lower_key = key_text.lower()
            rendered_header_keys.add(lower_key)
            value_text = "Bearer ***" if lower_key == "authorization" else str(value)
            curl_parts.extend(["-H", shlex.quote(f"{key_text}: {value_text}")])

        if body is not None:
            if "content-type" not in rendered_header_keys:
                curl_parts.extend(["-H", shlex.quote("Content-Type: application/json")])
            curl_parts.extend(
                ["-d", shlex.quote(json.dumps(body, ensure_ascii=True, separators=(",", ":")))]
            )
        return " ".join(curl_parts)

    def _shrink_failed_input_for_repro(
        self,
        *,
        client: Any,
        scenario: TestScenario,
        baseline_status: Optional[int],
    ) -> Dict[str, Any]:
        method = str(scenario.method).upper()
        endpoint_resolved = self._resolve_endpoint_path(
            scenario.endpoint,
            scenario.params,
            scenario.expected_status,
        )
        headers = self._render_headers(scenario.headers)
        query = dict(self._strip_path_params(scenario.endpoint, scenario.params))
        body = deepcopy(scenario.body) if isinstance(scenario.body, dict) else None
        target_status = int(baseline_status) if baseline_status is not None else None

        def _status(q: Dict[str, Any], b: Optional[Dict[str, Any]]) -> Optional[int]:
            try:
                response = client.request(
                    method=method,
                    url=endpoint_resolved,
                    headers=headers,
                    params=q or None,
                    json=b if method in {"POST", "PUT", "PATCH"} else None,
                )
                return int(response.status_code)
            except Exception:
                return None

        if target_status is None:
            target_status = _status(query, body)
            if target_status is None:
                return {"query": query, "body": body, "status": None}

        if isinstance(body, dict):
            for key in list(body.keys()):
                candidate = deepcopy(body)
                candidate.pop(key, None)
                status = _status(query, candidate)
                if status == target_status:
                    body = candidate

        if isinstance(query, dict):
            for key in list(query.keys()):
                candidate_query = dict(query)
                candidate_query.pop(key, None)
                status = _status(candidate_query, body)
                if status == target_status:
                    query = candidate_query

        return {"query": query, "body": body, "status": target_status}

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
        mutation_policy = report.get("mutation_policy", {})
        repair_policy = report.get("repair_policy", {})
        script_execution = report.get("generated_script_execution", {})
        scenario_context = report.get("scenario_context", {})
        spec_intelligence = report.get("spec_intelligence", {}) or {}
        oss_tooling = report.get("oss_tooling", {}) or {}

        lines = [
            "# QA Specialist Execution Report",
            "",
            "## Run Metadata",
            f"- Spec: `{metadata['spec_title']}` ({metadata['spec_version']})",
            f"- Spec Path: `{metadata['spec_path']}`",
            f"- Tenant: `{metadata['tenant_id']}`",
            f"- Workspace: `{metadata.get('workspace_id', '')}`",
            f"- Run ID: `{metadata.get('run_id', '')}`",
            f"- Spec Key: `{metadata.get('spec_key', '')}`",
            f"- Isolation: `{metadata['isolation_mode']}`",
            f"- Environment Profile: `{metadata.get('environment_profile', '')}`",
            f"- LLM Scenario Debug Log: `{metadata.get('llm_scenario_debug_log', '')}`",
            f"- Runtime Cap (sec): `{summary.get('max_runtime_sec', 'n/a')}`",
            f"- Runtime Cap Hit: `{summary.get('runtime_cap_hit', False)}`",
            f"- Runtime Skipped: `{summary.get('runtime_skipped_scenarios', 0)}`",
            f"- Unsafe Actions Blocked: `{summary.get('unsafe_actions_blocked', 0)}`",
            f"- LLM Token Cap: `{(metadata.get('runtime_caps', {}) or {}).get('llm_token_cap', 'n/a')}`",
            f"- Stage Metrics (ms): `{metadata.get('stage_metrics_ms', {})}`",
            f"- Generated At: `{metadata['generated_at']}`",
            f"- Execution Time: `{metadata['execution_seconds']}s`",
            "",
            "## Summary",
            f"- Total Scenarios: `{summary['total_scenarios']}`",
            f"- Passed: `{summary['passed_scenarios']}`",
            f"- Failed: `{summary['failed_scenarios']}`",
            f"- Suspect: `{summary.get('suspect_scenarios', 0)}`",
            f"- Blocked: `{summary.get('blocked_scenarios', 0)}`",
            f"- Pass Rate: `{summary['pass_rate']}`",
            f"- True Pass Rate: `{summary.get('true_pass_rate', summary['pass_rate'])}`",
            f"- Quality Gate ({summary['pass_threshold']}): `{summary['meets_quality_gate']}`",
            f"- Quality Gate Fail Reasons: `{summary.get('quality_gate_fail_reasons', [])}`",
            f"- Avg Duration: `{summary['average_duration_ms']} ms`",
            f"- Contract Checks Run: `{summary.get('contract_checks_run', 0)}`",
            f"- Contract Check Failures: `{summary.get('contract_check_failures', 0)}`",
            f"- Corrected Expectations: `{summary.get('corrected_expectations', 0)}`",
            "",
            "## Test Type Breakdown",
        ]

        for test_type, counts in summary["test_type_breakdown"].items():
            lines.append(
                f"- `{test_type}`: total={counts.get('total', 0)}, passed={counts.get('passed', 0)}, failed={counts.get('failed', 0)}, suspect={counts.get('suspect', 0)}, blocked={counts.get('blocked', 0)}"
            )

        lines.extend(
            [
                "",
                "## Generated Script Execution",
                f"- Script Kind: `{metadata.get('script_kind', '')}`",
                f"- RL Train Mode: `{metadata.get('rl_train_mode', '')}`",
                f"- Status: `{script_execution.get('status', 'n/a')}`",
                f"- Executed: `{script_execution.get('executed', False)}`",
                f"- Script Path: `{script_execution.get('script_path', '')}`",
                f"- Total Tests: `{script_execution.get('total_tests', 0)}`",
                f"- Passed Tests: `{script_execution.get('passed_tests', 0)}`",
                f"- Failed Tests: `{script_execution.get('failed_tests', 0)}`",
                f"- Pass Rate: `{script_execution.get('pass_rate', 0)}`",
            ]
        )
        if script_execution.get("error"):
            lines.append(f"- Error: `{script_execution.get('error')}`")

        lines.extend(
            [
                "",
                "## Learning Loop",
                f"- Run Reward: `{feedback.get('run_reward', 0.0)}`",
                f"- Average Decision Reward: `{feedback.get('average_decision_reward', 0.0)}`",
                f"- Rewarded Decisions: `{feedback.get('rewarded_decisions', 0)}`",
                f"- Penalized Decisions: `{feedback.get('penalized_decisions', 0)}`",
                f"- Learning Delta Status: `{learning.get('learning_delta_status', 'n/a')}`",
                f"- Learning Delta Reason: `{learning.get('learning_delta_reason', '')}`",
                f"- Learning Run Count: `{state_snapshot.get('run_count', 0)}`",
                f"- Learning State File: `{learning.get('state_file', '')}`",
                f"- RL Checkpoint File: `{learning.get('agent_lightning_checkpoint', '')}`",
            ]
        )
        weak_deltas = report.get("weak_pattern_deltas", {}) or {}
        lines.extend(
            [
                f"- Weak Pattern Deltas: `{weak_deltas.get('total_changed_patterns', 0)}`",
                f"- Weak Improved: `{weak_deltas.get('improved_count', 0)}`",
                f"- Weak Regressed: `{weak_deltas.get('regressed_count', 0)}`",
                f"- Repro Artifacts: `{len(report.get('repro_artifacts', []) or [])}`",
            ]
        )
        policy_movement = learning.get("policy_movement", {}) or {}
        if policy_movement:
            lines.extend(
                [
                    f"- Policy Movement Status: `{policy_movement.get('status', 'n/a')}`",
                    f"- Policy Turnover Ratio: `{policy_movement.get('turnover_ratio', 0)}`",
                    f"- Policy Jaccard Similarity: `{policy_movement.get('jaccard_similarity', 0)}`",
                ]
            )

        lines.extend(
            [
                "",
                "## Spec Intelligence",
                f"- Operations Total: `{spec_intelligence.get('operations_total', 0)}`",
                f"- Dependency Edge Count: `{(spec_intelligence.get('dependency_graph', {}) or {}).get('edge_count', 0)}`",
                f"- Workflow Candidates: `{len(spec_intelligence.get('workflow_candidates', []) or [])}`",
                "",
                "## OSS Tooling",
                f"- Python Packages: `{(oss_tooling.get('packages', {}) or {})}`",
                f"- CLI Binaries: `{(oss_tooling.get('binaries', {}) or {})}`",
                f"- Spec Validation: `{(oss_tooling.get('spec_validation', {}) or {})}`",
                "",
                "## Adaptive Selection Policy",
                f"- Algorithm: `{selection_policy.get('algorithm', 'n/a')}`",
                f"- Candidate Scenarios: `{selection_policy.get('candidate_count', 0)}`",
                f"- Base Candidates: `{selection_policy.get('base_candidate_count', 0)}`",
                f"- RL Mutated Candidates: `{selection_policy.get('mutated_candidate_count', 0)}`",
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

        scenario_counts = scenario_context.get("counts", {})
        source_breakdown = scenario_context.get("source_breakdown", {})
        selected_source_breakdown = source_breakdown.get("selected", {})
        lines.extend(
            [
                "",
                "## Scenario Learning Context",
                f"- Run Count Before: `{scenario_context.get('run_count_before', 'n/a')}`",
                f"- Run Count After: `{scenario_context.get('run_count_after', 'n/a')}`",
                f"- Base Generated: `{scenario_counts.get('base_generated', 0)}`",
                f"- Candidate Pool: `{scenario_counts.get('candidate_total', 0)}`",
                f"- Selected Total: `{scenario_counts.get('selected_total', 0)}`",
                f"- Selected New vs History: `{scenario_counts.get('selected_new_vs_history', 0)}`",
                f"- Selected Historical Weak Patterns: `{scenario_counts.get('selected_historical_weak_patterns', 0)}`",
                f"- Selected Source LLM Base: `{selected_source_breakdown.get('llm_base', 0)}`",
                f"- Selected Source RL Mutation: `{selected_source_breakdown.get('rl_mutation', 0)}`",
                f"- Selected Source RL History Seed: `{selected_source_breakdown.get('rl_history_seed', 0)}`",
            ]
        )

        selected_scenarios = scenario_context.get("selected_scenarios", [])
        if isinstance(selected_scenarios, list) and selected_scenarios:
            lines.append("- Sample Selected Scenarios:")
            for item in selected_scenarios[:10]:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    "- "
                    + f"`{item.get('name', '')}` "
                    + f"source={item.get('source', 'n/a')} "
                    + f"strategy={item.get('mutation_strategy', 'n/a')} "
                    + f"reason={item.get('selection_reason', 'n/a')} "
                    + f"expected={item.get('expected_status', 'n/a')} "
                    + f"actual={item.get('actual_status', 'n/a')} "
                    + f"passed={item.get('passed', 'n/a')}"
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
                "## RL Mutation Stage",
                f"- Enabled: `{mutation_policy.get('enabled', False)}`",
                f"- Base Candidates: `{mutation_policy.get('base_candidate_count', 0)}`",
                f"- Targeted Candidates: `{mutation_policy.get('targeted_candidates', 0)}`",
                f"- Mutated Added: `{mutation_policy.get('mutated_candidates_added', 0)}`",
                f"- Final Candidates: `{mutation_policy.get('final_candidate_count', 0)}`",
                f"- Priority Threshold: `{mutation_policy.get('priority_threshold', 0)}`",
            ]
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
        gam_diag = gam.get("diagnostics", {})
        gam_warnings = gam_diag.get("warnings", [])
        lines.extend(
            [
                "",
                "## GAM Memory",
                f"- Session ID: `{gam.get('session_id', '')}`",
                f"- Memo Page ID: `{gam.get('memo_page_id', '')}`",
                f"- Memo Title: `{gam.get('memo_title', '')}`",
                f"- Memory Store Path: `{gam.get('memory_store_path', '')}`",
                f"- Research Plan Items: `{gam.get('research_plan_count', 0)}`",
                f"- Research Excerpts: `{gam.get('research_excerpt_count', 0)}`",
                f"- Research Reflection: `{gam.get('research_reflection', '')}`",
                f"- Quality Score: `{gam_diag.get('quality_score', 'n/a')}`",
                f"- Convention Excerpts: `{gam_diag.get('convention_excerpts', 0)}`",
                f"- Non-Convention Excerpts: `{gam_diag.get('non_convention_excerpts', 0)}`",
                f"- Actionable Excerpts: `{gam_diag.get('actionable_excerpt_count', 0)}`",
                f"- Machine-Like Excerpts: `{gam_diag.get('machine_like_excerpt_count', 0)}`",
                "- Warnings: `"
                + (
                    ", ".join(str(item) for item in gam_warnings)
                    if isinstance(gam_warnings, list) and gam_warnings
                    else "none"
                )
                + "`",
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
                f"- Training Mode: `{training_stats.get('train_mode', 'periodic')}`",
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
        "--workspace-id",
        default=None,
        help="Workspace id for multi-user isolation (default: tenant-id)",
    )
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
        "--max-runtime-sec",
        type=int,
        default=None,
        help="Runtime cap in seconds for scenario execution",
    )
    parser.add_argument(
        "--llm-token-cap",
        type=int,
        default=None,
        help="Optional max token cap applied to GAM + scenario LLM calls",
    )
    parser.add_argument(
        "--environment-profile",
        default=DEFAULT_ENVIRONMENT_PROFILE,
        choices=sorted(SUPPORTED_ENVIRONMENT_PROFILES),
        help="Execution environment profile used for reward weighting",
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
    parser.add_argument(
        "--script-kind",
        default=DEFAULT_SCRIPT_KIND,
        choices=sorted(SUPPORTED_SCRIPT_KINDS),
        help="Generated script kind to produce and execute (default: python_pytest)",
    )
    parser.add_argument(
        "--rl-train-mode",
        default="periodic",
        choices=["periodic"],
        help="RL training mode (mandatory): periodic",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    agent = QASpecialistAgent(
        spec_path=args.spec,
        nlp_prompt=args.prompt,
        tenant_id=args.tenant_id,
        workspace_id=args.workspace_id,
        base_url=args.base_url,
        output_dir=args.output_dir,
        max_scenarios=args.max_scenarios,
        max_runtime_sec=args.max_runtime_sec,
        llm_token_cap=args.llm_token_cap,
        environment_profile=args.environment_profile,
        pass_threshold=args.pass_threshold,
        rl_checkpoint_path=args.rl_checkpoint,
        learning_state_path=args.learning_state,
        script_kind=args.script_kind,
        rl_train_mode=args.rl_train_mode,
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
    llm_debug_log = str((report.get("report_files", {}) or {}).get("llm_scenario_debug", "")).strip()
    if llm_debug_log:
        print(f"LLM scenario debug log: {llm_debug_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
