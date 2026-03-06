#!/usr/bin/env python3
"""
Multi-Language API Testing Agent
Comprehensive API testing agent that thinks and acts like a human tester
Supports multiple programming languages and testing frameworks
"""

import json
import logging
import os
import pprint
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from urllib.parse import quote, unquote, urlparse
import yaml
import requests
from pathlib import Path

logger = logging.getLogger(__name__)
MISSING_PATH_PARAM_SENTINEL = "__qa_missing_path_param__"


class TestLanguage(Enum):
    """Supported testing languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    CURL = "curl"
    POSTMAN = "postman"


class TestType(Enum):
    """Types of tests like a human tester would do."""
    HAPPY_PATH = "happy_path"
    ERROR_HANDLING = "error_handling"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INPUT_VALIDATION = "input_validation"
    BOUNDARY_TESTING = "boundary_testing"
    PERFORMANCE = "performance"
    SECURITY = "security"
    EDGE_CASES = "edge_cases"
    INTEGRATION = "integration"


@dataclass
class TestScenario:
    """A test scenario like a human tester would design."""
    name: str
    description: str
    test_type: TestType
    endpoint: str
    method: str
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    expected_status: int = 200
    expected_response_fields: List[str] = field(default_factory=list)
    assertions: List[str] = field(default_factory=list)


@dataclass
class APIEndpoint:
    """Parsed API endpoint information."""
    path: str
    method: str
    summary: str
    parameters: List[Dict[str, Any]]
    request_body: Optional[Dict[str, Any]]
    responses: Dict[str, Any]
    auth_required: bool = False
    auth_type: Optional[str] = None


class HumanTesterSimulator:
    """Simulates how a human tester would analyze and test an API with NLP prompts."""
    
    def __init__(
        self,
        api_spec: Dict[str, Any],
        base_url: str,
        llm_debug_log_path: Optional[str] = None,
    ):
        """Initialize with API specification and base URL."""
        self.api_spec = api_spec
        self.base_url = base_url.rstrip('/')
        self.endpoints = self._parse_endpoints()
        self.test_scenarios = []
        self.error_fixes = {}  # Store automatic fixes for errors
        self.workflow_chains = []  # Store workflow orchestrations
        self.last_generation_engine = "heuristic_default"
        self.llm_stats: Dict[str, int] = {
            "scenario_calls": 0,
            "scenario_success": 0,
            "scenario_errors": 0,
            "scenario_parse_failures": 0,
            "scenario_schema_rejections": 0,
        }
        self.last_llm_generation_diagnostics: Dict[str, Any] = {}
        self._llm_mode = str(os.getenv("QA_SCENARIO_LLM_MODE", "auto")).strip().lower() or "auto"
        self._llm_model = str(os.getenv("QA_SCENARIO_LLM_MODEL", "gpt-4.1-mini")).strip() or "gpt-4.1-mini"
        self._llm_temperature = self._safe_float(os.getenv("QA_SCENARIO_LLM_TEMPERATURE"), default=0.1)
        self._llm_max_tokens = self._safe_int(os.getenv("QA_SCENARIO_LLM_MAX_TOKENS"), default=2200)
        self._llm_timeout = self._safe_float(os.getenv("QA_SCENARIO_LLM_TIMEOUT_SECONDS"), default=20.0)
        self._llm_max_retries = self._safe_int(os.getenv("QA_SCENARIO_LLM_MAX_RETRIES"), default=1)
        self._llm_debug_log_path = self._resolve_llm_debug_log_path(llm_debug_log_path)
        self._llm_client = self._init_llm_client()
        self.llm_enabled = self._llm_client is not None
        
    def _parse_endpoints(self) -> List[APIEndpoint]:
        """Parse OpenAPI spec like a human tester would analyze it."""
        endpoints = []
        
        if 'paths' not in self.api_spec:
            return endpoints
            
        for path, path_info in self.api_spec['paths'].items():
            if not isinstance(path_info, dict):
                continue
            path_parameters = path_info.get('parameters', [])
            for method, method_info in path_info.items():
                if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                    operation_info = method_info if isinstance(method_info, dict) else {}
                    
                    # Extract parameters
                    parameters = self._merge_operation_parameters(
                        path_parameters,
                        operation_info.get('parameters', []),
                    )
                    
                    # Check authentication
                    auth_required = 'security' in operation_info or 'security' in self.api_spec
                    auth_type = None
                    if auth_required:
                        security = operation_info.get('security', self.api_spec.get('security', []))
                        if security:
                            auth_type = list(security[0].keys())[0] if security[0] else None
                    
                    endpoint = APIEndpoint(
                        path=path,
                        method=method.upper(),
                        summary=operation_info.get('summary', f"{method.upper()} {path}"),
                        parameters=parameters,
                        request_body=operation_info.get('requestBody'),
                        responses=operation_info.get('responses', {}),
                        auth_required=auth_required,
                        auth_type=auth_type
                    )
                    endpoints.append(endpoint)
        
        return endpoints

    def _merge_operation_parameters(
        self,
        path_parameters: Any,
        operation_parameters: Any,
    ) -> List[Dict[str, Any]]:
        """Merge path-level and operation-level OpenAPI parameters."""
        merged: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for source in (path_parameters, operation_parameters):
            if not isinstance(source, list):
                continue
            for param in source:
                if not isinstance(param, dict):
                    continue
                location = str(param.get("in", "")).lower()
                name = str(param.get("name", ""))
                if not location or not name:
                    continue
                merged[(location, name)] = param

        return list(merged.values())

    @staticmethod
    def _safe_float(value: Optional[str], default: float) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _safe_int(value: Optional[str], default: int) -> int:
        try:
            if value is None:
                return int(default)
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _resolve_llm_debug_log_path(path_value: Optional[str]) -> str:
        candidate = str(path_value or os.getenv("QA_SCENARIO_LLM_DEBUG_LOG", "")).strip()
        if not candidate:
            return ""
        path = Path(candidate).expanduser().resolve()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            return ""
        return str(path)

    @staticmethod
    def _text_preview(value: Any, max_chars: int = 240) -> str:
        cleaned = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 1] + "..."

    def _log_llm_debug(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        path = str(self._llm_debug_log_path or "").strip()
        if not path:
            return
        row: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": str(event),
            "llm_mode": str(self._llm_mode),
            "llm_model": str(self._llm_model),
        }
        if isinstance(payload, dict):
            row.update(payload)
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        except Exception as exc:
            logger.warning("Failed writing LLM scenario debug log (%s): %s", path, exc)

    def _init_llm_client(self) -> Any:
        if self._llm_mode == "off":
            self._log_llm_debug("llm_disabled", {"reason": "QA_SCENARIO_LLM_MODE=off"})
            return None
        api_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
        if not api_key:
            if self._llm_mode == "on":
                logger.warning(
                    "QA_SCENARIO_LLM_MODE=on but OPENAI_API_KEY is missing. "
                    "Falling back to heuristic scenario generation."
                )
            self._log_llm_debug("llm_unavailable", {"reason": "missing_openai_api_key"})
            return None
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=api_key,
                timeout=float(self._llm_timeout),
                max_retries=max(0, int(self._llm_max_retries)),
            )
            self._log_llm_debug("llm_client_initialized", {"timeout_sec": float(self._llm_timeout)})
            return client
        except Exception as exc:
            logger.warning("Failed to initialize OpenAI client for scenario generation: %s", exc)
            self._log_llm_debug(
                "llm_unavailable",
                {"reason": "client_init_failed", "error": self._text_preview(str(exc), 300)},
            )
            return None

    @staticmethod
    def _extract_bracketed_json(raw: str) -> str:
        text = str(raw or "")
        start_idx = -1
        opening = ""
        for idx, ch in enumerate(text):
            if ch in "{[":
                start_idx = idx
                opening = ch
                break
        if start_idx < 0:
            return ""
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start_idx, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    return text[start_idx : idx + 1]
        return ""

    def _extract_json_object(self, text: str) -> Optional[Any]:
        parsed, _ = self._extract_json_object_detailed(text)
        return parsed

    @staticmethod
    def _normalize_json_candidate(raw: str) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        # Some models prepend "json" before the object.
        if text.lower().startswith("json\n"):
            text = text[5:].lstrip()
        # Normalize typographic quotes and remove control chars.
        text = (
            text.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
        )
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        # Recover common invalid JSON emitted by LLMs.
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return text.strip()

    @staticmethod
    def _repair_truncated_json_candidate(raw: str) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        text = HumanTesterSimulator._normalize_json_candidate(text)
        if not text:
            return ""
        # If output was cut right after a key separator, inject null so we can close the structure.
        text = re.sub(r":\s*$", ": null", text)
        text = re.sub(r",\s*$", "", text)

        stack: List[str] = []
        in_string = False
        escaped = False
        for ch in text:
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch in "{[":
                stack.append(ch)
            elif ch == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif ch == "]" and stack and stack[-1] == "[":
                stack.pop()

        if in_string:
            # Best-effort close for unterminated JSON string.
            if text.endswith("\\"):
                text = text[:-1]
            text += '"'
        while stack:
            opener = stack.pop()
            text += "}" if opener == "{" else "]"
        return HumanTesterSimulator._normalize_json_candidate(text)

    def _extract_json_object_detailed(self, text: str) -> Tuple[Optional[Any], Dict[str, Any]]:
        raw = str(text or "").strip()
        if not raw:
            return None, {
                "status": "empty",
                "parse_attempts": 0,
                "parse_failures": 0,
                "strategy": "",
                "errors": [],
            }

        candidates: List[str] = []
        fenced_blocks = re.findall(
            r"```(?:json)?\s*([\s\S]*?)```",
            raw,
            flags=re.IGNORECASE,
        )
        candidates.extend(block.strip() for block in fenced_blocks if str(block).strip())
        candidates.append(raw)

        parse_attempts = 0
        parse_errors: List[Dict[str, str]] = []
        for candidate in candidates:
            variants: List[Tuple[str, str]] = [("raw", candidate)]
            bracketed = self._extract_bracketed_json(candidate)
            if bracketed and bracketed != candidate:
                variants.append(("bracketed", bracketed))
            normalized = self._normalize_json_candidate(candidate)
            if normalized and normalized != candidate:
                variants.append(("normalized", normalized))
                normalized_bracketed = self._extract_bracketed_json(normalized)
                if normalized_bracketed and normalized_bracketed != normalized:
                    variants.append(("normalized_bracketed", normalized_bracketed))
            repaired = self._repair_truncated_json_candidate(candidate)
            if repaired and repaired not in {candidate, normalized}:
                variants.append(("repaired", repaired))
                repaired_bracketed = self._extract_bracketed_json(repaired)
                if repaired_bracketed and repaired_bracketed not in {repaired, candidate, normalized}:
                    variants.append(("repaired_bracketed", repaired_bracketed))

            seen_variant_payloads: set[str] = set()
            for strategy, payload in variants:
                if payload in seen_variant_payloads:
                    continue
                seen_variant_payloads.add(payload)
                parse_attempts += 1
                try:
                    parsed = json.loads(payload)
                    if isinstance(parsed, (dict, list)):
                        return parsed, {
                            "status": "ok",
                            "parse_attempts": int(parse_attempts),
                            "parse_failures": int(len(parse_errors)),
                            "strategy": str(strategy),
                            "errors": parse_errors[:6],
                        }
                except Exception as exc:
                    if len(parse_errors) < 12:
                        parse_errors.append(
                            {
                                "strategy": str(strategy),
                                "error": self._text_preview(str(exc), 240),
                            }
                        )
                    continue
        return None, {
            "status": "parse_failed",
            "parse_attempts": int(parse_attempts),
            "parse_failures": int(len(parse_errors)),
            "strategy": "",
            "errors": parse_errors[:6],
        }

    def _extract_scenario_candidates(self, parsed: Any) -> List[Dict[str, Any]]:
        def _as_dict_list(value: Any) -> List[Dict[str, Any]]:
            if not isinstance(value, list):
                return []
            return [item for item in value if isinstance(item, dict)]

        if isinstance(parsed, list):
            return _as_dict_list(parsed)

        if not isinstance(parsed, dict):
            return []

        preferred_keys = (
            "scenarios",
            "tests",
            "test_scenarios",
            "cases",
            "items",
            "results",
            "data",
            "output",
        )
        for key in preferred_keys:
            value = parsed.get(key)
            if isinstance(value, list):
                rows = _as_dict_list(value)
                if rows:
                    return rows
            elif isinstance(value, dict):
                rows = self._extract_scenario_candidates(value)
                if rows:
                    return rows

        if any(
            key in parsed
            for key in (
                "method",
                "http_method",
                "verb",
                "endpoint",
                "path",
                "url",
                "operation",
                "endpoint_key",
            )
        ):
            return [parsed]

        for value in parsed.values():
            if isinstance(value, (dict, list)):
                rows = self._extract_scenario_candidates(value)
                if rows:
                    return rows
        return []

    def _normalize_llm_method_and_endpoint(self, item: Dict[str, Any]) -> Tuple[str, str]:
        method = str(
            item.get("method")
            or item.get("http_method")
            or item.get("verb")
            or ""
        ).strip().upper()
        endpoint = str(
            item.get("endpoint")
            or item.get("path")
            or item.get("url")
            or item.get("uri")
            or item.get("route")
            or ""
        ).strip()

        operation_hint = str(item.get("operation") or item.get("endpoint_key") or "").strip()
        for candidate in (endpoint, operation_hint):
            if not candidate:
                continue
            match = re.match(
                r"^\s*(GET|POST|PUT|PATCH|DELETE)\s+(.+?)\s*$",
                candidate,
                flags=re.IGNORECASE,
            )
            if match:
                if not method:
                    method = str(match.group(1)).upper()
                endpoint = str(match.group(2)).strip()
                break

        return method, endpoint

    @staticmethod
    def _segment_is_placeholder(segment: str) -> bool:
        seg = str(segment or "").strip()
        if not seg:
            return False
        if (seg.startswith("{") and seg.endswith("}")) or (seg.startswith("<") and seg.endswith(">")):
            return True
        if seg.startswith(":") and len(seg) > 1:
            return True
        return False

    @staticmethod
    def _normalize_static_segment(segment: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(segment or "").strip().lower())

    def _resolve_endpoint_template(self, endpoint: str, method: Optional[str] = None) -> str:
        candidate = str(endpoint or "").strip()
        method_filter = str(method or "").strip().upper()
        if not candidate:
            return ""

        op_match = re.match(
            r"^\s*(GET|POST|PUT|PATCH|DELETE)\s+(.+?)\s*$",
            candidate,
            flags=re.IGNORECASE,
        )
        if op_match:
            if not method_filter:
                method_filter = str(op_match.group(1)).upper()
            candidate = str(op_match.group(2)).strip()

        parsed_url = urlparse(candidate)
        if parsed_url.scheme and parsed_url.path:
            candidate = parsed_url.path
        candidate = unquote(str(candidate or "").strip())
        candidate = candidate.split("?", 1)[0].split("#", 1)[0]
        candidate = re.sub(r"[.,;]+$", "", candidate).strip()
        candidate = re.sub(r":([a-zA-Z_][a-zA-Z0-9_]*)", r"{\1}", candidate)
        candidate = re.sub(r"<([a-zA-Z_][a-zA-Z0-9_]*)>", r"{\1}", candidate)
        if candidate and not candidate.startswith("/"):
            candidate = "/" + candidate
        if not candidate:
            return ""

        known_paths = {
            str(ep.path).strip()
            for ep in self.endpoints
            if not method_filter or str(ep.method).upper() == method_filter
        }
        if not known_paths:
            known_paths = {str(ep.path).strip() for ep in self.endpoints}

        if candidate in known_paths:
            return candidate

        lower_exact = {path.lower(): path for path in known_paths}
        if candidate.lower() in lower_exact:
            return lower_exact[candidate.lower()]

        candidate_parts = [part for part in candidate.strip("/").split("/") if part]
        if not candidate_parts:
            return ""

        best_path = ""
        best_score = -1.0
        for path in known_paths:
            path_parts = [part for part in path.strip("/").split("/") if part]
            if len(path_parts) != len(candidate_parts):
                continue
            matched = True
            score = 0.0
            for path_seg, cand_seg in zip(path_parts, candidate_parts):
                if self._segment_is_placeholder(path_seg) or self._segment_is_placeholder(cand_seg):
                    score += 0.25
                    continue
                if self._normalize_static_segment(path_seg) == self._normalize_static_segment(cand_seg):
                    score += 1.0
                else:
                    matched = False
                    break
            if matched and score > best_score:
                best_path = path
                best_score = score

        return best_path

    def _coerce_test_type(self, value: Any) -> TestType:
        normalized = re.sub(r"[\s\-]+", "_", str(value or "").strip().lower())
        aliases = {
            "auth": TestType.AUTHENTICATION,
            "authn": TestType.AUTHENTICATION,
            "authz": TestType.AUTHORIZATION,
            "validation": TestType.INPUT_VALIDATION,
            "boundary": TestType.BOUNDARY_TESTING,
            "error": TestType.ERROR_HANDLING,
            "edge": TestType.EDGE_CASES,
        }
        if normalized in aliases:
            return aliases[normalized]
        for test_type in TestType:
            if normalized == test_type.value:
                return test_type
        return TestType.ERROR_HANDLING

    def _default_expected_status(self, test_type: TestType, method: str) -> int:
        if test_type in {TestType.AUTHENTICATION, TestType.AUTHORIZATION}:
            return 401
        if test_type in {TestType.ERROR_HANDLING, TestType.INPUT_VALIDATION, TestType.BOUNDARY_TESTING, TestType.SECURITY}:
            return 400
        if test_type == TestType.HAPPY_PATH:
            return 201 if method.upper() == "POST" else 200
        return 200

    def _documented_statuses_for_operation(self, method: str, endpoint: str) -> List[int]:
        method_u = str(method or "").strip().upper()
        endpoint_norm = str(endpoint or "").strip()
        for item in self.endpoints:
            if str(item.method).upper() != method_u:
                continue
            if str(item.path).strip() != endpoint_norm:
                continue
            statuses: List[int] = []
            responses = item.responses if isinstance(item.responses, dict) else {}
            for key in responses.keys():
                raw = str(key).strip()
                if raw.isdigit():
                    code = int(raw)
                    if 100 <= code <= 599:
                        statuses.append(code)
            return sorted(set(statuses))
        return []

    @staticmethod
    def _is_attack_like_security_signal(
        *,
        name: str,
        description: str,
        params: Dict[str, Any],
        body: Optional[Dict[str, Any]],
    ) -> bool:
        attack_markers = (
            "sql injection",
            "drop table",
            "union select",
            " or 1=1",
            "--",
            "<script",
            "xss",
            "javascript:",
            "../",
            "%3cscript",
        )
        haystack = " ".join(
            [
                str(name or ""),
                str(description or ""),
                json.dumps(params or {}, ensure_ascii=True),
                json.dumps(body or {}, ensure_ascii=True),
            ]
        ).lower()
        return any(marker in haystack for marker in attack_markers)

    def _normalize_llm_expected_status(
        self,
        *,
        test_type: TestType,
        method: str,
        endpoint: str,
        expected_status: int,
        name: str,
        description: str,
        params: Dict[str, Any],
        body: Optional[Dict[str, Any]],
    ) -> int:
        method_u = str(method or "").strip().upper()
        status = int(expected_status)
        if status < 100 or status > 599:
            status = self._default_expected_status(test_type, method_u)

        documented = self._documented_statuses_for_operation(method_u, endpoint)
        documented_negative = [s for s in documented if 400 <= s < 500]
        documented_non_auth_negative = [s for s in documented_negative if s not in {401, 403}]
        documented_happy = [s for s in documented if 200 <= s < 300]

        if test_type in {TestType.AUTHENTICATION, TestType.AUTHORIZATION}:
            return 401

        if test_type == TestType.HAPPY_PATH:
            if 200 <= status < 300:
                return status
            if documented_happy:
                return int(documented_happy[0])
            return 201 if method_u == "POST" else 200

        if test_type == TestType.SECURITY:
            is_attack = self._is_attack_like_security_signal(
                name=name,
                description=description,
                params=params,
                body=body,
            )
            if is_attack and 200 <= status < 300:
                if documented_non_auth_negative:
                    return int(documented_non_auth_negative[0])
                return 400
            return status

        if test_type in {
            TestType.ERROR_HANDLING,
            TestType.INPUT_VALIDATION,
            TestType.BOUNDARY_TESTING,
            TestType.EDGE_CASES,
        } and 200 <= status < 300:
            if documented_negative:
                return int(documented_negative[0])
            return 400
        return status

    def _generate_from_nlp_prompt_llm(self, prompt: str) -> List[TestScenario]:
        if self._llm_client is None:
            self.last_llm_generation_diagnostics = {
                "status": "skipped",
                "reason": "llm_client_missing",
            }
            self._log_llm_debug("planner_skip", {"reason": "llm_client_missing"})
            return []

        self.llm_stats["scenario_calls"] = int(self.llm_stats.get("scenario_calls", 0)) + 1
        self.last_llm_generation_diagnostics = {
            "status": "in_progress",
            "response_mode": "",
            "parse_diagnostics": {},
            "candidate_rows": 0,
            "accepted_rows": 0,
            "drop_counts": {},
        }
        spec_info = self.api_spec.get("info", {}) if isinstance(self.api_spec, dict) else {}
        operation_pairs = [
            {"method": str(ep.method).upper(), "path": str(ep.path), "auth_required": bool(ep.auth_required)}
            for ep in self.endpoints
        ][:40]
        payload = {
            "spec_title": str(spec_info.get("title", "")),
            "spec_version": str(spec_info.get("version", "")),
            "prompt": str(prompt),
            "available_operations": operation_pairs,
            "allowed_test_types": [t.value for t in TestType],
            "constraints": {
                "max_scenarios": max(12, min(24, len(operation_pairs) * 5)),
                "prefer_negative_tests": False,
                "require_happy_path_per_operation": True,
                "keep_endpoint_as_openapi_template": True,
            },
        }
        print(
            "   🧠 LLM scenario planner call "
            f"(model={self._llm_model}, timeout={self._llm_timeout:.1f}s, retries={self._llm_max_retries})"
        )
        self._log_llm_debug(
            "planner_start",
            {
                "prompt_chars": len(str(prompt or "")),
                "operations_available": len(operation_pairs),
                "timeout_sec": float(self._llm_timeout),
                "max_retries": int(self._llm_max_retries),
            },
        )

        def _request_llm(response_mode: str) -> Any:
            mode = str(response_mode or "plain").strip().lower()
            kwargs: Dict[str, Any] = {
                "model": self._llm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You generate API QA test scenarios from an OpenAPI operation list.\n"
                            "Return STRICT JSON with key 'scenarios' as an array.\n"
                            "Each scenario object keys: name, description, test_type, method, endpoint, "
                            "expected_status, headers, params, body.\n"
                            "Rules: endpoint must match available OpenAPI template path; "
                            "method should be a valid HTTP verb; no markdown.\n"
                            "Keep values concise to avoid long/truncated outputs."
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
                ],
                "temperature": float(self._llm_temperature),
                "max_tokens": int(self._llm_max_tokens),
                "timeout": float(self._llm_timeout),
            }
            if mode == "json_schema":
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "qa_scenarios",
                        "strict": False,
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["scenarios"],
                            "properties": {
                                "scenarios": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["test_type", "method", "endpoint", "expected_status"],
                                        "properties": {
                                            "name": {"type": "string"},
                                            "description": {"type": "string"},
                                            "test_type": {"type": "string"},
                                            "method": {"type": "string"},
                                            "endpoint": {"type": "string"},
                                            "expected_status": {"type": "integer"},
                                            "headers": {"type": "object"},
                                            "params": {"type": "object"},
                                            "body": {"type": ["object", "null"]},
                                        },
                                    },
                                }
                            },
                        },
                    },
                }
            elif mode == "json_object":
                kwargs["response_format"] = {"type": "json_object"}
            return self._llm_client.chat.completions.create(**kwargs)

        started_at = time.time()
        try:
            response = None
            response_mode_used = ""
            try:
                print('*************************************************************************************')
                response = _request_llm("json_schema")
                response_mode_used = "json_schema"
            except Exception as schema_error:
                logger.warning(
                    "Scenario planner json_schema call failed, retrying with json_object mode: %s",
                    schema_error,
                )
                self._log_llm_debug(
                    "planner_json_schema_failed",
                    {"error": self._text_preview(str(schema_error), 300)},
                )
                try:
                    response = _request_llm("json_object")
                    response_mode_used = "json_object"
                except Exception as json_mode_error:
                    logger.warning(
                        "Scenario planner json_object call failed, retrying without response_format: %s",
                        json_mode_error,
                    )
                    self._log_llm_debug(
                        "planner_json_mode_failed",
                        {"error": self._text_preview(str(json_mode_error), 300)},
                    )
                    response = _request_llm("plain")
                    response_mode_used = "plain"
            elapsed = time.time() - started_at
            print(f"   🧠 LLM scenario planner response received in {elapsed:.2f}s")
            content = ""
            if getattr(response, "choices", None):
                msg = getattr(response.choices[0], "message", None)
                content = str(getattr(msg, "content", "") or "")
            self._log_llm_debug(
                "planner_response",
                {
                    "elapsed_sec": round(float(elapsed), 3),
                    "content_chars": len(content),
                    "content_preview": self._text_preview(content, 320),
                    "response_mode": response_mode_used,
                },
            )
            parsed, parse_diagnostics = self._extract_json_object_detailed(content)
            if parse_diagnostics.get("status") != "ok":
                self.llm_stats["scenario_parse_failures"] = int(
                    self.llm_stats.get("scenario_parse_failures", 0)
                ) + 1
                self._log_llm_debug(
                    "planner_parse_failed",
                    {
                        "response_mode": response_mode_used,
                        "parse_diagnostics": parse_diagnostics,
                    },
                )
            raw_scenarios = self._extract_scenario_candidates(parsed)
            drop_counts: Dict[str, int] = {}
            drop_samples: List[Dict[str, Any]] = []
            scenarios: List[TestScenario] = []
            seen_keys: set[tuple[str, str, str, int]] = set()
            if not raw_scenarios:
                drop_counts["no_candidates_extracted"] = 1
                parsed_type = type(parsed).__name__ if parsed is not None else "none"
                parsed_keys: List[str] = []
                if isinstance(parsed, dict):
                    parsed_keys = list(parsed.keys())[:12]
                if len(drop_samples) < 6:
                    drop_samples.append(
                        {
                            "reason": "no_candidates_extracted",
                            "parsed_type": str(parsed_type),
                            "parsed_keys": parsed_keys,
                            "parse_diagnostics": parse_diagnostics,
                        }
                    )
            for item in raw_scenarios:
                if not isinstance(item, dict):
                    drop_counts["non_dict_row"] = int(drop_counts.get("non_dict_row", 0)) + 1
                    continue
                method, endpoint_raw = self._normalize_llm_method_and_endpoint(item)
                if not method:
                    method = "GET"
                if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                    drop_counts["invalid_method"] = int(drop_counts.get("invalid_method", 0)) + 1
                    if len(drop_samples) < 6:
                        drop_samples.append(
                            {
                                "reason": "invalid_method",
                                "method": str(method),
                                "endpoint": self._text_preview(endpoint_raw, 100),
                            }
                        )
                    continue
                endpoint = self._resolve_endpoint_template(endpoint_raw, method=method)
                if not endpoint:
                    drop_counts["endpoint_unresolved"] = int(
                        drop_counts.get("endpoint_unresolved", 0)
                    ) + 1
                    if len(drop_samples) < 6:
                        drop_samples.append(
                            {
                                "reason": "endpoint_unresolved",
                                "method": str(method),
                                "endpoint": self._text_preview(endpoint_raw, 100),
                            }
                        )
                    continue
                test_type = self._coerce_test_type(
                    item.get("test_type", item.get("type", item.get("category")))
                )
                expected_status_raw = item.get(
                    "expected_status",
                    item.get("status", item.get("expectedStatus")),
                )
                if expected_status_raw is None:
                    expected_status_raw = self._default_expected_status(test_type, method)
                try:
                    expected_status = int(expected_status_raw)
                except Exception:
                    expected_status = self._default_expected_status(test_type, method)
                raw_name = str(item.get("name", "")).strip()
                raw_description = str(item.get("description", item.get("reason", ""))).strip()
                headers = item.get("headers", item.get("request_headers", {}))
                params = item.get("params", item.get("query", item.get("query_params", {})))
                body = item.get(
                    "body",
                    item.get("payload", item.get("request_body", item.get("data", None))),
                )
                if not isinstance(headers, dict):
                    headers = {}
                if not isinstance(params, dict):
                    params = {}
                if body is not None and not isinstance(body, dict):
                    body = None
                expected_status = self._normalize_llm_expected_status(
                    test_type=test_type,
                    method=method,
                    endpoint=endpoint,
                    expected_status=int(expected_status),
                    name=raw_name,
                    description=raw_description,
                    params=dict(params),
                    body=dict(body) if isinstance(body, dict) else None,
                )
                key = (method, endpoint, test_type.value, int(expected_status))
                if key in seen_keys:
                    drop_counts["duplicate"] = int(drop_counts.get("duplicate", 0)) + 1
                    continue
                seen_keys.add(key)
                scenarios.append(
                    TestScenario(
                        name=raw_name
                        or f"test_{method.lower()}_{endpoint.replace('/', '_').replace('{', '').replace('}', '')}_{test_type.value}",
                        description=raw_description
                        or f"{test_type.value} scenario for {method} {endpoint}",
                        test_type=test_type,
                        endpoint=endpoint,
                        method=method,
                        headers={str(k): str(v) for k, v in headers.items()},
                        params=dict(params),
                        body=dict(body) if isinstance(body, dict) else None,
                        expected_status=int(expected_status),
                    )
                )
                if len(scenarios) >= 60:
                    break

            if scenarios:
                self.llm_stats["scenario_success"] = int(self.llm_stats.get("scenario_success", 0)) + 1
                self._log_llm_debug(
                    "planner_accepted",
                    {
                        "candidate_rows": len(raw_scenarios),
                        "accepted_rows": len(scenarios),
                        "drop_counts": drop_counts,
                        "response_mode": response_mode_used,
                        "parse_diagnostics": parse_diagnostics,
                    },
                )
                self.last_llm_generation_diagnostics = {
                    "status": "accepted",
                    "response_mode": str(response_mode_used),
                    "parse_diagnostics": parse_diagnostics,
                    "candidate_rows": int(len(raw_scenarios)),
                    "accepted_rows": int(len(scenarios)),
                    "drop_counts": dict(drop_counts),
                }
                return scenarios
            if raw_scenarios:
                logger.warning(
                    "LLM scenario planner returned %d candidate rows but none mapped to known operations. Falling back to heuristic mode.",
                    len(raw_scenarios),
                )
            self._log_llm_debug(
                "planner_rejected",
                {
                    "candidate_rows": len(raw_scenarios),
                    "accepted_rows": 0,
                    "drop_counts": drop_counts,
                    "drop_samples": drop_samples,
                    "response_mode": response_mode_used,
                    "parse_diagnostics": parse_diagnostics,
                    "fallback": "heuristic",
                },
            )
            self.llm_stats["scenario_schema_rejections"] = int(
                self.llm_stats.get("scenario_schema_rejections", 0)
            ) + 1
            self.last_llm_generation_diagnostics = {
                "status": "rejected",
                "response_mode": str(response_mode_used),
                "parse_diagnostics": parse_diagnostics,
                "candidate_rows": int(len(raw_scenarios)),
                "accepted_rows": 0,
                "drop_counts": dict(drop_counts),
                "drop_samples": drop_samples[:6],
                "fallback": "heuristic",
            }
            self.llm_stats["scenario_errors"] = int(self.llm_stats.get("scenario_errors", 0)) + 1
            return []
        except Exception as exc:
            self.llm_stats["scenario_errors"] = int(self.llm_stats.get("scenario_errors", 0)) + 1
            elapsed = time.time() - started_at
            print(f"   ⚠️ LLM scenario planner failed after {elapsed:.2f}s, switching to heuristic mode")
            logger.warning("LLM scenario generation failed. Falling back to heuristic mode: %s", exc)
            self._log_llm_debug(
                "planner_exception",
                {
                    "elapsed_sec": round(float(elapsed), 3),
                    "error": self._text_preview(str(exc), 320),
                    "fallback": "heuristic",
                },
            )
            self.last_llm_generation_diagnostics = {
                "status": "exception",
                "error": self._text_preview(str(exc), 320),
                "fallback": "heuristic",
            }
            return []
    
    def think_like_tester(self, nlp_prompt: Optional[str] = None) -> List[TestScenario]:
        """
        Think like a human tester with optional natural language prompt.
        
        Examples:
        - "Generate tests to validate status codes and response times"
        - "Create security tests for user authentication endpoints"
        - "Test error handling for invalid input data"
        - "Generate comprehensive test suite with boundary testing"
        """
        scenarios = []
        
        if nlp_prompt:
            print(f"🤖 AI Tester analyzing prompt: '{nlp_prompt}'")
            scenarios = self._generate_from_nlp_prompt(nlp_prompt)
            if not scenarios:
                self.last_generation_engine = "heuristic_fallback_empty"
        else:
            # Default comprehensive testing
            for endpoint in self.endpoints:
                # 1. Happy path testing (what should work)
                scenarios.extend(self._create_happy_path_tests(endpoint))
                
                # 2. Error handling (what should fail gracefully)
                scenarios.extend(self._create_error_tests(endpoint))
                
                # 3. Authentication/Authorization testing
                if endpoint.auth_required:
                    scenarios.extend(self._create_auth_tests(endpoint))
                
                # 4. Input validation testing
                scenarios.extend(self._create_validation_tests(endpoint))
                
                # 5. Boundary testing
                scenarios.extend(self._create_boundary_tests(endpoint))
                
                # 6. Security testing
                scenarios.extend(self._create_security_tests(endpoint))
                
                # 7. Edge cases
                scenarios.extend(self._create_edge_case_tests(endpoint))
            self.last_generation_engine = "heuristic_default"
        
        self.test_scenarios = scenarios
        return scenarios
    
    def _generate_from_nlp_prompt(self, prompt: str) -> List[TestScenario]:
        """Generate tests based on natural language prompt like Postman AI."""
        scenarios = []
        prompt_lower = prompt.lower()
        wants_comprehensive = any(
            token in prompt_lower
            for token in (
                "comprehensive",
                "complete",
                "full coverage",
                "end to end",
                "end-to-end",
            )
        )
        
        print(f"   🧠 Analyzing intent: {prompt}")

        llm_scenarios = self._generate_from_nlp_prompt_llm(prompt)
        if llm_scenarios:
            print(f"   🧠 LLM generated {len(llm_scenarios)} scenarios")
            self.last_generation_engine = "llm_primary"
            return llm_scenarios
        if self.llm_enabled:
            self.last_generation_engine = "llm_fallback_to_heuristic"
        else:
            self.last_generation_engine = "heuristic_prompt_mode"
        
        # Parse what user wants to test
        if 'status code' in prompt_lower or 'response time' in prompt_lower:
            print("   📊 Generating status code and performance tests")
            scenarios.extend(self._create_status_and_performance_tests())
            
        if 'security' in prompt_lower or 'sql inject' in prompt_lower or 'xss' in prompt_lower:
            print("   🔒 Generating security vulnerability tests")
            scenarios.extend(self._create_comprehensive_security_tests())
            
        if 'auth' in prompt_lower or 'login' in prompt_lower or 'token' in prompt_lower:
            print("   🔐 Generating authentication and authorization tests")
            scenarios.extend(self._create_comprehensive_auth_tests())

        if 'error' in prompt_lower or 'invalid' in prompt_lower or 'fail' in prompt_lower:
            print("   💥 Generating error handling and edge case tests")
            scenarios.extend(self._create_comprehensive_error_tests())
            
        if 'validation' in prompt_lower or 'input' in prompt_lower:
            print("   🛡️ Generating input validation tests")
            scenarios.extend(self._create_comprehensive_validation_tests())
            
        if 'boundary' in prompt_lower or 'limit' in prompt_lower or 'edge' in prompt_lower:
            print("   🎯 Generating boundary and limit tests")
            scenarios.extend(self._create_comprehensive_boundary_tests())

        if 'happy path' in prompt_lower or 'happy_path' in prompt_lower or wants_comprehensive:
            print("   ✅ Ensuring happy path coverage for each operation")
            for endpoint in self.endpoints:
                scenarios.extend(self._create_happy_path_tests(endpoint))
            
        # If no specific intent, generate comprehensive suite
        if not scenarios:
            print("   🎯 No specific intent detected, generating comprehensive test suite")
            self.last_generation_engine = "heuristic_prompt_to_default"
            return self.think_like_tester()  # Recursive call without prompt
            
        print(f"   ✅ Generated {len(scenarios)} test scenarios from prompt")
        return scenarios
    
    def analyze_error_and_suggest_fix(self, error_response: Dict[str, Any], original_scenario: TestScenario) -> Dict[str, Any]:
        """
        Analyze API errors and suggest automatic fixes like Postman AI.
        
        Handles common errors:
        - 401 Unauthorized: Suggests adding authentication
        - 403 Forbidden: Suggests checking permissions/scope  
        - 400 Bad Request: Analyzes and fixes request data
        - 404 Not Found: Suggests valid resource IDs
        """
        status_code = error_response.get('status_code', 0)
        error_body = error_response.get('body', {})
        
        print(f"🔍 AI Error Analysis: HTTP {status_code}")
        
        analysis = {
            'error_type': status_code,
            'root_cause': '',
            'suggested_fixes': [],
            'auto_fix_available': False,
            'fixed_scenario': None,
            'confidence': 0.0
        }
        
        if status_code == 401:
            analysis.update({
                'root_cause': 'Missing or invalid authentication token',
                'suggested_fixes': [
                    'Add Authorization header with Bearer token',
                    'Include API key in request headers',
                    'Verify authentication endpoint is working'
                ],
                'auto_fix_available': True,
                'confidence': 0.9
            })
            
            # Auto-fix: Add authentication
            fixed_scenario = TestScenario(
                name=f"{original_scenario.name}_with_auth",
                description=f"{original_scenario.description} (with authentication)",
                test_type=original_scenario.test_type,
                endpoint=original_scenario.endpoint,
                method=original_scenario.method,
                headers={'Authorization': 'Bearer {{auth_token}}'},
                params=original_scenario.params.copy(),
                body=original_scenario.body.copy() if original_scenario.body else None,
                expected_status=200
            )
            analysis['fixed_scenario'] = fixed_scenario
            
        elif status_code == 403:
            analysis.update({
                'root_cause': 'Valid authentication but insufficient permissions',
                'suggested_fixes': [
                    'Use token with elevated permissions',
                    'Check required scopes for this endpoint',
                    'Verify resource ownership'
                ],
                'auto_fix_available': True,
                'confidence': 0.8
            })
            
            # Auto-fix: Use admin token
            fixed_scenario = TestScenario(
                name=f"{original_scenario.name}_with_admin",
                description=f"{original_scenario.description} (with admin permissions)",
                test_type=original_scenario.test_type,
                endpoint=original_scenario.endpoint,
                method=original_scenario.method,
                headers={'Authorization': 'Bearer {{admin_token}}'},
                params=original_scenario.params.copy(),
                body=original_scenario.body.copy() if original_scenario.body else None,
                expected_status=200
            )
            analysis['fixed_scenario'] = fixed_scenario
            
        elif status_code == 400:
            analysis.update({
                'root_cause': 'Invalid request data or missing required fields',
                'suggested_fixes': [
                    'Check required fields are present',
                    'Validate data types match API spec',
                    'Remove invalid fields',
                    'Fix JSON formatting'
                ],
                'auto_fix_available': True,
                'confidence': 0.7
            })
            
            # Auto-fix: Add required fields
            fixed_body = original_scenario.body.copy() if original_scenario.body else {}
            if not fixed_body and original_scenario.method in ['POST', 'PUT', 'PATCH']:
                fixed_body = {'name': 'Test Item', 'description': 'Auto-generated test data'}
            elif fixed_body:
                # Add common required fields if missing
                if 'name' not in fixed_body:
                    fixed_body['name'] = 'Auto-fixed Name'
                if 'email' not in fixed_body and 'user' in original_scenario.endpoint.lower():
                    fixed_body['email'] = 'test@example.com'
                    
            fixed_scenario = TestScenario(
                name=f"{original_scenario.name}_fixed_data",
                description=f"{original_scenario.description} (with valid data)",
                test_type=original_scenario.test_type,
                endpoint=original_scenario.endpoint,
                method=original_scenario.method,
                headers=original_scenario.headers.copy(),
                params=original_scenario.params.copy(),
                body=fixed_body,
                expected_status=200
            )
            analysis['fixed_scenario'] = fixed_scenario
            
        elif status_code == 404:
            analysis.update({
                'root_cause': 'Resource not found or invalid ID',
                'suggested_fixes': [
                    'Use valid resource ID that exists',
                    'Create resource first if needed',
                    'Check URL path is correct'
                ],
                'auto_fix_available': True,
                'confidence': 0.85
            })
            
            # Auto-fix: Use valid ID
            fixed_endpoint = original_scenario.endpoint.replace('/999', '/123').replace('/invalid', '/1')
            fixed_scenario = TestScenario(
                name=f"{original_scenario.name}_valid_id",
                description=f"{original_scenario.description} (with valid ID)",
                test_type=original_scenario.test_type,
                endpoint=fixed_endpoint,
                method=original_scenario.method,
                headers=original_scenario.headers.copy(),
                params=original_scenario.params.copy(),
                body=original_scenario.body.copy() if original_scenario.body else None,
                expected_status=200
            )
            analysis['fixed_scenario'] = fixed_scenario
            
        else:
            analysis.update({
                'root_cause': f'Unexpected HTTP {status_code} error',
                'suggested_fixes': ['Check API documentation', 'Verify server status', 'Review request format'],
                'auto_fix_available': False,
                'confidence': 0.3
            })
        
        # Store fix for reuse
        self.error_fixes[f"{status_code}_{original_scenario.name}"] = analysis
        
        print(f"   📋 Root cause: {analysis['root_cause']}")
        print(f"   🔧 Auto-fix available: {analysis['auto_fix_available']}")
        print(f"   📊 Confidence: {analysis['confidence']:.1%}")
        
        return analysis
    
    def _create_status_and_performance_tests(self) -> List[TestScenario]:
        """Create tests focused on status codes and response times."""
        scenarios = []
        
        for endpoint in self.endpoints:
            # Status code validation test
            scenario = TestScenario(
                name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_')}_status_validation",
                description=f"Validate status codes and response times for {endpoint.method} {endpoint.path}",
                test_type=TestType.PERFORMANCE,
                endpoint=endpoint.path,
                method=endpoint.method,
                expected_status=200,
                headers={'Accept': 'application/json'},
                body=None
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def _create_comprehensive_security_tests(self) -> List[TestScenario]:
        """Create comprehensive security tests including SQL injection, XSS."""
        scenarios = []
        
        for endpoint in self.endpoints:
            # SQL injection test
            if endpoint.method in ['POST', 'PUT', 'PATCH']:
                sql_injection_scenario = TestScenario(
                    name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_')}_sql_injection",
                    description=f"Test SQL injection protection on {endpoint.method} {endpoint.path}",
                    test_type=TestType.SECURITY,
                    endpoint=endpoint.path,
                    method=endpoint.method,
                    expected_status=400,
                    body={'name': "'; DROP TABLE users; --", 'description': 'SQL injection attempt'}
                )
                scenarios.append(sql_injection_scenario)
                
            # XSS test
            if endpoint.method in ['POST', 'PUT', 'PATCH']:
                xss_scenario = TestScenario(
                    name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_')}_xss_protection",
                    description=f"Test XSS protection on {endpoint.method} {endpoint.path}",
                    test_type=TestType.SECURITY,
                    endpoint=endpoint.path,
                    method=endpoint.method,
                    expected_status=400,
                    body={'comment': '<script>alert("xss")</script>', 'content': 'XSS attempt'}
                )
                scenarios.append(xss_scenario)
        
        return scenarios
    
    def _create_comprehensive_auth_tests(self) -> List[TestScenario]:
        """Create comprehensive authentication and authorization tests."""
        scenarios = []
        
        for endpoint in self.endpoints:
            if endpoint.auth_required:
                # No auth test
                no_auth_scenario = TestScenario(
                    name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_')}_no_auth",
                    description=f"Test {endpoint.method} {endpoint.path} without authentication",
                    test_type=TestType.AUTHENTICATION,
                    endpoint=endpoint.path,
                    method=endpoint.method,
                    expected_status=401,
                    headers={},  # No auth headers
                    body=None
                )
                scenarios.append(no_auth_scenario)
                
                # Invalid auth test
                invalid_auth_scenario = TestScenario(
                    name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_')}_invalid_auth",
                    description=f"Test {endpoint.method} {endpoint.path} with invalid token",
                    test_type=TestType.AUTHENTICATION,
                    endpoint=endpoint.path,
                    method=endpoint.method,
                    expected_status=401,
                    headers={'Authorization': 'Bearer invalid_token'},
                    body=None
                )
                scenarios.append(invalid_auth_scenario)
        
        return scenarios
    
    def _create_comprehensive_error_tests(self) -> List[TestScenario]:
        """Create comprehensive error handling tests."""
        scenarios = []
        
        for endpoint in self.endpoints:
            # Invalid method test
            invalid_method_scenario = TestScenario(
                name=f"test_{endpoint.path.replace('/', '_')}_invalid_method",
                description=f"Test invalid method on {endpoint.path}",
                test_type=TestType.ERROR_HANDLING,
                endpoint=endpoint.path,
                method='DELETE' if endpoint.method != 'DELETE' else 'PATCH',
                expected_status=405
            )
            scenarios.append(invalid_method_scenario)
            
            # Malformed request test
            if endpoint.method in ['POST', 'PUT', 'PATCH']:
                malformed_scenario = TestScenario(
                    name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_')}_malformed_request",
                    description=f"Test malformed request data on {endpoint.method} {endpoint.path}",
                    test_type=TestType.ERROR_HANDLING,
                    endpoint=endpoint.path,
                    method=endpoint.method,
                    expected_status=400,
                    body={'invalid_json': 'missing_quotes: true, broken: }'}
                )
                scenarios.append(malformed_scenario)
        
        return scenarios
    
    def _create_comprehensive_validation_tests(self) -> List[TestScenario]:
        """Create comprehensive input validation tests."""
        scenarios = []
        
        for endpoint in self.endpoints:
            if endpoint.method in ['POST', 'PUT', 'PATCH']:
                # Empty body test
                empty_body_scenario = TestScenario(
                    name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_')}_empty_body",
                    description=f"Test empty request body on {endpoint.method} {endpoint.path}",
                    test_type=TestType.INPUT_VALIDATION,
                    endpoint=endpoint.path,
                    method=endpoint.method,
                    expected_status=400,
                    body={}
                )
                scenarios.append(empty_body_scenario)
                
                # Type mismatch test
                type_mismatch_scenario = TestScenario(
                    name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_')}_type_mismatch", 
                    description=f"Test type mismatch on {endpoint.method} {endpoint.path}",
                    test_type=TestType.INPUT_VALIDATION,
                    endpoint=endpoint.path,
                    method=endpoint.method,
                    expected_status=400,
                    body={'age': 'not_a_number', 'count': 'invalid_integer'}
                )
                scenarios.append(type_mismatch_scenario)
        
        return scenarios
    
    def _create_comprehensive_boundary_tests(self) -> List[TestScenario]:
        """Create comprehensive boundary and limit tests."""
        scenarios = []
        
        for endpoint in self.endpoints:
            if endpoint.method != 'GET':
                continue

            query_params = [
                p for p in endpoint.parameters
                if str((p or {}).get("in", "")).lower() == "query"
            ]
            if not query_params:
                # Avoid hallucinated pagination checks on endpoints without query parameters.
                continue

            query_name_map = {
                str((p or {}).get("name", "")).lower(): str((p or {}).get("name", ""))
                for p in query_params
                if str((p or {}).get("name", ""))
            }
            pagination_aliases = {
                "limit",
                "page",
                "offset",
                "per_page",
                "perpage",
                "page_size",
                "pagesize",
            }
            if not (set(query_name_map.keys()) & pagination_aliases):
                # Skip pagination boundary cases if the endpoint does not expose pagination params.
                continue

            invalid_params: Dict[str, Any] = {}
            if "limit" in query_name_map:
                invalid_params[query_name_map["limit"]] = 99999
            if "page" in query_name_map:
                invalid_params[query_name_map["page"]] = -1
            if "offset" in query_name_map:
                invalid_params[query_name_map["offset"]] = -1
            if "per_page" in query_name_map:
                invalid_params[query_name_map["per_page"]] = 0
            if "perpage" in query_name_map:
                invalid_params[query_name_map["perpage"]] = 0
            if "page_size" in query_name_map:
                invalid_params[query_name_map["page_size"]] = 0
            if "pagesize" in query_name_map:
                invalid_params[query_name_map["pagesize"]] = 0

            if not invalid_params:
                continue

            limit_scenario = TestScenario(
                name=f"test_get_{endpoint.path.replace('/', '_')}_pagination_limit",
                description=f"Test pagination limits on GET {endpoint.path}",
                test_type=TestType.BOUNDARY_TESTING,
                endpoint=endpoint.path,
                method='GET',
                expected_status=400,
                params=invalid_params,
            )
            scenarios.append(limit_scenario)
        
        return scenarios
    
    def _create_happy_path_tests(self, endpoint: APIEndpoint) -> List[TestScenario]:
        """Create happy path tests - the basic functionality."""
        scenarios = []
        
        # Basic successful request
        scenario = TestScenario(
            name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}_success",
            description=f"Test successful {endpoint.method} request to {endpoint.path}",
            test_type=TestType.HAPPY_PATH,
            endpoint=endpoint.path,
            method=endpoint.method,
            expected_status=200 if endpoint.method == 'GET' else (201 if endpoint.method == 'POST' else 200)
        )
        
        # Add required parameters
        for param in endpoint.parameters:
            if param.get('required', False):
                location = str(param.get('in', '')).lower()
                if location in {'query', 'path'}:
                    scenario.params[param['name']] = self._generate_sample_value(param)
                elif location == 'header':
                    scenario.headers[param['name']] = self._generate_sample_value(param)
        
        # Add request body for POST/PUT
        if endpoint.method in ['POST', 'PUT', 'PATCH'] and endpoint.request_body:
            scenario.body = self._generate_sample_body(endpoint.request_body)
        
        scenarios.append(scenario)
        return scenarios
    
    def _create_error_tests(self, endpoint: APIEndpoint) -> List[TestScenario]:
        """Create error handling tests."""
        scenarios = []
        
        # Test 404 - Non-existent resource
        if '{id}' in endpoint.path or '{' in endpoint.path:
            unresolved_path = re.sub(r"\{[^}]+\}", "nonexistent_dependency_id", endpoint.path)
            scenarios.append(TestScenario(
                name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}_not_found",
                description=f"Test {endpoint.method} request with non-existent ID",
                test_type=TestType.ERROR_HANDLING,
                endpoint=unresolved_path.replace('{id}', '99999').replace('{userId}', '99999'),
                method=endpoint.method,
                expected_status=404,
                assertions=["Response contains error message", "Error format is consistent"]
            ))
        
        # Test 400 - Bad request
        if endpoint.method in ['POST', 'PUT', 'PATCH']:
            scenarios.append(TestScenario(
                name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}_bad_request",
                description=f"Test {endpoint.method} request with invalid data",
                test_type=TestType.ERROR_HANDLING,
                endpoint=endpoint.path,
                method=endpoint.method,
                body={"invalid_field": "invalid_value"},
                expected_status=400,
                assertions=["Response explains validation errors", "Error details are provided"]
            ))
        
        return scenarios
    
    def _create_auth_tests(self, endpoint: APIEndpoint) -> List[TestScenario]:
        """Create authentication/authorization tests."""
        scenarios = []
        
        # Test 401 - No authentication
        scenarios.append(TestScenario(
            name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}_unauthorized",
            description=f"Test {endpoint.method} request without authentication",
            test_type=TestType.AUTHENTICATION,
            endpoint=endpoint.path,
            method=endpoint.method,
            expected_status=401,
            assertions=["Unauthorized access is properly rejected"]
        ))
        
        # Test 403 - Invalid token
        scenarios.append(TestScenario(
            name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}_forbidden",
            description=f"Test {endpoint.method} request with invalid token",
            test_type=TestType.AUTHORIZATION,
            endpoint=endpoint.path,
            method=endpoint.method,
            headers={"Authorization": "Bearer invalid_token_12345"},
            expected_status=403,
            assertions=["Invalid token is rejected", "Proper error message returned"]
        ))
        
        return scenarios
    
    def _create_validation_tests(self, endpoint: APIEndpoint) -> List[TestScenario]:
        """Create input validation tests."""
        scenarios = []
        
        # Test missing required parameters
        for param in endpoint.parameters:
            if param.get('required', False):
                scenario = TestScenario(
                    name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}_missing_{param['name']}",
                    description=f"Test {endpoint.method} request missing required parameter {param['name']}",
                    test_type=TestType.INPUT_VALIDATION,
                    endpoint=endpoint.path,
                    method=endpoint.method,
                    expected_status=400,
                    assertions=[f"Missing {param['name']} parameter is detected"]
                )
                scenarios.append(scenario)
        
        return scenarios
    
    def _create_boundary_tests(self, endpoint: APIEndpoint) -> List[TestScenario]:
        """Create boundary value tests."""
        scenarios = []
        
        # Test very long strings, extreme numbers, etc.
        if endpoint.method in ['POST', 'PUT', 'PATCH']:
            scenarios.append(TestScenario(
                name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}_max_length",
                description=f"Test {endpoint.method} with maximum length inputs",
                test_type=TestType.BOUNDARY_TESTING,
                endpoint=endpoint.path,
                method=endpoint.method,
                body={"description": "A" * 10000},  # Very long string
                expected_status=400,
                assertions=["Long inputs are handled gracefully"]
            ))
        
        return scenarios
    
    def _create_security_tests(self, endpoint: APIEndpoint) -> List[TestScenario]:
        """Create security-focused tests."""
        scenarios = []
        
        # SQL Injection test
        if endpoint.parameters:
            scenarios.append(TestScenario(
                name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}_sql_injection",
                description=f"Test {endpoint.method} for SQL injection vulnerability",
                test_type=TestType.SECURITY,
                endpoint=endpoint.path,
                method=endpoint.method,
                params={"q": "'; DROP TABLE users; --"},
                expected_status=400,
                assertions=["SQL injection attempts are blocked"]
            ))
        
        # XSS test
        if endpoint.method in ['POST', 'PUT', 'PATCH']:
            scenarios.append(TestScenario(
                name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}_xss_protection",
                description=f"Test {endpoint.method} for XSS vulnerability",
                test_type=TestType.SECURITY,
                endpoint=endpoint.path,
                method=endpoint.method,
                body={"comment": "<script>alert('xss')</script>"},
                expected_status=400,
                assertions=["XSS attempts are sanitized"]
            ))
        
        return scenarios
    
    def _create_edge_case_tests(self, endpoint: APIEndpoint) -> List[TestScenario]:
        """Create edge case tests."""
        scenarios = []
        
        # Empty body test
        if endpoint.method in ['POST', 'PUT', 'PATCH']:
            scenarios.append(TestScenario(
                name=f"test_{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}_empty_body",
                description=f"Test {endpoint.method} with empty request body",
                test_type=TestType.EDGE_CASES,
                endpoint=endpoint.path,
                method=endpoint.method,
                body={},
                expected_status=400,
                assertions=["Empty body is handled appropriately"]
            ))
        
        return scenarios
    
    def _generate_sample_value(self, param: Dict[str, Any]) -> Any:
        """Generate sample values for parameters."""
        schema = param.get('schema', {}) if isinstance(param.get('schema', {}), dict) else {}
        param_type = str(param.get('type') or schema.get('type') or 'string').lower()
        param_name = param.get('name', '').lower()
        
        if param_type == 'integer':
            if 'id' in param_name:
                return 123
            minimum = schema.get('minimum')
            if isinstance(minimum, (int, float)):
                return int(minimum) if float(minimum).is_integer() else minimum
            return 10
        elif param_type == 'number':
            minimum = schema.get('minimum')
            if isinstance(minimum, (int, float)):
                return float(minimum) if float(minimum) > 0 else 1.0
            return 10.5
        elif param_type == 'boolean':
            return True
        elif param_type == 'array':
            return ["sample1", "sample2"]
        else:  # string
            if 'email' in param_name:
                return "test@example.com"
            elif 'id' in param_name:
                return "abc123"
            elif 'token' in param_name:
                return "bearer_token_12345"
            return "sample_value"
    
    def _generate_sample_body(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sample request body."""
        if not isinstance(request_body, dict):
            return {
                "name": "Test User",
                "email": "test@example.com",
                "description": "Sample description",
            }

        content = request_body.get("content", {})
        if not isinstance(content, dict):
            content = {}
        media = content.get("application/json")
        if not isinstance(media, dict):
            # Fallback to the first media type if JSON is not explicit.
            media = next((value for value in content.values() if isinstance(value, dict)), {})
        schema = media.get("schema", {}) if isinstance(media, dict) else {}
        resolved = self._resolve_schema(schema)
        if not isinstance(resolved, dict):
            resolved = {}

        generated = self._sample_from_schema(resolved, field_name_hint="")
        if isinstance(generated, dict) and generated:
            return generated

        # Stable fallback for underspecified schemas.
        return {
            "name": "Test User",
            "email": "test@example.com",
            "description": "Sample description",
        }

    def _resolve_schema(self, schema: Any) -> Any:
        if not isinstance(schema, dict):
            return schema
        if "$ref" not in schema:
            return schema
        ref = str(schema.get("$ref", ""))
        if not ref.startswith("#/components/schemas/"):
            return schema
        name = ref.split("/")[-1]
        components = self.api_spec.get("components", {}) if isinstance(self.api_spec, dict) else {}
        schemas = components.get("schemas", {}) if isinstance(components, dict) else {}
        resolved = schemas.get(name, {}) if isinstance(schemas, dict) else {}
        return resolved if isinstance(resolved, dict) else schema

    def _sample_from_schema(self, schema: Any, field_name_hint: str = "") -> Any:
        schema = self._resolve_schema(schema)
        if not isinstance(schema, dict):
            return "sample_value"

        if isinstance(schema.get("enum"), list) and schema["enum"]:
            return schema["enum"][0]

        schema_type = str(schema.get("type", "")).lower()
        if schema_type == "object" or (not schema_type and isinstance(schema.get("properties"), dict)):
            properties = schema.get("properties", {}) if isinstance(schema.get("properties", {}), dict) else {}
            required = schema.get("required", []) if isinstance(schema.get("required", []), list) else []
            payload: Dict[str, Any] = {}
            for key in required:
                if key in properties:
                    payload[str(key)] = self._sample_from_schema(properties[key], str(key))
            # Add one optional field for richer happy-path payloads when available.
            if properties and not payload:
                first_key = next(iter(properties.keys()))
                payload[str(first_key)] = self._sample_from_schema(
                    properties[first_key],
                    str(first_key),
                )
            return payload

        if schema_type == "array":
            item_schema = schema.get("items", {}) if isinstance(schema.get("items", {}), dict) else {}
            return [self._sample_from_schema(item_schema, field_name_hint)]

        if schema_type == "integer":
            minimum = schema.get("minimum")
            if isinstance(minimum, (int, float)):
                min_int = int(minimum)
                return min_int if min_int >= 0 else 0
            if "quantity" in field_name_hint.lower():
                return 1
            if "id" in field_name_hint.lower():
                return 123
            return 1

        if schema_type == "number":
            minimum = schema.get("minimum")
            if isinstance(minimum, (int, float)):
                min_num = float(minimum)
                return min_num if min_num > 0 else 0.01
            if "price" in field_name_hint.lower():
                return 19.99
            return 1.0

        if schema_type == "boolean":
            return True

        min_length = schema.get("minLength")
        if "email" in field_name_hint.lower():
            return "test@example.com"
        if "token" in field_name_hint.lower():
            return "valid_token_123"
        if "id" in field_name_hint.lower():
            return "id_123"
        if isinstance(min_length, int) and min_length > 0:
            return "x" * min(min_length, 8)
        if "name" in field_name_hint.lower():
            return "Sample Name"
        return "sample_value"


class MultiLanguageTestGenerator:
    """Generates tests in multiple programming languages."""
    
    def __init__(self, scenarios: List[TestScenario], base_url: str):
        """Initialize with test scenarios."""
        self.scenarios = scenarios
        self.base_url = base_url
        self._python_name_counts: Dict[str, int] = {}
        self._java_name_counts: Dict[str, int] = {}

    def _default_path_value(self, name: str) -> str:
        lowered = str(name or "").strip().lower()
        if lowered.endswith("id") or lowered == "id":
            return "123"
        return "sample"

    def _render_endpoint_and_query(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        rendered = str(endpoint or "")
        query_params = dict(params or {})
        for path_name in re.findall(r"\{([^}]+)\}", rendered):
            key = str(path_name)
            value = query_params.pop(key, self._default_path_value(key))
            placeholder = "{" + key + "}"
            if str(value).strip() == MISSING_PATH_PARAM_SENTINEL:
                rendered = rendered.replace("/" + placeholder, "")
                rendered = rendered.replace(placeholder + "/", "")
                rendered = rendered.replace(placeholder, "")
                continue
            rendered = rendered.replace(placeholder, quote(str(value), safe=""))
        rendered = re.sub(r"/{2,}", "/", rendered)
        if rendered and not rendered.startswith("/"):
            rendered = "/" + rendered
        return rendered, query_params
    
    def generate_python_tests(self) -> str:
        """Generate Python pytest tests."""
        code = '''import pytest
import requests
import json

BASE_URL = "{}"

class TestAPI:
    """Comprehensive API tests generated by AI agent."""

'''.format(self.base_url)
        
        for scenario in self.scenarios:
            method_name = self._python_method_name_for_scenario(scenario)
            code += self._generate_python_test_method(scenario, method_name)
        
        return code
    
    def _generate_python_test_method(self, scenario: TestScenario, method_name: str) -> str:
        """Generate a single Python test method."""
        rendered_endpoint, query_params = self._render_endpoint_and_query(
            scenario.endpoint,
            scenario.params,
        )
        method_code = f'''    def {method_name}(self):
        """{scenario.description}"""
        url = BASE_URL + "{rendered_endpoint}"
        
'''
        
        # Add headers
        if scenario.headers:
            method_code += f"        headers = {self._to_python_literal(scenario.headers)}\n"
        else:
            method_code += "        headers = {}\n"
        
        # Add params
        if query_params:
            method_code += f"        params = {self._to_python_literal(query_params)}\n"
        else:
            method_code += "        params = {}\n"
        
        # Add request
        if scenario.method == 'GET':
            method_code += "        response = requests.get(url, headers=headers, params=params)\n"
        elif scenario.method == 'POST':
            if scenario.body:
                method_code += f"        data = {self._to_python_literal(scenario.body)}\n"
                method_code += "        response = requests.post(url, headers=headers, params=params, json=data)\n"
            else:
                method_code += "        response = requests.post(url, headers=headers, params=params)\n"
        elif scenario.method == 'PUT':
            if scenario.body:
                method_code += f"        data = {self._to_python_literal(scenario.body)}\n"
                method_code += "        response = requests.put(url, headers=headers, params=params, json=data)\n"
            else:
                method_code += "        response = requests.put(url, headers=headers, params=params)\n"
        elif scenario.method == 'DELETE':
            method_code += "        response = requests.delete(url, headers=headers, params=params)\n"
        
        # Add assertions
        method_code += f"        assert response.status_code == {scenario.expected_status}\n"
        
        for assertion in scenario.assertions:
            if "error message" in assertion.lower():
                method_code += '        assert "error" in response.json() or "message" in response.json()\n'
            elif "response contains" in assertion.lower():
                method_code += "        assert response.json() is not None\n"
        
        method_code += "\n"
        return method_code

    @staticmethod
    def _to_python_literal(value: Any) -> str:
        """Render Python literals (None/True/False) instead of JSON (null/true/false)."""
        return pprint.pformat(value, sort_dicts=False)

    def _sanitize_identifier(self, raw_name: str, prefix: str) -> str:
        cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", str(raw_name or ""))
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        if not cleaned:
            cleaned = prefix
        if cleaned[0].isdigit():
            cleaned = f"{prefix}_{cleaned}"
        return cleaned

    def _reserve_name(self, base: str, counter: Dict[str, int]) -> str:
        count = int(counter.get(base, 0)) + 1
        counter[base] = count
        if count == 1:
            return base
        return f"{base}_{count}"

    def _python_method_name_for_scenario(self, scenario: TestScenario) -> str:
        base = self._sanitize_identifier(scenario.name, "test_case")
        if not base.startswith("test_"):
            base = f"test_{base}"
        return self._reserve_name(base, self._python_name_counts)

    def _java_method_name_for_scenario(self, scenario: TestScenario) -> str:
        base = self._sanitize_identifier(str(scenario.name).replace("test_", ""), "case")
        return self._reserve_name(base, self._java_name_counts)
    
    def generate_javascript_tests(self) -> str:
        """Generate JavaScript tests using Jest/Axios."""
        code = f'''const axios = require('axios');

const BASE_URL = '{self.base_url}';

describe('API Tests', () => {{

'''
        
        for scenario in self.scenarios:
            code += self._generate_javascript_test_method(scenario)
        
        code += "});\n"
        return code
    
    def _generate_javascript_test_method(self, scenario: TestScenario) -> str:
        """Generate JavaScript test method."""
        rendered_endpoint, query_params = self._render_endpoint_and_query(
            scenario.endpoint,
            scenario.params,
        )
        method_code = f'''  test('{scenario.description}', async () => {{
    const url = BASE_URL + '{rendered_endpoint}';
    const config = {{
      method: '{scenario.method.lower()}',
      url: url,
'''
        
        if scenario.headers:
            method_code += f"      headers: {json.dumps(scenario.headers, indent=6)},\n"
        
        if query_params:
            method_code += f"      params: {json.dumps(query_params, indent=6)},\n"
        
        if scenario.body:
            method_code += f"      data: {json.dumps(scenario.body, indent=6)},\n"
        
        method_code += "      validateStatus: () => true  // Don't throw on non-2xx\n"
        method_code += "    };\n\n"
        method_code += "    const response = await axios(config);\n"
        method_code += f"    expect(response.status).toBe({scenario.expected_status});\n"
        
        for assertion in scenario.assertions:
            if "error message" in assertion.lower():
                method_code += "    expect(response.data).toHaveProperty('error');\n"
        
        method_code += "  });\n\n"
        return method_code
    
    def generate_curl_tests(self) -> str:
        """Generate cURL commands for manual testing."""
        code = "#!/bin/bash\n# API Test Suite - cURL Commands\n\n"
        
        for scenario in self.scenarios:
            code += f"# {scenario.description}\n"
            
            curl_cmd = f"curl -X {scenario.method}"
            
            # Add headers
            for key, value in scenario.headers.items():
                curl_cmd += f' -H "{key}: {value}"'
            
            # Add data
            if scenario.body:
                curl_cmd += f" -d '{json.dumps(scenario.body)}'"
                curl_cmd += ' -H "Content-Type: application/json"'
            
            # Add URL
            rendered_endpoint, query_params = self._render_endpoint_and_query(
                scenario.endpoint,
                scenario.params,
            )
            url = self.base_url + rendered_endpoint
            if query_params:
                param_str = "&".join([f"{k}={v}" for k, v in query_params.items()])
                url += f"?{param_str}"
            
            curl_cmd += f' "{url}"'
            
            code += f"{curl_cmd}\n"
            code += f"# Expected status: {scenario.expected_status}\n\n"
        
        return code
    
    def generate_java_tests(self) -> str:
        """Generate Java tests using RestAssured."""
        code = f'''import io.restassured.RestAssured;
import org.junit.jupiter.api.Test;
import static io.restassured.RestAssured.*;
import static org.hamcrest.Matchers.*;

public class APITests {{
    private static final String BASE_URL = "{self.base_url}";

'''
        
        for scenario in self.scenarios:
            code += self._generate_java_test_method(scenario)
        
        code += "}\n"
        return code
    
    def _generate_java_test_method(self, scenario: TestScenario) -> str:
        """Generate Java test method."""
        method_name = self._java_method_name_for_scenario(scenario)
        rendered_endpoint, query_params = self._render_endpoint_and_query(
            scenario.endpoint,
            scenario.params,
        )
        method_code = f'''    @Test
    public void {method_name}() {{
        given()
            .baseUri(BASE_URL)
'''
        
        # Add headers
        for key, value in scenario.headers.items():
            method_code += f'            .header("{key}", "{value}")\n'
        
        # Add params
        for key, value in query_params.items():
            method_code += f'            .param("{key}", "{value}")\n'
        
        # Add body
        if scenario.body:
            method_code += f'            .body({json.dumps(scenario.body)})\n'
            method_code += '            .contentType("application/json")\n'
        
        method_code += "        .when()\n"
        method_code += f'            .{scenario.method.lower()}("{rendered_endpoint}")\n'
        method_code += "        .then()\n"
        method_code += f"            .statusCode({scenario.expected_status});\n"
        method_code += "    }\n\n"
        
        return method_code


class APITestingSandbox:
    """Sandbox environment for executing API tests safely."""
    
    def __init__(self, api_spec_path: str, base_url: str = "https://api.example.com"):
        """Initialize testing sandbox."""
        self.api_spec_path = api_spec_path
        self.base_url = base_url
        self.sandbox_dir = Path(tempfile.mkdtemp(prefix="api_testing_sandbox_"))
        self.results = []
        
        # Load API spec
        with open(api_spec_path, 'r') as f:
            if api_spec_path.endswith('.yaml') or api_spec_path.endswith('.yml'):
                self.api_spec = yaml.safe_load(f)
            else:
                self.api_spec = json.load(f)
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run complete test suite like a professional tester."""
        print("🤖 AI API Testing Agent Starting...")
        print("=" * 50)
        
        # 1. Think like a human tester
        print("🧠 Phase 1: Analyzing API like a human tester...")
        tester = HumanTesterSimulator(self.api_spec, self.base_url)
        scenarios = tester.think_like_tester()
        
        print(f"   Generated {len(scenarios)} test scenarios:")
        test_type_counts = {}
        for scenario in scenarios:
            test_type = scenario.test_type.value
            test_type_counts[test_type] = test_type_counts.get(test_type, 0) + 1
        
        for test_type, count in test_type_counts.items():
            print(f"   - {test_type.replace('_', ' ').title()}: {count} tests")
        
        # 2. Generate tests in multiple languages
        print("\n⚡ Phase 2: Generating tests in multiple languages...")
        generator = MultiLanguageTestGenerator(scenarios, self.base_url)
        
        # Generate Python tests
        python_tests = generator.generate_python_tests()
        python_file = self.sandbox_dir / "test_api.py"
        with open(python_file, 'w') as f:
            f.write(python_tests)
        print(f"   ✅ Python tests: {python_file}")
        
        # Generate JavaScript tests
        js_tests = generator.generate_javascript_tests()
        js_file = self.sandbox_dir / "test_api.test.js"
        with open(js_file, 'w') as f:
            f.write(js_tests)
        print(f"   ✅ JavaScript tests: {js_file}")
        
        # Generate cURL tests
        curl_tests = generator.generate_curl_tests()
        curl_file = self.sandbox_dir / "test_api.sh"
        with open(curl_file, 'w') as f:
            f.write(curl_tests)
        os.chmod(curl_file, 0o755)
        print(f"   ✅ cURL tests: {curl_file}")
        
        # Generate Java tests
        java_tests = generator.generate_java_tests()
        java_file = self.sandbox_dir / "APITests.java"
        with open(java_file, 'w') as f:
            f.write(java_tests)
        print(f"   ✅ Java tests: {java_file}")
        
        # 3. Create test documentation
        print("\n📝 Phase 3: Creating test documentation...")
        doc_content = self._generate_test_documentation(scenarios)
        doc_file = self.sandbox_dir / "TEST_PLAN.md"
        with open(doc_file, 'w') as f:
            f.write(doc_content)
        print(f"   ✅ Test documentation: {doc_file}")
        
        # 4. Create package files
        print("\n📦 Phase 4: Creating package files...")
        self._create_package_files()
        
        return {
            "sandbox_directory": str(self.sandbox_dir),
            "scenarios_generated": len(scenarios),
            "test_files": {
                "python": str(python_file),
                "javascript": str(js_file),
                "curl": str(curl_file),
                "java": str(java_file),
                "documentation": str(doc_file)
            },
            "test_breakdown": test_type_counts,
            "total_endpoints": len(tester.endpoints)
        }
    
    def _generate_test_documentation(self, scenarios: List[TestScenario]) -> str:
        """Generate comprehensive test documentation."""
        doc = """# API Testing Plan
Generated by AI Testing Agent

## Overview
This test suite provides comprehensive coverage of the API from the perspective of a professional tester.

## Test Categories

"""
        
        # Group by test type
        by_type = {}
        for scenario in scenarios:
            test_type = scenario.test_type.value
            if test_type not in by_type:
                by_type[test_type] = []
            by_type[test_type].append(scenario)
        
        for test_type, tests in by_type.items():
            doc += f"### {test_type.replace('_', ' ').title()}\n"
            doc += f"**Purpose**: {self._get_test_type_description(test_type)}\n"
            doc += f"**Test Count**: {len(tests)}\n\n"
            
            for test in tests[:3]:  # Show first 3 as examples
                doc += f"- **{test.name}**: {test.description}\n"
            
            if len(tests) > 3:
                doc += f"- ... and {len(tests) - 3} more\n"
            
            doc += "\n"
        
        doc += """## How to Run Tests

### Python (pytest)
```bash
pip install pytest requests
pytest test_api.py -v
```

### JavaScript (Jest)
```bash
npm install jest axios
npm test
```

### cURL (Manual)
```bash
chmod +x test_api.sh
./test_api.sh
```

### Java (Maven)
```bash
mvn test
```

## Expected Behavior
- All happy path tests should pass
- Error tests should return appropriate error codes
- Security tests should block malicious inputs
- Authentication tests should enforce proper access control
"""
        
        return doc
    
    def _get_test_type_description(self, test_type: str) -> str:
        """Get description for test type."""
        descriptions = {
            "happy_path": "Verify basic functionality works as expected",
            "error_handling": "Ensure graceful handling of error conditions",
            "authentication": "Verify authentication mechanisms work properly", 
            "authorization": "Test access control and permissions",
            "input_validation": "Ensure invalid inputs are properly rejected",
            "boundary_testing": "Test limits and edge values",
            "performance": "Verify response times and throughput",
            "security": "Test for common security vulnerabilities",
            "edge_cases": "Handle unusual but valid scenarios",
            "integration": "Test end-to-end workflows"
        }
        return descriptions.get(test_type, "Test specific functionality")
    
    def _create_package_files(self):
        """Create package management files."""
        
        # Python requirements.txt
        with open(self.sandbox_dir / "requirements.txt", 'w') as f:
            f.write("pytest>=7.0.0\nrequests>=2.25.0\n")
        
        # JavaScript package.json
        package_json = {
            "name": "api-tests",
            "version": "1.0.0",
            "scripts": {
                "test": "jest"
            },
            "dependencies": {
                "axios": "^0.27.0",
                "jest": "^28.0.0"
            }
        }
        with open(self.sandbox_dir / "package.json", 'w') as f:
            json.dump(package_json, f, indent=2)
        
        # Java pom.xml
        pom_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>api-tests</artifactId>
    <version>1.0.0</version>
    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
    </properties>
    <dependencies>
        <dependency>
            <groupId>io.rest-assured</groupId>
            <artifactId>rest-assured</artifactId>
            <version>5.1.1</version>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.8.2</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
</project>'''
        with open(self.sandbox_dir / "pom.xml", 'w') as f:
            f.write(pom_xml)
    
    def cleanup(self):
        """Clean up sandbox directory."""
        import shutil
        try:
            shutil.rmtree(self.sandbox_dir)
        except Exception:
            pass


if __name__ == "__main__":
    # Example usage
    sandbox = APITestingSandbox("examples/banking_api.yaml", "https://api.bankingexample.com")
    results = sandbox.run_full_test_suite()
    
    print("\n🎉 AI API Testing Complete!")
    print("=" * 30)
    print(f"Sandbox Directory: {results['sandbox_directory']}")
    print(f"Total Scenarios: {results['scenarios_generated']}")
    print(f"Endpoints Covered: {results['total_endpoints']}")
    print()
    print("Generated Test Files:")
    for lang, file_path in results['test_files'].items():
        print(f"  {lang.title()}: {file_path}")
    
    input("\nPress Enter to cleanup sandbox...")
    sandbox.cleanup()
    print("✅ Cleanup complete!")
