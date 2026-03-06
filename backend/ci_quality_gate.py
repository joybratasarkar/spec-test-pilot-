#!/usr/bin/env python3
"""CI quality gate evaluator for QA specialist reports.

Evaluates:
- pass-rate floor
- flaky scenario ratio vs previous run
- regression trend deltas (pass-rate + run-reward)
- optional GAM context quality floor
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"report_not_found:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _scenario_key(item: Dict[str, Any]) -> Tuple[str, str, str, int]:
    method = str(item.get("method", "")).upper()
    endpoint = str(item.get("endpoint_template", item.get("endpoint", "")))
    test_type = str(item.get("test_type", ""))
    try:
        expected = int(item.get("expected_status", 0) or 0)
    except Exception:
        expected = 0
    return (method, endpoint, test_type, expected)


def _extract_scenario_verdict_map(report: Dict[str, Any]) -> Dict[Tuple[str, str, str, int], str]:
    result: Dict[Tuple[str, str, str, int], str] = {}
    for item in report.get("scenario_results", []) or []:
        if not isinstance(item, dict):
            continue
        result[_scenario_key(item)] = str(item.get("verdict", "fail")).strip().lower()
    return result


def _flaky_ratio(current: Dict[str, Any], previous: Dict[str, Any]) -> float:
    curr = _extract_scenario_verdict_map(current)
    prev = _extract_scenario_verdict_map(previous)
    overlap = set(curr.keys()) & set(prev.keys())
    if not overlap:
        return 0.0
    changed = sum(1 for key in overlap if curr.get(key) != prev.get(key))
    return float(changed) / float(len(overlap))


def _nested_float(payload: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    node: Any = payload
    for key in keys:
        if not isinstance(node, dict):
            return float(default)
        node = node.get(key)
    try:
        return float(node)
    except Exception:
        return float(default)


def evaluate(
    *,
    report: Dict[str, Any],
    previous_report: Optional[Dict[str, Any]],
    pass_rate_floor: float,
    flaky_threshold: float,
    max_pass_rate_drop: float,
    max_run_reward_drop: float,
    min_context_quality: float,
    require_summary_quality_gate: bool,
) -> Dict[str, Any]:
    reasons = []
    warnings = []

    pass_rate = _nested_float(report, "summary", "pass_rate", default=0.0)
    if pass_rate < float(pass_rate_floor):
        reasons.append(
            f"pass_rate_floor_failed:{pass_rate:.4f}<{float(pass_rate_floor):.4f}"
        )

    summary_gate = bool(report.get("summary", {}).get("meets_quality_gate", False))
    if require_summary_quality_gate and not summary_gate:
        reasons.append("summary_quality_gate_failed")

    context_quality = _nested_float(report, "gam_context_pack", "quality_score", default=0.0)
    if context_quality < float(min_context_quality):
        reasons.append(
            f"gam_context_quality_failed:{context_quality:.4f}<{float(min_context_quality):.4f}"
        )

    run_reward = _nested_float(report, "learning", "feedback", "run_reward", default=0.0)
    pass_rate_delta = 0.0
    run_reward_delta = 0.0
    flaky_overlap = 0.0
    flaky_in_run = _nested_float(report, "summary", "flaky_ratio", default=0.0)

    if previous_report is not None:
        prev_pass_rate = _nested_float(previous_report, "summary", "pass_rate", default=0.0)
        prev_run_reward = _nested_float(
            previous_report, "learning", "feedback", "run_reward", default=0.0
        )
        pass_rate_delta = pass_rate - prev_pass_rate
        run_reward_delta = run_reward - prev_run_reward
        if pass_rate_delta < -abs(float(max_pass_rate_drop)):
            reasons.append(
                f"pass_rate_regression_failed:{pass_rate_delta:.4f}<{-abs(float(max_pass_rate_drop)):.4f}"
            )
        if run_reward_delta < -abs(float(max_run_reward_drop)):
            reasons.append(
                f"run_reward_regression_failed:{run_reward_delta:.4f}<{-abs(float(max_run_reward_drop)):.4f}"
            )
        flaky_overlap = _flaky_ratio(report, previous_report)
        if flaky_overlap > float(flaky_threshold):
            reasons.append(
                f"flaky_overlap_ratio_failed:{flaky_overlap:.4f}>{float(flaky_threshold):.4f}"
            )
    else:
        warnings.append("previous_report_missing_skip_regression_and_flaky_checks")
    if flaky_in_run > float(flaky_threshold):
        reasons.append(
            f"flaky_in_run_ratio_failed:{flaky_in_run:.4f}>{float(flaky_threshold):.4f}"
        )

    status = "pass" if not reasons else "fail"
    return {
        "status": status,
        "reasons": reasons,
        "warnings": warnings,
        "metrics": {
            "pass_rate": round(pass_rate, 6),
            "run_reward": round(run_reward, 6),
            "context_quality": round(context_quality, 6),
            "pass_rate_delta": round(pass_rate_delta, 6),
            "run_reward_delta": round(run_reward_delta, 6),
            "flaky_overlap_ratio": round(flaky_overlap, 6),
            "flaky_in_run_ratio": round(flaky_in_run, 6),
            "summary_quality_gate": bool(summary_gate),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CI quality gates for QA report JSON.")
    parser.add_argument("--report", required=True, help="Current qa_execution_report.json path")
    parser.add_argument("--previous-report", default="", help="Previous report for trend/flaky checks")
    parser.add_argument("--pass-rate-floor", type=float, default=0.70)
    parser.add_argument("--flaky-threshold", type=float, default=0.15)
    parser.add_argument("--max-pass-rate-drop", type=float, default=0.08)
    parser.add_argument("--max-run-reward-drop", type=float, default=0.10)
    parser.add_argument("--min-context-quality", type=float, default=0.55)
    parser.add_argument("--require-summary-quality-gate", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    current_path = Path(args.report).expanduser()
    previous_path = Path(args.previous_report).expanduser() if args.previous_report else None

    current = _load_json(current_path)
    previous = _load_json(previous_path) if previous_path and previous_path.exists() else None

    result = evaluate(
        report=current,
        previous_report=previous,
        pass_rate_floor=float(args.pass_rate_floor),
        flaky_threshold=float(args.flaky_threshold),
        max_pass_rate_drop=float(args.max_pass_rate_drop),
        max_run_reward_drop=float(args.max_run_reward_drop),
        min_context_quality=float(args.min_context_quality),
        require_summary_quality_gate=bool(args.require_summary_quality_gate),
    )
    print(json.dumps(result, ensure_ascii=True))
    return 0 if result.get("status") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
