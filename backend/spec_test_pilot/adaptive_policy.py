"""
Adaptive scenario policy for QA specialist selection.

This module implements a lightweight contextual bandit (linear UCB) policy
to move scenario selection beyond static rules. The policy learns from
rewarded/penalized outcomes and prioritizes uncertain/high-risk scenarios.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Optional

import numpy as np


DEFAULT_FEATURE_DIM = 96
DEFAULT_ALPHA = 0.35
DEFAULT_REGULARIZATION = 1.0


def scenario_fingerprint(
    *,
    test_type: str,
    method: str,
    endpoint: str,
    expected_status: int,
    has_body: bool,
    has_params: bool,
) -> str:
    """Stable fingerprint used for scenario-level outcome tracking."""
    return (
        f"{method.upper()}|{endpoint}|{test_type}|{int(expected_status)}|"
        f"body={int(bool(has_body))}|params={int(bool(has_params))}"
    )


class AdaptiveScenarioPolicy:
    """
    Contextual linear-UCB policy with persisted state.

    State:
    - A, b: linear regression posterior parameters
    - scenario_stats: rolling performance by scenario fingerprint
    """

    def __init__(
        self,
        *,
        feature_dim: int = DEFAULT_FEATURE_DIM,
        alpha: float = DEFAULT_ALPHA,
        regularization: float = DEFAULT_REGULARIZATION,
        state: Optional[Dict[str, Any]] = None,
    ):
        self.feature_dim = int(max(16, feature_dim))
        self.alpha = float(max(0.01, alpha))
        self.regularization = float(max(1e-6, regularization))

        self.A = np.eye(self.feature_dim, dtype=float) * self.regularization
        self.b = np.zeros(self.feature_dim, dtype=float)
        self.scenario_stats: Dict[str, Dict[str, Any]] = {}

        if state:
            self._load_state(state)

    @classmethod
    def from_state(
        cls,
        state: Optional[Dict[str, Any]],
        fallback_scenario_stats: Optional[Dict[str, Any]] = None,
    ) -> "AdaptiveScenarioPolicy":
        policy = cls(state=state or {})
        if fallback_scenario_stats and not policy.scenario_stats:
            if isinstance(fallback_scenario_stats, dict):
                policy.scenario_stats = dict(fallback_scenario_stats)
        return policy

    def to_state(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "alpha": self.alpha,
            "regularization": self.regularization,
            "A": self.A.tolist(),
            "b": self.b.tolist(),
            "scenario_stats": self.scenario_stats,
        }

    def score(
        self,
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
    ) -> Dict[str, float]:
        """
        Return a decomposed score for one scenario candidate.

        `score` is the final ranking value used by the planner.
        """
        x = self.vectorize(
            test_type=test_type,
            method=method,
            endpoint=endpoint,
            expected_status=expected_status,
            has_body=has_body,
            has_params=has_params,
        )

        inv = np.linalg.pinv(self.A)
        theta = inv @ self.b

        expected = float(np.dot(theta, x))
        uncertainty = float(math.sqrt(max(0.0, float(x.T @ inv @ x))))

        fp = scenario_fingerprint(
            test_type=test_type,
            method=method,
            endpoint=endpoint,
            expected_status=expected_status,
            has_body=has_body,
            has_params=has_params,
        )
        stats = self.scenario_stats.get(fp, {})
        failure_rate = float(stats.get("failure_rate", 0.0))
        historical_reward = float(stats.get("avg_reward", 0.0))

        failure_focus_bonus = 0.40 * failure_rate
        exploration_bonus = self.alpha * uncertainty

        total = (
            expected
            + exploration_bonus
            + failure_focus_bonus
            + float(rl_risk)
            + float(novelty_bonus)
            + float(legacy_weight_bonus)
            - float(diversity_penalty)
        )

        return {
            "score": float(total),
            "expected_reward": expected,
            "uncertainty": uncertainty,
            "exploration_bonus": float(exploration_bonus),
            "failure_focus_bonus": float(failure_focus_bonus),
            "historical_reward": float(historical_reward),
        }

    def observe(
        self,
        *,
        test_type: str,
        method: str,
        endpoint: str,
        expected_status: int,
        has_body: bool,
        has_params: bool,
        reward: float,
        passed: bool,
    ) -> None:
        """Update policy posterior and scenario statistics from one outcome."""
        x = self.vectorize(
            test_type=test_type,
            method=method,
            endpoint=endpoint,
            expected_status=expected_status,
            has_body=has_body,
            has_params=has_params,
        )
        self.A += np.outer(x, x)
        self.b += float(reward) * x

        fp = scenario_fingerprint(
            test_type=test_type,
            method=method,
            endpoint=endpoint,
            expected_status=expected_status,
            has_body=has_body,
            has_params=has_params,
        )

        stats = self.scenario_stats.setdefault(
            fp,
            {
                "attempts": 0,
                "passes": 0,
                "failures": 0,
                "avg_reward": 0.0,
                "failure_rate": 0.0,
            },
        )
        attempts = int(stats.get("attempts", 0)) + 1
        passes = int(stats.get("passes", 0)) + (1 if passed else 0)
        failures = int(stats.get("failures", 0)) + (0 if passed else 1)
        old_avg = float(stats.get("avg_reward", 0.0))
        new_avg = old_avg + ((float(reward) - old_avg) / max(1, attempts))

        stats["attempts"] = attempts
        stats["passes"] = passes
        stats["failures"] = failures
        stats["avg_reward"] = round(new_avg, 6)
        stats["failure_rate"] = round(failures / max(1, attempts), 6)

    def vectorize(
        self,
        *,
        test_type: str,
        method: str,
        endpoint: str,
        expected_status: int,
        has_body: bool,
        has_params: bool,
    ) -> np.ndarray:
        """Encode scenario features into fixed-size vector."""
        vec = np.zeros(self.feature_dim, dtype=float)

        tokens = [
            f"type:{test_type}",
            f"method:{method.upper()}",
            f"endpoint:{endpoint}",
            f"status:{self._status_bucket(int(expected_status))}",
            f"has_body:{int(bool(has_body))}",
            f"has_params:{int(bool(has_params))}",
            f"path_depth:{endpoint.count('/')}",
        ]

        for part in self._tokenize_path(endpoint):
            tokens.append(f"path_token:{part}")

        for token in tokens:
            vec[self._index(token)] += 1.0

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def _tokenize_path(self, endpoint: str) -> list[str]:
        raw = endpoint.replace("{", "/").replace("}", "/")
        parts = [p for p in raw.replace("-", "/").replace("_", "/").split("/") if p]
        return [p.lower()[:32] for p in parts]

    def _status_bucket(self, status: int) -> str:
        if 200 <= status < 300:
            return "2xx"
        if 300 <= status < 400:
            return "3xx"
        if 400 <= status < 500:
            return "4xx"
        if 500 <= status < 600:
            return "5xx"
        return "other"

    def _index(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % self.feature_dim

    def _load_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        self.feature_dim = int(max(16, state.get("feature_dim", self.feature_dim)))
        self.alpha = float(max(0.01, state.get("alpha", self.alpha)))
        self.regularization = float(
            max(1e-6, state.get("regularization", self.regularization))
        )

        default_A = np.eye(self.feature_dim, dtype=float) * self.regularization
        default_b = np.zeros(self.feature_dim, dtype=float)

        raw_A = state.get("A")
        raw_b = state.get("b")

        try:
            A = np.array(raw_A, dtype=float)
            b = np.array(raw_b, dtype=float)
            if A.shape == (self.feature_dim, self.feature_dim):
                self.A = A
            else:
                self.A = default_A
            if b.shape == (self.feature_dim,):
                self.b = b
            else:
                self.b = default_b
        except Exception:
            self.A = default_A
            self.b = default_b

        stats = state.get("scenario_stats", {})
        if isinstance(stats, dict):
            self.scenario_stats = dict(stats)
        else:
            self.scenario_stats = {}

