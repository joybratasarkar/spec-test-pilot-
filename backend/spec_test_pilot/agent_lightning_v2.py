#!/usr/bin/env python3
"""
Agent Lightning v2: Proper Implementation based on arXiv:2508.03680
Microsoft Research's Agent Lightning framework for training ANY AI agents with RL

Key Features:
- Training-Agent Disaggregation architecture
- Zero code changes to existing agents
- Hierarchical RL with LightningRL algorithm
- Sidecar monitoring for observability
- Works with any agent framework (LangChain, AutoGen, etc.)
"""

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
import numpy as np
from collections import deque, defaultdict
from pathlib import Path

# Optional ML dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class AgentMessage:
    """Standardized message format for agent communication."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass 
class AgentTrace:
    """Execution trace for agent actions - core of Agent Lightning."""
    trace_id: str
    session_id: str
    agent_id: str
    timestamp: float
    trace_type: str  # "action", "observation", "thought", "tool_call"
    content: Dict[str, Any]
    parent_trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingTransition:
    """RL training transition following Agent Lightning spec."""
    state: Dict[str, Any]
    action: Dict[str, Any] 
    reward: float
    next_state: Dict[str, Any]
    done: bool
    trace_sequence: List[AgentTrace]
    session_id: str
    agent_id: str


class ObservabilityCollector:
    """
    Sidecar monitoring system - key innovation of Agent Lightning.
    Collects agent traces without modifying agent code.
    """
    
    def __init__(self, buffer_size: int = 10000):
        self.traces: deque[AgentTrace] = deque(maxlen=buffer_size)
        self.active_sessions: Dict[str, List[AgentTrace]] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def start_session(self, session_id: str, agent_id: str) -> None:
        """Start monitoring session for agent."""
        with self.lock:
            self.active_sessions[session_id] = []
            self.logger.info(f"Started observability session {session_id} for agent {agent_id}")
    
    def collect_trace(
        self, 
        session_id: str,
        agent_id: str,
        trace_type: str,
        content: Dict[str, Any],
        parent_trace_id: Optional[str] = None
    ) -> str:
        """Collect agent execution trace."""
        trace = AgentTrace(
            trace_id=str(uuid.uuid4()),
            session_id=session_id,
            agent_id=agent_id,
            timestamp=time.time(),
            trace_type=trace_type,
            content=content,
            parent_trace_id=parent_trace_id
        )
        
        with self.lock:
            self.traces.append(trace)
            if session_id in self.active_sessions:
                self.active_sessions[session_id].append(trace)
        
        return trace.trace_id
    
    def end_session(self, session_id: str) -> List[AgentTrace]:
        """End session and return collected traces."""
        with self.lock:
            traces = self.active_sessions.pop(session_id, [])
            self.logger.info(f"Ended session {session_id}, collected {len(traces)} traces")
            return traces


class CreditAssignmentModule:
    """
    Hierarchical RL credit assignment - core of LightningRL algorithm.
    Assigns rewards across multi-step agent trajectories.
    """
    
    def __init__(self, gamma: float = 0.99, lambda_gae: float = 0.95):
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.logger = logging.getLogger(__name__)
    
    def assign_credit(
        self, 
        traces: List[AgentTrace], 
        final_reward: float,
        success: bool
    ) -> List[float]:
        """
        Assign credit across trace sequence using hierarchical RL.
        
        This is the key innovation - converts ANY agent trajectory 
        into RL training data.
        """
        if not traces:
            return []
        
        n = len(traces)
        rewards = [0.0] * n
        
        # Terminal reward
        rewards[-1] = final_reward if success else -abs(final_reward)
        
        # Backward credit assignment with GAE
        for i in range(n - 2, -1, -1):
            # Check if this is an action trace (vs observation)
            if traces[i].trace_type in ["action", "tool_call"]:
                # Apply discounted future reward
                rewards[i] = self.gamma * rewards[i + 1]
                
                # Add immediate reward based on trace quality
                immediate_reward = self._compute_immediate_reward(traces[i])
                rewards[i] += immediate_reward
            else:
                # Observation traces get minimal reward
                rewards[i] = 0.1 * rewards[i + 1]
        
        self.logger.debug(f"Assigned credit to {n} traces, total reward: {sum(rewards):.3f}")
        return rewards
    
    def _compute_immediate_reward(self, trace: AgentTrace) -> float:
        """Compute immediate reward for individual trace."""
        base_reward = 0.1
        
        # Reward based on trace type
        if trace.trace_type == "action":
            base_reward = 0.2
        elif trace.trace_type == "tool_call":
            base_reward = 0.15
        elif trace.trace_type == "thought":
            base_reward = 0.05
        
        # Bonus for successful tool calls
        if (trace.trace_type == "tool_call" and 
            trace.content.get("status") == "success"):
            base_reward += 0.1
        
        return base_reward


class LightningRLAlgorithm:
    """
    LightningRL: Hierarchical RL algorithm from Agent Lightning paper.
    Trains on agent trajectories with credit assignment.
    """
    
    def __init__(
        self,
        state_dim: int = 768,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        batch_size: int = 32,
        buffer_size: int = 100000
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.state_encoder = "stable_sha256_v2"
        self.logger = logging.getLogger(__name__)
        
        # Experience replay buffer
        self.replay_buffer: deque[TrainingTransition] = deque(maxlen=buffer_size)
        self.training_steps = 0
        
        # Initialize neural networks if PyTorch available
        if TORCH_AVAILABLE:
            self._init_networks()
        else:
            self.logger.warning("PyTorch not available - RL training disabled")
    
    def _init_networks(self):
        """Initialize value and policy networks."""
        # Value network for state evaluation
        self.value_net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # Policy network for action prediction (optional)
        self.policy_net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.state_dim)  # Output action probabilities
        )
        
        # Optimizers
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Loss functions
        # Huber loss is more stable than pure MSE on noisy rewards.
        self.value_loss_fn = nn.SmoothL1Loss()
        self.policy_loss_fn = nn.CrossEntropyLoss()
    
    def add_transition(self, transition: TrainingTransition) -> None:
        """Add training transition to replay buffer."""
        self.replay_buffer.append(transition)
    
    def train_step(self) -> Dict[str, Any]:
        """
        Execute one RL training step using proper Agent Lightning methodology.
        
        Key: Agent Lightning uses smaller batch sizes and trains more frequently
        based on Microsoft Research implementation.
        """
        if not TORCH_AVAILABLE:
            return {"status": "skipped", "reason": "pytorch_unavailable"}
        
        # Require a minimum batch to reduce high-variance updates.
        min_batch_size = max(4, min(self.batch_size, 8))
        
        if len(self.replay_buffer) < min_batch_size:
            return {"status": "skipped", "reason": f"need_{min_batch_size}_samples"}
        
        actual_batch_size = min(self.batch_size, len(self.replay_buffer))

        # Run a few mini-updates when replay memory is sufficiently large.
        gradient_steps = 1
        if len(self.replay_buffer) >= actual_batch_size * 2:
            gradient_steps = 2
        if len(self.replay_buffer) >= actual_batch_size * 4:
            gradient_steps = 4

        losses: List[float] = []
        for _ in range(gradient_steps):
            batch = self._sample_batch(actual_batch_size)

            # Convert to tensors
            states = torch.FloatTensor([self._encode_state(t.state) for t in batch])
            rewards = torch.FloatTensor([t.reward for t in batch])
            next_states = torch.FloatTensor([self._encode_state(t.next_state) for t in batch])
            dones = torch.BoolTensor([t.done for t in batch])

            # Normalize rewards to stabilize training across domains.
            reward_mean = rewards.mean()
            reward_std = rewards.std()
            if float(reward_std.item()) > 1e-6:
                rewards = (rewards - reward_mean) / (reward_std + 1e-8)
            else:
                rewards = rewards - reward_mean

            current_values = self.value_net(states).squeeze(-1)
            with torch.no_grad():
                next_values = self.value_net(next_states).squeeze(-1)
                target_values = rewards + 0.99 * next_values * (~dones)

            value_loss = self.value_loss_fn(current_values, target_values)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()
            losses.append(float(value_loss.item()))

        self.training_steps += 1
        mean_loss = float(sum(losses) / max(1, len(losses)))
        
        # Log training progress every 10 steps
        if self.training_steps % 10 == 0:
            print(f"🧠 RL Training Step {self.training_steps}: Loss={mean_loss:.4f}")
        
        return {
            "status": "trained",
            "value_loss": mean_loss,
            "batch_size": actual_batch_size,
            "gradient_steps": gradient_steps,
            "min_batch_size": min_batch_size,
            "training_steps": self.training_steps,
            "buffer_size": len(self.replay_buffer),
            "learning_rate": self.learning_rate,
            "weights_updated": True
        }

    def _sample_batch(self, batch_size: int) -> List[TrainingTransition]:
        if batch_size >= len(self.replay_buffer):
            return list(self.replay_buffer)
        batch_indices = np.random.choice(
            len(self.replay_buffer),
            batch_size,
            replace=False,
        )
        return [self.replay_buffer[i] for i in batch_indices]
    
    def _encode_state(self, state: Dict[str, Any]) -> List[float]:
        """Encode state dictionary to fixed-size vector."""
        vector = [0.0] * self.state_dim
        if not isinstance(state, dict):
            return vector

        # Stable hashing keeps RL feature mapping consistent across process restarts.
        for key, value in sorted(state.items(), key=lambda kv: str(kv[0])):
            payload = self._stable_json({"k": key, "v": value})
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            index = int(digest[:16], 16) % self.state_dim
            vector[index] += 1.0

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = [float(v / norm) for v in vector]
        return vector

    def _stable_json(self, value: Any) -> str:
        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
        except Exception:
            return str(value)

    def predict_state_value(self, state: Dict[str, Any]) -> Optional[float]:
        """Predict scalar value for a state; returns None when model is unavailable."""
        if not TORCH_AVAILABLE or not hasattr(self, "value_net"):
            return None
        try:
            encoded = self._encode_state(state)
            with torch.no_grad():
                value = self.value_net(torch.FloatTensor([encoded])).squeeze().item()
            return float(value)
        except Exception:
            return None
    
    def _encode_action(self, action: Dict[str, Any]) -> List[float]:
        """Encode action dictionary to vector."""
        return self._encode_state(action)  # Same encoding for simplicity

    def _trace_to_dict(self, trace: AgentTrace) -> Dict[str, Any]:
        return {
            "trace_id": trace.trace_id,
            "session_id": trace.session_id,
            "agent_id": trace.agent_id,
            "timestamp": float(trace.timestamp),
            "trace_type": trace.trace_type,
            "content": trace.content,
            "parent_trace_id": trace.parent_trace_id,
            "metadata": trace.metadata,
        }

    def _trace_from_dict(self, payload: Dict[str, Any]) -> AgentTrace:
        return AgentTrace(
            trace_id=str(payload.get("trace_id", str(uuid.uuid4()))),
            session_id=str(payload.get("session_id", "")),
            agent_id=str(payload.get("agent_id", "")),
            timestamp=float(payload.get("timestamp", time.time())),
            trace_type=str(payload.get("trace_type", "observation")),
            content=dict(payload.get("content", {})),
            parent_trace_id=payload.get("parent_trace_id"),
            metadata=dict(payload.get("metadata", {})),
        )

    def _transition_to_dict(self, transition: TrainingTransition) -> Dict[str, Any]:
        return {
            "state": transition.state,
            "action": transition.action,
            "reward": float(transition.reward),
            "next_state": transition.next_state,
            "done": bool(transition.done),
            "trace_sequence": [self._trace_to_dict(t) for t in transition.trace_sequence],
            "session_id": transition.session_id,
            "agent_id": transition.agent_id,
        }

    def _transition_from_dict(self, payload: Dict[str, Any]) -> TrainingTransition:
        trace_sequence = [
            self._trace_from_dict(item) for item in payload.get("trace_sequence", [])
            if isinstance(item, dict)
        ]
        return TrainingTransition(
            state=dict(payload.get("state", {})),
            action=dict(payload.get("action", {})),
            reward=float(payload.get("reward", 0.0)),
            next_state=dict(payload.get("next_state", {})),
            done=bool(payload.get("done", False)),
            trace_sequence=trace_sequence,
            session_id=str(payload.get("session_id", "")),
            agent_id=str(payload.get("agent_id", "")),
        )

    def build_checkpoint_payload(
        self,
        include_replay_buffer: bool = True,
        max_replay_items: Optional[int] = None,
    ) -> Dict[str, Any]:
        transitions: List[TrainingTransition]
        if include_replay_buffer:
            transitions = list(self.replay_buffer)
            if max_replay_items and max_replay_items > 0:
                transitions = transitions[-max_replay_items:]
        else:
            transitions = []

        payload: Dict[str, Any] = {
            "metadata": {
                "algorithm": "LightningRLAlgorithm",
                "state_dim": self.state_dim,
                "hidden_dim": self.hidden_dim,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "training_steps": int(self.training_steps),
                "buffer_size": len(self.replay_buffer),
                "state_encoder": self.state_encoder,
                "checkpoint_time": time.time(),
            },
            "replay_buffer": [self._transition_to_dict(t) for t in transitions],
        }

        if TORCH_AVAILABLE and hasattr(self, "value_net"):
            payload["model"] = {
                "value_net": self.value_net.state_dict(),
                "policy_net": self.policy_net.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
            }
        return payload

    def load_checkpoint_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {"status": "error", "reason": "invalid_checkpoint_payload"}

        meta = payload.get("metadata", {})
        self.training_steps = int(meta.get("training_steps", self.training_steps))
        loaded_encoder = str(meta.get("state_encoder", "legacy_python_hash"))
        if loaded_encoder != self.state_encoder:
            self.logger.warning(
                "Loaded checkpoint with state encoder '%s' into '%s'.",
                loaded_encoder,
                self.state_encoder,
            )

        replay_items = payload.get("replay_buffer", [])
        loaded_replay = 0
        if isinstance(replay_items, list):
            for item in replay_items:
                if not isinstance(item, dict):
                    continue
                try:
                    self.replay_buffer.append(self._transition_from_dict(item))
                    loaded_replay += 1
                except Exception:
                    continue

        loaded_models = False
        if TORCH_AVAILABLE and hasattr(self, "value_net"):
            model = payload.get("model", {})
            if isinstance(model, dict) and model:
                try:
                    if "value_net" in model:
                        self.value_net.load_state_dict(model["value_net"])
                    if "policy_net" in model:
                        self.policy_net.load_state_dict(model["policy_net"])
                    if "value_optimizer" in model:
                        self.value_optimizer.load_state_dict(model["value_optimizer"])
                    if "policy_optimizer" in model:
                        self.policy_optimizer.load_state_dict(model["policy_optimizer"])
                    loaded_models = True
                except Exception:
                    loaded_models = False

        return {
            "status": "loaded",
            "training_steps": self.training_steps,
            "replay_loaded": loaded_replay,
            "models_loaded": loaded_models,
            "state_encoder": loaded_encoder,
        }

    def save_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        include_replay_buffer: bool = True,
        max_replay_items: Optional[int] = None,
    ) -> Dict[str, Any]:
        path = Path(checkpoint_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.build_checkpoint_payload(
            include_replay_buffer=include_replay_buffer,
            max_replay_items=max_replay_items,
        )
        if TORCH_AVAILABLE:
            torch.save(payload, str(path))
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        return {
            "status": "saved",
            "path": str(path),
            "training_steps": self.training_steps,
            "buffer_size": len(self.replay_buffer),
        }

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        allow_missing: bool = True,
    ) -> Dict[str, Any]:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.exists():
            status = {"status": "missing", "path": str(path)}
            if allow_missing:
                return status
            raise FileNotFoundError(str(path))

        if TORCH_AVAILABLE:
            payload = torch.load(str(path), map_location="cpu")
        else:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

        loaded = self.load_checkpoint_payload(payload)
        loaded["path"] = str(path)
        return loaded


class AgentLightningTrainer:
    """
    Main Agent Lightning trainer - implements Training-Agent Disaggregation.
    
    This is the core class that enables training ANY agent with RL
    without modifying the agent code.
    """
    
    def __init__(
        self,
        collector: Optional[ObservabilityCollector] = None,
        credit_assignment: Optional[CreditAssignmentModule] = None,
        rl_algorithm: Optional[LightningRLAlgorithm] = None,
        gam_memory_system = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_autosave: bool = True,
    ):
        # Initialize components
        self.collector = collector or ObservabilityCollector()
        self.credit_assignment = credit_assignment or CreditAssignmentModule()
        self.rl_algorithm = rl_algorithm or LightningRLAlgorithm()
        self.gam_memory = gam_memory_system
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.checkpoint_autosave = bool(checkpoint_autosave)
        
        # Registered agents
        self.agents: Dict[str, Callable] = {}
        self.reward_functions: Dict[str, Callable] = {}
        
        # Training state
        self.training_enabled = True
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Agent Lightning trainer initialized")
        if self.checkpoint_path:
            load_result = self.load_checkpoint(self.checkpoint_path, allow_missing=True)
            if load_result.get("status") == "loaded":
                self.logger.info(
                    "Loaded Agent Lightning checkpoint from %s (steps=%s, replay=%s)",
                    self.checkpoint_path,
                    load_result.get("training_steps", 0),
                    load_result.get("replay_loaded", 0),
                )
    
    def register_agent(
        self, 
        agent_id: str, 
        agent_function: Callable,
        reward_function: Optional[Callable] = None
    ) -> None:
        """
        Register an agent for training.
        
        This is the key API - ANY agent can be registered with zero code changes.
        """
        self.agents[agent_id] = agent_function
        if reward_function:
            self.reward_functions[agent_id] = reward_function
        
        self.logger.info(f"Registered agent '{agent_id}' for RL training")
    
    async def train_agent(
        self, 
        agent_id: str, 
        task_data: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train agent with RL using Agent Lightning methodology.
        
        This implements the core training loop with observability collection
        and hierarchical RL.
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent '{agent_id}' not registered")
        
        session_id = session_id or str(uuid.uuid4())
        agent_function = self.agents[agent_id]
        
        # Start GAM session if available
        gam_session_id = None
        if self.gam_memory:
            gam_session_id = self.gam_memory.start_session(
                tenant_id=task_data.get("tenant_id", "default")
            )
        
        # Start observability collection
        self.collector.start_session(session_id, agent_id)
        rl_training_result = None
        
        try:
            # Collect pre-execution trace
            self.collector.collect_trace(
                session_id, agent_id, "action",
                {
                    "type": "task_start",
                    "task_data": task_data,
                    "agent_id": agent_id
                }
            )
            
            # Execute agent (ZERO CODE CHANGES to existing agent)
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(agent_function):
                result = await agent_function(task_data)
            else:
                result = agent_function(task_data)
            
            execution_time = time.time() - start_time

            decision_signals = task_data.get("decision_signals", [])
            if isinstance(decision_signals, list) and decision_signals:
                self._collect_decision_signal_traces(
                    session_id=session_id,
                    agent_id=agent_id,
                    decision_signals=decision_signals,
                )
            
            # Collect post-execution trace
            self.collector.collect_trace(
                session_id, agent_id, "observation",
                {
                    "type": "task_result", 
                    "result": result,
                    "execution_time": execution_time,
                    "success": self._evaluate_success(result)
                }
            )
            
            # Process traces for RL training
            if self.training_enabled:
                rl_training_result = await self._process_training_data(
                    session_id, agent_id, task_data, result, execution_time
                )
            
            # Update GAM memory
            if self.gam_memory and gam_session_id:
                self.gam_memory.add_to_session(
                    gam_session_id, "agent", 
                    f"Agent {agent_id} executed task with result: {result}"
                )
                
                self.gam_memory.end_session_with_memo(
                    gam_session_id,
                    spec_title=task_data.get("spec_title", "Agent Task"),
                    endpoints_count=1,
                    tests_generated=1, 
                    key_decisions=[f"Agent {agent_id} training executed"],
                    issues_found=[]
                )
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "session_id": session_id,
                "traces_collected": len(self.collector.active_sessions.get(session_id, [])),
                "training_enabled": self.training_enabled,
                "rl_training_result": rl_training_result,
            }
            
        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}")
            
            # Collect failure trace
            self.collector.collect_trace(
                session_id, agent_id, "observation",
                {"type": "task_failure", "error": str(e)}
            )
            
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id,
                "traces_collected": len(self.collector.active_sessions.get(session_id, [])),
                "training_enabled": self.training_enabled,
                "rl_training_result": rl_training_result,
            }
        
        finally:
            # End observability session
            traces = self.collector.end_session(session_id)

    def _collect_decision_signal_traces(
        self,
        session_id: str,
        agent_id: str,
        decision_signals: List[Dict[str, Any]],
    ) -> None:
        """Attach per-scenario decision traces so RL gets dense training feedback."""
        for idx, signal in enumerate(decision_signals):
            if not isinstance(signal, dict):
                continue

            reward_signal = float(signal.get("reward", 0.0))
            self.collector.collect_trace(
                session_id,
                agent_id,
                "action",
                {
                    "type": "scenario_decision",
                    "index": idx,
                    "name": str(signal.get("name", f"scenario_{idx}")),
                    "test_type": str(signal.get("test_type", "unknown")),
                    "method": str(signal.get("method", "GET")).upper(),
                    "endpoint_template": str(signal.get("endpoint_template", "")),
                    "endpoint_key": str(signal.get("endpoint_key", "")),
                    "has_body": bool(signal.get("has_body", False)),
                    "has_params": bool(signal.get("has_params", False)),
                    "expected_status": int(signal.get("expected_status", 0)),
                    "actual_status": signal.get("actual_status"),
                    "passed": bool(signal.get("passed", False)),
                    "reward_signal": reward_signal,
                },
            )
    
    async def _process_training_data(
        self, 
        session_id: str, 
        agent_id: str, 
        task_data: Dict[str, Any],
        result: Any, 
        execution_time: float
    ) -> None:
        """Process collected traces into RL training data."""
        traces = self.collector.active_sessions.get(session_id, [])
        
        if len(traces) < 2:
            return
        
        # Calculate final reward
        success = self._evaluate_success(result)
        final_reward = self._calculate_reward(task_data, result, execution_time, success)
        
        # Assign credit across traces
        rewards = self.credit_assignment.assign_credit(traces, final_reward, success)
        
        # Create RL transitions
        for i in range(len(traces) - 1):
            current_trace = traces[i]
            next_trace = traces[i + 1]
            
            # Only create transitions for action traces
            if current_trace.trace_type == "action":
                transition_reward = rewards[i]
                if current_trace.content.get("type") == "scenario_decision":
                    transition_reward = float(
                        current_trace.content.get("reward_signal", transition_reward)
                    )

                transition = TrainingTransition(
                    state=current_trace.content,
                    action={"type": current_trace.trace_type, "content": current_trace.content},
                    reward=transition_reward,
                    next_state=next_trace.content,
                    done=(i == len(traces) - 2),
                    trace_sequence=traces[i:i+2],
                    session_id=session_id,
                    agent_id=agent_id
                )
                
                # Add to RL algorithm
                self.rl_algorithm.add_transition(transition)
        
        # Train RL model (Agent Lightning: train more frequently with smaller batches)
        training_result = self.rl_algorithm.train_step()  # Always try to train
        
        if training_result.get("status") == "trained":
            self.logger.info(f"🧠 RL training executed: Loss={training_result['value_loss']:.4f}")
            print(f"🧠 RL TRAINING ACTIVE: Step {training_result['training_steps']}, Loss={training_result['value_loss']:.4f}")
        elif training_result.get("status") == "skipped":
            self.logger.debug(f"RL training skipped: {training_result.get('reason', 'unknown')}")

        if self.checkpoint_autosave and self.checkpoint_path:
            try:
                checkpoint_result = self.save_checkpoint(self.checkpoint_path)
                training_result["checkpoint"] = checkpoint_result
            except Exception as e:
                self.logger.warning("Failed to save Agent Lightning checkpoint: %s", e)
                training_result["checkpoint_error"] = str(e)
        
        # Store training result in the overall result
        return training_result
    
    def _evaluate_success(self, result: Any) -> bool:
        """Evaluate if agent execution was successful."""
        if isinstance(result, dict):
            return result.get("success", True)
        return result is not None
    
    def _calculate_reward(
        self, 
        task_data: Dict[str, Any], 
        result: Any, 
        execution_time: float, 
        success: bool
    ) -> float:
        """Calculate reward for agent performance."""
        base_reward = 1.0 if success else -0.5
        
        # Time-based penalty
        time_penalty = min(execution_time / 10.0, 0.5)
        
        # Quality bonus
        quality_bonus = 0.0
        if isinstance(result, dict) and "quality_score" in result:
            quality_bonus = result["quality_score"] * 0.3
        
        return base_reward - time_penalty + quality_bonus

    def save_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        include_replay_buffer: bool = True,
        max_replay_items: Optional[int] = None,
    ) -> Dict[str, Any]:
        path = checkpoint_path or self.checkpoint_path
        if not path:
            return {"status": "skipped", "reason": "checkpoint_path_not_set"}
        return self.rl_algorithm.save_checkpoint(
            checkpoint_path=path,
            include_replay_buffer=include_replay_buffer,
            max_replay_items=max_replay_items,
        )

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        allow_missing: bool = True,
    ) -> Dict[str, Any]:
        path = checkpoint_path or self.checkpoint_path
        if not path:
            return {"status": "skipped", "reason": "checkpoint_path_not_set"}
        return self.rl_algorithm.load_checkpoint(path, allow_missing=allow_missing)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        rl_buffer_size = len(self.rl_algorithm.replay_buffer)
        rl_training_steps = self.rl_algorithm.training_steps
        return {
            "registered_agents": len(self.agents),
            "total_traces": len(self.collector.traces),
            "active_sessions": len(self.collector.active_sessions),
            "rl_buffer_size": rl_buffer_size,
            "rl_training_steps": rl_training_steps,
            "training_enabled": self.training_enabled,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_autosave": self.checkpoint_autosave,
            "rl_model_ready": bool(rl_training_steps >= 3 and rl_buffer_size >= 32),
            "state_encoder": getattr(self.rl_algorithm, "state_encoder", "unknown"),
        }


# Integration adapter for existing SpecTestPilot agent
class SpecTestPilotAdapter:
    """Adapter to integrate SpecTestPilot with Agent Lightning."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def run_spec_test_agent(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrap existing SpecTestPilot functionality for Agent Lightning.
        This enables RL training with ZERO changes to existing code.
        """
        try:
            # Import existing components
            from .multi_language_tester import HumanTesterSimulator, MultiLanguageTestGenerator
            from .sandbox import AgentLightningSandbox
            
            # Run existing agent logic
            sandbox = AgentLightningSandbox()
            result = sandbox.execute_agent_task(task_data)
            
            return {
                "success": result.get("success", True),
                "result": result,
                "quality_score": 0.9 if result.get("success") else 0.3
            }
            
        except Exception as e:
            self.logger.error(f"SpecTestPilot execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "quality_score": 0.0
            }


# Example usage with existing SpecTestPilot
def create_agent_lightning_system(
    gam_memory_system=None,
    checkpoint_path: Optional[str] = None,
    checkpoint_autosave: bool = True,
):
    """Create Agent Lightning system integrated with existing components."""
    
    # Initialize Agent Lightning
    trainer = AgentLightningTrainer(
        gam_memory_system=gam_memory_system,
        checkpoint_path=checkpoint_path,
        checkpoint_autosave=checkpoint_autosave,
    )
    
    # Create adapter for existing SpecTestPilot
    adapter = SpecTestPilotAdapter()
    
    # Register SpecTestPilot with Agent Lightning (ZERO CODE CHANGES)
    trainer.register_agent("spec_test_pilot", adapter.run_spec_test_agent)
    
    return trainer


# Main training function for integration
async def train_with_agent_lightning(
    task_data: Dict[str, Any],
    gam_memory_system=None,
    checkpoint_path: Optional[str] = None,
    checkpoint_autosave: bool = True,
) -> Dict[str, Any]:
    """
    Train SpecTestPilot using Agent Lightning with zero code changes.
    
    This is the main entry point that demonstrates how Agent Lightning
    enables RL training of ANY existing agent.
    """
    
    # Create system
    trainer = create_agent_lightning_system(
        gam_memory_system,
        checkpoint_path=checkpoint_path,
        checkpoint_autosave=checkpoint_autosave,
    )
    
    # Train agent
    result = await trainer.train_agent("spec_test_pilot", task_data)
    
    # Return comprehensive results
    stats = trainer.get_training_stats()
    result.update(stats)
    
    return result


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def demo():
        task_data = {
            "openapi_spec": "examples/banking_api.yaml",
            "spec_title": "Banking API",
            "tenant_id": "demo_corp"
        }
        
        result = await train_with_agent_lightning(task_data)
        print("Agent Lightning training result:", json.dumps(result, indent=2))
    
    asyncio.run(demo())
