"""Tests for Agent Lightning checkpoint save/load persistence."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from spec_test_pilot.agent_lightning_v2 import (
    AgentLightningTrainer,
    AgentTrace,
    LightningRLAlgorithm,
    TORCH_AVAILABLE,
    TrainingTransition,
)


def _make_transition(index: int) -> TrainingTransition:
    trace = AgentTrace(
        trace_id=f"trace-{index}",
        session_id=f"session-{index}",
        agent_id="qa_specialist",
        timestamp=time.time(),
        trace_type="action",
        content={"step": index, "signal": f"s{index}"},
    )
    return TrainingTransition(
        state={"k": f"state-{index}", "n": index},
        action={"type": "action", "k": f"action-{index}"},
        reward=float(index) * 0.1 + 0.3,
        next_state={"k": f"next-{index}", "n": index + 1},
        done=(index == 3),
        trace_sequence=[trace],
        session_id=trace.session_id,
        agent_id=trace.agent_id,
    )


def test_rl_algorithm_checkpoint_roundtrip(tmp_path: Path) -> None:
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch unavailable in environment")

    import torch

    ckpt = tmp_path / "lightning_algo.pt"
    algo = LightningRLAlgorithm(state_dim=32, hidden_dim=16, batch_size=4, buffer_size=100)

    for i in range(4):
        algo.add_transition(_make_transition(i))
    train_result = algo.train_step()
    assert train_result["status"] == "trained"

    save_result = algo.save_checkpoint(str(ckpt))
    assert save_result["status"] == "saved"
    assert ckpt.exists()

    restored = LightningRLAlgorithm(state_dim=32, hidden_dim=16, batch_size=4, buffer_size=100)
    load_result = restored.load_checkpoint(str(ckpt), allow_missing=False)
    assert load_result["status"] == "loaded"
    assert restored.training_steps == algo.training_steps
    assert len(restored.replay_buffer) == len(algo.replay_buffer)

    for p1, p2 in zip(algo.value_net.parameters(), restored.value_net.parameters()):
        assert torch.allclose(p1, p2)


def test_trainer_auto_loads_checkpoint(tmp_path: Path) -> None:
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch unavailable in environment")

    ckpt = tmp_path / "trainer_auto_load.pt"
    trainer = AgentLightningTrainer(
        rl_algorithm=LightningRLAlgorithm(state_dim=24, hidden_dim=12, batch_size=4, buffer_size=50),
        checkpoint_path=str(ckpt),
        checkpoint_autosave=False,
    )
    trainer.rl_algorithm.add_transition(_make_transition(0))
    save_result = trainer.save_checkpoint()
    assert save_result["status"] == "saved"

    restored = AgentLightningTrainer(
        rl_algorithm=LightningRLAlgorithm(state_dim=24, hidden_dim=12, batch_size=4, buffer_size=50),
        checkpoint_path=str(ckpt),
        checkpoint_autosave=False,
    )
    assert len(restored.rl_algorithm.replay_buffer) == 1
    stats = restored.get_training_stats()
    assert stats["checkpoint_path"] == str(ckpt)
