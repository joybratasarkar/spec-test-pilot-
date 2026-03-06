#!/usr/bin/env python3
"""Periodic RL trainer for Agent Lightning checkpoints.

Usage:
  python backend/rl_periodic_trainer.py \
    --checkpoint /path/to/agent_lightning_checkpoint.pt \
    --max-steps 25 \
    --min-buffer 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spec_test_pilot.agent_lightning_v2 import AgentLightningTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run periodic RL training on a saved checkpoint buffer")
    parser.add_argument("--checkpoint", required=True, help="Path to Agent Lightning checkpoint (.pt or .json)")
    parser.add_argument(
        "--train-mode",
        default="periodic",
        choices=["periodic"],
        help="Trainer mode (mandatory): periodic",
    )
    parser.add_argument("--max-steps", type=int, default=25, help="Maximum train_step calls in this batch")
    parser.add_argument("--min-buffer", type=int, default=32, help="Minimum replay buffer size to start training")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checkpoint_path = str(Path(args.checkpoint).expanduser().resolve())

    trainer = AgentLightningTrainer(
        checkpoint_path=checkpoint_path,
        checkpoint_autosave=True,
        gam_writeback=False,
        train_mode=args.train_mode,
    )
    result = trainer.run_periodic_training(
        max_steps=max(1, int(args.max_steps)),
        min_buffer_size=max(1, int(args.min_buffer)),
    )
    stats = trainer.get_training_stats()

    print("RL periodic training complete")
    print(json.dumps({"result": result, "stats": stats}, indent=2))

    status = str(result.get("status", ""))
    if status in {"completed", "skipped"}:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
