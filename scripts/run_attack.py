#!/usr/bin/env python3
"""
Phase 1: Run the evolutionary framing attack.

Usage:
    python scripts/run_attack.py
    python scripts/run_attack.py --config config/custom.yaml
    python scripts/run_attack.py --dataset custom --max-samples 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.attack.pipeline import AttackPipeline
from src.data import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Run the framing attack")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to override config YAML"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset name (triviaqa, custom, combined)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of questions to attack"
    )
    parser.add_argument(
        "--include-already-wrong", action="store_true",
        help="Include questions the model already gets wrong"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.dataset:
        config.dataset.name = args.dataset
    if args.max_samples:
        config.dataset.max_samples = args.max_samples

    pipeline = AttackPipeline(config)

    qa_items = load_dataset(
        name=config.dataset.name,
        split=config.dataset.split,
        max_samples=config.dataset.max_samples,
        seed=config.experiment.seed,
    )

    print(f"Loaded {len(qa_items)} questions")
    print(f"Target model: {config.target_model.provider}/{config.target_model.model_name}")
    print(f"Auxiliary model: {config.auxiliary_model.provider}/{config.auxiliary_model.model_name}")
    print()

    pipeline.run_attack(
        qa_items,
        skip_already_wrong=not args.include_already_wrong,
    )


if __name__ == "__main__":
    main()
