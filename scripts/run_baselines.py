#!/usr/bin/env python3
"""
Phase 0: Measure baseline error rates.

Usage:
    python scripts/run_baselines.py
    python scripts/run_baselines.py --config config/custom.yaml
    python scripts/run_baselines.py --dataset custom --max-samples 10
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.attack.pipeline import AttackPipeline
from src.data import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Measure baseline error rates")
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
        help="Maximum number of questions"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Questions measured in parallel (rate limits and budget are shared)"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.dataset:
        config.dataset.name = args.dataset
    if args.max_samples:
        config.dataset.max_samples = args.max_samples

    # Run
    pipeline = AttackPipeline(config)

    # Load data
    qa_items = load_dataset(
        name=config.dataset.name,
        split=config.dataset.split,
        max_samples=config.dataset.max_samples,
        seed=config.experiment.seed,
    )

    print(f"Loaded {len(qa_items)} questions")
    print(f"Target model: {config.target_model.provider}/{config.target_model.model_name}")
    print()

    pipeline.run_baselines(qa_items, workers=args.workers)


if __name__ == "__main__":
    main()
