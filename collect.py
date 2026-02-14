"""
Data collection script for sort-vs-reverse experiment.
Run in tmux: python collect.py [--num_datasets 500] [--max_steps 50] [--debug]
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path.home() / ".env")

from lib import (
    ExperimentConfig, InnerModelManager,
    collect_data, samples_to_dataframe,
)


def main():
    parser = argparse.ArgumentParser(description="Collect sort-vs-reverse finetune data")
    parser.add_argument("--num_datasets", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="collected_data.csv")
    parser.add_argument("--debug", action="store_true", help="Enable wandb for inner runs")
    args = parser.parse_args()

    config = ExperimentConfig(
        num_datasets=args.num_datasets,
        inner_max_steps=args.max_steps,
        inner_learning_rate=args.learning_rate,
        debug=args.debug,
    )

    print(f"Config: {args.num_datasets} datasets, {args.max_steps} steps, lr={args.learning_rate}")
    print(f"Output: {args.output}")

    manager = InnerModelManager(config)

    print("\n=== Baseline Evaluation ===")
    manager.evaluate_baseline(seed=0)

    print("\n=== Collecting Data ===")
    samples = collect_data(config, manager, start_seed=args.start_seed)

    df = samples_to_dataframe(samples)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} samples to {args.output}")
    print(f"Label distribution:\n{df.label.value_counts()}")


if __name__ == "__main__":
    main()
