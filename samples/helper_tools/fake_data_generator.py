# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""
Generates fake data for demonstration purposes.
Creates 'attributes.csv' and 'formula_features.csv'.
- attributes.csv: Contains target variables for 3 regression and 2 classification tasks, plus an ID and split.
- formula_features.csv: Contains dummy formula features, plus an ID.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_NUM_SAMPLES = 1000
DEFAULT_NUM_FORMULA_FEATURES = 256
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "fake_data"

# For reproducibility of fake data
DEFAULT_RANDOM_SEED = 42


def generate_fake_data(
    num_samples: int,
    num_formula_features: int,
    output_dir: Path,
    random_seed: int = DEFAULT_RANDOM_SEED,
):
    """
    Generates and saves fake attributes and formula features CSV files.

    Args:
        num_samples (int): Number of samples to generate.
        num_formula_features (int): Number of formula features to generate.
        output_dir (Path): Directory to save the CSV files.
        random_seed (int): Seed for random number generation.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    ids = [f"sample_{i:04d}" for i in range(num_samples)]

    # --- Generate attributes.csv ---
    attributes_data = {"id": ids}

    # 3 Regression tasks
    attributes_data["target_regression_1"] = np.random.rand(num_samples) * 10 - 5  # Range [-5, 5]
    attributes_data["target_regression_2"] = np.random.randn(num_samples) * 2 + 1  # Normal dist, mean 1, std 2
    attributes_data["target_regression_3"] = np.random.gamma(2, 2, num_samples)  # Gamma distribution

    # 2 Classification tasks
    # Binary classification
    attributes_data["target_classification_A"] = np.random.randint(0, 2, num_samples)  # Values: 0 or 1
    # Multi-class classification (3 classes)
    attributes_data["target_classification_B"] = np.random.randint(0, 3, num_samples)  # Values: 0, 1, or 2

    # Add a 'split' column
    splits = []
    for _ in range(num_samples):
        rand_val = random.random()
        if rand_val < 0.7:
            splits.append("train")
        elif rand_val < 0.85:
            splits.append("val")
        else:
            splits.append("test")
    attributes_data["split"] = splits

    attributes_df = pd.DataFrame(attributes_data)
    attributes_path = output_dir / "attributes.csv"
    attributes_df.to_csv(attributes_path, index=False)
    print(f"Generated {attributes_path} with {len(attributes_df)} samples.")

    # --- Generate formula_features.csv ---
    formula_features_data = {"id": ids}
    for i in range(num_formula_features):
        # Simulate different types of features
        if i % 3 == 0:  # Integer-like features
            formula_features_data[f"feature_{i + 1}"] = np.random.randint(0, 100, num_samples)
        elif i % 3 == 1:  # Features with a wider range
            formula_features_data[f"feature_{i + 1}"] = np.random.rand(num_samples) * 100
        else:  # Standard normal features
            formula_features_data[f"feature_{i + 1}"] = np.random.randn(num_samples)

    formula_features_df = pd.DataFrame(formula_features_data)
    formula_features_path = output_dir / "formula_features.csv"
    formula_features_df.to_csv(formula_features_path, index=False)
    print(
        f"Generated {formula_features_path} with {len(formula_features_df)} samples and {num_formula_features} features."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fake data for foundation model examples.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of samples to generate (default: {DEFAULT_NUM_SAMPLES})",
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=DEFAULT_NUM_FORMULA_FEATURES,
        help=f"Number of formula features to generate (default: {DEFAULT_NUM_FORMULA_FEATURES})",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the generated CSV files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_SEED})",
    )

    args = parser.parse_args()

    print(f"Generating fake data with seed {args.seed}...")
    generate_fake_data(
        num_samples=args.num_samples,
        num_formula_features=args.num_features,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )
    print("Fake data generation complete.")
    print(f"Data saved in: {args.output_dir.resolve()}")

    # To run this script:
    # python samples/helper_tools/fake_data_generator.py
    #
    # To install dependencies if not already present (using uv, as per project setup):
    # uv add pandas numpy
