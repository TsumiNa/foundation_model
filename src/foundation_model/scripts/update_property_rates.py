#!/usr/bin/env python3
import argparse
from dataclasses import field

from ..configs.model_config import ExperimentConfig


def update_property_rates(rate: float) -> None:
    """Update property sampling rates based on predefined groups.

    Args:
        rate: New rate value for mp_props group (0.0 to 1.0)
    """
    # Create a new property_fractions dictionary
    new_fractions = {}

    # Set ac_qc_starry_props to 1.0 (fixed)
    for prop in ExperimentConfig.ac_qc_starry_props:
        new_fractions[prop] = 1.0

    # Set mp_props to specified rate
    for prop in ExperimentConfig.mp_props:
        new_fractions[prop] = rate

    # Update ExperimentConfig's property_fractions default factory
    ExperimentConfig.property_fractions = field(
        default_factory=lambda: dict(new_fractions)
    )
    print("Updated property rates:")
    print("- Fixed properties (rate=1.0):")
    for prop in ExperimentConfig.ac_qc_starry_props:
        print(f"  - {prop}")
    print(f"\n- Variable properties (rate={rate}):")
    for prop in ExperimentConfig.mp_props:
        print(f"  - {prop}")


def main():
    parser = argparse.ArgumentParser(
        description="Update property sampling rates for mp_props while keeping ac_qc_starry_props fixed at 1.0"
    )
    parser.add_argument(
        "--rate",
        type=float,
        required=True,
        help="Rate value for mp_props (between 0.0 and 1.0)",
    )
    args = parser.parse_args()

    if not 0.0 <= args.rate <= 1.0:
        raise ValueError("Rate must be between 0.0 and 1.0")

    update_property_rates(args.rate)


if __name__ == "__main__":
    main()
