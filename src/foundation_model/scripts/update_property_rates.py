#!/usr/bin/env python
import argparse
from dataclasses import field

from foundation_model.configs.model_config import ExperimentConfig


def update_attribute_rates(rate: float) -> None:
    """Update attribute sampling rates based on predefined groups.

    Parameters
    ----------
    rate : float
        New sampling rate for mp_attrs group
    """
    # Create a new attribute_rates dictionary
    new_rates = {}

    # Set ac_qc_starry_attrs to fixed rate 1.0
    for attr in ExperimentConfig.ac_qc_starry_attrs:
        new_rates[attr] = 1.0

    # Set mp_attrs to specified rate
    for attr in ExperimentConfig.mp_attrs:
        new_rates[attr] = rate

    # Update ExperimentConfig's attribute_rates default factory
    ExperimentConfig.attribute_rates = field(default_factory=lambda: dict(new_rates))

    print("Updated attribute rates:")
    print("- Fixed attributes (rate=1.0):")
    for attr in ExperimentConfig.ac_qc_starry_attrs:
        print(f"  - {attr}")

    print(f"\n- Variable attributes (rate={rate}):")
    for attr in ExperimentConfig.mp_attrs:
        print(f"  - {attr}")


def main():
    parser = argparse.ArgumentParser(
        description="Update attribute sampling rates for mp_attrs while keeping ac_qc_starry_attrs fixed at 1.0"
    )
    parser.add_argument(
        "--rate",
        type=float,
        required=True,
        help="Sampling rate for mp_attrs group (between 0 and 1)",
    )
    args = parser.parse_args()
    update_attribute_rates(args.rate)


if __name__ == "__main__":
    main()
