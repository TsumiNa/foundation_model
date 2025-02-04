from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from multi_task import (
    CompoundDataModule,
    CompoundDataset,
    MultiTaskPropertyPredictor,
    plot_predictions,
    training,
)


def scaling_laws_test(
    descriptor,
    property_data,
    target_property,
    mode="vary_target",
    device="cpu",
    num_runs=4,
):
    """
    Test scaling laws for multi-task learning.

    Parameters
    ----------
    descriptor : pd.DataFrame
        Input feature DataFrame
    property_data : pd.DataFrame
        Property values DataFrame
    target_property : str
        The property to analyze scaling laws for
    mode : str, optional
        Either 'vary_target' or 'vary_others'
        - 'vary_target': Fix other properties at 100% and vary target property fraction
        - 'vary_others': Fix target property at 100% and vary other properties fraction
    device : str, optional
        Device to run the model on, by default "cpu"
    num_runs : int, optional
        Number of runs for each configuration, by default 4

    Returns
    -------
    tuple
        (test_losses, test_losses_std, fractions) containing:
        - test_losses: Mean test loss for each fraction
        - test_losses_std: Standard deviation of test losses
        - fractions: List of fractions tested
    """
    # Create a temporary dataset to get the list of attributes
    temp_dataset = CompoundDataset(descriptor, property_data)
    if target_property not in temp_dataset.attributes:
        raise ValueError(f"target_property must be one of {temp_dataset.attributes}")

    target_idx = temp_dataset.attributes.index(target_property)

    # Define fractions to test
    fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    test_losses = []
    test_losses_std = []

    for fraction in fractions:
        losses = []
        print(f"\nTesting with fraction {fraction}:")

        all_stat = []
        # Plot and save predictions for this run
        save_dir = Path(
            f"images/multi_tasks/scaling_laws/{mode}_{target_property}/fraction_{fraction}"
        )
        for run in range(num_runs):
            # Split data using indices to ensure alignment
            indices = np.arange(len(descriptor))
            train_indices, test_indices = train_test_split(
                indices, test_size=0.2, random_state=run
            )

            train_data = descriptor.iloc[train_indices]
            test_data = descriptor.iloc[test_indices]
            train_prop = property_data.iloc[train_indices]
            test_prop = property_data.iloc[test_indices]

            # Set up property fractions based on mode
            if mode == "vary_target":
                # Fix other properties at 100% and vary target property
                property_fractions = {attr: 1.0 for attr in temp_dataset.attributes}
                property_fractions[target_property] = fraction
            else:  # vary_others
                # Fix target property at 100% and vary other properties
                property_fractions = {
                    attr: fraction for attr in temp_dataset.attributes
                }
                property_fractions[target_property] = 1.0

            # Create datasets with specified property fractions
            train_dataset = CompoundDataset(
                train_data, train_prop, **property_fractions
            )
            test_dataset = CompoundDataset(test_data, test_prop)

            # Create data module
            data_module = CompoundDataModule(
                descriptor=train_data,
                property_data=train_prop,
                splitter=lambda x: (np.arange(len(x)), []),  # No validation split
                property_fractions=property_fractions,
                batch_size=32,
            )
            data_module.test_dataset = test_dataset  # Set test dataset directly

            # Initialize model
            model = MultiTaskPropertyPredictor(
                shared_block_dims=[train_data.shape[1], 512, 256, 128],
                task_block_dims=[128, 64, 32, 1],
                n_tasks=len(train_dataset.attributes),
                shared_block_lr=0.0005,
                task_block_lr=0.0005,
            )

            # Train and evaluate
            try:
                # Set up experiment directory
                exp_dir = save_dir / f"run_{run}"
                exp_dir.mkdir(parents=True, exist_ok=True)

                avg_test_losses, all_preds, all_targets, all_masks = training(
                    model=model,
                    datamodule=data_module,
                    max_epochs=100,
                    accelerator=device if device == "cpu" else "gpu",
                    devices=1,
                    default_root_dir=str(exp_dir),
                )

                # Record loss for target property
                losses.append(avg_test_losses[target_idx])
                stat = plot_predictions(
                    all_preds,
                    all_targets,
                    all_masks,
                    test_dataset,
                    savefig=str(save_dir),
                    suffix=f"run_{run}",
                    return_stat=True,
                    no_show=True,
                )

                stat = stat.assign(run=run)
                all_stat.append(stat)
            except ValueError as e:
                print(f"{e} in {save_dir} - run {run}")
            except Exception as e:
                # Handle the exception
                print(f"An unexpected error occurred: {e}")

        pd.concat(all_stat).reset_index(drop=True).to_csv(save_dir / "test_stat.csv")
        test_losses.append(np.mean(losses))
        test_losses_std.append(np.std(losses))

    # Plotting with confidence intervals
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Plot mean loss
    plt.plot(fractions, test_losses, marker="o", label="Mean Loss")

    # Add confidence intervals
    ci = 1.96 * np.array(test_losses_std) / np.sqrt(num_runs)
    plt.fill_between(
        fractions,
        np.array(test_losses) - ci,
        np.array(test_losses) + ci,
        alpha=0.3,
        label="95% CI",
    )

    plt.xlabel("Data Fraction")
    plt.ylabel(f"Test Loss for {target_property}")
    title = (
        f"Effect of {'Target' if mode == 'vary_target' else 'Other Properties'} "
        f"Data Size on {target_property} Prediction"
    )
    plt.title(title)
    plt.legend()

    # Save plot
    save_dir = Path("images/multi_tasks/scaling_laws")
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        save_dir / f"scaling_law_{mode}_{target_property}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    return test_losses, test_losses_std, fractions
