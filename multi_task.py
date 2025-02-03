from typing import Sequence
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt
from plot_utils import plot_scatter_comparison


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CompoundDataset(Dataset):
    def __init__(
        self,
        descriptor: pd.DataFrame,
        property: pd.DataFrame,
        **known_attributes,
    ):
        """
        Custom dataset for compounds.

        Parameters
        ----------
        descriptor : pd.DataFrame
            Input features for the compounds
        property : pd.DataFrame
            Target properties for the compounds
        known_attributes : dict
            Dictionary specifying the percentage of known values for each property
            e.g., {"property_name": 0.8} means 80% of values for that property are known
        """
        # Ensure descriptor and property have matching indices
        if not descriptor.index.equals(property.index):
            raise ValueError("descriptor and property must have matching indices")

        # Get attributes from property columns
        self.attributes = list(property.columns)
        if not self.attributes:
            raise ValueError("property DataFrame must have at least one column")

        # Input features
        self.x = descriptor.values

        # Output attributes - select all columns from property
        self.y = property.values.astype(np.float32)

        # Create initial masks based on non-nan values
        self.mask = (~np.isnan(self.y)).astype(int)

        # Initialize known_attributes with default values
        self._known_attributes = {attr: 1.0 for attr in self.attributes}

        # Validate and update known_attributes if provided
        if known_attributes:
            # Validate attributes
            invalid_attrs = set(known_attributes.keys()) - set(self.attributes)
            if invalid_attrs:
                raise ValueError(
                    f"Invalid attributes in known_attributes: {invalid_attrs}. "
                    f"Valid attributes are: {self.attributes}"
                )

            # Validate percentages
            invalid_percents = [
                (attr, percent)
                for attr, percent in known_attributes.items()
                if not 0 <= percent <= 1
            ]
            if invalid_percents:
                raise ValueError(
                    "Percentages must be between 0 and 1. Invalid values: "
                    f"{invalid_percents}"
                )

            # Update with provided values
            self._known_attributes.update(known_attributes)

            # Apply percentages only to non-nan values
            for attr_name, percent in self._known_attributes.items():
                attr_idx = self.attributes.index(attr_name)
                # Get indices where values are not nan
                valid_indices = np.where(~np.isnan(self.y[:, attr_idx]))[0]
                if len(valid_indices) > 0:  # Only proceed if there are valid values
                    # Calculate number of known samples from valid data
                    num_known = int(len(valid_indices) * percent)
                    # Randomly select indices to keep
                    keep_indices = np.random.choice(
                        valid_indices, num_known, replace=False
                    )
                    # Mask all valid indices except those we keep
                    mask_indices = np.setdiff1d(valid_indices, keep_indices)
                    self.mask[mask_indices, attr_idx] = 0

        # Fill all nan to 0 after masking
        self.y = np.nan_to_num(self.y)

        # Convert to tensors
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        self.mask = torch.tensor(self.mask, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask[idx]

    @property
    def property(self) -> dict[str, float]:
        """
        Get the known attributes percentages.

        Returns
        -------
        dict[str, float]
            Dictionary containing all attributes and their percentages.
        """
        return self._known_attributes.copy()


class LinearLayer(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        normalization=True,
        activation: None | nn.Module = nn.LeakyReLU(0.1),
    ):
        """
        Parameters
        ----------
        n_in: int
            Size of each input sample.
        n_out: int
            Size of each output sample
        """
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.normal = nn.BatchNorm1d(n_out) if normalization else None
        self.activation = nn.LeakyReLU(0.1)
        self.activation = activation

    def forward(self, x):
        _out = self.layer(x)
        if self.normal:
            _out = self.normal(_out)
        if self.activation:
            _out = self.activation(_out)

        return _out


class ResidualBlock(nn.Module):
    def __init__(
        self,
        n_features,
        normalization=True,
        n_layers=2,
        layer_activation: None | nn.Module = nn.LeakyReLU(0.1),
        output_active: None | nn.Module = nn.LeakyReLU(0.1),
    ):
        super().__init__()
        self.layers = nn.Sequential(
            *[
                LinearLayer(
                    n_features,
                    n_features,
                    normalization=normalization,
                    activation=layer_activation,
                )
                if i != n_layers - 1
                else LinearLayer(
                    n_features, n_features, normalization=normalization, activation=None
                )
                for i in range(n_layers)
            ]
        )
        self.output_active = output_active

    def forward(self, x):
        y = self.layers(x)
        y += x
        if self.output_active:
            return self.output_active(y)
        return y


class LinearBlock(nn.Module):
    def __init__(
        self,
        shared_layer_dims: Sequence[int],
        normalization=True,
        residual=False,
        layer_activation: None | nn.Module = nn.LeakyReLU(0.1),
        output_active: None | nn.Module = nn.LeakyReLU(0.1),
    ):
        super().__init__()
        counter = len(shared_layer_dims) - 1
        if counter < 1:
            raise ValueError("shared_layer_dims must have at least 2 elements")

        if residual:
            self.layers = nn.Sequential(
                *[
                    # Add residual block after each layer
                    nn.Sequential(
                        LinearLayer(
                            shared_layer_dims[i],
                            shared_layer_dims[i + 1],
                            normalization=normalization,
                            activation=layer_activation,
                        ),
                        ResidualBlock(
                            shared_layer_dims[i + 1],
                            normalization=normalization,
                            layer_activation=layer_activation,
                            output_active=None,
                        ),
                    )
                    if i == counter - 1 and output_active is None
                    else nn.Sequential(
                        LinearLayer(
                            shared_layer_dims[i],
                            shared_layer_dims[i + 1],
                            normalization=normalization,
                            activation=layer_activation,
                        ),
                        ResidualBlock(
                            shared_layer_dims[i + 1],
                            normalization=normalization,
                            layer_activation=layer_activation,
                            output_active=output_active,
                        ),
                    )
                    for i in range(counter)
                ]
            )
        else:
            self.layers = nn.Sequential(
                *[
                    LinearLayer(
                        shared_layer_dims[i],
                        shared_layer_dims[i + 1],
                        normalization=normalization,
                        activation=None,
                    )
                    if i == counter - 1 and output_active is None
                    else LinearLayer(
                        shared_layer_dims[i],
                        shared_layer_dims[i + 1],
                        normalization=normalization,
                        activation=layer_activation,
                    )
                    for i in range(counter)
                ]
            )

    def forward(self, x):
        return self.layers(x)


class CompoundPropertyPredictor(nn.Module):
    def __init__(
        self,
        shared_block_dims: Sequence[int],
        task_block_dims: Sequence[int],
        n_tasks: int,
        *,
        norm_shared: bool = True,
        residual_shared: bool = False,
        norm_tasks: bool = True,
        residual_tasks: bool = False,
    ):
        super(CompoundPropertyPredictor, self).__init__()
        self.norm_shared = norm_shared
        self.residual_shared = residual_shared

        # Create shared block
        self.shard_block = LinearBlock(
            shared_block_dims,
            normalization=norm_shared,
            residual=residual_shared,
        )

        # Create intermediate layers
        self.deposit_layer = nn.Sequential(
            nn.Linear(shared_block_dims[-1], task_block_dims[0]),
            nn.Tanh(),
        )

        # Create task blocks
        self.task_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    LinearBlock(
                        task_block_dims[:-1],
                        normalization=norm_tasks,
                        residual=residual_tasks,
                    ),
                    LinearLayer(
                        task_block_dims[-2],
                        task_block_dims[-1],
                        activation=None,
                        normalization=False,
                    ),
                )
                for _ in range(n_tasks)
            ]
        )

        # Initialize parameters
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.shard_block(x)  # shared block
        x = self.deposit_layer(x)  # intermediate layer
        x = torch.cat([layer(x) for layer in self.task_blocks], dim=1)  # task blocks

        return x


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    num_epochs: int = 100,
):
    criterion = nn.MSELoss(reduction="none")
    print(f"Optimizer:\n{optimizer}\n")
    print(f"Model:\n{model}\n")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_known = 0
        for inputs, targets, masks in train_loader:
            inputs, targets, masks = (
                inputs.to(device),
                targets.to(device),
                masks.to(device),
            )

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets) * masks
            loss = loss.sum() / masks.sum()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_known += masks.sum().item()

            if np.any(np.isnan(running_loss)):
                raise ValueError("Running loss contains NaN")

        if epoch % 10 == 0:
            avg_train_loss = running_loss / total_known
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}"
            )

    # Evaluation
    model.eval()
    with torch.no_grad():
        # Get number of attributes from the first batch
        num_attributes = next(iter(test_loader))[1].shape[1]
        total_losses = np.zeros(num_attributes)
        total_known = np.zeros(num_attributes)
        all_preds = []
        all_targets = []
        all_masks = []

        for inputs, targets, masks in test_loader:
            inputs, targets, masks = (
                inputs.to(device),
                targets.to(device),
                masks.to(device),
            )
            outputs = model(inputs)

            # Calculate loss for each attribute
            losses = criterion(outputs, targets)  # Shape: [batch_size, num_attributes]
            masked_losses = losses * masks

            # Sum losses and counts for each attribute
            total_losses += masked_losses.sum(dim=0).cpu().numpy()
            total_known += masks.sum(dim=0).cpu().numpy()

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_masks.append(masks.cpu().numpy())

        # Calculate average loss for each attribute
        avg_test_losses = total_losses / total_known

        # Get dataset attributes from test_loader
        test_dataset = test_loader.dataset
        print("\nTest Losses by Attribute:")
        for attr_idx, attr_name in enumerate(test_dataset.attributes):
            print(f"{attr_name}: {avg_test_losses[attr_idx]:.4f}")

    return avg_test_losses, all_preds, all_targets, all_masks


def plot_predictions(
    all_preds,
    all_targets,
    all_masks,
    dataset: CompoundDataset,
    *,
    savefig=None,
    suffix=None,
    no_show=False,
    return_stat=False,
):
    """
    Plot prediction results for each attribute.

    Parameters
    ----------
    all_preds : list
        List of prediction arrays
    all_targets : list
        List of target arrays
    all_masks : list
        List of mask arrays
    dataset : CompoundDataset
        Dataset containing the attributes information
    savefig : str, optional
        Path to save the figures
    suffix : str, optional
        Suffix to append to saved figure names
    no_show : bool, optional
        If True, close figures after saving
    return_stat : bool, optional
        If True, return statistics DataFrame

    Returns
    -------
    pd.DataFrame
        Statistics of the predictions if return_stat is True
    """
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    all_stat = []
    for m in range(min(6, len(dataset.attributes))):
        mask_m = all_masks[:, m] == 1
        preds_m = all_preds[mask_m, m]
        targets_m = all_targets[mask_m, m]

        # Create DataFrame for plotting
        fig, (ax,), stat = plot_scatter_comparison(
            targets_m, preds_m, title=dataset.attributes[m], return_stat=True
        )
        if savefig and isinstance(savefig, str):
            savefig_ = f"{savefig}/{suffix if suffix else ''}"
            _ = Path(savefig_).mkdir(parents=True, exist_ok=True)
            fig.savefig(f"{savefig_}/{dataset.attributes[m]}.png", bbox_inches="tight")
        stat["property"] = dataset.attributes[m]
        all_stat.append(stat)

        if no_show:
            plt.cla()
            plt.clf()
            plt.close()

    return pd.DataFrame(all_stat)


def scaling_laws_test(
    descriptor,
    property_data,
    target_property,
    mode="vary_target",
    device="cpu",
    num_runs=5,
):
    """
    Test scaling laws for multi-task learning.

    Args:
        descriptor: Input feature DataFrame
        property_data: Property values DataFrame
        target_property: The property to analyze scaling laws for
        mode: Either 'vary_target' or 'vary_others'
            - 'vary_target': Fix other properties at 100% and vary target property fraction
            - 'vary_others': Fix target property at 100% and vary other properties fraction
        device: Device to run the model on
        num_runs: Number of runs for each configuration
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

            # Set up known_attributes based on mode
            if mode == "vary_target":
                # Fix other properties at 100% and vary target property
                known_attributes = {attr: 1.0 for attr in temp_dataset.attributes}
                known_attributes[target_property] = fraction
            else:  # vary_others
                # Fix target property at 100% and vary other properties
                known_attributes = {attr: fraction for attr in temp_dataset.attributes}
                known_attributes[target_property] = 1.0

            # Create datasets with specified known_attributes
            train_dataset = CompoundDataset(train_data, train_prop, **known_attributes)
            test_dataset = CompoundDataset(test_data, test_prop)

            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # Initialize model
            model = CompoundPropertyPredictor(
                shared_block_dims=[train_data.shape[1], 512, 256, 128],
                task_block_dims=[128, 64, 32, 1],
                n_tasks=len(train_dataset.attributes),
            ).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.0005)

            # Train and evaluate
            try:
                avg_test_losses, all_preds, all_targets, all_masks = train_and_evaluate(
                    model, train_loader, test_loader, optimizer, device, num_epochs=100
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
