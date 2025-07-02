#!/usr/bin/env python3
"""
Comprehensive diagnostic tool for tracking sample counts throughout the prediction pipeline.
This tool helps identify where samples are being lost in the predict_idx="all" scenario.

Usage:
    python diagnostic_prediction_flow.py
"""

import ast
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from foundation_model.data.datamodule import CompoundDataModule
from foundation_model.data.dataset import CompoundDataset
from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel


class PredictionDiagnosticTool:
    """
    Comprehensive diagnostic tool for tracking sample flow in prediction pipeline.
    """

    def __init__(self):
        self.sample_counts = {}
        self.processing_log = []
        self.data_quality_stats = {}
        self.original_methods = {}  # Store original methods for restoration

    def log_step(self, step_name: str, count: int, details: str = ""):
        """Log a processing step with sample count."""
        self.sample_counts[step_name] = count
        log_entry = f"[{step_name}] Samples: {count}"
        if details:
            log_entry += f" | {details}"
        self.processing_log.append(log_entry)
        logger.info(log_entry)

    def analyze_raw_data_quality(self, data_path: str) -> Dict[str, Any]:
        """Analyze the quality of raw data files."""
        logger.info("=== Analyzing Raw Data Quality ===")

        stats = {}

        try:
            # Load the attributes data
            if data_path.endswith(".parquet"):
                df = pd.read_parquet(data_path)
            elif data_path.endswith(".pkl"):
                df = pd.read_pickle(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")

            stats["total_samples"] = len(df)
            self.log_step("raw_data_loaded", len(df), f"from {data_path}")

            # Analyze DOS-related columns
            dos_columns = [col for col in df.columns if "DOS" in col.upper()]
            logger.info(f"Found DOS-related columns: {dos_columns}")

            for col in dos_columns:
                if col in df.columns:
                    # Count non-null values
                    non_null_count = df[col].notna().sum()
                    stats[f"{col}_non_null"] = non_null_count

                    # For DOS density, analyze the data structure
                    if "density" in col.lower():
                        # Sample some values to understand the structure
                        sample_values = df[col].dropna().head(10)
                        logger.info(f"Sample {col} values:")
                        for i, val in enumerate(sample_values):
                            if isinstance(val, (list, np.ndarray)):
                                logger.info(f"  Sample {i}: length={len(val)}, type={type(val)}")
                            else:
                                logger.info(f"  Sample {i}: {val}, type={type(val)}")

                    # For DOS energy, analyze the structure
                    if "energy" in col.lower():
                        sample_values = df[col].dropna().head(10)
                        logger.info(f"Sample {col} values:")
                        for i, val in enumerate(sample_values):
                            if isinstance(val, (list, np.ndarray)):
                                logger.info(f"  Sample {i}: length={len(val)}, type={type(val)}")
                            else:
                                logger.info(f"  Sample {i}: {val}, type={type(val)}")

            self.data_quality_stats = stats

        except Exception as e:
            logger.error(f"Error analyzing raw data: {e}")
            stats["error"] = str(e)

        return stats

    def hook_datamodule_methods(self, datamodule: CompoundDataModule):
        """Hook DataModule methods to track sample counts."""
        logger.info("=== Hooking DataModule Methods ===")

        # Store original methods
        self.original_methods["datamodule_setup"] = datamodule.setup
        self.original_methods["datamodule_predict_dataloader"] = datamodule.predict_dataloader

        def tracked_setup(stage: Optional[str] = None):
            """Tracked version of setup method."""
            logger.info(f"DataModule.setup called with stage: {stage}")

            # Call original setup
            result = self.original_methods["datamodule_setup"](stage)

            # Track sample counts after setup
            if stage == "predict" or stage is None:
                if hasattr(datamodule, "predict_idx") and datamodule.predict_idx is not None:
                    self.log_step(
                        "datamodule_predict_idx_resolved",
                        len(datamodule.predict_idx),
                        f"predict_idx type: {type(datamodule.predict_idx)}",
                    )

                if hasattr(datamodule, "predict_dataset") and datamodule.predict_dataset is not None:
                    self.log_step("datamodule_predict_dataset_created", len(datamodule.predict_dataset))

                # Log the original data sizes
                if hasattr(datamodule, "formula_df") and datamodule.formula_df is not None:
                    self.log_step("datamodule_formula_df", len(datamodule.formula_df))

                if hasattr(datamodule, "attributes_df") and datamodule.attributes_df is not None:
                    self.log_step("datamodule_attributes_df", len(datamodule.attributes_df))

            return result

        def tracked_predict_dataloader():
            """Tracked version of predict_dataloader method."""
            logger.info("DataModule.predict_dataloader called")

            # Call original method
            dataloader = self.original_methods["datamodule_predict_dataloader"]()

            if dataloader is not None:
                dataset_size = len(dataloader.dataset)
                batch_size = dataloader.batch_size
                num_batches = len(dataloader)
                self.log_step(
                    "datamodule_predict_dataloader_created",
                    dataset_size,
                    f"batch_size={batch_size}, num_batches={num_batches}",
                )
            else:
                self.log_step("datamodule_predict_dataloader_created", 0, "dataloader is None")

            return dataloader

        # Apply hooks
        datamodule.setup = tracked_setup
        datamodule.predict_dataloader = tracked_predict_dataloader

    def hook_dataset_methods(self, dataset):
        """Hook Dataset methods to track sample access patterns."""
        logger.info("=== Hooking Dataset Methods ===")

        # Verify it's a CompoundDataset
        if not isinstance(dataset, CompoundDataset):
            logger.warning(f"Expected CompoundDataset, got {type(dataset)}. Skipping dataset hooks.")
            return

        # Store original methods
        self.original_methods["dataset_getitem"] = dataset.__getitem__
        self.original_methods["dataset_len"] = dataset.__len__

        # Track getitem calls
        self.getitem_call_count = 0

        def tracked_getitem(idx):
            """Tracked version of __getitem__ method."""
            self.getitem_call_count += 1

            # Call original method
            result = self.original_methods["dataset_getitem"](idx)

            # Log every 1000 calls to avoid spam
            if self.getitem_call_count % 1000 == 0:
                logger.info(f"Dataset.__getitem__ called {self.getitem_call_count} times")

            return result

        def tracked_len():
            """Tracked version of __len__ method."""
            length = self.original_methods["dataset_len"]()
            logger.info(f"Dataset.__len__ returned: {length}")
            return length

        # Apply hooks
        dataset.__getitem__ = tracked_getitem
        dataset.__len__ = tracked_len

    def hook_model_methods(self, model: FlexibleMultiTaskModel):
        """Hook Model methods to track prediction processing."""
        logger.info("=== Hooking Model Methods ===")

        # Store original methods
        self.original_methods["model_predict_step"] = model.predict_step
        self.original_methods["model_expand_for_extend_regression"] = model._expand_for_extend_regression

        # Track predict_step calls
        self.predict_step_call_count = 0
        self.predict_step_input_samples = 0
        self.predict_step_output_samples = 0

        def tracked_predict_step(batch, batch_idx, dataloader_idx=0, tasks_to_predict=None):
            """Tracked version of predict_step method."""
            self.predict_step_call_count += 1

            # Count input samples
            if isinstance(batch[0], torch.Tensor):
                batch_input_size = batch[0].shape[0]
            elif isinstance(batch[0], (list, tuple)) and len(batch[0]) > 0:
                batch_input_size = batch[0][0].shape[0] if isinstance(batch[0][0], torch.Tensor) else len(batch[0])
            else:
                batch_input_size = 0

            self.predict_step_input_samples += batch_input_size

            logger.info(f"predict_step batch {batch_idx}: input_size={batch_input_size}")

            # Call original method
            result = self.original_methods["model_predict_step"](batch, batch_idx, dataloader_idx, tasks_to_predict)

            # Count output samples
            if result:
                # For ExtendRegression tasks, count list elements
                for task_name, predictions in result.items():
                    if isinstance(predictions, list):
                        output_count = len(predictions)
                        logger.info(f"  {task_name}: {output_count} predictions (list format)")
                        if task_name.endswith("_value"):  # Likely the main prediction
                            self.predict_step_output_samples += output_count
                    elif isinstance(predictions, (torch.Tensor, np.ndarray)):
                        output_count = len(predictions)
                        logger.info(f"  {task_name}: {output_count} predictions (tensor format)")
                        if task_name.endswith("_value"):  # Likely the main prediction
                            self.predict_step_output_samples += output_count

            return result

        # Track _expand_for_extend_regression calls
        self.expand_call_count = 0
        self.expand_input_samples = 0
        self.expand_output_samples = 0

        def tracked_expand_for_extend_regression(h_task, t_sequence):
            """Tracked version of _expand_for_extend_regression method."""
            self.expand_call_count += 1

            # Count input samples
            input_batch_size = h_task.shape[0] if isinstance(h_task, torch.Tensor) else 0
            self.expand_input_samples += input_batch_size

            # Analyze t_sequence structure
            if isinstance(t_sequence, list):
                t_sequence_info = f"List[Tensor] with {len(t_sequence)} elements"
                # Count non-empty sequences
                non_empty_count = sum(1 for t in t_sequence if t.numel() > 0 and (t != 0.0).any())
                placeholder_count = len(t_sequence) - non_empty_count
            elif isinstance(t_sequence, torch.Tensor):
                t_sequence_info = f"Tensor with shape {t_sequence.shape}"
                # Count non-zero elements per sample
                non_empty_count = 0
                placeholder_count = 0
                for i in range(t_sequence.shape[0]):
                    if (t_sequence[i] != 0.0).any():
                        non_empty_count += 1
                    else:
                        placeholder_count += 1
            else:
                t_sequence_info = f"Unknown type: {type(t_sequence)}"
                non_empty_count = 0
                placeholder_count = 0

            logger.info(f"_expand_for_extend_regression call {self.expand_call_count}:")
            logger.info(f"  Input batch size: {input_batch_size}")
            logger.info(f"  t_sequence: {t_sequence_info}")
            logger.info(f"  Non-empty sequences: {non_empty_count}")
            logger.info(f"  Placeholder sequences: {placeholder_count}")

            # Call original method
            expanded_h_task, expanded_t = self.original_methods["model_expand_for_extend_regression"](
                h_task, t_sequence
            )

            # Count output samples
            output_size = expanded_h_task.shape[0] if isinstance(expanded_h_task, torch.Tensor) else 0
            self.expand_output_samples += output_size

            logger.info(f"  Output size: {output_size}")

            return expanded_h_task, expanded_t

        # Apply hooks
        model.predict_step = tracked_predict_step
        model._expand_for_extend_regression = tracked_expand_for_extend_regression

    def restore_original_methods(self, datamodule, dataset, model):
        """Restore original methods after diagnosis."""
        logger.info("=== Restoring Original Methods ===")

        if "datamodule_setup" in self.original_methods:
            datamodule.setup = self.original_methods["datamodule_setup"]
        if "datamodule_predict_dataloader" in self.original_methods:
            datamodule.predict_dataloader = self.original_methods["datamodule_predict_dataloader"]

        if "dataset_getitem" in self.original_methods:
            dataset.__getitem__ = self.original_methods["dataset_getitem"]
        if "dataset_len" in self.original_methods:
            dataset.__len__ = self.original_methods["dataset_len"]

        if "model_predict_step" in self.original_methods:
            model.predict_step = self.original_methods["model_predict_step"]
        if "model_expand_for_extend_regression" in self.original_methods:
            model._expand_for_extend_regression = self.original_methods["model_expand_for_extend_regression"]

    def analyze_prediction_output(self, output_path: str):
        """Analyze the final prediction output file."""
        logger.info("=== Analyzing Prediction Output ===")

        try:
            if os.path.exists(output_path):
                df = pd.read_csv(output_path, index_col=0)
                self.log_step("final_output_file", len(df), f"saved to {output_path}")

                # Analyze DOS predictions
                dos_columns = [col for col in df.columns if "dos" in col.lower()]
                for col in dos_columns:
                    if col in df.columns:
                        # Parse DOS values and analyze lengths
                        dos_lengths = []
                        valid_predictions = 0
                        placeholder_predictions = 0

                        for i, dos_str in enumerate(df[col].head(1000)):  # Sample first 1000
                            try:
                                dos_list = ast.literal_eval(dos_str)
                                dos_lengths.append(len(dos_list))
                                if len(dos_list) > 1:
                                    valid_predictions += 1
                                else:
                                    placeholder_predictions += 1
                            except:
                                dos_lengths.append(0)

                        logger.info(f"DOS column '{col}' analysis (first 1000 samples):")
                        logger.info(f"  Valid predictions (length > 1): {valid_predictions}")
                        logger.info(f"  Placeholder predictions (length = 1): {placeholder_predictions}")
                        logger.info(f"  Unique lengths: {set(dos_lengths)}")

            else:
                self.log_step("final_output_file", 0, f"file not found: {output_path}")

        except Exception as e:
            logger.error(f"Error analyzing prediction output: {e}")

    def generate_diagnostic_report(self) -> str:
        """Generate a comprehensive diagnostic report."""
        report = []
        report.append("=" * 80)
        report.append("PREDICTION PIPELINE DIAGNOSTIC REPORT")
        report.append("=" * 80)

        report.append("\n1. SAMPLE COUNT TRACKING:")
        for step, count in self.sample_counts.items():
            report.append(f"   {step}: {count}")

        report.append("\n2. DATA QUALITY STATISTICS:")
        for key, value in self.data_quality_stats.items():
            report.append(f"   {key}: {value}")

        report.append("\n3. PROCESSING STATISTICS:")
        if hasattr(self, "getitem_call_count"):
            report.append(f"   Dataset.__getitem__ calls: {self.getitem_call_count}")
        if hasattr(self, "predict_step_call_count"):
            report.append(f"   Model.predict_step calls: {self.predict_step_call_count}")
            report.append(f"   predict_step input samples: {self.predict_step_input_samples}")
            report.append(f"   predict_step output samples: {self.predict_step_output_samples}")
        if hasattr(self, "expand_call_count"):
            report.append(f"   _expand_for_extend_regression calls: {self.expand_call_count}")
            report.append(f"   expand input samples: {self.expand_input_samples}")
            report.append(f"   expand output samples: {self.expand_output_samples}")

        report.append("\n4. PROCESSING LOG:")
        for log_entry in self.processing_log:
            report.append(f"   {log_entry}")

        report.append("\n5. POTENTIAL ISSUES:")
        issues = []

        # Check for sample count mismatches
        if "raw_data_loaded" in self.sample_counts and "final_output_file" in self.sample_counts:
            input_count = self.sample_counts["raw_data_loaded"]
            output_count = self.sample_counts["final_output_file"]
            if input_count != output_count:
                issues.append(f"Sample count mismatch: input={input_count}, output={output_count}")

        # Check for expansion issues
        if hasattr(self, "expand_input_samples") and hasattr(self, "expand_output_samples"):
            if self.expand_output_samples == 0 and self.expand_input_samples > 0:
                issues.append("_expand_for_extend_regression produced no output despite having input")

        if issues:
            for issue in issues:
                report.append(f"   ⚠️  {issue}")
        else:
            report.append("   ✅ No obvious issues detected")

        report.append("=" * 80)

        return "\n".join(report)


def main():
    """Main diagnostic function."""
    logger.info("Starting Prediction Pipeline Diagnostic")

    # Initialize diagnostic tool
    diagnostic = PredictionDiagnosticTool()

    # Configuration paths
    config_path = "samples/configs/test_2steps_t_depends_fc/predict_config.yaml"
    data_path = "/data/foundation_model/data/qc_ac_te_mp_dos_reformat_20250615.pd.parquet"

    try:
        # Step 1: Analyze raw data quality
        diagnostic.analyze_raw_data_quality(data_path)

        # Step 2: Create and hook DataModule
        logger.info("=== Creating DataModule ===")

        # Import task configs (simplified version for testing)
        from foundation_model.models.model_config import ExtendRegressionTaskConfig, RegressionTaskConfig, TaskType

        task_configs = [
            RegressionTaskConfig(
                name="density",
                type=TaskType.REGRESSION,
                data_column="Density (normalized)",
                dims=[128, 64, 32, 1],
                enabled=True,
            ),
            RegressionTaskConfig(
                name="formation_energy",
                type=TaskType.REGRESSION,
                data_column="Formation energy per atom (normalized)",
                dims=[128, 64, 32, 1],
                enabled=True,
            ),
            RegressionTaskConfig(
                name="volume",
                type=TaskType.REGRESSION,
                data_column="Volume (normalized)",
                dims=[128, 64, 32, 1],
                enabled=True,
            ),
            ExtendRegressionTaskConfig(
                name="dos",
                type=TaskType.ExtendRegression,
                data_column="DOS density (normalized)",
                t_column="DOS energy",
                x_dim=[128, 32, 16],
                t_dim=[32, 16],
                t_encoding_method="fc",
                enabled=True,
            ),
        ]

        datamodule = CompoundDataModule(
            formula_desc_source="/data/foundation_model/data/qc_ac_te_mp_dos_composition_desc_trans_20250615.pd.parquet",
            attributes_source=data_path,
            task_configs=task_configs,
            predict_idx="all",
            batch_size=256,
            num_workers=0,
        )

        # Hook DataModule methods
        diagnostic.hook_datamodule_methods(datamodule)

        # Step 3: Setup DataModule
        logger.info("=== Setting up DataModule ===")
        datamodule.setup(stage="predict")

        # Step 4: Create DataLoader and hook Dataset
        logger.info("=== Creating DataLoader ===")
        predict_dataloader = datamodule.predict_dataloader()

        if predict_dataloader is not None:
            dataset = predict_dataloader.dataset
            diagnostic.hook_dataset_methods(dataset)

            # Step 5: Create a minimal model for testing (we don't need a trained model for this diagnostic)
            logger.info("=== Creating Model ===")
            model = FlexibleMultiTaskModel(
                shared_block_dims=[290, 128],
                task_configs=task_configs,
                enable_learnable_loss_balancer=False,  # Simplified for testing
            )

            # Hook model methods
            diagnostic.hook_model_methods(model)

            # Step 6: Run a few prediction steps to track the flow
            logger.info("=== Running Prediction Steps ===")
            model.eval()

            with torch.no_grad():
                for batch_idx, batch in enumerate(predict_dataloader):
                    if batch_idx >= 3:  # Only test first 3 batches
                        break

                    try:
                        predictions = model.predict_step(batch, batch_idx)
                        logger.info(f"Batch {batch_idx} completed successfully")
                    except Exception as e:
                        logger.error(f"Error in batch {batch_idx}: {e}")
                        break

            # Step 7: Restore original methods
            diagnostic.restore_original_methods(datamodule, dataset, model)

        else:
            logger.error("Failed to create predict_dataloader")

        # Step 8: Analyze existing prediction output if available
        output_path = "samples/example_logs/basic_run/basic_experiment_20250702_003437/predict/predictions.csv"
        diagnostic.analyze_prediction_output(output_path)

    except Exception as e:
        logger.error(f"Diagnostic failed: {e}")
        import traceback

        traceback.print_exc()

    # Generate and save diagnostic report
    report = diagnostic.generate_diagnostic_report()

    # Save report to file
    report_path = "diagnostic_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # Print report
    print(report)

    logger.info(f"Diagnostic complete. Report saved to {report_path}")


if __name__ == "__main__":
    main()
