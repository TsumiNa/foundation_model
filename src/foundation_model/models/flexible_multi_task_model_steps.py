"""Mixin containing the step methods for FlexibleMultiTaskModel."""

from __future__ import annotations

from typing import List, Optional

import torch

from .task_head.kernel_regression import KernelRegressionHead


class FlexibleMultiTaskStepsMixin:
    """Provides Lightning step implementations shared by FlexibleMultiTaskModel."""

    def training_step(self, batch, batch_idx):
        """Training step implementation for supervised multi-task learning."""
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        lr_schedulers = self.lr_schedulers()
        if not isinstance(lr_schedulers, list):
            lr_schedulers = [lr_schedulers]

        x, y_dict_batch, task_masks_batch, task_sequence_data_batch = batch
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor inputs in training_step, received {type(x)}")

        train_logs: dict[str, torch.Tensor] = {}
        supervised_loss_contribution = torch.zeros((), device=x.device)

        preds = self(x, task_sequence_data_batch)

        raw_supervised_losses = {}
        for name, pred_tensor in preds.items():
            if name not in y_dict_batch or not self.task_configs_map[name].enabled:
                continue

            head = self.task_heads[name]
            target = y_dict_batch[name]
            sample_mask = task_masks_batch.get(name)

            if isinstance(head, KernelRegressionHead):
                if isinstance(target, list):
                    target = torch.cat(target, dim=0)
                if sample_mask is not None and isinstance(sample_mask, list):
                    sample_mask = torch.cat(sample_mask, dim=0)
                elif sample_mask is None:
                    self._log_warning(
                        f"Mask not found for KernelRegression task {name} in training_step. Assuming all valid."
                    )
                    sample_mask = torch.ones_like(target, dtype=torch.bool, device=target.device)
            else:
                if sample_mask is None:
                    self._log_warning(f"Mask not found for task {name} in training_step. Assuming all valid.")
                    sample_mask = torch.ones_like(target, dtype=torch.bool, device=target.device)

            raw_loss_t = head.compute_loss(pred_tensor, target, sample_mask)

            if raw_loss_t is None:
                if self.allow_all_missing_in_batch:
                    self._log_debug(f"Task '{name}' has no valid samples in this batch. Skipping loss calculation.")
                    train_logs[f"train_{name}_all_missing"] = 1.0
                    continue
                raise ValueError(
                    f"Task '{name}' has no valid samples in this batch and allow_all_missing_in_batch is False."
                )

            raw_supervised_losses[name] = raw_loss_t
            train_logs[f"train_{name}_raw_loss"] = raw_loss_t.detach()
            train_logs[f"train_{name}_all_missing"] = 0.0

        for name, raw_loss_t in raw_supervised_losses.items():
            static_weight = self._get_task_static_weight(name)
            if self.enable_learnable_loss_balancer and name in self.task_log_sigmas:
                current_log_sigma_t = self.task_log_sigmas[name]
                precision_factor_t = torch.exp(-2 * current_log_sigma_t)
                final_task_loss_component = (
                    static_weight * 0.5 * precision_factor_t * raw_loss_t
                ) + current_log_sigma_t

                supervised_loss_contribution += final_task_loss_component
                train_logs[f"train_{name}_sigma_t"] = torch.exp(current_log_sigma_t).detach()
                train_logs[f"train_{name}_final_loss_contrib"] = final_task_loss_component.detach()
            else:
                final_task_loss_component = static_weight * raw_loss_t
                supervised_loss_contribution += final_task_loss_component
                train_logs[f"train_{name}_final_loss_contrib"] = final_task_loss_component.detach()
            train_logs[f"train_{name}_static_weight"] = torch.tensor(static_weight, device=x.device)

        train_logs["train_final_supervised_loss"] = supervised_loss_contribution.detach()

        total_loss = supervised_loss_contribution

        self.log_dict(train_logs, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_final_loss", total_loss.detach(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        if total_loss.requires_grad:
            self.manual_backward(total_loss)
            for opt in optimizers:
                opt.step()

            for scheduler in lr_schedulers:
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(total_loss.detach())
                    else:
                        scheduler.step()
        else:
            self._log_warning(
                f"total_loss does not require grad and has no grad_fn at batch_idx {batch_idx}. "
                "Skipping backward pass and optimizer step. "
                "This might indicate all parameters are frozen, loss contributions are zero, "
                "or an issue with the computation graph.",
            )
            for opt in optimizers:
                opt.step()

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step implementation mirroring training_step without gradient updates.

        Parameters
        ----------
        batch : tuple
            A tuple containing (x, y_dict_batch, task_masks_batch, task_sequence_data_batch)
        batch_idx : int
            Index of the current batch

        Returns
        -------
        None
            This method logs metrics using self.log_dict() and does not return a value.
        """
        x, y_dict_batch, task_masks_batch, task_sequence_data_batch = batch
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor inputs in validation_step, received {type(x)}")

        val_logs: dict[str, torch.Tensor] = {}
        final_val_loss = torch.zeros((), device=x.device)
        val_supervised_loss_contribution = torch.zeros_like(final_val_loss)
        val_sum_supervised_raw_loss = torch.zeros_like(final_val_loss)

        preds = self(x, task_sequence_data_batch)

        raw_val_supervised_losses = {}
        for name, pred_tensor in preds.items():
            if name not in y_dict_batch or not self.task_configs_map[name].enabled:
                continue

            head = self.task_heads[name]
            target = y_dict_batch[name]
            sample_mask = task_masks_batch.get(name)

            if isinstance(head, KernelRegressionHead):
                if isinstance(target, list):
                    target = torch.cat(target, dim=0)
                if sample_mask is not None and isinstance(sample_mask, list):
                    sample_mask = torch.cat(sample_mask, dim=0)
                elif sample_mask is None:
                    self._log_warning(
                        f"Mask not found for KernelRegression task {name} in validation_step. Assuming all valid."
                    )
                    sample_mask = torch.ones_like(target, dtype=torch.bool, device=target.device)
            else:
                if sample_mask is None:
                    self._log_warning(f"Mask not found for task {name} in validation_step. Assuming all valid.")
                    sample_mask = torch.ones_like(target, dtype=torch.bool, device=target.device)

            raw_loss_t = head.compute_loss(pred_tensor, target, sample_mask)

            if raw_loss_t is None:
                if self.allow_all_missing_in_batch:
                    self._log_debug(f"Task '{name}' has no valid samples in this batch. Skipping loss calculation.")
                    val_logs[f"val_{name}_all_missing"] = 1.0
                    continue
                raise ValueError(
                    f"Task '{name}' has no valid samples in this batch and allow_all_missing_in_batch is False."
                )

            raw_val_supervised_losses[name] = raw_loss_t
            val_sum_supervised_raw_loss += raw_loss_t.detach()
            val_logs[f"val_{name}_raw_loss"] = raw_loss_t.detach()
            val_logs[f"val_{name}_all_missing"] = 0.0

        val_logs["val_sum_supervised_raw_loss"] = val_sum_supervised_raw_loss

        for name, raw_loss_t in raw_val_supervised_losses.items():
            static_weight = self._get_task_static_weight(name)
            if self.enable_learnable_loss_balancer and name in self.task_log_sigmas:
                current_log_sigma_t = self.task_log_sigmas[name]
                precision_factor_t = torch.exp(-2 * current_log_sigma_t)
                final_task_loss_component = (
                    static_weight * 0.5 * precision_factor_t * raw_loss_t
                ) + current_log_sigma_t

                val_supervised_loss_contribution += final_task_loss_component.detach()
                val_logs[f"val_{name}_sigma_t"] = torch.exp(current_log_sigma_t).detach()
                val_logs[f"val_{name}_final_loss_contrib"] = final_task_loss_component.detach()
            else:
                final_task_loss_component = static_weight * raw_loss_t
                val_supervised_loss_contribution += final_task_loss_component.detach()
                val_logs[f"val_{name}_final_loss_contrib"] = final_task_loss_component.detach()
            val_logs[f"val_{name}_static_weight"] = torch.tensor(static_weight, device=x.device)

        val_logs["val_final_supervised_loss"] = val_supervised_loss_contribution.detach()
        final_val_loss = final_val_loss + val_supervised_loss_contribution

        self.log_dict(val_logs, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_final_loss", final_val_loss.detach(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return None

    def test_step(self, batch, batch_idx):
        """
        Test step implementation mirroring validation_step but logging to the test namespace.

        Parameters
        ----------
        batch : tuple
            A tuple containing (x, y_dict_batch, task_masks_batch, task_sequence_data_batch)
        batch_idx : int
            Index of the current batch

        Returns
        -------
        None
        """
        x, y_dict_batch, task_masks_batch, task_sequence_data_batch = batch
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor inputs in test_step, received {type(x)}")

        test_logs: dict[str, torch.Tensor] = {}
        final_test_loss = torch.zeros((), device=x.device)
        test_supervised_loss_contribution = torch.zeros_like(final_test_loss)
        test_sum_supervised_raw_loss = torch.zeros_like(final_test_loss)

        preds = self(x, task_sequence_data_batch)

        raw_test_supervised_losses = {}
        for name, pred_tensor in preds.items():
            if name not in y_dict_batch or not self.task_configs_map[name].enabled:
                continue

            head = self.task_heads[name]
            target = y_dict_batch[name]
            sample_mask = task_masks_batch.get(name)

            if isinstance(head, KernelRegressionHead):
                if isinstance(target, list):
                    target = torch.cat(target, dim=0)
                if sample_mask is not None and isinstance(sample_mask, list):
                    sample_mask = torch.cat(sample_mask, dim=0)
                elif sample_mask is None:
                    self._log_warning(
                        f"Mask not found for KernelRegression task {name} in test_step. Assuming all valid."
                    )
                    sample_mask = torch.ones_like(target, dtype=torch.bool, device=target.device)
            else:
                if sample_mask is None:
                    self._log_warning(f"Mask not found for task {name} in test_step. Assuming all valid.")
                    sample_mask = torch.ones_like(target, dtype=torch.bool, device=target.device)

            raw_loss_t = head.compute_loss(pred_tensor, target, sample_mask)

            if raw_loss_t is None:
                if self.allow_all_missing_in_batch:
                    self._log_debug(f"Task '{name}' has no valid samples in this batch. Skipping loss calculation.")
                    test_logs[f"test_{name}_all_missing"] = 1.0
                    continue
                raise ValueError(
                    f"Task '{name}' has no valid samples in this batch and allow_all_missing_in_batch is False."
                )

            raw_test_supervised_losses[name] = raw_loss_t
            test_sum_supervised_raw_loss += raw_loss_t.detach()
            test_logs[f"test_{name}_raw_loss"] = raw_loss_t.detach()
            test_logs[f"test_{name}_all_missing"] = 0.0

        test_logs["test_sum_supervised_raw_loss"] = test_sum_supervised_raw_loss

        for name, raw_loss_t in raw_test_supervised_losses.items():
            static_weight = self._get_task_static_weight(name)
            if self.enable_learnable_loss_balancer and name in self.task_log_sigmas:
                current_log_sigma_t = self.task_log_sigmas[name]
                precision_factor_t = torch.exp(-2 * current_log_sigma_t)
                final_task_loss_component = (
                    static_weight * 0.5 * precision_factor_t * raw_loss_t
                ) + current_log_sigma_t

                test_supervised_loss_contribution += final_task_loss_component.detach()
                test_logs[f"test_{name}_sigma_t"] = torch.exp(current_log_sigma_t).detach()
                test_logs[f"test_{name}_final_loss_contrib"] = final_task_loss_component.detach()
            else:
                final_task_loss_component = static_weight * raw_loss_t
                test_supervised_loss_contribution += final_task_loss_component.detach()
                test_logs[f"test_{name}_final_loss_contrib"] = final_task_loss_component.detach()
            test_logs[f"test_{name}_static_weight"] = torch.tensor(static_weight, device=x.device)

        test_logs["test_final_supervised_loss"] = test_supervised_loss_contribution.detach()
        final_test_loss = final_test_loss + test_supervised_loss_contribution

        self.log_dict(test_logs, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "test_final_loss", final_test_loss.detach(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )

        return None

    def predict_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
        tasks_to_predict: Optional[List[str]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Prediction step that forwards inputs through the model and post-processes the outputs.

        Parameters
        ----------
        batch : tuple
            Typically contains (x_formula, _, _, task_sequence_data_batch). Only x_formula and
            task_sequence_data_batch are used.
        batch_idx : int
            Index of the current batch.
        dataloader_idx : int, optional
            Index of the dataloader (if multiple).
        tasks_to_predict : list[str] | None, optional
            A list of task names to predict. If None, predicts all enabled tasks.

        Returns
        -------
        dict[str, torch.Tensor]
            Flat dictionary containing head-specific prediction outputs.
        """
        x_formula = batch[0]
        if not isinstance(x_formula, torch.Tensor):
            raise TypeError(f"Expected batch[0] to be a Tensor (x_formula), but got {type(x_formula)}")

        task_sequence_data_batch = batch[3] if len(batch) > 3 else {}

        kernel_regression_sequence_lengths = {}
        for task_name, sequence_data in task_sequence_data_batch.items():
            if task_name in self.task_heads and isinstance(self.task_heads[task_name], KernelRegressionHead):
                if isinstance(sequence_data, list):
                    kernel_regression_sequence_lengths[task_name] = [len(seq) for seq in sequence_data]
                elif isinstance(sequence_data, torch.Tensor):
                    lengths = []
                    for sample in sequence_data:
                        valid_mask = sample != 0.0
                        lengths.append(int(valid_mask.sum().item()))
                    kernel_regression_sequence_lengths[task_name] = lengths

        raw_preds = self(x_formula, task_sequence_data_batch)

        final_predictions: dict[str, torch.Tensor] = {}

        if tasks_to_predict is None:
            tasks_to_iterate = [(name, tensor) for name, tensor in raw_preds.items() if name in self.task_heads]
        else:
            tasks_to_iterate = []
            for task_name in tasks_to_predict:
                if task_name not in self.task_heads:
                    self._log_warning(
                        f"Task '{task_name}' requested for prediction but not found or not enabled in the model. Skipping."
                    )
                    continue
                if task_name not in raw_preds:
                    self._log_warning(
                        f"Task '{task_name}' requested for prediction, found in model heads, but not present in raw output. Skipping."
                    )
                    continue
                tasks_to_iterate.append((task_name, raw_preds[task_name]))

        for task_name, raw_pred_tensor in tasks_to_iterate:
            head = self.task_heads[task_name]
            processed_pred_dict = head.predict(raw_pred_tensor)  # type: ignore

            if isinstance(head, KernelRegressionHead) and task_name in kernel_regression_sequence_lengths:
                sequence_lengths = kernel_regression_sequence_lengths[task_name]
                processed_pred_dict = self._reshape_kernel_regression_predictions(processed_pred_dict, sequence_lengths)

            final_predictions.update(processed_pred_dict)

        return final_predictions
