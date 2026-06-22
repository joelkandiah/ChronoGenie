from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_lr_finder import LRFinder
import yaml

from dataset import ProcessedData
from distribution_construction import MixedLoss
from train_test_helpers import get_batched_graph, apply_poisson_noise_to_contexts


class _LRFinderDataset(Dataset):
    def __init__(self, base_dataset: ProcessedData, num_context: int):
        self.base_dataset = base_dataset
        self.num_context = num_context

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        sample = self.base_dataset[idx]
        static, current_t, current_dow, *contexts_and_targets = sample
        contexts = contexts_and_targets[: self.num_context]
        targets = contexts_and_targets[self.num_context :]
        target_tensor = torch.stack(targets, dim=-1)
        return (static, current_t, current_dow, *contexts, target_tensor), torch.tensor(0.0)


class _LRFinderLossModel(nn.Module):
    def __init__(
        self,
        cfg: Any,
        temporal_model: nn.ModuleList,
        local_profile_encoder: Optional[nn.Module],
        interaction_encoder: nn.Module,
        temp_graph_data: Any,
        spatial_graph_data: Any,
        dataset_directory: Any,
        poisson_noise_context: bool = False,
        augmentation_thinning_factor: int = 5,
    ):
        super().__init__()
        self.cfg = cfg
        self.temporal_model = temporal_model
        self.local_profile_encoder = local_profile_encoder
        self.interaction_encoder = interaction_encoder
        self.temp_graph_data = temp_graph_data
        self.spatial_graph_data = spatial_graph_data
        self.dataset_directory = dataset_directory
        self.poisson_noise_context = poisson_noise_context
        self.augmentation_thinning_factor = augmentation_thinning_factor
        self._forward_batch_idx = 0

    def forward(self, batch):
        static, current_t_batch, current_dow_batch, *contexts_and_target = batch
        _ = current_t_batch

        target = contexts_and_target[-1]
        contexts = contexts_and_target[:-1]

        apply_augmented_context = (
            self.poisson_noise_context
            and (self.augmentation_thinning_factor <= 1 or self._forward_batch_idx % self.augmentation_thinning_factor == 0)
        )
        if apply_augmented_context:
            contexts = apply_poisson_noise_to_contexts(
                contexts=contexts,
                columns_for_context=self.dataset_directory.columns_for_context,
                columns_with_data=self.dataset_directory.columns_with_data,
                temporal_scalers=self.dataset_directory.temporal_scalers,
                prediction_distribution_types=self.dataset_directory.prediction_distribution_types,
                columns_for_prediction=self.dataset_directory.columns_for_prediction,
                device=str(contexts[0].device),
                model_type=self.cfg.model_type,
            )

        self._forward_batch_idx += 1

        context_combined = torch.cat(contexts, dim=2)

        if self.local_profile_encoder is not None:
            if self.cfg.spatial_encoder_type == "gnn":
                spatial_encoder_batch = get_batched_graph(static, self.spatial_graph_data, self.cfg.device)
                h_spatial = self.local_profile_encoder(spatial_encoder_batch)
            else:
                h_spatial = self.local_profile_encoder(static)

            temporal_encoder_batch = get_batched_graph(context_combined, self.temp_graph_data, self.cfg.device)
            h_temporal = self.interaction_encoder(temporal_encoder_batch)
        elif self.cfg.spatial_ablation:
            combined_batch = get_batched_graph(context_combined, self.temp_graph_data, self.cfg.device)
            h_combined = self.interaction_encoder(combined_batch)
        else:
            combined_node_features = torch.cat([static, context_combined], dim=2)
            combined_batch = get_batched_graph(combined_node_features, self.temp_graph_data, self.cfg.device)
            h_combined = self.interaction_encoder(combined_batch)

        dow_oh = F.one_hot(current_dow_batch.long(), num_classes=7).float()
        day_of_week = dow_oh.unsqueeze(1).repeat(1, self.dataset_directory.num_geographies, 1)

        feature_list = []
        if self.cfg.include_latest_context:
            latest_vals = torch.cat([c[:, :, -1:].contiguous() for c in contexts], dim=2) if len(contexts) > 0 else None
            if latest_vals is not None:
                feature_list.append(latest_vals)

        if self.local_profile_encoder is not None:
            feature_list.extend([h_spatial, h_temporal, day_of_week])
        else:
            feature_list.extend([h_combined, day_of_week])

        latent_input = torch.cat(feature_list, dim=-1)

        temporal_heads = self.temporal_model._orig_mod if hasattr(self.temporal_model, "_orig_mod") else self.temporal_model
        batch_loss = torch.tensor(0.0, device=latent_input.device)

        for step_idx in range(self.cfg.multistep_k):
            if hasattr(temporal_heads, "__getitem__"):
                head_model = temporal_heads[step_idx]
            else:
                head_model = temporal_heads

            if self.cfg.multistep_k == 1:
                target_k = target
            else:
                target_k = target[:, :, step_idx, :]

            if self.cfg.model_type in ("flow_matching", "diffusion", "normalising_flow", "normalising_flow_nsf", "normalising_flow_maf", "normalising_flow_cnf"):
                step_loss = head_model.training_step(latent_input, target_k)
            elif self.cfg.model_type == "variational_flow_matching":
                step_loss, _ = head_model.training_step(latent_input, target_k, is_epoch_diag=False)
            else:
                mu_pred, var_pred = head_model(latent_input)
                step_loss, _ = MixedLoss(mu_pred, var_pred, target_k, self.dataset_directory.prediction_distribution_types)

            weight = self.cfg.multistep_weights[step_idx] if step_idx < len(self.cfg.multistep_weights) else 0.0
            batch_loss = batch_loss + step_loss * weight

        return batch_loss


class _RecordingIdentityCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_losses: list[float] = []

    def forward(self, output, target):
        _ = target
        self.raw_losses.append(float(output.detach().item()))
        return output


def _find_steepest_point(history: dict[str, list[float]]) -> dict[str, float]:
    """Return the point with the steepest loss decrease over log10 learning rate."""
    lrs = history.get("lr", [])
    losses = history.get("loss", [])

    if len(lrs) < 2 or len(losses) < 2:
        raise ValueError("LR finder history is too short to estimate a steepest point")

    lr_values = torch.tensor(lrs, dtype=torch.float64)
    loss_values = torch.tensor(losses, dtype=torch.float64)

    valid_mask = torch.isfinite(lr_values) & torch.isfinite(loss_values) & (lr_values > 0)
    lr_values = lr_values[valid_mask]
    loss_values = loss_values[valid_mask]

    if lr_values.numel() < 2:
        raise ValueError("LR finder history does not contain enough valid positive learning rates")

    log_lr = torch.log10(lr_values)
    slopes = (loss_values[1:] - loss_values[:-1]) / (log_lr[1:] - log_lr[:-1])
    steepest_idx = torch.argmin(slopes).item()

    return {
        "steepest_lr": float(lr_values[steepest_idx + 1].item()),
        "steepest_loss": float(loss_values[steepest_idx + 1].item()),
        "steepest_slope": float(slopes[steepest_idx].item()),
        "steepest_prev_lr": float(lr_values[steepest_idx].item()),
        "steepest_prev_loss": float(loss_values[steepest_idx].item()),
    }


def run_lr_finder(
    cfg: Any,
    temporal_model: nn.ModuleList,
    local_profile_encoder: Optional[nn.Module],
    interaction_encoder: nn.Module,
    graph_data,
    dataset_directory,
    poisson_noise_context: bool = False,
    augmentation_thinning_factor: int = 5,
) -> None:
    """Run a PyTorch LR finder sweep for any GENIE model family."""

    lr_finder_cfg = cfg.lr_finder
    lr_finder_dir = os.path.join(cfg.output_dir, lr_finder_cfg.output_subdir)
    os.makedirs(lr_finder_dir, exist_ok=True)

    if lr_finder_cfg.num_iter <= 1 or lr_finder_cfg.start_lr <= 0 or lr_finder_cfg.end_lr <= 0:
        raise ValueError("lr_finder requires positive start_lr/end_lr and num_iter > 1")

    base_dataset = ProcessedData(
        data_directory=dataset_directory,
        past_context_size=cfg.context_size,
        type="training",
        multistep_k=cfg.multistep_k,
    )
    lr_dataset = _LRFinderDataset(base_dataset, len(dataset_directory.columns_for_context))
    train_loader = DataLoader(
        lr_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    temporal_state = copy.deepcopy(temporal_model.state_dict())
    interaction_state = copy.deepcopy(interaction_encoder.state_dict())
    local_state = copy.deepcopy(local_profile_encoder.state_dict()) if local_profile_encoder is not None else None

    model = _LRFinderLossModel(
        cfg=cfg,
        temporal_model=temporal_model,
        local_profile_encoder=local_profile_encoder,
        interaction_encoder=interaction_encoder,
        temp_graph_data=graph_data[0],
        spatial_graph_data=graph_data[1],
        dataset_directory=dataset_directory,
        poisson_noise_context=poisson_noise_context,
        augmentation_thinning_factor=augmentation_thinning_factor,
    ).to(cfg.device)

    criterion = _RecordingIdentityCriterion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_finder_cfg.start_lr, weight_decay=cfg.adamw_weight_decay)

    finder = LRFinder(model, optimizer, criterion, device=cfg.device)

    try:
        finder.range_test(
            train_loader,
            end_lr=lr_finder_cfg.end_lr,
            num_iter=lr_finder_cfg.num_iter,
            step_mode="exp",
            smooth_f=1-lr_finder_cfg.beta,
            diverge_th=lr_finder_cfg.diverge_th,
        )

        finder.plot(skip_start=lr_finder_cfg.skip_start, skip_end=lr_finder_cfg.skip_end)
        plot_path = os.path.join(lr_finder_dir, lr_finder_cfg.plot_filename)
        csv_path = os.path.join(lr_finder_dir, lr_finder_cfg.csv_filename)

        import matplotlib.pyplot as plt

        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        history = finder.history
        raw_losses = criterion.raw_losses[: len(history.get("loss", []))]
        history_for_csv = dict(history)
        history_for_csv["raw_loss"] = raw_losses
        pd.DataFrame(history_for_csv).to_csv(csv_path, index=False)

        steepest_point = _find_steepest_point(history)
        summary_path = os.path.join(lr_finder_dir, "lr_finder_summary.yaml")
        with open(summary_path, "w") as summary_file:
            yaml.safe_dump(steepest_point, summary_file, sort_keys=False)

        try:
            import matplotlib.pyplot as plt

            plt.figure()
            lrs = history.get("lr", [])
            smoothed_losses = history.get("loss", [])
            plt.semilogx(lrs, smoothed_losses, label="smoothed loss")
            if raw_losses:
                plt.semilogx(lrs[: len(raw_losses)], raw_losses, ".", linestyle="None", alpha=0.2, label="raw loss")
            plt.scatter(
                [steepest_point["steepest_lr"]],
                [steepest_point["steepest_loss"]],
                color="red",
                zorder=5,
                label="steepest point",
            )
            plt.legend()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception:
            pass
    finally:
        finder.reset()
        temporal_model.load_state_dict(temporal_state)
        interaction_encoder.load_state_dict(interaction_state)
        if local_profile_encoder is not None and local_state is not None:
            local_profile_encoder.load_state_dict(local_state)

    print(f"LR finder results saved to: {lr_finder_dir}")
