from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from chronos.chronos2 import Chronos2Pipeline


def _repeat_static_rows(static_features: torch.Tensor, history_length: int) -> torch.Tensor:
    """Repeat per-node static features across the time axis and flatten to rows."""
    if static_features.ndim != 2:
        raise ValueError(f"Expected static_features with shape [M, F], got {tuple(static_features.shape)}")
    repeated = static_features.unsqueeze(-1).expand(-1, -1, history_length)
    return repeated.reshape(-1, history_length)


def pack_chronos_input(
    series_tensors: Iterable[torch.Tensor],
    static_features: torch.Tensor,
) -> torch.Tensor:
    """Pack direct Chronos inputs as a multivariate tensor [V, T]."""
    series_list = [tensor for tensor in series_tensors]
    if len(series_list) == 0:
        raise ValueError("series_tensors is empty")

    history_length = series_list[0].shape[-1]
    if any(tensor.ndim != 2 for tensor in series_list):
        raise ValueError("Each series tensor must have shape [M, T]")
    if any(tensor.shape[-1] != history_length for tensor in series_list):
        raise ValueError("All series tensors must share the same history length")

    stacked = torch.cat(series_list, dim=0)
    static_rows = _repeat_static_rows(static_features, history_length)
    return torch.cat([stacked, static_rows], dim=0)


def add_count_smoothing_noise(
    packed_input: torch.Tensor,
    count_row_count: int,
    low: float = 0.0,
    high: float = 1.0,
) -> torch.Tensor:
    """Add uniform noise to the leading count rows while leaving static context untouched."""
    if packed_input.ndim != 2:
        raise ValueError(f"Expected packed_input with shape [R, T], got {tuple(packed_input.shape)}")
    if count_row_count < 0 or count_row_count > packed_input.shape[0]:
        raise ValueError("count_row_count is out of range")

    noisy = packed_input.clone()
    if count_row_count > 0:
        noise = torch.empty_like(noisy[:count_row_count]).uniform_(low, high)
        noisy[:count_row_count] = torch.floor(noisy[:count_row_count] + noise).clamp_min(0.0)
    return noisy


def unpack_prediction_blocks(
    quantile_tensor: torch.Tensor,
    context_names: list[str],
    prediction_names: list[str],
    num_nodes: int,
) -> torch.Tensor:
    """Extract prediction-variable blocks from a Chronos tensor output.

    Returns a tensor with shape [T, M, V, Q].
    """
    if quantile_tensor.ndim != 3:
        raise ValueError(f"Expected Chronos output with shape [V, T, Q], got {tuple(quantile_tensor.shape)}")

    blocks = []
    for name in prediction_names:
        if name not in context_names:
            raise ValueError(f"Prediction column '{name}' must be present in context columns for direct Chronos input")
        ctx_idx = context_names.index(name)
        start = ctx_idx * num_nodes
        end = start + num_nodes
        blocks.append(quantile_tensor[start:end])

    return torch.stack(blocks, dim=0).permute(2, 1, 0, 3).contiguous()


def sample_from_quantiles(
    quantile_values: torch.Tensor,
    quantile_levels: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """Sample trajectories from monotone quantile forecasts using inverse-CDF interpolation."""
    if quantile_values.ndim != 4:
        raise ValueError(f"Expected quantile_values with shape [T, M, V, Q], got {tuple(quantile_values.shape)}")

    if quantile_levels.ndim != 1:
        raise ValueError("quantile_levels must be 1-D")

    if quantile_levels.numel() == 1:
        return quantile_values[..., 0].unsqueeze(0).expand(num_samples, -1, -1, -1).contiguous()

    device = quantile_values.device
    q_levels = quantile_levels.to(device=device, dtype=quantile_values.dtype)
    q_count = q_levels.numel()
    u = torch.rand((num_samples,) + quantile_values.shape[:-1], device=device, dtype=quantile_values.dtype)
    idx = torch.searchsorted(q_levels, u, right=True).clamp(1, q_count - 1)
    left = idx - 1
    right = idx

    q_left = q_levels[left]
    q_right = q_levels[right]
    expanded = quantile_values.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)
    v_left = torch.gather(expanded, -1, left.unsqueeze(-1)).squeeze(-1)
    v_right = torch.gather(expanded, -1, right.unsqueeze(-1)).squeeze(-1)

    denom = torch.clamp(q_right - q_left, min=torch.finfo(quantile_values.dtype).eps)
    weight = (u - q_left) / denom
    return v_left + weight * (v_right - v_left)


@dataclass
class ChronosTemporalAdapter:
    model_id: str = "amazon/chronos-2"
    device: str = "cuda"

    def __post_init__(self) -> None:
        self.pipeline = Chronos2Pipeline.from_pretrained(self.model_id)
        self.is_chronos_model = True
        self.is_generative_model = False

    def eval(self) -> "ChronosTemporalAdapter":
        self.pipeline.model.eval()
        return self

    def to(self, device: str) -> "ChronosTemporalAdapter":
        self.device = device
        self.pipeline.model.to(device)
        return self

    def fit(self, *args, **kwargs):
        self.pipeline = self.pipeline.fit(*args, **kwargs)
        return self

    def fine_tune(self, *args, **kwargs):
        count_row_count = kwargs.pop("count_row_count", None)
        count_noise_low = kwargs.pop("count_noise_low", 0.0)
        count_noise_high = kwargs.pop("count_noise_high", 1.0)

        if count_row_count is not None and "inputs" in kwargs:
            inputs = kwargs["inputs"]
            if isinstance(inputs, list):
                kwargs["inputs"] = [
                    add_count_smoothing_noise(tensor, count_row_count, count_noise_low, count_noise_high)
                    if isinstance(tensor, torch.Tensor)
                    else tensor
                    for tensor in inputs
                ]

        self.pipeline = self.pipeline.fit(*args, **kwargs)
        return self

    def predict_quantiles(self, *args, **kwargs):
        return self.pipeline.predict_quantiles(*args, **kwargs)
