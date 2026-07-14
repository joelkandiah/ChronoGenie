from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from chronos.chronos2 import Chronos2Pipeline

# ─── Spatial neighbor configuration ──────────────────────────────────────────
NUM_SPATIAL_NEIGHBORS = 9


def build_neighbor_map(
    static_features_tensor: torch.Tensor,
    raw_static_features_tensor: torch.Tensor | None = None,
    n_neighbors: int = NUM_SPATIAL_NEIGHBORS,
) -> dict[int, list[int]]:
    """Build a deterministic neighbor lookup from static feature and geo spaces.

    The original graph-building path combined two independent 9-NN sets: one in
    the full static feature space and one in geographic coordinate space. This
    helper reproduces that union deterministically and preserves the order in
    which neighbours are discovered.

    Args:
        static_features_tensor: Float tensor of shape ``[M, F]`` used for the
            feature-space KNN.
        raw_static_features_tensor: Optional unscaled static features tensor. When
            provided, the first two columns are used for the geographic KNN.
        n_neighbors: Exact number of neighbors to return per node (default 9).

    Returns:
        dict mapping each MSOA node id (``int``, 0-indexed) to a list of neighbor
        node ids in deterministic proximity order. Self-loops are excluded.
    """
    if static_features_tensor.ndim != 2:
        raise ValueError(f"Expected static_features_tensor with shape [M, F], got {tuple(static_features_tensor.shape)}")

    def _kneighbors(feature_tensor: torch.Tensor, metric: str) -> list[list[int]]:
        feature_array = feature_tensor.detach().cpu().numpy()
        if metric == "haversine":
            if feature_array.shape[1] < 2:
                raise ValueError("Geographic neighbour search requires at least two columns")
            feature_array = np.radians(feature_array[:, :2])
        k = min(n_neighbors + 1, feature_array.shape[0])
        nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(feature_array)
        _, indices = nbrs.kneighbors(feature_array)
        return [[int(j) for j in row if int(j) != i][:n_neighbors] for i, row in enumerate(indices)]

    feature_candidates = _kneighbors(static_features_tensor, metric="euclidean")
    geo_source = raw_static_features_tensor if raw_static_features_tensor is not None else static_features_tensor[:, :2]
    geo_candidates = _kneighbors(geo_source, metric="haversine" if raw_static_features_tensor is not None else "euclidean")

    neighbor_map: dict[int, list[int]] = {}
    for i in range(static_features_tensor.shape[0]):
        neighbors: list[int] = []
        seen = {i}
        for candidate in feature_candidates[i] + geo_candidates[i]:
            if candidate not in seen:
                neighbors.append(candidate)
                seen.add(candidate)
        neighbor_map[i] = neighbors

    return neighbor_map


def build_anchor_spatial_input(
    series_tensors: Iterable[torch.Tensor],
    anchor_id: int,
    neighbor_map: dict[int, list[int]],
) -> torch.Tensor:
    """Pack one MSOA anchor and its neighbours into a Chronos [V, T] tensor."""
    series_list = [tensor for tensor in series_tensors]
    if len(series_list) == 0:
        raise ValueError("series_tensors is empty")

    history_length = series_list[0].shape[-1]
    if any(tensor.ndim != 2 for tensor in series_list):
        raise ValueError("Each series tensor must have shape [M, T]")
    if any(tensor.shape[-1] != history_length for tensor in series_list):
        raise ValueError("All series tensors must share the same history length")

    node_ids = [anchor_id] + list(neighbor_map[anchor_id])
    rows = [series[node_id] for node_id in node_ids for series in series_list]
    return torch.stack(rows, dim=0)


def build_anchor_spatial_input_from_stacked(
    stacked_series: torch.Tensor,
    anchor_id: int,
    neighbor_map: dict[int, list[int]],
) -> torch.Tensor:
    """Pack one MSOA anchor from a stacked [C, M, T] context tensor."""
    if stacked_series.ndim != 3:
        raise ValueError(f"Expected stacked_series with shape [C, M, T], got {tuple(stacked_series.shape)}")

    node_ids = [anchor_id] + list(neighbor_map[anchor_id])
    packed = stacked_series[:, node_ids, :].permute(1, 0, 2).contiguous()
    return packed.reshape(-1, stacked_series.shape[-1])


def build_anchor_spatial_inputs(
    series_tensors: Iterable[torch.Tensor],
    anchor_ids: Iterable[int],
    neighbor_map: dict[int, list[int]],
) -> list[torch.Tensor]:
    """Build one Chronos input tensor per MSOA anchor."""
    return [build_anchor_spatial_input(series_tensors, anchor_id, neighbor_map) for anchor_id in anchor_ids]


def _repeat_static_rows(static_features: torch.Tensor, history_length: int) -> torch.Tensor:
    """Repeat per-node static features across the time axis and flatten to rows."""
    if static_features.ndim != 2:
        raise ValueError(f"Expected static_features with shape [M, F], got {tuple(static_features.shape)}")
    repeated = static_features.unsqueeze(-1).expand(-1, -1, history_length)
    return repeated.reshape(-1, history_length)


def pack_chronos_input(
    series_tensors: Iterable[torch.Tensor],
    static_features: torch.Tensor | None = None,
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
    if static_features is None:
        return stacked

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
    cache_dir: str | None = None

    def __post_init__(self) -> None:
        cache_dir = self.cache_dir or os.environ.get("CHRONOS_CACHE_DIR") or os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HOME") or os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "huggingface",
        )
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self.pipeline = Chronos2Pipeline.from_pretrained(self.model_id, cache_dir=cache_dir)
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
                if isinstance(count_row_count, (list, tuple)):
                    if len(count_row_count) != len(inputs):
                        raise ValueError("count_row_count must match the number of training inputs")
                    kwargs["inputs"] = [
                        add_count_smoothing_noise(tensor, row_count, count_noise_low, count_noise_high)
                        if isinstance(tensor, torch.Tensor)
                        else tensor
                        for tensor, row_count in zip(inputs, count_row_count)
                    ]
                else:
                    kwargs["inputs"] = [
                        add_count_smoothing_noise(tensor, count_row_count, count_noise_low, count_noise_high)
                        if isinstance(tensor, torch.Tensor)
                        else tensor
                        for tensor in inputs
                    ]

        self.pipeline = self.pipeline.fit(*args, **kwargs)
        # Save logs to an attribute instead of returning them!
        self.run_logs = None
        if hasattr(self.pipeline, "trainer") and hasattr(self.pipeline.trainer, "state"):
            self.run_logs = self.pipeline.trainer.state.log_history
            if self.run_logs is not None:
                print(f"[ChronosTemporalAdapter] Saved {len(self.run_logs)} log entries")
            else:
                print("[ChronosTemporalAdapter] Warning: No log history available")
        else:
            print("[ChronosTemporalAdapter] Warning: trainer or state not found")
            
        return self

    def predict_quantiles(self, *args, **kwargs):
        return self.pipeline.predict_quantiles(*args, **kwargs)
