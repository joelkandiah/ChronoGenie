#!/usr/bin/env python3
"""Spatial neighbourhood construction and Chronos-2 covariate packing.

This module is the single source of truth for *what spatial information each
MSOA sees*. It is shared by fine-tuning and by autoregressive testing so that
the two always agree on the row layout handed to Chronos-2 (a mismatch between
the two silently destroys forecast quality, so there is exactly one builder).

Neighbourhood definition
------------------------
``build_neighbour_map`` uses the same two k-NN searches as the GNN models in
``../GENIE`` (``graph_construction.get_graph`` with
``use_k_nearest_features=True``, ``use_k_nearest_distance=True``, ``knn_k=9``):

1. k-NN (k=9, Euclidean) in the *scaled* static-feature space,
2. k-NN (k=9, haversine) in raw geographic space,
3. the union of the two sets, **per node**.

Degree therefore varies between 9 and 18 depending on how much the two k-NN sets
overlap. ``symmetrise=True`` additionally closes the relation both ways, which
reproduces the undirected graph the GENIE GNNs message-pass over; it is off by
default because a covariate list should be "the MSOAs this one is nearest to".
Chronos-2 groups rows per input item, so a variable number of covariate rows per
anchor is fine -- items in a batch need not share a row count.

Covariate modes
---------------
Each input item is one *anchor* MSOA. Its own burden series are the forecast
targets; the spatial information enters as past-only covariates:

``self_only``
    No spatial covariates. Ablation baseline.
``neighbours``
    One covariate row per (neighbour, burden). Highest fidelity, most rows.
``mean_all_others``
    Per burden, the mean series over every *other* MSOA.
``neighbour_mean``
    Per burden, the mean series over the anchor's neighbours.
``neighbour_and_nonneighbour_means``
    Per burden, two rows: mean over neighbours and mean over non-neighbours.

Optionally a day-of-week pair (sin/cos) is appended as a *known-future*
covariate, mirroring the one-hot day-of-week input the GENIE models receive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

EARTH_RADIUS_KM = 6371.0

COVARIATE_MODES = (
    "self_only",
    "neighbours",
    "mean_all_others",
    "neighbour_mean",
    "neighbour_and_nonneighbour_means",
)

DEFAULT_KNN_K = 9


# ─── Neighbourhood construction ──────────────────────────────────────────────


def haversine_matrix(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """Great-circle distance in km between every pair of points. Shape ``[M, M]``."""
    lat = np.radians(np.asarray(latitudes, dtype=np.float64))
    lon = np.radians(np.asarray(longitudes, dtype=np.float64))

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _knn_neighbours(features: np.ndarray, knn_k: int, metric: str = "euclidean") -> list[list[int]]:
    """Each node's own ``knn_k`` nearest neighbours, self excluded, nearest-first."""
    num_nodes = features.shape[0]
    if num_nodes < 2 or knn_k < 1:
        return [[] for _ in range(num_nodes)]

    num_k = min(knn_k + 1, num_nodes)  # +1 so self can be dropped
    finder = NearestNeighbors(n_neighbors=num_k, metric=metric).fit(features)
    _, indices = finder.kneighbors(features)
    return [[int(j) for j in indices[i] if int(j) != i][:knn_k] for i in range(num_nodes)]


def _knn_neighbours_from_distance(distance_matrix: np.ndarray, knn_k: int) -> list[list[int]]:
    """Each node's own ``knn_k`` nearest neighbours from a precomputed distance matrix."""
    num_nodes = distance_matrix.shape[0]
    if num_nodes < 2 or knn_k < 1:
        return [[] for _ in range(num_nodes)]

    k = min(knn_k, num_nodes - 1)
    return [
        [int(j) for j in np.argsort(distance_matrix[i], kind="stable") if int(j) != i][:k]
        for i in range(num_nodes)
    ]


def build_neighbour_map(
    scaled_static_features: torch.Tensor,
    raw_static_features: torch.Tensor,
    static_feature_names: Sequence[str] | None = None,
    knn_k: int = DEFAULT_KNN_K,
    use_k_nearest_features: bool = True,
    use_k_nearest_distance: bool = True,
    symmetrise: bool = False,
    max_neighbours: int | None = None,
) -> tuple[dict[int, list[int]], np.ndarray]:
    """Build the neighbourhood used for spatial covariates.

    Node ``i``'s neighbourhood is the union of **its own** two k-NN sets: the
    ``knn_k`` nearest in scaled static-feature space (Euclidean) and the ``knn_k``
    nearest geographically (haversine). Degree is therefore between ``knn_k`` and
    ``2 * knn_k``, depending on how much the two sets overlap.

    Args:
        scaled_static_features: ``[M, F]`` min-max scaled static features.
        raw_static_features: ``[M, F]`` unscaled static features, used to locate lat/lon.
        static_feature_names: Column names of the static feature tensors. When given,
            ``latitude``/``longitude`` are looked up by name; otherwise columns 0 and 1
            are assumed (the order produced by ``dataset.load_static_features``).
        knn_k: Neighbours per metric before the union (GENIE default 9).
        use_k_nearest_features: Include the static-feature-space k-NN set.
        use_k_nearest_distance: Include the geographic (haversine) k-NN set.
        symmetrise: Also treat ``i`` as a neighbour of ``j`` whenever ``j`` is a
            neighbour of ``i``. This reproduces the *graph* the GENIE GNNs message-pass
            over (``graph_construction.get_graph`` ends in ``to_undirected``), which is
            denser -- mean degree ~20 rather than ~14 on the 84-MSOA data. Off by
            default: as a covariate list, "the MSOAs this one is nearest to" is the
            intended relation, not "plus every MSOA that happens to pick this one".
        max_neighbours: Optional cap on the degree, keeping the nearest neighbours.
            The main lever on the cost of the ``neighbours`` covariate mode.

    Returns:
        ``(neighbour_map, distance_matrix)`` where ``neighbour_map[i]`` lists node ids
        adjacent to ``i``, nearest-first by great-circle distance (ties broken by node
        id), and ``distance_matrix`` is the ``[M, M]`` haversine matrix in km.
    """
    if scaled_static_features.ndim != 2 or raw_static_features.ndim != 2:
        raise ValueError("Static feature tensors must be 2-D with shape [M, F]")
    if scaled_static_features.shape[0] != raw_static_features.shape[0]:
        raise ValueError("Scaled and raw static features must describe the same nodes")
    if not (use_k_nearest_features or use_k_nearest_distance):
        raise ValueError("At least one of use_k_nearest_features / use_k_nearest_distance must be True")

    num_nodes = scaled_static_features.shape[0]
    raw = raw_static_features.detach().cpu().numpy()

    lat_idx, lon_idx = 0, 1
    if static_feature_names is not None:
        names = list(static_feature_names)
        if "latitude" in names and "longitude" in names:
            lat_idx, lon_idx = names.index("latitude"), names.index("longitude")
    distances = haversine_matrix(raw[:, lat_idx], raw[:, lon_idx])

    empty: list[list[int]] = [[] for _ in range(num_nodes)]
    by_feature = (
        _knn_neighbours(scaled_static_features.detach().cpu().numpy(), knn_k, metric="euclidean")
        if use_k_nearest_features
        else empty
    )
    by_distance = _knn_neighbours_from_distance(distances, knn_k) if use_k_nearest_distance else empty

    neighbours: list[set[int]] = [
        set(by_feature[i]) | set(by_distance[i]) for i in range(num_nodes)
    ]
    for node in range(num_nodes):
        neighbours[node].discard(node)

    if symmetrise:
        for node in range(num_nodes):
            for other in list(neighbours[node]):
                neighbours[other].add(node)

    neighbour_map = {
        i: sorted(neighbours[i], key=lambda j: (float(distances[i, j]), j)) for i in range(num_nodes)
    }
    if max_neighbours is not None:
        if max_neighbours < 1:
            raise ValueError("max_neighbours must be >= 1 when set")
        neighbour_map = {i: js[:max_neighbours] for i, js in neighbour_map.items()}
    return neighbour_map, distances


def neighbour_adjacency_matrix(neighbour_map: dict[int, list[int]], num_nodes: int) -> torch.Tensor:
    """Boolean ``[M, M]`` adjacency (no self-loops) from a neighbour map.

    Row ``i`` marks the neighbours *of* ``i``. The map need not be symmetric, so
    ``adjacency[i, j]`` and ``adjacency[j, i]`` can differ; every consumer works
    row-wise, which is what "the covariates node ``i`` sees" means.
    """
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    for i, neighbours in neighbour_map.items():
        for j in neighbours:
            adjacency[i, j] = True
    adjacency.fill_diagonal_(False)
    return adjacency


def neighbour_map_summary(neighbour_map: dict[int, list[int]]) -> dict[str, float]:
    """Degree statistics, logged so that runs can be checked for graph drift."""
    degrees = np.array([len(v) for v in neighbour_map.values()], dtype=np.float64)
    directed = {(i, j) for i, js in neighbour_map.items() for j in js}
    reciprocal = sum(1 for (i, j) in directed if (j, i) in directed)
    return {
        "num_nodes": float(degrees.size),
        "min_degree": float(degrees.min()) if degrees.size else 0.0,
        "mean_degree": float(degrees.mean()) if degrees.size else 0.0,
        "max_degree": float(degrees.max()) if degrees.size else 0.0,
        "num_directed_edges": float(len(directed)),
        "fraction_reciprocal": float(reciprocal / len(directed)) if directed else 0.0,
    }


# ─── Covariate packing ───────────────────────────────────────────────────────


def day_of_week_features(start_dow: int, timesteps: torch.Tensor | np.ndarray | Sequence[int]) -> torch.Tensor:
    """Sin/cos encoding of the weekday for absolute timesteps. Shape ``[2, len(timesteps)]``."""
    steps = torch.as_tensor(np.asarray(timesteps), dtype=torch.float32)
    angle = 2.0 * np.pi * ((steps + float(start_dow)) % 7.0) / 7.0
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=0)


@dataclass
class SpatialCovariateBuilder:
    """Turns a stacked context tensor into per-anchor Chronos-2 inputs.

    Args:
        mode: One of :data:`COVARIATE_MODES`.
        neighbour_map: Output of :func:`build_neighbour_map`.
        num_nodes: Number of MSOAs ``M``.
        num_channels: Number of burden channels ``C`` per MSOA.
        channel_names: Names of the ``C`` channels, used for covariate naming/logging.
        include_day_of_week: Append a sin/cos weekday pair as a known-future covariate.
    """

    mode: str
    neighbour_map: dict[int, list[int]]
    num_nodes: int
    num_channels: int
    channel_names: Sequence[str] = ()
    include_day_of_week: bool = False

    _neighbour_weights: torch.Tensor | None = field(default=None, init=False, repr=False)
    _non_neighbour_weights: torch.Tensor | None = field(default=None, init=False, repr=False)
    _others_weights: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in COVARIATE_MODES:
            raise ValueError(f"Unknown covariate mode '{self.mode}'. Expected one of {COVARIATE_MODES}.")
        if self.num_nodes < 1 or self.num_channels < 1:
            raise ValueError("num_nodes and num_channels must be positive")

        adjacency = neighbour_adjacency_matrix(self.neighbour_map, self.num_nodes).float()

        if self.mode in ("neighbour_mean", "neighbour_and_nonneighbour_means"):
            self._neighbour_weights = self._row_normalise(adjacency)
        if self.mode == "neighbour_and_nonneighbour_means":
            non_neighbour = 1.0 - adjacency
            non_neighbour.fill_diagonal_(0.0)  # the anchor is never its own "non-neighbour"
            self._non_neighbour_weights = self._row_normalise(non_neighbour)
        if self.mode == "mean_all_others":
            others = torch.ones((self.num_nodes, self.num_nodes))
            others.fill_diagonal_(0.0)
            self._others_weights = self._row_normalise(others)

    @staticmethod
    def _row_normalise(weights: torch.Tensor) -> torch.Tensor:
        """Rows sum to 1; empty rows stay all-zero (mean of nothing is reported as 0)."""
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)

    # -- introspection ------------------------------------------------------

    @property
    def channel_labels(self) -> list[str]:
        if len(self.channel_names) == self.num_channels:
            return list(self.channel_names)
        return [f"channel_{i}" for i in range(self.num_channels)]

    def covariate_names(self, anchor_id: int) -> list[str]:
        """Names of the past-only covariate rows for one anchor, in row order."""
        labels = self.channel_labels
        if self.mode == "self_only":
            names: list[str] = []
        elif self.mode == "neighbours":
            names = [f"nbr{rank}_{label}" for rank, _ in enumerate(self.neighbour_map[anchor_id]) for label in labels]
        elif self.mode == "mean_all_others":
            names = [f"mean_others_{label}" for label in labels]
        elif self.mode == "neighbour_mean":
            names = [f"mean_nbrs_{label}" for label in labels]
        else:
            names = [f"mean_nbrs_{label}" for label in labels] + [f"mean_nonnbrs_{label}" for label in labels]
        return names

    def num_covariates(self, anchor_id: int) -> int:
        if self.mode == "self_only":
            return 0
        if self.mode == "neighbours":
            return len(self.neighbour_map[anchor_id]) * self.num_channels
        if self.mode == "neighbour_and_nonneighbour_means":
            return 2 * self.num_channels
        return self.num_channels

    def rows_per_anchor(self, anchor_id: int) -> int:
        """Total Chronos rows for one anchor: targets + covariates (+ weekday pair)."""
        extra = 2 if self.include_day_of_week else 0
        return self.num_channels + self.num_covariates(anchor_id) + extra

    def total_rows(self) -> int:
        return sum(self.rows_per_anchor(m) for m in range(self.num_nodes))

    # -- packing ------------------------------------------------------------

    def _spatial_covariates(self, stacked: torch.Tensor) -> list[torch.Tensor] | None:
        """Per-anchor covariate blocks ``[R_m, L]`` from a ``[C, M, L]`` context tensor."""
        if self.mode == "self_only":
            return None

        num_channels, num_nodes, length = stacked.shape

        if self.mode == "neighbours":
            return [
                stacked[:, self.neighbour_map[m], :].permute(1, 0, 2).reshape(-1, length)
                for m in range(num_nodes)
            ]

        # Mean-based modes: one [M, M] @ [M, C*L] product covers every anchor at once.
        flat = stacked.permute(1, 0, 2).reshape(num_nodes, num_channels * length)
        blocks = []
        if self.mode == "mean_all_others":
            blocks.append(self._others_weights @ flat)
        elif self.mode == "neighbour_mean":
            blocks.append(self._neighbour_weights @ flat)
        else:
            blocks.append(self._neighbour_weights @ flat)
            blocks.append(self._non_neighbour_weights @ flat)

        stacked_blocks = torch.cat(
            [b.reshape(num_nodes, num_channels, length) for b in blocks], dim=1
        )  # [M, R, L]
        return [stacked_blocks[m] for m in range(num_nodes)]

    def build_inputs(
        self,
        stacked_context: torch.Tensor,
        prediction_length: int,
        anchor_ids: Iterable[int] | None = None,
        past_dow: torch.Tensor | None = None,
        future_dow: torch.Tensor | None = None,
    ) -> list[dict]:
        """Build one Chronos-2 ``PreparedInput`` per anchor MSOA.

        Args:
            stacked_context: ``[C, M, L]`` history, channel-major, raw (unscaled) units.
            prediction_length: Forecast horizon of the call the inputs are built for.
            anchor_ids: Anchors to build, defaults to all ``M`` in order.
            past_dow: ``[2, L]`` weekday encoding aligned with ``stacked_context``.
            future_dow: ``[2, prediction_length]`` weekday encoding of the horizon.

        Returns:
            A list of ``PreparedInput`` dicts: ``context`` rows are
            ``[targets (C) | spatial covariates (R) | weekday (2)]``, with the weekday
            rows flagged as known into the future.
        """
        if stacked_context.ndim != 3:
            raise ValueError(f"Expected stacked_context [C, M, L], got {tuple(stacked_context.shape)}")
        num_channels, num_nodes, length = stacked_context.shape
        if num_channels != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {num_channels}")
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {num_nodes}")
        if length < 1:
            raise ValueError("stacked_context must have at least one timestep of history")

        stacked = stacked_context.detach().to(dtype=torch.float32, device="cpu")
        covariates = self._spatial_covariates(stacked)
        anchors = list(range(num_nodes)) if anchor_ids is None else list(anchor_ids)

        if self.include_day_of_week:
            if past_dow is None or future_dow is None:
                raise ValueError("include_day_of_week=True requires past_dow and future_dow")
            past_dow = past_dow.to(dtype=torch.float32, device="cpu")
            future_dow = future_dow.to(dtype=torch.float32, device="cpu")
            if past_dow.shape != (2, length):
                raise ValueError(f"past_dow must have shape [2, {length}], got {tuple(past_dow.shape)}")
            if future_dow.shape != (2, prediction_length):
                raise ValueError(
                    f"future_dow must have shape [2, {prediction_length}], got {tuple(future_dow.shape)}"
                )

        num_future_covariates = 2 if self.include_day_of_week else 0
        inputs: list[dict] = []
        for anchor_id in anchors:
            rows = [stacked[:, anchor_id, :]]
            if covariates is not None:
                rows.append(covariates[anchor_id])
            if self.include_day_of_week:
                rows.append(past_dow)
            context = torch.cat(rows, dim=0).contiguous()

            num_rows = context.shape[0]
            future = torch.full((num_rows, prediction_length), float("nan"), dtype=torch.float32)
            if self.include_day_of_week:
                future[-2:] = future_dow

            inputs.append(
                {
                    "context": context,
                    "future_covariates": future,
                    "n_targets": num_channels,
                    "n_covariates": num_rows - num_channels,
                    "n_future_covariates": num_future_covariates,
                }
            )
        return inputs
