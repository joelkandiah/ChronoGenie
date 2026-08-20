#!/usr/bin/env python3
"""Thin wrapper around ``Chronos2Pipeline`` plus the sampling utilities it needs.

The adapter keeps three responsibilities:

* resolving a model source (base checkpoint or a fine-tuned LoRA directory),
* fine-tuning while capturing the trainer's log history for plotting,
* inverse-CDF sampling from the quantile forecasts Chronos-2 returns.

Chronos-2 emits *marginal* quantiles per row and per horizon step; it does not
emit joint sample paths. Trajectories are therefore produced by the caller
(:mod:`testing_sliding_window`) via a 1-step autoregressive rollout in which each
path draws its own inverse-CDF realisation and feeds it back into its own
context -- the same scheme the GENIE models use.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from chronos.chronos2 import Chronos2Pipeline

logger = logging.getLogger(__name__)

FINETUNED_CKPT_NAME = "finetuned-ckpt"


# ─── Sampling ────────────────────────────────────────────────────────────────


def sample_from_quantiles(
    quantile_values: torch.Tensor,
    quantile_levels: torch.Tensor,
    num_samples: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw samples from monotone quantile forecasts by inverse-CDF interpolation.

    Args:
        quantile_values: ``[..., Q]`` quantile forecasts, increasing along the last axis.
        quantile_levels: ``[Q]`` the levels the values correspond to, strictly increasing.
        num_samples: Number of independent draws per leading position.
        generator: Optional RNG for reproducible draws.

    Returns:
        ``[num_samples, ...]`` samples. Draws with ``u`` outside
        ``[min(levels), max(levels)]`` are linearly extrapolated from the outermost
        pair of quantiles rather than clipped, so the tails keep some spread.
    """
    if quantile_values.ndim < 1:
        raise ValueError("quantile_values must have at least one dimension")
    if quantile_levels.ndim != 1:
        raise ValueError("quantile_levels must be 1-D")
    if quantile_values.shape[-1] != quantile_levels.numel():
        raise ValueError(
            f"quantile_values last dim ({quantile_values.shape[-1]}) must match "
            f"quantile_levels ({quantile_levels.numel()})"
        )

    device = quantile_values.device
    dtype = quantile_values.dtype
    levels = quantile_levels.to(device=device, dtype=dtype)
    num_levels = levels.numel()

    if num_levels == 1:
        return quantile_values[..., 0].unsqueeze(0).expand(num_samples, *quantile_values.shape[:-1]).contiguous()

    u = torch.rand(
        (num_samples, *quantile_values.shape[:-1]), device=device, dtype=dtype, generator=generator
    )
    right = torch.searchsorted(levels, u.contiguous(), right=True).clamp(1, num_levels - 1)
    left = right - 1

    expanded = quantile_values.unsqueeze(0).expand(num_samples, *quantile_values.shape)
    value_left = torch.gather(expanded, -1, left.unsqueeze(-1)).squeeze(-1)
    value_right = torch.gather(expanded, -1, right.unsqueeze(-1)).squeeze(-1)

    level_left = levels[left]
    level_right = levels[right]
    denominator = torch.clamp(level_right - level_left, min=torch.finfo(dtype).eps)
    weight = (u - level_left) / denominator
    return value_left + weight * (value_right - value_left)


def dequantise_counts(
    values: torch.Tensor,
    count_row_mask: torch.Tensor,
    low: float = 0.0,
    high: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add uniform jitter to integer count rows: ``y -> floor(y + U(low, high))``.

    Discrete counts are a poor fit for a continuous quantile loss. Jittering them
    during fine-tuning lets the model learn a continuous density whose floor
    recovers the count, which is why sampled counts are floored at test time.

    Args:
        values: ``[R, T]`` rows of a Chronos context.
        count_row_mask: ``[R]`` bool, True for rows holding counts.
        low: Lower bound of the jitter.
        high: Upper bound of the jitter.
        generator: Optional RNG for reproducible jitter.

    Returns:
        A copy of ``values`` with the masked rows jittered and clamped at zero.
    """
    if values.ndim != 2:
        raise ValueError(f"Expected values with shape [R, T], got {tuple(values.shape)}")
    if count_row_mask.shape[0] != values.shape[0]:
        raise ValueError("count_row_mask must have one entry per row of values")

    noisy = values.clone()
    if bool(count_row_mask.any()):
        rows = noisy[count_row_mask]
        jitter = torch.empty(rows.shape, dtype=rows.dtype, device=rows.device)
        jitter.uniform_(low, high, generator=generator)
        noisy[count_row_mask] = torch.floor(rows + jitter).clamp_min(0.0)
    return noisy


# ─── Model source resolution ─────────────────────────────────────────────────


def resolve_model_source(base_model_id: str, checkpoint_dir: str | os.PathLike | None) -> str:
    """Return the path/id Chronos should load, preferring a fine-tuned checkpoint.

    ``Chronos2Pipeline.fit`` writes its adapter to ``<output_dir>/finetuned-ckpt``,
    so both that directory and its parent are accepted.
    """
    if checkpoint_dir is None:
        return base_model_id

    candidate = Path(checkpoint_dir)
    for path in (candidate / FINETUNED_CKPT_NAME, candidate):
        if (path / "adapter_config.json").is_file() or (path / "config.json").is_file():
            return str(path)
    return base_model_id


def find_finetuned_checkpoint(output_dir: str | os.PathLike) -> str | None:
    """Path of the fine-tuned checkpoint under ``output_dir``, or None if absent."""
    resolved = resolve_model_source("__missing__", output_dir)
    return None if resolved == "__missing__" else resolved


# ─── Adapter ─────────────────────────────────────────────────────────────────


@dataclass
class ChronosTemporalAdapter:
    """Chronos-2 pipeline wrapper with the interface the runner and tester expect."""

    model_id: str = "amazon/chronos-2"
    device: str = "cuda"
    cache_dir: str | None = None
    run_logs: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        cache_dir = (
            self.cache_dir
            or os.environ.get("CHRONOS_CACHE_DIR")
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
            or os.environ.get("HF_HOME")
            or os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "huggingface")
        )
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

        logger.info("Loading Chronos-2 from %s (cache_dir=%s)", self.model_id, cache_dir)
        self.pipeline = Chronos2Pipeline.from_pretrained(self.model_id, cache_dir=cache_dir)
        self.is_chronos_model = True
        self.is_generative_model = False

    # -- lifecycle ----------------------------------------------------------

    def eval(self) -> "ChronosTemporalAdapter":
        self.pipeline.model.eval()
        return self

    def to(self, device: str) -> "ChronosTemporalAdapter":
        self.device = device
        self.pipeline.model.to(device)
        return self

    @property
    def native_quantile_levels(self) -> list[float]:
        """The quantile grid the model was trained on; sampling on it avoids interpolation."""
        return list(self.pipeline.quantiles)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.pipeline.model.parameters())

    # -- fine-tuning --------------------------------------------------------

    def fine_tune(self, *args, **kwargs) -> "ChronosTemporalAdapter":
        """Fine-tune the pipeline, capturing the HF trainer log history."""
        from transformers.trainer_callback import TrainerCallback

        captured: list[dict] = []

        class _LogCapture(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **_):
                if logs:
                    entry = dict(logs)
                    entry.setdefault("step", state.global_step)
                    entry.setdefault("epoch", state.epoch)
                    captured.append(entry)

        callbacks = list(kwargs.pop("callbacks", []) or [])
        callbacks.append(_LogCapture())

        self.pipeline = self.pipeline.fit(*args, callbacks=callbacks, **kwargs)
        self.run_logs = captured
        logger.info("Fine-tuning finished; captured %d trainer log entries", len(captured))
        return self

    def save_training_logs(self, output_dir: str | os.PathLike) -> str | None:
        """Write the captured trainer logs to ``training_log.jsonl``."""
        if not self.run_logs:
            logger.warning("No trainer logs captured; nothing to save")
            return None
        path = os.path.join(str(output_dir), "training_log.jsonl")
        with open(path, "w") as handle:
            for entry in self.run_logs:
                handle.write(json.dumps(entry) + "\n")
        logger.info("Wrote trainer logs to %s", path)
        return path

    # -- inference ----------------------------------------------------------

    def predict_quantiles(self, *args, **kwargs):
        return self.pipeline.predict_quantiles(*args, **kwargs)
