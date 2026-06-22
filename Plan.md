# Chrono-Genie / Chronos-2 plan

The goal is not to invent a new forecasting pipeline. It is to keep the existing GENIE data contract, the current autoregressive testing pattern, and the current scoring contract, then swap in Chronos-2 as the temporal model while preserving comparable outputs.

## 1. Keep the existing node and graph contract

Use the same MSOA ordering already built by the dataset and graph code, with nodes aligned by `gid`. Reuse the same spatial neighborhoods from the current graph construction path in `graph_construction.py`, including the default `knn_k=9` Haversine neighborhood logic that GENIE already uses for geographic adjacency.

The static node context should continue to come from the existing processed static feature table returned by `load_static_features`. In practice that means Chronos should receive the same per-node static context that the current code already attaches to each geography, rather than a new feature set or a reindexed node table.

## 2. Reuse the current autoregressive test shape

The testing path already rolls forecasts forward in sliding windows through `testing.py` and `testing_sliding_window.py`. Chronos should fit into that same loop structure:

- use the same context windows and the same batch/window boundaries
- keep the same neighbor-history sequences that the current pipeline already feeds into the temporal path
- preserve the same static context alongside the temporal input so the model sees the same node identity signal on every step
- keep the same per-window, per-sample rollout shape so the output contract stays comparable

This is the right place to preserve the current one-step autoregressive behavior: predict one step, write the prediction back into the rolling context, shift forward, and repeat until the horizon is covered. The Chronos variant should update the same rolling context state rather than building a separate decoding scheme.

## 3. Fine-tune Chronos-2 with PEFT only

Load Chronos-2 in `torch.bfloat16`, freeze the base weights, and add LoRA adapters to the attention projections. Keep the fine-tuning setup single-GPU and lightweight, so the Chronos branch stays practical inside the current project instead of becoming a separate training stack.

The important part is output compatibility, not a new training recipe. The Chronos head should be wired so it produces the same forecast variables and sample counts expected by the current evaluation code.

## 4. Match the autoregressive rollout behavior

Use Chronos sampling in a way that mirrors the current recursive testing loop:

- sample one forecast step
- feed that sampled value back into the next context window
- keep the static context fixed for the node across steps
- keep using the same neighboring past sequences rather than switching to a different sequence-selection rule

If Chronos returns logits or token bins, convert them into sampled real-valued forecasts before the update step, so the recursive loop can continue to operate on the same kind of values the existing code expects.

## 5. Keep the evaluation path unchanged

The code already evaluates forecasting quality with the existing proper scoring implementations, so the Chronos strategy should reuse those exact functions instead of introducing a parallel metric stack. That means the Chronos outputs should be shaped to work with the current energy score and variogram score computations, along with the existing interval score and CRPS paths.

The target is matching outputs, not just matching interface names. Chronos should emit prediction samples in the same layout the current testing code consumes, so the saved results and downstream metrics stay directly comparable to the GENIE baseline.