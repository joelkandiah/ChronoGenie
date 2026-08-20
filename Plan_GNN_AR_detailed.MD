# Plan: GNN-AR Spatial Neighbor Autoregressive Inference

Restructure the testing and training pipeline per **Plan_GNN_AR.md** to support:
- Per-MSOA 1-step autoregressive rollouts that pull 9 nearest-neighbor context sequences
- Consistent `neighbor_map` built from static feature KNN (matching original graph construction logic)
- Aligned training inputs (same row layout: self + 9 neighbors, all burdens)
- Correct prediction-folder naming per day index

---

## User Review Required

> [!IMPORTANT]
> **Training input layout change**: The current `build_chronos_training_inputs` packs **all MSOAs** for a simulation into a single tensor `[M * C + F_static, T]`. Under the new scheme, the tensor for each training sim becomes `[(1_self + 9_neighbors) * num_burdens * M + F_static, T]` — i.e. each MSOA acts as an anchor and provides itself + its 9 neighbors across all burdens as rows. **This changes the number of rows per training input** significantly. The model must be fine-tuned (or re-fine-tuned) on this layout before testing with the same layout. Confirm whether you want training inputs rebuilt to match, or whether you'd prefer to keep a simpler training scheme and only restructure inference.

> [!WARNING]
> **The plan references `spatial_graph_data.edge_index`** (passed via `graph_data[1]`). Currently `run_experiment.py` passes `graph_data=[None, None]` and `testing.py` immediately `del`s it. The new pipeline instead builds `neighbor_map` from static features (lat/lon KNN), not a pre-built graph object — so no extra argument plumbing is needed. Verify that `static_features_tensor` columns include `latitude` and `longitude` in positions 0 and 1 (they do, from `dataset.py`'s `preferred_columns` order).

> [!NOTE]
> The plan says `autoregressive_windows: [60]` in the yaml. The existing test-only config has `[7]`. I'll set it to `[60]` in `chronos2_test_only.yaml`. The semantic here (per the plan) is that windows = days to forecast ahead from `min_day` — each step forecasts 1 day.

---

## Open Questions

> [!IMPORTANT]
> **Training alignment**: Should I also update `build_chronos_training_inputs` in `run_experiment.py` to use the neighbor-stacked layout? This makes training and inference fully consistent (as required by Note §4.2 of the plan), but it will break any existing fine-tuned checkpoint trained with the old layout. Please confirm before I proceed with training changes, or I can make it a toggled config option.

---

## Proposed Changes

### Phase A — Configuration

#### [MODIFY] [chronos2_test_only.yaml](file:///home/joelk/ProgrammingProjects/Python/ChronoGenie/configs/experiments/chronos2_test_only.yaml)
- Change `testing.autoregressive_windows: [7]` → `[60]`
- Matches plan Task 1: "60-day evaluation horizon"

---

### Phase B — Spatial Graph Extraction (neighbor_map)

#### [MODIFY] [testing_sliding_window.py](file:///home/joelk/ProgrammingProjects/Python/ChronoGenie/testing_sliding_window.py)

Add a **`build_neighbor_map`** helper function at module level that:
1. Accepts `static_features_tensor: torch.Tensor` of shape `[M, F]`
2. Extracts the first two columns (lat, lon) for Euclidean KNN
3. Uses `sklearn.neighbors.NearestNeighbors(n_neighbors=10, metric='euclidean')` (10 so we can drop self, keeping exactly 9)
4. Returns `neighbor_map: dict[int, list[int]]` where each key is `msoa_id` (0-indexed) and value is a list of exactly 9 neighbor `node_ids` in proximity order (nearest-first, matching the original `graph_construction.py` logic)

This completely replaces GNN aggregation for inference while remaining faithful to the graph topology.

---

### Phase C — Step-by-Step Autoregressive Ingestion Loop

#### [MODIFY] [testing_sliding_window.py](file:///home/joelk/ProgrammingProjects/Python/ChronoGenie/testing_sliding_window.py)

**Restructure `get_sliding_window_predictions`**:

**Signature change**: Accept `graph_data` but build `neighbor_map` internally from `dataset_directory.static_features_tensor`. No new external arguments needed.

**Init block** (before autoregressive loop):
```python
neighbor_map = build_neighbor_map(
    static_features_tensor=dataset_directory.static_features_tensor,
    n_neighbors=9,
)
```

**New inner loop structure** (Task 3 & 4):

For each `current_test_window` step (1-day rollout):
- For each `sim_id` in `dataset_directory.test_sims`:
  - Build a **single batched Chronos input tensor** for all MSOAs at once
  - For each target MSOA `m` (0 to `num_msoas-1`):
    - Fetch history slice `[context_start:context_end]` for: 
      - All `num_burdens` channels of **self** (m) → `num_burdens` rows
      - All `num_burdens` channels of each of its 9 **neighbors** → `9 * num_burdens` rows
    - Total rows per MSOA = `10 * num_burdens`
  - Stack all MSOAs: final tensor shape `[num_msoas * 10 * num_burdens + F_static, context_size]`
  - Call `temporal_model.predict_quantiles(inputs=[packed], prediction_length=1)`
  - **Extract only the self-MSOA rows** from the output (rows `[m * 10 * num_burdens : m * 10 * num_burdens + num_burdens]` for each m)
  - Apply `sample_from_quantiles` → take 1 stochastic path (ICDF sample index 0)
  - Write `icdf_sample` back into `predictions` for next step

> [!NOTE]
> Packing strategy: Rather than one giant tensor, we pack one tensor **per sim** with all MSOA anchor sequences stacked, then call predict_quantiles once. This is memory-efficient and avoids a per-MSOA inference loop. The `unpack_prediction_blocks` helper currently assumes a different row order, so we'll bypass it and do direct index slicing instead.

---

### Phase D — Stochastic Feedback & Path Unpacking

#### [MODIFY] [testing_sliding_window.py](file:///home/joelk/ProgrammingProjects/Python/ChronoGenie/testing_sliding_window.py)

- **Task 5**: Use `sample_from_quantiles(..., num_samples=1)` to draw a single ICDF path instead of `sampled.mean(dim=1)`. This is already partially done (line 112: `icdf_sample = sampled[:, 0, :]`), but must be tied to the new per-MSOA extraction logic
- **Task 6**: Extract only the target MSOA's generated values (self-block rows), discard neighbor prediction rows, stitch into `predictions` rolling context

---

### Phase E — Non-Overwriting File Structure

#### [MODIFY] [testing_sliding_window.py](file:///home/joelk/ProgrammingProjects/Python/ChronoGenie/testing_sliding_window.py)

- **Task 7**: Prediction folder path already uses `str(pred_time_idx)` in the current code (line 141), so this fix is already applied. Verify it tracks `pred_time_idx = from_index + context_size + current_test_window` correctly.

---

### Training Alignment (Phase F — Conditional on user confirmation)

#### [MODIFY] [run_experiment.py](file:///home/joelk/ProgrammingProjects/Python/ChronoGenie/run_experiment.py)

Update `build_chronos_training_inputs` to use the same **neighbor-stacked row layout** as inference:
- For each training sim, for each MSOA `m`: pack `[self + 9 neighbors] * num_burdens` rows
- Results in a list of tensors matching the inference input shape
- This requires passing `static_features_tensor` and `neighbor_map` into the builder

---

## Verification Plan

### Automated Checks
- Run a smoke test with `chronos2_test_only.yaml` (context_size=14, 60 windows) to verify shapes don't throw
- Add shape assertions in the code for key tensors

### Manual Verification
- Print `neighbor_map[0]` → should show 9 integer node IDs
- Print input tensor shape before `predict_quantiles` → should be `[num_msoas * 10 * num_burdens + F_static, context_size]`
- Confirm `predictions[prediction_indices, sim_idx, :, pred_time_idx]` is updated correctly each step
- Confirm output feather files appear under `predictions/{pred_time_idx}/` for each day
