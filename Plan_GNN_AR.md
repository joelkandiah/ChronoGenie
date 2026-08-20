
# Plan: Integrating Dynamic Nearest Neighbor Sequences into Chronos-2 Autoregressive Rollout

## 1. Objective

Modify the `Chronos-2` time-series test environment (`testing_sliding_window.py`) to transition from a generalized univariate inference loop into a true **1-step autoregressive rolling forecast** over a 60-day horizon. Instead of evaluating MSOAs independently or pooling spatial context via GNN means, the updated pipeline will dynamically pull the **9 nearest neighbor sequences** from the spatial graph topology at each step, sample stochastically using **Inverse CDF selection**, and roll the values back into the context for all MSOAs concurrently.

---

## 2. Structural Requirements & Tasks

### Phase A: Configuration Adjustments

* **Task 1:** Modify `chronos2_test_only.yaml` to run a 60-day evaluation horizon instead of the smoke-test default.
* *Update:* Change `testing.autoregressive_windows` to describe after min day how many forecast windows to start. I.e. a 7 day forcast [testing.autoregressive_windows] times from each of the days from min_day:testing.autoregressive_windows+min_day



### Phase B: Spatial Graph Extraction (No-Graph Architecture)

* **Task 2:** Extract neighbor indexes directly from `spatial_graph_data.edge_index` to completely bypass GNN aggregation blocks inside the inference loop while keeping neighbor identification faithful to the original graph construction.
* *Implementation:* Build a deterministic lookup dictionary `neighbor_map` where `neighbor_map[msoa_id]` contains an array of exactly its 9 nearest neighbor `node_ids` (sorted by geographic proximity or spatial edge rank).
* Note this spatial fraph data was implemented before so I want you to find the old version of the code via git history (which we removed very early on as we transitioned to using chronos only)



### Phase C: Step-by-Step Autoregressive Ingestion Loop

* **Task 3:** Restructure  the main loop inside `get_sliding_window_predictions` to advance step-by-step for a 1-day forecast window.
* **Task 4:** Construct multivariate rows for Chronos by appending the raw history tensors of the target MSOA and its 9 neighbors sequentially.
* *Format:* For every target MSOA, pack its inputs into Chronos with a total vertical axis of:

$$\text{Total Rows} = \text{num\_msoas} \times ((1 \text{ [Self]} + 9 \text{ [Neighbors]}) * num_burdens) \times \text{num\_prediction}$$


* *Execution:* Call `temporal_model.predict_quantiles` strictly with `prediction_length=1`.
* I.e. for each msoa and burden we should predict the one step ahead predictio for that msoa given it's context history (for the prediction burden) and covariate sequences in attention which are the 9 neighbours sequences for every burden type (and the other burdens for the target msoa) 



### Phase D: Stochastic Feedback & Path Unpacking

* **Task 5:** Implement Inverse-CDF trajectory progression instead of squashing variance using execution means.
* *Implementation:* Call `sample_from_quantiles(quantiles_list, num_samples=1)` to pick an explicit realization path.


* **Task 6:** Extract exclusively the target MSOA's generated values (discarding predicted neighbor rows) and stitch them back into `contexts_multi_sample`, shifting the sliding time context window forward.

### Phase E: Non-Overwriting File Structure

* **Task 7:** Update the dynamic naming convention of output folders to match daily outputs rather than an aggregated base context timestamp.
* *Fix:* Change `prediction_folder = os.path.join(sim_root_folder, "predictions", str(from_index + context_size))` to depend on `str(pred_time_idx)`.



---

## 3. Execution Checklist & Code Map

| Target File | Target Section | Description of Action |
| --- | --- | --- |
| `chronos2_test_only.yaml` | `testing:` block | Set `autoregressive_windows: [60]`. |
| `testing_sliding_window.py` | Init / Setup | Build `neighbor_map` using `graph_data[1].edge_index`. Truncate/pad to 9 neighbors. |
| `testing_sliding_window.py` | Loop `current_test_window` | Change Chronos logic to fetch target + 9 neighbor sequences explicitly; set `prediction_length=1`. |
| `testing_sliding_window.py` | Update Context Block | Replace `sampled.mean(dim=1)` with path index `0` of Inverse-CDF `sample_from_quantiles`. |
| `testing_sliding_window.py` | Saving / Output | Change `prediction_folder` sub-path to track the iterating variable `pred_time_idx`. |

---

## 4. Important Implementation Notes for the Engineering Agent

> ⚠️ **CRITICAL SOURCE CODE NOTICE:**
> The original spatial-GNN testing logic, spatial aggregation parameters, and the structural design of the historical rolling forecast framework can be recovered directly by inspecting the code from the **initial repository commit** (or the commit immediately following the initialization setup).
> Reviewing `testing_sliding_window_old.py` and checking earlier versions of `testing_sliding_window.py` from your git history will provide absolute clarity on how the model tracks tracking tensors, constructs `contexts_multi_sample`, manages the index transformations `ctx_to_pred_idx`, and utilizes `spatial_graph_data`.

* **Fine-Tuning Dimensionality Constraint:** The engineering agent must verify that the fine-tuning setup used in `training.py` or the `ChronosTemporalAdapter` aligns perfectly with this input dimension configuration ($10 \times \text{variables}$ per target sequence block). If the adapter model was not trained to parse stacked spatial multi-series sequences, a parallel training adjustment using this exact layout template will be mandatory prior to inference evaluation.
* **Neighborhood Consistency:** Since Chronos relies on row ordering positioning within the batch tensor to separate distinct components, neighbors inside `neighbor_map` must be organized in a deterministic manner (e.g., matching the order extracted from `NearestNeighbors` or the graph index topology) across both training and testing steps.