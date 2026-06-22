import os
from datetime import datetime
import importlib.util
import pandas as pd
import numpy as np
import random
import copy
import contextlib

import torch
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, OneCycleLR
import muon

from dataset import ProcessedData
from distribution_construction import MixedLoss
from train_test_helpers import get_batched_graph, plot_train_val_loss, plot_lr_vs_iteration, apply_poisson_noise_to_contexts

best_val_loss = float("inf")
best_models = {}


# Adding random seed for replication
torch.manual_seed(26)
random.seed(26)
np.random.seed(26)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(26)

torch.set_num_threads(20)

def train(temporal_model, local_profile_encoder, interaction_encoder,
          graph_data, dataset_directory, context_size,
          num_epochs=500, lr=1e-3, num_batches=10, batch_size=64,
          device="cuda", output_folder="MODEL_OUTPUT",
          lpe_ablation = False, include_latest_context = False, spatial_encoder_type = "gnn", resume_from_checkpoint=None,
          model_type="mlp", ema_alpha=0.1, use_compile=False,
          multistep_k=1, multistep_weights=None, validation_thinning_factor=5,
          validation_batch_size=None, poisson_noise_context=False,
          augmentation_thinning_factor=5,
          optimiser_type="adamw", muon_lr=0.02, muon_momentum=0.95, muon_weight_decay=0.0,
          adamw_lr=None, adamw_weight_decay=1e-2, gradient_accumulation_steps=1, use_bfloat16=False,
          scheduler_type="onecycle", cosine_eta_min=0.0, cosine_warmup_fraction=0.05, max_grad_norm=1.0,
          gradient_clip_value=1.0):
    
    if multistep_weights is None:
        multistep_weights = [1.0]
    if adamw_lr is None:
        adamw_lr = lr
    # Unpacking graph data, make sure that the order is correct.
    temp_graph_data, spatial_graph_data = graph_data[0], graph_data[1]

    # Creating the output folder to save the trained models
    trained_models_subfolder = f"{output_folder}/trained_models/"
    os.makedirs(trained_models_subfolder, exist_ok=True)
    training_loss_path = os.path.join(trained_models_subfolder, "loss_history.csv")
    loss_header_written = os.path.exists(training_loss_path)
    use_multistep_metrics = multistep_k > 1
    loss_columns = ["train_loss", "val_loss_1step", "lr_muon", "lr_adamw"]
    if use_multistep_metrics:
        loss_columns.append("val_loss_multistep")

    ##########################
    # Moving Model to Device #
    ##########################
    temporal_model.to(device)
    interaction_encoder.to(device)
    if local_profile_encoder is not None: # Cases where we don't use the LPE i) Ablation ii) Unified Encoder
        local_profile_encoder.to(device)

    ########
    # Data #
    ########
    # Training Data Loader --> Using RandomSampler for stochastic gradient descent.
    train_dataset = ProcessedData(data_directory=dataset_directory, past_context_size=context_size, type="training", multistep_k=multistep_k)
    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=num_batches*batch_size)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=0, pin_memory=False)
    
    # Validation data loader.
    # Even when split into mini-batches, we still evaluate the full validation set by
    # looping over every batch and aggregating metrics over all samples.
    validation_dataset = ProcessedData(data_directory=dataset_directory, past_context_size=context_size, type="validation", multistep_k=multistep_k)
    if validation_batch_size is None:
        validation_batch_size = min(batch_size, len(validation_dataset))
    val_loader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=False, num_workers=0, pin_memory=False)

    #################################
    # Optimiser, Loss and Scheduler #
    #################################
    # Setting up optimiser, passing all model's parameters.
    named_params = []
    for i, m in enumerate(temporal_model):
        for name, param in m.named_parameters():
            named_params.append((f"temporal_model.{i}.{name}", param))
    for name, param in interaction_encoder.named_parameters():
        named_params.append((f"interaction_encoder.{name}", param))
    if local_profile_encoder is not None:
        for name, param in local_profile_encoder.named_parameters():
            named_params.append((f"local_profile_encoder.{name}", param))
            
    params_to_clip = [p for _, p in named_params]

    if optimiser_type == "muon":
        # Simply import/use the module directly
        from muon import SingleDeviceMuon
        
        exclude_keywords = [
            "mu_head", "disp_head", "velocity_head", "noise_head",
            "mu0_head", "mu1_head", "logvar_head"
        ]
        muon_params = []
        adamw_params = []
        for name, p in named_params:
            is_head = any(keyword in name for keyword in exclude_keywords)
            if p.ndim >= 2 and not is_head:
                muon_params.append(p)
            else:
                adamw_params.append(p)

        print(
            f"Using Muon optimiser. Muon parameters: {len(muon_params)} "
            f"(lr={muon_lr}, momentum={muon_momentum}, weight_decay={muon_weight_decay}), "
            f"AdamW parameters: {len(adamw_params)} (lr={adamw_lr}, weight_decay={adamw_weight_decay})"
        )
        
        opt_muon = SingleDeviceMuon(muon_params, lr=muon_lr, momentum=muon_momentum, weight_decay=muon_weight_decay) if muon_params else None
        opt_adamw = torch.optim.AdamW(adamw_params, lr=adamw_lr, weight_decay=adamw_weight_decay) if adamw_params else None
        optimiser = (opt_muon, opt_adamw)
    else:
        architecture_parameters = [p for _, p in named_params]
        opt_adamw = torch.optim.AdamW(architecture_parameters, lr=adamw_lr, weight_decay=adamw_weight_decay)
        optimiser = (None, opt_adamw)
    
    # NOTE: Scheduler is constructed AFTER checkpoint loading (below) so that
    # we know start_iteration and can build OneCycleLR only for the *remaining*
    # epochs.  This prevents load_state_dict from overwriting total_steps with
    # a stale value from a previous (possibly shorter) run.

    ############
    # TRAINING #
    ############
    # Initialising tracking variables for early stopping and loss history
    best_val_loss = float('inf') # Current best validation loss (for checkpointing)
    final_val_loss = float('inf')
    training_loss = []  # Lists to store loss values per iterations
    validation_loss_1step = []
    validation_loss_multistep = [] if use_multistep_metrics else None
    lr_history_muon = []
    lr_history_adamw = []

    # Initialize EMA model if requested
    temporal_model_ema = None
    if ema_alpha < 1.0:
        print(f"Initializing EMA model with alpha={ema_alpha}")
        temporal_model_ema = copy.deepcopy(temporal_model)
        temporal_model_ema.to(device)
        temporal_model_ema.eval() # EMA model is only for inference
    if loss_header_written:
        try:
            existing_loss_df = pd.read_csv(training_loss_path)
            existing_loss_df = existing_loss_df.reindex(columns=loss_columns)
            existing_loss_df.to_csv(training_loss_path, index=False)
            training_loss = existing_loss_df.get("train_loss", pd.Series([], dtype=float)).tolist()
            if "val_loss_1step" in existing_loss_df.columns:
                validation_loss_1step = existing_loss_df["val_loss_1step"].tolist()
            else:
                validation_loss_1step = existing_loss_df.get("val_loss", pd.Series([], dtype=float)).tolist()
            if use_multistep_metrics:
                if "val_loss_multistep" in existing_loss_df.columns:
                    validation_loss_multistep = existing_loss_df["val_loss_multistep"].tolist()
                else:
                    validation_loss_multistep = existing_loss_df.get("val_loss", pd.Series([], dtype=float)).tolist()
        except Exception as exc:
            print(f"Warning: could not load existing loss history from {training_loss_path}: {exc}")
            training_loss = []
            validation_loss_1step = []
            validation_loss_multistep = [] if use_multistep_metrics else None
    iteration_time = [] # List to store time taken per iteration
    
    # Initialise per-burden NLL tracking
    num_burdens = len(dataset_directory.columns_for_prediction)
    train_nll_per_burden = [] # list of numpy arrays, one per iteration
    val_nll_per_burden = [] # list of numpy arrays, one per iteration

    start_iteration = 0
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        ckpt = torch.load(resume_from_checkpoint, map_location=device)
        temporal_model.load_state_dict(ckpt["temporal_model"])
        interaction_encoder.load_state_dict(ckpt["interaction_encoder"])
        if local_profile_encoder is not None and ckpt["local_profile_encoder"] is not None:
            local_profile_encoder.load_state_dict(ckpt["local_profile_encoder"])
        
        # Handle both single and dual optimizer checkpoints
        ckpt_optimiser = ckpt["optimiser"]
        if isinstance(optimiser, tuple):
            opt_muon, opt_adamw = optimiser
            if isinstance(ckpt_optimiser, dict) and "muon" in ckpt_optimiser:
                if opt_muon is not None and ckpt_optimiser["muon"] is not None:
                    opt_muon.load_state_dict(ckpt_optimiser["muon"])
                opt_adamw.load_state_dict(ckpt_optimiser["adamw"])
            else:
                # Old checkpoint format - try to load as adamw only
                opt_adamw.load_state_dict(ckpt_optimiser)
        else:
            optimiser.load_state_dict(ckpt_optimiser)
        
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        start_iteration = ckpt["iteration"] + 1
        if start_iteration > 0: print("Warning: Resuming mid-run. Changes to num_epochs (from original run) will alter the OneCycleLR schedule geometry compared to the original run. This may mean an unusual learning curve as a result")
        # Scheduler state is intentionally NOT restored from the checkpoint.
        # OneCycleLR embeds total_steps in its state dict; restoring it would
        # lock us into the old step budget even if num_epochs has changed.
        # Instead we build a fresh scheduler for the remaining epochs below.

        # 20/01/26
        # Ensuring loss_history.csv is continued correctly
        # The csv may contain
        # - extra rows if wrote to CSV then crashed before saving checkpoint
        # - fewer rows if checkpoint saved but not yet written to CSV
        # We align the CSV to match the checkpoint iteration
        if os.path.exists(training_loss_path):
            try:
                existing_loss_df = pd.read_csv(training_loss_path)
                if len(existing_loss_df) > start_iteration:
                    existing_loss_df = existing_loss_df.iloc[:start_iteration]
                existing_loss_df = existing_loss_df.reindex(columns=loss_columns)
                existing_loss_df.to_csv(training_loss_path, index=False)
                loss_header_written = True
                training_loss = existing_loss_df.get("train_loss", pd.Series([], dtype=float)).tolist()
                if "val_loss_1step" in existing_loss_df.columns:
                    validation_loss_1step = existing_loss_df["val_loss_1step"].tolist()
                else:
                    validation_loss_1step = existing_loss_df.get("val_loss", pd.Series([], dtype=float)).tolist()
                if use_multistep_metrics:
                    if "val_loss_multistep" in existing_loss_df.columns:
                        validation_loss_multistep = existing_loss_df["val_loss_multistep"].tolist()
                    else:
                        validation_loss_multistep = existing_loss_df.get("val_loss", pd.Series([], dtype=float)).tolist()
            except Exception as exc:
                print(f"Warning: could not align existing loss history from {training_loss_path}: {exc}")
                # If we can't read/trim, start fresh to avoid corrupt history.
                try:
                    os.replace(training_loss_path, training_loss_path + ".corrupt")
                    print(f"Moved corrupt loss history to {training_loss_path}.corrupt")
                except Exception:
                    pass
                loss_header_written = False
                training_loss = []
                validation_loss_1step = []
                validation_loss_multistep = [] if use_multistep_metrics else None

    ##################
    # Torch Compile  #
    ##################
    # Compile only after any resume checkpoint has been restored into the
    # original modules. This avoids state_dict key mismatches with OptimisedModule.
    if use_compile and hasattr(torch, "compile"):
        try:
            try:
                import torch._dynamo as torch_dynamo
                torch_dynamo.config.suppress_errors = True
            except Exception:
                pass
            print("Compiling models with torch.compile...")
            compiled_temporal_model = torch.compile(temporal_model, dynamic=True)
            compiled_interaction_encoder = torch.compile(interaction_encoder, dynamic=True)
            compiled_local_profile_encoder = None
            if local_profile_encoder is not None:
                compiled_local_profile_encoder = torch.compile(local_profile_encoder, dynamic=True)

            temporal_model = compiled_temporal_model
            interaction_encoder = compiled_interaction_encoder
            if local_profile_encoder is not None:
                local_profile_encoder = compiled_local_profile_encoder
        except Exception as exc:
            print(f"Warning: torch.compile failed, continuing in eager mode: {exc}")

    # Set up autocast context
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if use_bfloat16 else contextlib.nullcontext()

    # Calculate steps
    steps_per_epoch = (len(train_loader) + gradient_accumulation_steps - 1) // gradient_accumulation_steps
    total_steps = steps_per_epoch * num_epochs

    if optimiser_type == "muon":
        opt_muon, opt_adamw = optimiser
        
        # Define the shared scheduler factory
        def get_shared_scheduler(optimizer, peak_lr, final_lr_ratio):
            warmup_steps = int(0.05 * total_steps)
            decay_steps = total_steps - warmup_steps
        
            # Linear Warmup: 0 to peak_lr
            warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        
            # Cosine Decay: peak_lr to (final_lr_ratio * peak_lr)
            decay = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=final_lr_ratio * peak_lr)
        
            return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])

        # Apply the SAME schedule to both
        # Apply the schedules: Muon decays to 0, AdamW stays at 10%
        scheduler_muon = get_shared_scheduler(opt_muon, muon_lr, final_lr_ratio=0.0) if opt_muon is not None else None
        scheduler_adamw = get_shared_scheduler(opt_adamw, adamw_lr, final_lr_ratio=0.1)
        
        scheduler = (scheduler_muon, scheduler_adamw)
    else:
        _, opt_adamw = optimiser
        if scheduler_type == "cosine":
            warmup_steps = int(cosine_warmup_fraction * total_steps)
            if total_steps > 1:
                warmup_steps = min(max(warmup_steps, 0), total_steps - 1)
            else:
                warmup_steps = 0

            if warmup_steps > 0:
                warmup = LinearLR(
                    opt_adamw,
                    start_factor=1e-8,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                decay = CosineAnnealingLR(
                    opt_adamw,
                    T_max=max(1, total_steps - warmup_steps),
                    eta_min=cosine_eta_min,
                )
                scheduler_adamw = SequentialLR(
                    opt_adamw,
                    schedulers=[warmup, decay],
                    milestones=[warmup_steps],
                )
            else:
                scheduler_adamw = CosineAnnealingLR(
                    opt_adamw,
                    T_max=max(1, total_steps),
                    eta_min=cosine_eta_min,
                )
        else:
            scheduler_adamw = OneCycleLR(
                opt_adamw,
                max_lr=lr,
                total_steps=total_steps,
            )
        scheduler = (None, scheduler_adamw)

    pbar_epochs = tqdm(range(start_iteration, num_epochs), desc="Submitting Epochs", dynamic_ncols=True, position=0)
    
    for epoch in pbar_epochs:
        # Setting all models to training mode
        temporal_model.train()
        interaction_encoder.train()
        if local_profile_encoder is not None:
            local_profile_encoder.train()

        # Use a global batch counter so thinning is consistent across epochs.
        batches_per_epoch = len(train_loader)

        train_loss = 0.0 # Accumulator for this iterations, training loss
        # per-iteration accumulator for per-burden NLL
        train_nll_acc = torch.zeros(num_burdens, dtype=torch.float32, device="cpu")

        # Tracking time per iteration
        iteration_start_time = datetime.now()

        # Zero gradients at the start of each epoch
        if isinstance(optimiser, tuple):
            opt_muon, opt_adamw = optimiser
            if opt_muon is not None:
                opt_muon.zero_grad()
            opt_adamw.zero_grad()
        else:
            optimiser.zero_grad()

        # Inner progress bar for samples in current epoch
        pbar_samples = tqdm(total=len(train_loader) * batch_size, desc=f"Epoch {epoch+1}/{num_epochs}", 
                            dynamic_ncols=True, leave=False, unit="samples", position=1)

        # Iterate over batches from the training DataLoader
        for i, batch in enumerate(train_loader):
            global_batch_idx = epoch * batches_per_epoch + i
            batch_loss = torch.tensor(0.0, device=device) # Initialise batch loss

            # Unpack the batch (IMPORTANTLY THIS IS DEFINED BY THE ORDER OF columns_for_prediction and columns_for_context)
            static, current_t_batch, current_dow_batch, *contexts_and_targets = batch

            num_context = len(dataset_directory.columns_for_context)

            # Isolating the context and target tensors
            # contexts_and_targets are all the remaining tensors after the first three fixed ones
            contexts = contexts_and_targets[:num_context] # Takes the first num_context tensors, each [B, M, context_size]
            y_t_list = contexts_and_targets[num_context:] # Takes the remaining tensors each [B, M]

            # Move to device
            static = static.to(device)
            current_t_batch = current_t_batch.to(device)
            current_dow_batch = current_dow_batch.to(device)
            contexts = [c.to(device) for c in contexts]
            y_t_list = [y.to(device) for y in y_t_list]

            # --- Poisson Noise Data Augmentation (training only) ---
            # Perturb the context history of discrete/count (NB) variables by sampling
            # from Poisson(lambda = raw_count). Targets (y_t_truth) are never modified.
            # Disabled for validation — see validation loop below.
            apply_augmented_context = (
                poisson_noise_context
                and (augmentation_thinning_factor <= 1 or global_batch_idx % augmentation_thinning_factor == 0)
            )
            if apply_augmented_context:
                contexts = apply_poisson_noise_to_contexts(
                    contexts=contexts,
                    columns_for_context=dataset_directory.columns_for_context,
                    columns_with_data=dataset_directory.columns_with_data,
                    temporal_scalers=dataset_directory.temporal_scalers,
                    prediction_distribution_types=dataset_directory.prediction_distribution_types,
                    columns_for_prediction=dataset_directory.columns_for_prediction,
                    device=device,
                    model_type=model_type,
                )

            with autocast_ctx:
                context_combined = torch.cat(contexts, dim=2) # [B, M, num_context * context_size]
                
                # y_t_list prior to stacking is a list of [B, M] for each burden
                # It would be y_t_list = [
                # [B, M]  # burden 1
                # [B, M]  # burden 2
                # ...
                # [B, M]  # burden N
                # ]
                # After stacking along dim=-1 we get [B, M, columns_for_prediction]
                y_t_truth = torch.stack(y_t_list, dim=-1)

                if local_profile_encoder is not None:
                    # If we are using a GNN for LPE then we want to create a batched graph
                    # If using MLP, we can directly pass the static features tensor
                    if spatial_encoder_type == "gnn":
                        spatial_encoder_batch = get_batched_graph(static, spatial_graph_data, device)
                        h_spatial = local_profile_encoder(spatial_encoder_batch)
                    else:
                        h_spatial = local_profile_encoder(static)  # [B, M, static_spatial_dim] 

                    # Interaction encoder always uses GNN so we create batched graph
                    temporal_encoder_batch = get_batched_graph(context_combined, temp_graph_data, device)
                    h_temporal = interaction_encoder(temporal_encoder_batch)

                elif local_profile_encoder is None and lpe_ablation:
                    # If spatial ablation (meaning no static features used) we only use context features
                    combined_node_features = torch.cat([context_combined], dim=2)
                    combined_batch = get_batched_graph(combined_node_features, temp_graph_data, device)
                    
                    # Forward pass through interaction encoder
                    h_combined = interaction_encoder(combined_batch)
                
                elif local_profile_encoder is None and not lpe_ablation:
                    # If no local profile encoder and no spatial ablation, combine static and context features
                    combined_node_features = torch.cat([static, context_combined], dim=2)
                    combined_batch = get_batched_graph(combined_node_features, temp_graph_data, device)

                    # Forward pass through interaction encoder
                    h_combined = interaction_encoder(combined_batch)
                
                # Create a per-sample, per-node, day-of-week feature tensor
                dow_oh = F.one_hot(current_dow_batch.long(), num_classes=7).float() # One-hot encode day of week
                day_of_week = dow_oh.unsqueeze(1).repeat(1, dataset_directory.num_geographies, 1) # shape [B, M, 7]

                feature_list = []
                # If we want to include the latest context values as features
                if include_latest_context:
                    # contexts -> list of [B, M, context_size]
                    if len(contexts) == 0:
                        latest_vals = None
                    else:
                        # produce [B, M, num_context]
                        latest_vals = torch.cat([c[:, :, -1:].contiguous() for c in contexts], dim=2)
                    if latest_vals is not None:
                        feature_list.append(latest_vals)

                ######################################
                # Creating the Latent Representation #
                ######################################
                if local_profile_encoder is not None:
                    feature_list.extend([h_spatial, h_temporal, day_of_week])
                else:
                    feature_list.extend([h_combined, day_of_week])

                # Final concatenation
                latent_input = torch.cat(feature_list, dim=-1)
                    
                # --- Multistep Loss Computation ---
                batch_loss = torch.tensor(0.0, device=device)
                primary_per_burden_nll = None
                temporal_heads = temporal_model._orig_mod if hasattr(temporal_model, "_orig_mod") else temporal_model
                
                for step_idx in range(multistep_k):
                    if hasattr(temporal_heads, "__getitem__"):
                        head_model = temporal_heads[step_idx]
                    else:
                        head_model = temporal_heads
                    if multistep_k == 1:
                        target_k = y_t_truth
                    else:
                        target_k = y_t_truth[:, :, step_idx, :]
                    
                    if model_type in ("flow_matching", "diffusion", "normalising_flow", "normalising_flow_nsf", "normalising_flow_maf", "normalising_flow_cnf"):
                        step_loss = head_model.training_step(latent_input, target_k)
                        step_nll = torch.full((num_burdens,), float("nan"), device=device)
                    elif model_type == "variational_flow_matching":
                        is_epoch_diag = (epoch % 5 == 0 and i == 0 and step_idx == 0)
                        step_loss, step_nll = head_model.training_step(latent_input, target_k, is_epoch_diag=is_epoch_diag)
                    else:
                        mu_pred, var_pred = head_model(latent_input)
                        step_loss, step_nll = MixedLoss(mu_pred, var_pred, target_k, dataset_directory.prediction_distribution_types)
                    
                    weight = multistep_weights[step_idx] if step_idx < len(multistep_weights) else 0.0
                    batch_loss += step_loss * weight
                    
                    if step_idx == 0:
                        primary_per_burden_nll = step_nll
                
                per_burden_nll = primary_per_burden_nll

            accumulated_loss = batch_loss / gradient_accumulation_steps
            accumulated_loss.backward()

            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                if params_to_clip and gradient_clip_value is not None:
                    torch.nn.utils.clip_grad_value_(params_to_clip, gradient_clip_value)
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params_to_clip, max_grad_norm)
                if isinstance(optimiser, tuple):
                    opt_muon, opt_adamw = optimiser
                    if opt_muon is not None:
                        opt_muon.step()
                    opt_adamw.step()
                    if isinstance(scheduler, tuple):
                        scheduler_muon, scheduler_adamw = scheduler
                        if scheduler_muon is not None:
                            scheduler_muon.step()
                        scheduler_adamw.step()
                    # Zero grads for both optimizers
                    if opt_muon is not None:
                        opt_muon.zero_grad()
                    opt_adamw.zero_grad()
                else:
                    optimiser.step()
                    scheduler.step()
                    optimiser.zero_grad()

            # Update EMA model weights
            if temporal_model_ema is not None:
                with torch.no_grad():
                    for p, p_ema in zip(temporal_model.parameters(), temporal_model_ema.parameters()):
                        p_ema.data.mul_(1.0 - ema_alpha).add_(p.data, alpha=ema_alpha)
            
            # Accumulate both total loss and per-burden NLLs
            current_batch_loss = batch_loss.item()
            train_loss += current_batch_loss
            train_nll_acc += per_burden_nll.detach().cpu()

            pbar_samples.update(batch_size)
# Get LR for display
        if isinstance(scheduler, tuple):
            scheduler_muon, scheduler_adamw = scheduler
            display_lr = scheduler_adamw.get_last_lr()[0] if scheduler_adamw is not None else (scheduler_muon.get_last_lr()[0] if scheduler_muon is not None else 0.0)
        else:
            display_lr = scheduler.get_last_lr()[0]
        
        pbar_samples.set_postfix({
            'Loss': f"{current_batch_loss:.4f}",
            'Avg': f"{train_loss/(i+1):.4f}",
            'LR': f"{display_lr:.2e}"
            })

        pbar_samples.close()

        # Average the accumulated loss over the number of training batches
        train_loss /= num_batches
        training_loss.append(train_loss)

        # Average per-burden NLL for this iteration and store (average over batches)
        train_iter_nll = (train_nll_acc / num_batches).numpy()
        train_nll_per_burden.append(train_iter_nll)

        ##############
        # VALIDATION #
        ##############
        should_validate = ((epoch + 1) % validation_thinning_factor == 0) or ((epoch + 1) == num_epochs)

        if should_validate:
            # Switch all models to evaluation mode (disables dropout, etc.)
            temporal_model.eval()
            interaction_encoder.eval()

            if local_profile_encoder is not None:
                local_profile_encoder.eval()

            # Sample-weighted accumulators over the entire validation set.
            val_loss_1step_weighted_sum = 0.0
            val_loss_multistep_weighted_sum = 0.0
            val_nll_weighted_sum = torch.zeros(num_burdens, dtype=torch.float32, device='cpu')
            val_sample_count = 0

            with torch.inference_mode(): # Lower-overhead eval mode, no gradients or version counter updates
                for batch in val_loader:
                    with autocast_ctx:
                        # Unpack the batch
                        static, current_t_batch, current_dow_batch, *contexts_and_targets = batch
                        num_context = len(dataset_directory.columns_for_context)

                        # Split into lists
                        contexts = contexts_and_targets[:num_context] # list of tensors, each [B, M, context_size]
                        y_t_list = contexts_and_targets[num_context:] # list of tensors, each [B, M]

                        # Move to device
                        static = static.to(device)
                        current_t_batch = current_t_batch.to(device)
                        current_dow_batch = current_dow_batch.to(device)
                        contexts = [c.to(device) for c in contexts]
                        y_t_list = [y.to(device) for y in y_t_list]

                        # Combine all context series
                        # Result: [B, M, num_context * past_context_size]
                        context_combined = torch.cat(contexts, dim=2)
                        y_t_truth = torch.stack(y_t_list, dim=-1)

                        if local_profile_encoder is not None:
                            if spatial_encoder_type == "gnn":
                                # Do not cache full-validation-size graph shells: they are huge and one-shot.
                                spatial_encoder_batch = get_batched_graph(static, spatial_graph_data, device, use_data_cache=False, use_topology_cache=False)
                                h_spatial = local_profile_encoder(spatial_encoder_batch)
                            else:
                                h_spatial = local_profile_encoder(static)  # [B, M, static_spatial_dim] 

                            temporal_encoder_batch = get_batched_graph(context_combined, temp_graph_data, device, use_data_cache=False, use_topology_cache=False)
                            h_temporal = interaction_encoder(temporal_encoder_batch)
                        
                        elif local_profile_encoder is None and lpe_ablation:
                            combined_node_features = torch.cat([context_combined], dim=2)
                            combined_batch = get_batched_graph(combined_node_features, temp_graph_data, device, use_data_cache=False, use_topology_cache=False)
                            h_combined = interaction_encoder(combined_batch)
                        else:
                            combined_node_features = torch.cat([static, context_combined], dim=2)
                            combined_batch = get_batched_graph(combined_node_features, temp_graph_data, device, use_data_cache=False, use_topology_cache=False)
                            h_combined = interaction_encoder(combined_batch)

                        dow_oh = F.one_hot(current_dow_batch.long(), num_classes=7).float()
                        day_of_week = dow_oh.unsqueeze(1).repeat(1, dataset_directory.num_geographies, 1) # shape [B, M, 7]
                        
                        feature_list = []
                        if include_latest_context:
                            latest_vals = torch.cat([c[:, :, -1:].contiguous() for c in contexts], dim=2)
                            feature_list.append(latest_vals)

                        if local_profile_encoder is not None:
                            feature_list.extend([h_spatial, h_temporal, day_of_week])
                        else:
                            feature_list.extend([h_combined, day_of_week])

                        # Final concatenation
                        y_aug = torch.cat(feature_list, dim=-1)
                        
                        batch_loss_1step = None
                        batch_loss_multistep = torch.tensor(0.0, device=device)
                        primary_per_burden_nll = None
                        temporal_heads = temporal_model._orig_mod if hasattr(temporal_model, "_orig_mod") else temporal_model

                        for step_idx in range(multistep_k):
                            if hasattr(temporal_heads, "__getitem__"):
                                head_model = temporal_heads[step_idx]
                            else:
                                head_model = temporal_heads
                            if multistep_k == 1:
                                target_k = y_t_truth
                            else:
                                target_k = y_t_truth[:, :, step_idx, :]
                            
                            if model_type in ("flow_matching", "diffusion", "normalising_flow", "normalising_flow_nsf", "normalising_flow_maf", "normalising_flow_cnf"):
                                step_loss = head_model.training_step(y_aug, target_k)
                                step_nll = torch.full((num_burdens,), float("nan"), device=device)
                            elif model_type == "variational_flow_matching":
                                step_loss, step_nll = head_model.training_step(y_aug, target_k)
                            else:
                                mu_pred, var_pred = head_model(y_aug)
                                step_loss, step_nll = MixedLoss(mu_pred, var_pred, target_k, dataset_directory.prediction_distribution_types)
                            
                            if step_idx == 0:
                                batch_loss_1step = step_loss
                                primary_per_burden_nll = step_nll
                            
                            weight = multistep_weights[step_idx] if step_idx < len(multistep_weights) else 0.0
                            batch_loss_multistep += step_loss * weight

                    # static has shape [B, M, ...], so dim 0 is the number of samples in this validation batch.
                    val_batch_sample_count = int(static.shape[0])
                    val_batch_loss_1step_weighted = batch_loss_1step.item() * val_batch_sample_count
                    val_batch_loss_multistep_weighted = batch_loss_multistep.item() * val_batch_sample_count
                    val_batch_nll_weighted = primary_per_burden_nll.detach().cpu() * val_batch_sample_count

                    # Accumulate weighted sums so final metrics are true sample-wise averages over the full validation set.
                    val_loss_1step_weighted_sum += val_batch_loss_1step_weighted
                    val_loss_multistep_weighted_sum += val_batch_loss_multistep_weighted
                    val_nll_weighted_sum += val_batch_nll_weighted
                    val_sample_count += val_batch_sample_count

            # Full-validation averages across all samples, regardless of batch split.
            val_loss_1step = val_loss_1step_weighted_sum / max(val_sample_count, 1)
            val_loss_multistep = val_loss_multistep_weighted_sum / max(val_sample_count, 1)
            final_val_loss = val_loss_multistep if use_multistep_metrics else val_loss_1step

            # Average per-burden NLL for this iteration over all validation samples.
            val_iter_nll = (val_nll_weighted_sum / max(val_sample_count, 1)).numpy()
            val_nll_per_burden.append(val_iter_nll)
            
            validation_loss_1step.append(val_loss_1step)
            if use_multistep_metrics:
                validation_loss_multistep.append(val_loss_multistep)
        else:
            val_loss_1step = float('nan')
            val_loss_multistep = float('nan')
            validation_loss_1step.append(val_loss_1step)
            if use_multistep_metrics:
                validation_loss_multistep.append(val_loss_multistep)
            val_iter_nll = np.full(num_burdens, np.nan)
            val_nll_per_burden.append(val_iter_nll)

        current_lr_muon = np.nan
        current_lr_adamw = np.nan
        if isinstance(scheduler, tuple):
            scheduler_muon, scheduler_adamw = scheduler
            current_lr_muon = scheduler_muon.get_last_lr()[0] if scheduler_muon is not None else np.nan
            current_lr_adamw = scheduler_adamw.get_last_lr()[0] if scheduler_adamw is not None else np.nan
        else:
            current_lr_adamw = scheduler.get_last_lr()[0]

        lr_history_muon.append(current_lr_muon)
        lr_history_adamw.append(current_lr_adamw)

        loss_row = {
            "train_loss": [train_loss],
            "val_loss_1step": [val_loss_1step],
            "lr_muon": [current_lr_muon],
            "lr_adamw": [current_lr_adamw],
        }
        if use_multistep_metrics:
            loss_row["val_loss_multistep"] = [val_loss_multistep]

        pd.DataFrame(loss_row).to_csv(
            training_loss_path,
            mode="a",
            header=not loss_header_written,
            index=False
        )
        loss_header_written = True

        # Getting the time taken for this iteration
        iteration_end_time = datetime.now()
        iteration_duration = (iteration_end_time - iteration_start_time).total_seconds()
        iteration_time.append(iteration_duration)

        # Update epoch progress bar with summaries
        # Get LR for display
        if isinstance(scheduler, tuple):
            scheduler_muon, scheduler_adamw = scheduler
            display_lr = current_lr_adamw if not np.isnan(current_lr_adamw) else (current_lr_muon if not np.isnan(current_lr_muon) else 0.0)
        else:
            display_lr = current_lr_adamw
        
        pbar_epochs.set_postfix({
            'TrainAvg': f"{train_loss:.4f}",
            'Val1Step': f"{val_loss_1step:.4f}",
            'ValMulti': f"{val_loss_multistep:.4f}" if use_multistep_metrics else 'N/A',
            'BestVal': f"{best_val_loss:.4f}" if best_val_loss != float('inf') else "N/A",
            'LR': f"{display_lr:.2e}"
        })
        
        checkpoint_path = os.path.join(trained_models_subfolder, "checkpoint_latest.pth")
        
        # Prepare optimizer and scheduler state dicts
        if isinstance(optimiser, tuple):
            opt_muon, opt_adamw = optimiser
            optimiser_state = {
                "muon": opt_muon.state_dict() if opt_muon is not None else None,
                "adamw": opt_adamw.state_dict()
            }
        else:
            optimiser_state = optimiser.state_dict()
        
        if isinstance(scheduler, tuple):
            scheduler_muon, scheduler_adamw = scheduler
            scheduler_state = {
                "muon": scheduler_muon.state_dict() if scheduler_muon is not None else None,
                "adamw": scheduler_adamw.state_dict()
            }
        else:
            scheduler_state = scheduler.state_dict()
        
        torch.save({
            "iteration": epoch,
            "temporal_model": temporal_model.state_dict(),
            "interaction_encoder": interaction_encoder.state_dict(),
            "local_profile_encoder": (
                local_profile_encoder.state_dict() if local_profile_encoder is not None else None
            ),
            "optimiser": optimiser_state,
            "scheduler": scheduler_state,
            "best_val_loss": best_val_loss,
        }, checkpoint_path)

        current_val_loss = val_loss_multistep if use_multistep_metrics else val_loss_1step
        if should_validate and current_val_loss < best_val_loss:
            best_val_loss = current_val_loss

            temp_model_path = f"{trained_models_subfolder}best_temporal_model.pth"
            # If EMA is active, save the EMA weights as the "best" model for smoother inference.
            # We assume temporal_model_ema has been updated in the loop above.
            if 'temporal_model_ema' in locals() and temporal_model_ema is not None:
                torch.save(temporal_model_ema[0].state_dict(), temp_model_path)
            else:
                torch.save(temporal_model[0].state_dict(), temp_model_path)

            interaction_encoder_path = f"{trained_models_subfolder}best_interaction_encoder.pth"
            torch.save(interaction_encoder.state_dict(), interaction_encoder_path)

            if local_profile_encoder is not None:
                local_profile_encoder_path = f"{trained_models_subfolder}best_location_specific_encoder.pth"
                torch.save(local_profile_encoder.state_dict(), local_profile_encoder_path)

    # Get the total training time by summing iteration times
    total_training_time = sum(iteration_time)
    print(f"It took, {total_training_time/3600:.2f} hours to train the model OR {total_training_time/60:.2f} minutes.")

    # Saving iteration times
    iteration_time_path = os.path.join(trained_models_subfolder, "iteration_times.csv")
    pd.DataFrame({"iteration_time_seconds": iteration_time}).to_csv(iteration_time_path, index=False)

    # Build per-burden dataframe from per-iteration lists and save
    burden_names = dataset_directory.columns_for_prediction
    train_arr = np.stack(train_nll_per_burden, axis=0) if len(train_nll_per_burden) > 0 else np.zeros((0, num_burdens))
    val_arr = np.stack(val_nll_per_burden, axis=0) if len(val_nll_per_burden) > 0 else np.zeros((0, num_burdens))

    n_iters_recorded = train_arr.shape[0]
    data = {"iteration": np.arange(n_iters_recorded)}
    for i, name in enumerate(burden_names):
        data[f"train_{name}_nll"] = train_arr[:, i] if train_arr.size else np.full(n_iters_recorded, np.nan)
        data[f"val_{name}_nll"] = val_arr[:, i] if val_arr.size else np.full(n_iters_recorded, np.nan)

    per_burden_df = pd.DataFrame(data)
    burden_loss_path = os.path.join(trained_models_subfolder, "per_burden_loss.csv")
    per_burden_df.to_csv(burden_loss_path, index=False)

    # Plotting train and validation loss curves
    plot_train_val_loss(
        training_loss=training_loss,
        validation_loss_1step=validation_loss_1step,
        validation_loss_multistep=validation_loss_multistep,
        trained_models_subfolder=trained_models_subfolder
    )

    plot_lr_vs_iteration(
        lr_history_muon=lr_history_muon,
        lr_history_adamw=lr_history_adamw,
        trained_models_subfolder=trained_models_subfolder
    )

    final_temporal_model_path = os.path.join(trained_models_subfolder, "final_temporal_model.pth")
    torch.save(temporal_model[0].state_dict(), final_temporal_model_path)

    final_interaction_encoder_path = os.path.join(trained_models_subfolder, "final_interaction_encoder.pth")
    torch.save(interaction_encoder.state_dict(), final_interaction_encoder_path)

    if local_profile_encoder is not None:
        final_location_specific_encoder_path = os.path.join(trained_models_subfolder, "final_location_specific_encoder.pth")
        torch.save(local_profile_encoder.state_dict(), final_location_specific_encoder_path)

    return temporal_model, local_profile_encoder, interaction_encoder, final_val_loss
