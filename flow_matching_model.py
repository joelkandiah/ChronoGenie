import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Based on the paper: https://arxiv.org/abs/2412.06264
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper


## Class to represent the learning of the velocity field. In particular this module should take as an input the context vector (LIE + LPE embeddings + time + current state) and output the velocity vector.
## The velocity field is a function v(t, x_t, context) that represents the direction and magnitude of the flow at a given time t and state x_t.
## Each timestep we solve the ODE at will call this module
class VelocityField(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, dropout=0.1):
        """
        Velocity field network v(t, x_t, context).
        
        Args:
            input_dim: Dimension of context: (x_t + context + time)
            hidden_layers: List of hidden layer dimensions
            output_dim: Dimension of the velocity (same as x_t)
            dropout: Dropout probability
        """
        super().__init__()
        layers = []
        prev_dim = input_dim + 1 + output_dim # context + t + x_t
        
        for layer_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, layer_dim))
            layers.append(nn.LayerNorm(layer_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = layer_dim
        
        # Output layer for velocity
        self.net = nn.Sequential(*layers)
        self.velocity_head = nn.Linear(prev_dim, output_dim)

    def forward(self, t, x, context):
        """
        Args:
            t: [N, 1] scalar time
            x: [N, D] or [B, M, D] current state
            context: [N, C] or [B, M, C] context features
        Returns:
            v: Same shape as x
        
        Call Flow & Backprop:
        1. During TRAINING (via FlowMatchingModel.training_step):
           - 'x' is the sampled intermediate state 'xt'
           - 't' is the sampled time from IndependentConditionalFlowMatcher
           - Backward pass: Gradients flow through 'v' -> self.net -> self.velocity_head
             back into the network parameters. NO gradients flow through 'xt', 't', 
             or 'context' (they are treated as targets/inputs).
        2. During INFERENCE (via FlowMatchingModel.sample):
           - Called iteratively (e.g., 50 steps) to solve the ODE.
           - 'x' is the current state of the integration.
           - NO backprop occurs during inference.
        """
        is_3d = (x.ndim == 3)
        if is_3d:
            B, M, D = x.shape
            x = x.view(-1, D)
            context = context.view(-1, context.size(-1))
            
        # Ensure t is [N, 1] and matches batch size of x
        if t.ndim == 0:
            t = t.view(1, 1).expand(x.size(0), 1)
        elif t.ndim == 1:
            if t.size(0) == 1:
                t = t.view(1, 1).expand(x.size(0), 1)
            else:
                t = t.view(-1, 1)
        
        # Concatenate x, t, and context along last dimension
        h = torch.cat([x, t, context], dim=-1)
        
        out = self.net(h)
        v = self.velocity_head(out)
        
        if is_3d:
            v = v.view(B, M, D)
        return v

## Class to represent the flow matching model. This module will take as an input the context vector and the target data and will output the velocity field.
## It will also be responsible for the dequantization of the target data and the scaling of the target data.
## It does this by learning the velocity field v(t, x_t, context) that represents the direction and magnitude of the flow at a given time t and state x_t.
## The loss function is the mean squared error between the predicted velocity and the true velocity.
## The true velocity is the diffence between the scaled target data and the scaled noise.
## The flow matching model is a wrapper around the velocity field and the flow matcher.
## The IndependentConditionalFlowMatcher is a class that is responsible for the sampling of the flow matching components.
## This means it takes in the scaled target data and the scaled noise and outputs the velocity field.
## The velocity field is then used to generate the target data.
class FlowMatchingModel(nn.Module):
    def __init__(self, hidden_layers, input_dim, output_dim, 
                 prediction_distribution_types, scalers, 
                 sigma=0.1, dropout=0.2):
        """
        Flow Matching Model wrapper.
        
        Args:
            hidden_layers: MLP hidden layers for velocity field
            input_dim: Dimension of the context features (C)
            output_dim: Dimension of the target (D)
            prediction_distribution_types: List of 'nb' or 'lognormal'
            scalers: List of StandardScaler objects for each output column
            sigma: Noise level for the flow matcher
            dropout: Dropout for velocity field
        """
        super().__init__()
        self.output_dim = output_dim
        self.prediction_distribution_types = prediction_distribution_types
        
        # Initialize velocity field
        # Note: VelocityField needs (input_dim + 1 + output_dim) as input
        self.v_field = VelocityField(input_dim, hidden_layers, output_dim, dropout=dropout)
        
        self.sigma = sigma
        
        # Initialize native flow matching components
        self.prob_path = CondOTProbPath()
        
        # Solver will be initialized during inference in sample()
        
        # Extract and store scaling parameters as buffers
        means = []
        stds = []
        is_discrete = []
        for i, dist_type in enumerate(prediction_distribution_types):
            if dist_type == 'nb':
                is_discrete.append(1.0)
            else:
                is_discrete.append(0.0)
                
            if scalers[i] is not None and hasattr(scalers[i], 'mean_'):
                means.append(scalers[i].mean_[0])
                stds.append(scalers[i].scale_[0])
            else:
                means.append(0.0)
                stds.append(1.0)
        
        self.register_buffer("means", torch.tensor(means, dtype=torch.float32))
        self.register_buffer("stds", torch.tensor(stds, dtype=torch.float32))
        self.register_buffer("is_discrete", torch.tensor(is_discrete, dtype=torch.float32))

    def scale(self, x_raw, indices=None):
        """Scales raw data using stored means/stds buffers."""
        means = self.means if indices is None else self.means[indices]
        stds = self.stds if indices is None else self.stds[indices]
        
        shape_view = [1] * (x_raw.ndim - 1) + [-1]
        x_scaled = (x_raw - means.view(*shape_view)) / stds.view(*shape_view)
        return x_scaled

    def unscale(self, x_scaled, indices=None):
        """Unscales data using stored means/stds buffers."""
        means = self.means if indices is None else self.means[indices]
        stds = self.stds if indices is None else self.stds[indices]
        
        shape_view = [1] * (x_scaled.ndim - 1) + [-1]
        x_unscaled = x_scaled * stds.view(*shape_view) + means.view(*shape_view)
        return x_unscaled

    def dequantize(self, x_raw):
        """Dequantizes raw counts (if training) and then scales them."""
        # Handle both [B, M, D] and [N, D]
        shape_view = [1] * (x_raw.ndim - 1) + [-1]

        if self.training:
            noise = torch.rand_like(x_raw) * self.is_discrete.view(*shape_view)
            x_deq = x_raw + noise
        else:
            x_deq = x_raw
            
        return self.scale(x_deq)

    def training_step(self, context, x1_raw):
        """ 
        Full Training Logic: Dequantize -> Scale -> Sample Flow -> Predict Velocity -> Compute MSE.
        
        Args:
            context: [B, M, C] context features
            x1_raw: [B, M, D] target data (raw integer counts for 'nb')
        Returns:
            loss: scalar MSE loss
        
        Call Flow & Backprop:
        1. Dequantization: Raw integer counts (x1_raw) are made continuous with Unif(0, 1) noise.
        2. Scaling: Targets are scaled using training means and std devs.
        3. Flow Sampling: The CFM logic samples a random time 't' and interpolates 'xt' 
           between noise and data.
        4. Velocity Prediction: The internal VelocityField predicts 'v_pred' based on ('t', 'xt', 'context').
        5. Loss: MSE is computed against the ground truth flow 'ut'.
        
        Backpropagation runs solely through the VelocityField network parameters.
        """
        B, M, D = x1_raw.shape
        
        # 1. Dequantize and scale target
        x1_scaled = self.dequantize(x1_raw)
        
        # 2. Reshape for flow matching (combine B and M)
        x1_flat = x1_scaled.view(-1, D)
        context_flat = context.reshape(-1, context.size(-1))
        
        # 3. Sample flow matching components
        # We sample Gaussian noise x0 scaled by sigma and interpolate to xt using the optimal transport path.
        # ut is the target velocity (x1 - x0).
        x0 = torch.randn_like(x1_flat) * self.sigma
        
        # Sample time t uniformly for each sample in the batch
        t = torch.rand(x1_flat.size(0), device=x1_flat.device)
        
        # Sample the path (xt = alpha_t * x1 + sigma_t * x0)
        path_sample = self.prob_path.sample(x_0=x0, x_1=x1_flat, t=t)
        xt = path_sample.x_t
        ut = path_sample.dx_t
        
        # 4. Predict velocity
        # v_field handles [N, D] and [N, C] and [N, 1]
        v_pred = self.v_field(t.view(-1, 1), xt, context_flat)
        
        # 5. Compute MSE loss
        loss = F.mse_loss(v_pred, ut)
        
        return loss

    @torch.no_grad()
    def sample(self, context, num_steps=50, device=None):
        """
        Core Inference Entry Point.
        
        Call Flow:
        1. Starts with Gaussian noise x0 ~ N(0, I).
        2. Iteratively calls v_field to get velocity.
        3. Updates state x using Euler integration: x = x + v * dt.
        4. Final result x_1 represents a sample from the target distribution.
        
        Note: Backprop does NOT run through this ODE solver.
        """
        if device is None:
            device = context.device

        # Handle both [B, M, C] and [N, C]
        is_3d = (context.ndim == 3)
        if is_3d:
            B, M, C = context.shape
            context_flat = context.reshape(B * M, -1)
        else:
            N, C = context.shape
            context_flat = context
            
        D = self.output_dim
        num_samples_total = context_flat.size(0)
        
        # Initial noise x0 ~ N(0, sigma^2 * I)
        x0 = torch.randn(num_samples_total, D, device=device) * self.sigma
        
        # Wrap the velocity field.
        model_wrapped = ModelWrapper(self.v_field)
            
        # Solve the ODE from t=0 to t=1
        t_span = torch.linspace(0, 1, num_steps + 1, device=device)
        solver = ODESolver(velocity_model=model_wrapped)
        
        # Pass context as part of model_extras
        x1_flat = solver.sample(
            x_init=x0,
            step_size=None,
            method='euler',
            time_grid=t_span,
            return_intermediates=False,
            context=context_flat
        )
        
        # --- Un-scaling and Quantization ---
        # x1_flat is [N, D] in scaled space.
        # Scale back to raw space: x_raw = x_scaled * std + mean
        x1_unscaled = self.unscale(x1_flat)
        
        # Quantize discrete columns (nb) using floor
        # is_discrete is a buffer [D] with 1.0 for 'nb' and 0.0 for 'lognormal'
        shape_view = [1] * (x1_unscaled.ndim - 1) + [-1]
        discrete_mask = self.is_discrete.view(*shape_view)
        x1_quantized = torch.where(
            discrete_mask > 0.5,
            torch.floor(x1_unscaled.clamp(min=0)), # Floor discrete counts, ensure non-negative
            x1_unscaled # Keep continuous as is
        )
        
        if is_3d:
            return x1_quantized.view(B, M, D)
        return x1_quantized

    def forward(self, context, num_steps=50):
        if self.training:
            raise RuntimeError("Use model.training_step(context, targets) for training, not model(context)")
        return self.sample(context, num_steps=num_steps, device=context.device)
