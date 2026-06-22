import torch
import torch.nn as nn
import numpy as np

import zuko

# Normalising Flow model using Zuko's CNF (Continuous Normalizing Flow / NODEs)
# CNF represents the transformation as an ODE, which is solved during training and inference.
# It is often more flexible and parameter-efficient than discrete flows.


class CNFFlowModel(nn.Module):
    def __init__(self, hidden_layers, input_dim, output_dim,
                 prediction_distribution_types, scalers,
                 sigma=0.1, dropout=0.2,
                 activation=nn.GELU):
        """
        Normalising Flow Model using Zuko's CNF.

        Args:
            hidden_layers: List of hidden layer widths for the ODE velocity field
            input_dim: Dimension of the context features (C)
            output_dim: Dimension of the target (D)
            prediction_distribution_types: List of 'nb' or 'lognormal'
            scalers: List of StandardScaler objects for each output column
            sigma: Noise level for dequantisation (unused by the flow itself,
                   kept for API consistency with FlowMatchingModel)
            dropout: Dropout for velocity field (unused by Zuko's built-in
                     architectures, kept for API consistency)
            activation: Activation function for the velocity field (default: nn.GELU)
        """
        super().__init__()
        self.output_dim = output_dim
        self.prediction_distribution_types = prediction_distribution_types

        # Build the normalising flow
        # CNF in Zuko takes features, context, and optional architecture params
        self.flow = zuko.flows.CNF(
            features=output_dim,
            context=input_dim,
            hidden_features=hidden_layers,
            activation=activation,
        )

        self.sigma = sigma

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
        """Dequantizes raw counts (if training) and then scales them.

        - During training, adds Uniform(0,1) noise to discrete (nb) columns
        - Scales all columns to model space: (x - mu) / sigma
        """
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
        Full Training Logic: Dequantize -> Scale -> Compute NLL.

        Args:
            context: [B, M, C] context features
            x1_raw: [B, M, D] target data (raw integer counts for 'nb')
        Returns:
            loss: scalar negative log-likelihood loss

        Call Flow & Backprop:
        1. Dequantization: Raw integer counts (x1_raw) are made continuous
           with Unif(0, 1) noise.
        2. Scaling: Targets are scaled using training means and std devs.
        3. NLL: The flow computes -log p(x_scaled | context) and we average.
           - Note: The `log_prob` call masks the fact that the Jacobian determinant 
             is computed "lazily" during the transformation pass.

        Backpropagation runs through the CNF velocity field parameters.
        """
        B, M, D = x1_raw.shape

        # 1. Dequantize and scale target
        x1_scaled = self.dequantize(x1_raw)

        # 2. Reshape for flow (combine B and M)
        x1_flat = x1_scaled.view(-1, D)
        context_flat = context.reshape(-1, context.size(-1))

        # 3. Compute negative log-likelihood
        # flow(context) returns a conditional distribution p(x | context)
        dist = self.flow(context_flat)
        log_prob = dist.log_prob(x1_flat)
        loss = -log_prob.mean()

        return loss

    @torch.no_grad()
    def sample(self, context, num_steps=None, device=None):
        """
        Core Inference Entry Point.

        Call Flow:
        1. Passes context through the flow to get a conditional distribution.
        2. Draws one sample per context vector from the learned distribution.
        3. Un-scales and quantizes as in FlowMatchingModel.

        Args:
            context: [B, M, C] or [N, C] context features
            num_steps: Unused, kept for API compatibility with FlowMatchingModel.
            device: Device override.
        Returns:
            Samples in raw (un-scaled, quantized for discrete) space.

        Note: Backprop does NOT run through sampling.
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

        # Get conditional distribution and sample
        dist = self.flow(context_flat)
        # sample((1,)) returns [1, N, D], squeeze to [N, D]
        x1_flat = dist.sample((1,)).squeeze(0)

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
            torch.floor(x1_unscaled.clamp(min=0)),  # Floor discrete counts, ensure non-negative
            x1_unscaled  # Keep continuous as is
        )

        if is_3d:
            return x1_quantized.view(B, M, -1)
        return x1_quantized

    def forward(self, context, num_steps=None):
        if self.training:
            raise RuntimeError("Use model.training_step(context, targets) for training, not model(context)")
        return self.sample(context, num_steps=num_steps, device=context.device)
