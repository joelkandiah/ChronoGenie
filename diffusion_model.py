import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Map discrete diffusion timesteps to a fixed sinusoidal feature space.

    This follows the standard transformer-style positional encoding used by many
    diffusion models. The embedding provides the noise predictor with a smooth,
    periodic representation of the current diffusion step.
    """

    def __init__(self, embedding_dim):
        """Create a sinusoidal embedding module.

        Args:
            embedding_dim: Size of the returned time embedding.
        """
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        """Convert integer timesteps into sinusoidal embeddings.

        Args:
            timesteps: Scalar, [N], or [N, 1] diffusion steps.

        Returns:
            A tensor of shape [N, embedding_dim] containing the time features.
        """
        if timesteps.ndim == 0:
            timesteps = timesteps.view(1)

        timesteps = timesteps.float().view(-1, 1)
        half_dim = self.embedding_dim // 2
        if half_dim == 0:
            return timesteps

        device = timesteps.device
        exponent = torch.arange(half_dim, device=device, dtype=timesteps.dtype)
        exponent = -math.log(10000.0) * exponent / max(half_dim - 1, 1)
        freqs = torch.exp(exponent).view(1, -1)
        args = timesteps * freqs

        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class NoisePredictor(nn.Module):
    """MLP that predicts the diffusion noise epsilon_theta(x_t, t, context).

    The network consumes three pieces of information:
    - the current noisy state x_t
    - a time embedding for the diffusion step t
    - the conditioning context produced by the rest of the GENIE pipeline

    This is the learnable denoiser used for DDPM training and DDIM sampling.
    """

    def __init__(self, input_dim, hidden_layers, output_dim, time_embed_dim=128, dropout=0.2):
        """Build the denoising network.

        Args:
            input_dim: Dimension of the conditioning context vector.
            hidden_layers: Hidden layer widths for the MLP trunk.
            output_dim: Number of target dimensions to denoise.
            time_embed_dim: Width of the sinusoidal timestep embedding.
            dropout: Dropout probability applied between hidden layers.
        """
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)

        layers = []
        prev_dim = input_dim + time_embed_dim + output_dim

        for layer_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, layer_dim))
            layers.append(nn.LayerNorm(layer_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = layer_dim

        self.net = nn.Sequential(*layers)
        self.noise_head = nn.Linear(prev_dim, output_dim)

    def forward(self, timesteps, x, context):
        """Predict the noise component for a noisy diffusion state.

        Args:
            timesteps: Diffusion step(s) associated with the current state.
            x: Current noisy sample x_t.
            context: Conditioning features from the GENIE encoders.

        Returns:
            Predicted noise tensor with the same shape as x.
        """
        is_3d = (x.ndim == 3)
        if is_3d:
            batch_size, num_nodes, output_dim = x.shape
            x = x.view(-1, output_dim)
            context = context.view(-1, context.size(-1))

        if timesteps.ndim == 0:
            timesteps = timesteps.view(1)
        if timesteps.ndim == 2 and timesteps.size(-1) == 1:
            timesteps = timesteps.view(-1)
        if timesteps.ndim == 1 and timesteps.size(0) == 1 and x.size(0) > 1:
            timesteps = timesteps.expand(x.size(0))

        time_emb = self.time_embedding(timesteps)
        if time_emb.size(0) == 1 and x.size(0) > 1:
            time_emb = time_emb.expand(x.size(0), -1)

        h = torch.cat([x, time_emb, context], dim=-1)
        out = self.net(h)
        noise = self.noise_head(out)

        if is_3d:
            noise = noise.view(batch_size, num_nodes, output_dim)
        return noise


class DiffusionModel(nn.Module):
    """Custom conditional diffusion wrapper with DDPM training and DDIM sampling.

    The module mirrors the API of the other GENIE temporal models so it can be
    plugged into run_experiment.py, training.py, and testing_sliding_window.py
    without changing the rest of the experiment pipeline.

    Core pieces:
    - dequantize/scale/unscale: matches the existing generative models so raw
      count-like targets are converted into a continuous model space.
    - betas: the forward-process variance schedule. Each beta_t controls how
      much Gaussian noise is injected at step t. Small early betas preserve
      structure; larger late betas destroy information more aggressively.
    - alpha_cumprod: the accumulated product of (1 - beta_t), used to derive
      the closed-form noisy sample x_t during training and sampling.
    - noise_predictor: the MLP that learns epsilon_theta(x_t, t, context).
    """

    def __init__(self, hidden_layers, input_dim, output_dim,
                 prediction_distribution_types, scalers,
                 time_embed_dim=128, num_train_timesteps=1000,
                 beta_start=1e-4, beta_end=2e-2, dropout=0.2):
        """Construct the diffusion model and its variance schedule.

        Args:
            hidden_layers: Hidden layer widths for the denoising MLP.
            input_dim: Conditioning context dimension.
            output_dim: Number of target dimensions to generate.
            prediction_distribution_types: Target distribution metadata used
                to mark discrete count columns for dequantization/quantization.
            scalers: Per-target scalers used to normalize and unnormalize data.
            time_embed_dim: Width of the timestep embedding.
            num_train_timesteps: Number of discrete forward-process steps.
            beta_start: Initial beta value in the linear noise schedule.
            beta_end: Final beta value in the linear noise schedule.
            dropout: Dropout applied inside the denoiser MLP.

        Notes:
            The linear beta schedule defines the forward diffusion process.
            At each step, the model gradually replaces signal with Gaussian
            noise. The cumulative product of alphas determines how much of the
            original sample survives at a given timestep.
        """
        super().__init__()
        self.output_dim = output_dim
        self.prediction_distribution_types = prediction_distribution_types
        self.num_train_timesteps = num_train_timesteps

        self.noise_predictor = NoisePredictor(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            output_dim=output_dim,
            time_embed_dim=time_embed_dim,
            dropout=dropout,
        )

        # Beta schedule for the forward diffusion process: beta_t is the amount
        # of noise injected at step t. We use a simple linear schedule here.
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))

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
        """Standardize raw targets with the stored per-feature statistics."""
        means = self.means if indices is None else self.means[indices]
        stds = self.stds if indices is None else self.stds[indices]

        shape_view = [1] * (x_raw.ndim - 1) + [-1]
        return (x_raw - means.view(*shape_view)) / stds.view(*shape_view)

    def unscale(self, x_scaled, indices=None):
        """Invert the stored standardization back to raw target space."""
        means = self.means if indices is None else self.means[indices]
        stds = self.stds if indices is None else self.stds[indices]

        shape_view = [1] * (x_scaled.ndim - 1) + [-1]
        return x_scaled * stds.view(*shape_view) + means.view(*shape_view)

    def dequantize(self, x_raw):
        """Add uniform noise to discrete columns before scaling.

        Count-like outputs are treated as discrete during training. Adding
        Uniform(0, 1) noise makes them continuous so the diffusion objective
        operates on a smooth space rather than integer-valued atoms.
        """
        shape_view = [1] * (x_raw.ndim - 1) + [-1]

        if self.training:
            noise = torch.rand_like(x_raw) * self.is_discrete.view(*shape_view)
            x_deq = x_raw + noise
        else:
            x_deq = x_raw

        return self.scale(x_deq)

    def _extract(self, values, timesteps, target_shape):
        """Gather timestep-dependent coefficients and broadcast them.

        Args:
            values: 1-D schedule tensor such as sqrt(alpha_bar_t).
            timesteps: Per-sample step indices.
            target_shape: Shape whose rank is used for broadcasting.

        Returns:
            The selected schedule values reshaped for arithmetic with a batch.
        """
        out = values[timesteps.long()]
        while out.ndim < len(target_shape):
            out = out.unsqueeze(-1)
        return out

    def training_step(self, context, x1_raw):
        """Compute the DDPM noise-prediction loss for one training batch.

        Training procedure:
        1. Dequantize and scale the raw targets.
        2. Sample a random diffusion step t for each item in the batch.
        3. Form x_t using the closed-form forward diffusion equation.
        4. Predict the injected noise epsilon with the denoiser MLP.
        5. Minimize mean-squared error between predicted and true noise.

        Args:
            context: Conditioning features produced by the GENIE encoders.
            x1_raw: Raw target values in the original data scale.

        Returns:
            Scalar training loss.
        """
        x1_scaled = self.dequantize(x1_raw)

        x1_flat = x1_scaled.reshape(-1, x1_scaled.size(-1))
        context_flat = context.reshape(-1, context.size(-1))

        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            (x1_flat.size(0),),
            device=x1_flat.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(x1_flat)

        sqrt_alpha_bar = self._extract(self.sqrt_alpha_cumprod, timesteps, x1_flat.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alpha_cumprod, timesteps, x1_flat.shape)
        x_t = sqrt_alpha_bar * x1_flat + sqrt_one_minus_alpha_bar * noise

        noise_pred = self.noise_predictor(timesteps, x_t, context_flat)
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def _get_sampling_timesteps(self, num_steps, device):
        """Build the reverse-time schedule used for DDIM sampling.

        If num_steps matches or exceeds the training horizon, we use every
        discrete timestep. Otherwise we subsample the schedule to get a shorter
        deterministic trajectory, which is the DDIM-style acceleration path.
        """
        total_steps = self.num_train_timesteps
        if num_steps is None or num_steps >= total_steps:
            return torch.arange(total_steps - 1, -1, -1, device=device, dtype=torch.long)

        timesteps = torch.linspace(total_steps - 1, 0, steps=num_steps, device=device)
        timesteps = timesteps.round().long()
        return torch.unique_consecutive(timesteps)

    @torch.no_grad()
    def sample(self, context, num_steps=50, device=None):
        """Generate samples by iteratively denoising from Gaussian noise.

        Args:
            context: Conditioning features from the rest of the GENIE model.
            num_steps: Number of reverse steps to use for DDIM sampling.
            device: Optional device override.

        Returns:
            Raw-space predictions with discrete columns re-quantized.
        """
        if device is None:
            device = context.device

        is_3d = (context.ndim == 3)
        if is_3d:
            batch_size, num_nodes, _ = context.shape
            context_flat = context.reshape(batch_size * num_nodes, -1)
        else:
            context_flat = context

        num_samples_total = context_flat.size(0)
        x = torch.randn(num_samples_total, self.output_dim, device=device)

        timesteps = self._get_sampling_timesteps(num_steps, device)
        num_time_steps = timesteps.size(0)

        for idx, timestep in enumerate(timesteps):
            timestep_batch = torch.full((num_samples_total,), timestep, device=device, dtype=torch.long)
            noise_pred = self.noise_predictor(timestep_batch, x, context_flat)

            alpha_bar_t = self.alpha_cumprod[timestep].view(1, 1)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
            x0_pred = (x - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t.clamp(min=1e-8)

            if idx == num_time_steps - 1:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)
            else:
                prev_timestep = timesteps[idx + 1]
                alpha_bar_prev = self.alpha_cumprod[prev_timestep].view(1, 1)

            x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1.0 - alpha_bar_prev).clamp(min=0.0) * noise_pred

        x_unscaled = self.unscale(x)

        shape_view = [1] * (x_unscaled.ndim - 1) + [-1]
        discrete_mask = self.is_discrete.view(*shape_view)
        x_quantized = torch.where(
            discrete_mask > 0.5,
            torch.floor(x_unscaled.clamp(min=0)),
            x_unscaled,
        )

        if is_3d:
            return x_quantized.view(batch_size, num_nodes, self.output_dim)
        return x_quantized

    def forward(self, context, num_steps=50):
        """Inference-only forward pass that delegates to sample()."""
        if self.training:
            raise RuntimeError("Use model.training_step(context, targets) for training, not model(context)")
        return self.sample(context, num_steps=num_steps, device=context.device)