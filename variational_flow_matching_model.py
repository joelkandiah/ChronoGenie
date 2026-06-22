import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from distribution_construction import MixedLoss
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper


class VRFMPosteriorEncoder(nn.Module):
    """
    Posterior Encoder q_phi(z | x_t, x_0, x_1, t).
    
    Only used during training. Maps the full trajectory information
    (current state, noise origin, data destination, time) to a latent
    distribution from which z is sampled via reparameterisation.
    """
    def __init__(self, state_dim, context_dim, latent_dim, hidden_dims=[128, 128]):
        super().__init__()
        # Input: x_t [D] + x_0 [D] + x_1 [D] + t [1] + context [C]
        input_dim = 3 * state_dim + 1 + context_dim
        
        layers = []
        curr = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(curr, h),
                nn.LayerNorm(h),
                nn.GELU()
            ])
            curr = h
        self.net = nn.Sequential(*layers)
        
        self.mu_head = nn.Linear(curr, latent_dim)
        self.logvar_head = nn.Linear(curr, latent_dim)
        
        # Initialise logvar head near 0 (unit variance prior)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)
    
    def forward(self, x_t, x_0, x_1, t, context):
        """
        Args:
            x_t: [N, D] current interpolated state
            x_0: [N, D] noise sample
            x_1: [N, D] data target (in scaled space)
            t: [N, 1] time
            context: [N, C] spatial/temporal context
        Returns:
            mu_z: [N, latent_dim]
            logvar_z: [N, latent_dim]
        """
        h = torch.cat([x_t, x_0, x_1, t, context], dim=-1)
        shared = self.net(h)
        return self.mu_head(shared), self.logvar_head(shared)
    
    def reparameterize(self, mu, logvar):
        """Sample z = mu + exp(0.5 * logvar) * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps


class VRFMFlowModel(nn.Module):
    """
    Two-Sided Flow Model f_theta(x_t, t, z, context) -> (mu_0, mu_1, dispersion).
    
    Predicts the endpoint means and dispersion parameters conditioned on the 
    current state, time, latent variable, and spatial/temporal context.
    Used during both training and inference.
    """
    def __init__(self, state_dim, latent_dim, context_dim, 
                 hidden_layers=[256, 256, 128, 128], dropout=0.2, use_skip_connection=False):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.use_skip_connection = use_skip_connection
        
        if not use_skip_connection:
            # Standard Joint Architecture
            input_dim = state_dim + 1 + latent_dim + context_dim
            layers = []
            curr = input_dim
            for h in hidden_layers:
                layers.extend([
                    nn.Linear(curr, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ])
                curr = h
            self.net = nn.Sequential(*layers)
            fusion_dim = curr
        else:
            # Context-Skip Architecture
            # 1. State Encoder (noisy/dynamic flow state)
            state_in_dim = state_dim + 1 + latent_dim
            state_layers = []
            curr = state_in_dim
            for h in hidden_layers:
                state_layers.extend([
                    nn.Linear(curr, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ])
                curr = h
            self.state_net = nn.Sequential(*state_layers)
            
            # 2. Context Encoder (stable historical features)
            self.context_net = nn.Sequential(
                nn.Linear(context_dim, hidden_layers[0]),
                nn.LayerNorm(hidden_layers[0]),
                nn.GELU(),
                nn.Linear(hidden_layers[0], curr),
                nn.LayerNorm(curr),
                nn.GELU()
            )
            fusion_dim = curr * 2
            
        # Output Heads
        self.mu0_head = nn.Linear(fusion_dim, state_dim)
        self.mu1_head = nn.Linear(fusion_dim, state_dim)
        self.disp_head = nn.Linear(fusion_dim, state_dim)
        
        # Initialization
        nn.init.zeros_(self.disp_head.bias)
        nn.init.zeros_(self.disp_head.weight)
    
    def forward(self, x_t, t, z, context):
        """
        Args:
            x_t: [N, D] current state
            t: [N, 1] or scalar time
            z: [N, latent_dim] latent variable
            context: [N, C] spatial/temporal context
        Returns:
            mu0: [N, D] predicted noise endpoint
            mu1: [N, D] predicted data endpoint  
            disp: [N, D] dispersion parameters
        """
        N = x_t.size(0)
        
        # Handle t shape
        if t.ndim == 0:
            t = t.view(1, 1).expand(N, 1)
        elif t.ndim == 1:
            t = t.view(-1, 1)
        
        if not self.use_skip_connection:
            h = torch.cat([x_t, t, z, context], dim=-1)
            fused = self.net(h)
        else:
            # Split Pathway
            state_in = torch.cat([x_t, t, z], dim=-1)
            state_h = self.state_net(state_in)
            ctx_h = self.context_net(context)
            fused = torch.cat([state_h, ctx_h], dim=-1)
        
        return self.mu0_head(fused), self.mu1_head(fused), self.disp_head(fused)


class VFMFlowMatchingModel(nn.Module):
    """
    Rectified Variational Flow Matching Model with Two-Sided Endpoint Prediction.
    
    Architecture:
        - Posterior Encoder q_phi(z | x_t, x_0, x_1, t): learns trajectory-level latent z
        - Flow Model f_theta(x_t, t, z, context): predicts both endpoints conditioned on z
    
    Training Objective (ELBO):
        L = L_NLL(x1) + L_NLL(x0) + lambda * L_KL
        - L_NLL(x1): Negative Binomial NLL for count data fidelity
        - L_NLL(x0): Gaussian NLL for noise endpoint (known sigma)
        - L_KL: KL divergence between posterior q_phi and prior N(0, I)
    
    Velocity Field:
        v_t = mu_1(x_t, t, z, ctx) - mu_0(x_t, t, z, ctx)
        Stable two-sided estimator, no division by (1-t).
    
    Inference:
        1. Sample z ~ N(0, I) once
        2. Hold z fixed during ODE integration t: 0 -> 1
        3. Unscale via expm1 and quantise for count data
    """
    def __init__(self, hidden_layers, input_dim, output_dim, 
                 prediction_distribution_types, raw_scalers, 
                 log_means, log_stds,
                 sigma=1.0, dropout=0.2, parameterization="mean",
                 latent_dim=8, kl_weight=1.0, encoder_hidden=[128, 128],
                 debug=False, **kwargs):
        """
        Args:
            hidden_layers: MLP hidden layers for flow model
            input_dim: Dimension of the context features (C)
            output_dim: Dimension of the target (D)
            prediction_distribution_types: List of 'nb' or 'lognormal'
            raw_scalers: List of StandardScaler objects
            log_means: Per-column log-space means
            log_stds: Per-column log-space stds
            sigma: Noise level for the base distribution
            dropout: Dropout for flow model
            latent_dim: Dimension of the latent variable z
            kl_weight: Weight for the KL divergence term (lambda)
            encoder_hidden: Hidden layers for the posterior encoder
        """
        super().__init__()
        self.output_dim = output_dim
        self.prediction_distribution_types = prediction_distribution_types
        self.parameterization = parameterization
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.kl_warmup_steps = kwargs.get("kl_warmup_steps", 2000)
        self.debug = debug
        self.sigma = sigma
        self._train_step_count = 0
        
        # Skip connection configuration
        self.use_skip_connection = kwargs.get("vfm_skip_connection", False)
        
        # Posterior encoder: q_phi(z | x_t, x_0, x_1, t, context)
        self.posterior_encoder = VRFMPosteriorEncoder(
            state_dim=output_dim, 
            context_dim=input_dim,
            latent_dim=latent_dim, 
            hidden_dims=encoder_hidden
        )
        
        # Two-sided flow model: f_theta(x_t, t, z, context) -> (mu0, mu1, disp)
        self.flow_model = VRFMFlowModel(
            state_dim=output_dim,
            latent_dim=latent_dim,
            context_dim=input_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            use_skip_connection=self.use_skip_connection
        )
        
        # Store log-scalers as buffers
        self.register_buffer("log_means", torch.from_numpy(np.array(log_means, dtype=np.float32)))
        self.register_buffer("log_stds", torch.from_numpy(np.array(log_stds, dtype=np.float32)).clamp(min=1e-6))
        
        # Store raw means for diagnostics
        raw_means_np = np.array([s.mean_ if s is not None else 0.0 for s in raw_scalers], dtype=np.float32)
        self.register_buffer("raw_means_buffer", torch.from_numpy(raw_means_np))
        
        # Precompute masks for distribution types
        self.register_buffer("ln_mask", torch.tensor([t.lower() == 'lognormal' for t in prediction_distribution_types], dtype=torch.bool))
        self.register_buffer("nb_mask", torch.tensor([t.lower() == 'nb' for t in prediction_distribution_types], dtype=torch.bool))

    def scale(self, x_raw, indices=None):
        """Map raw values to scaled log-values. 
        Compatible with testing_sliding_window.py signature.
        """
        means = self.log_means if indices is None else self.log_means[indices]
        stds = self.log_stds if indices is None else self.log_stds[indices]
        
        shape_view = [1] * (x_raw.ndim - 1) + [-1]
        z_raw = torch.log1p(x_raw)
        z_scaled = (z_raw - means.view(*shape_view)) / stds.view(*shape_view)
        return z_scaled

    def unscale(self, x_scaled, indices=None):
        """Map scaled log-values back to raw count values."""
        means = self.log_means if indices is None else self.log_means[indices]
        stds = self.log_stds if indices is None else self.log_stds[indices]
        
        shape_view = [1] * (x_scaled.ndim - 1) + [-1]
        z_raw = x_scaled * stds.view(*shape_view) + means.view(*shape_view)
        return torch.expm1(z_raw)

    def training_step(self, latent_input, target, is_epoch_diag=False):
        """
        Rectified Variational Flow Matching training step.
        
        This implements the combined ELBO: L = L_NLL(x1) + L_NLL(x0) + lambda * L_KL
        where both endpoints are predicted conditioned on a trajectory-level latent z.
        """
        device = latent_input.device
        B, M, D = target.shape
        
        # 1. Transform target to standardised log-space (x1_scaled)
        # Note: we use 'x1_scaled' to avoid confusion with the latent variable 'z'
        x1_log_raw = torch.log1p(target)
        x1_scaled = (x1_log_raw - self.log_means.view(1, 1, -1)) / self.log_stds.view(1, 1, -1)
        
        # 2. Sample noise (x0) and time (t)
        t = torch.rand(B, device=device)
        x0 = torch.randn_like(x1_scaled) * self.sigma
        
        # 3. Interpolate trajectory: x_t = (1-t)*x0 + t*x1 (Straight-Line OT)
        t_view = t.view(-1, 1, 1)
        x_t = (1.0 - t_view) * x0 + t_view * x1_scaled
        
        # Flatten spatial dimension for network batching
        context_flat = latent_input.view(B * M, -1)     # [B*M, C]
        t_expanded = t.view(B, 1, 1).expand(B, M, 1).reshape(B * M, 1)  # [B*M, 1]
        xt_flat = x_t.view(B * M, -1)                   # [B*M, D]
        x0_flat = x0.view(B * M, -1)                    # [B*M, D]
        x1_flat = x1_scaled.view(B * M, -1)             # [B*M, D]
        
        # 4. Encoder: q_phi(z | x_t, x_0, x_1, t, context) 
        mu_z, log_var_z = self.posterior_encoder(xt_flat, x0_flat, x1_flat, t_expanded, context_flat)
        z = self.posterior_encoder.reparameterize(mu_z, log_var_z)  # [B*M, latent_dim]
        
        # 5. Flow Model: f_theta(x_t, t, z, context) -> Predicted Endpoints
        mu0_flat, mu1_flat, disp_flat = self.flow_model(xt_flat, t_expanded, z, context_flat)
        
        # Reshape for loss calculation [B, M, D]
        mu0_3d = mu0_flat.view(B, M, D)
        mu1_3d = mu1_flat.view(B, M, D)
        disp_3d = disp_flat.view(B, M, D)
        
        # KL Divergence: KL(q(z|...) || N(0, I))
        loss_kl = -0.5 * torch.sum(1 + log_var_z - mu_z.pow(2) - log_var_z.exp())
        loss_kl = loss_kl / (B * M)
        
        # === LOSSES ===
        
        # a) NB/LogNormal NLL for x1 endpoint (data fidelity)
        # 
        # CRITICAL: The model predicts in log1p(count) space (standardised).
        # After unscaling we get log1p_mu = log(1 + count_pred).
        # But the NB/LN NLL expects log(count_pred), NOT log(1 + count_pred).
        #   exp(log1p(x)) = 1 + x  ← wrong rate (biased by +1)
        #   exp(log(x))   = x      ← correct rate
        # So we must convert: log(count) = log(expm1(log1p_count))
        #
        log1p_mu1 = mu1_3d * self.log_stds.view(1, 1, -1) + self.log_means.view(1, 1, -1)
        mu1_count = torch.expm1(log1p_mu1).clamp(min=1e-3)  # actual count prediction
        log_mu1_for_nll = torch.log(mu1_count)  # log(count) for NLL
        
        # Diagnostics: raw predicted counts (same as mu1_count, no grad)
        with torch.no_grad():
            mu1_raw_pred = mu1_count.detach()
        
        loss_nll, per_burden_nll = MixedLoss(
            log_mu1_for_nll, disp_3d, target,
            self.prediction_distribution_types,
            parameterization=self.parameterization,
            transform="exp",
            use_log_nb=True 
        )
        
        # b) Gaussian NLL for x0 endpoint (base distribution fidelity)
        loss_x0 = F.gaussian_nll_loss(mu0_3d, x0, torch.full_like(mu0_3d, self.sigma**2))
        
        # KL Warmup logic
        if self.training:
            warmup_factor = min(1.0, self._train_step_count / max(1, self.kl_warmup_steps))
        else:
            warmup_factor = 1.0
        
        current_kl_weight = self.kl_weight * warmup_factor
        total_loss = loss_nll + loss_x0 + current_kl_weight * loss_kl
        
        # === DIAGNOSTICS ===
        do_diag = is_epoch_diag or (self.training and self._train_step_count % 50 == 0)
        
        if do_diag:
            with torch.no_grad():
                # Context Usage Diagnostic: How sensitive is the model to the specific context?
                # We shuffle the context indices and measure the impact on mu1.
                shuffled_idx = torch.randperm(context_flat.size(0))
                mu1_shuffled, _, _ = self.flow_model(xt_flat, t_expanded, z, context_flat[shuffled_idx])
                ctx_sens = (mu1_flat - mu1_shuffled).abs().mean().item()
                
                ctx_mean = context_flat.mean().item()
                ctx_std = context_flat.std().item()

                print(f"\n[VRFM DIAG] Step {self._train_step_count} (KL Factor: {warmup_factor:.2f})")
                print(f"  Context: mean={ctx_mean:.3f} std={ctx_std:.3f} | Sensitivity (shuffle diff)={ctx_sens:.4f}")
                print(f"  Global Losses: NLL(x1)={loss_nll.item():.4f}  NLL(x0)={loss_x0.item():.4f}  KL_raw={loss_kl.item():.4f}")
                print(f"  Latent z: mu_z mean={mu_z.mean().item():.3f} std={mu_z.std().item():.3f}")
                
                # Iterate through individual burdens
                for d in range(D):
                    dist_type = self.prediction_distribution_types[d].lower()
                    col_name = f"Col {d}"
                    
                    # Individual endpoint means
                    m1_raw = mu1_raw_pred[:, :, d].mean().item()
                    t_raw = target[:, :, d].mean().item()
                    col_loss = per_burden_nll[d].item()
                    
                    # DISPERSION SYNC: Exact same logic as MixedLoss
                    d_raw = disp_3d[:, :, d]
                    if dist_type == 'nb':
                        if self.parameterization == "mean":
                            alpha = (F.softplus(d_raw) + 0.01).mean().item()
                            r = 1.0 / alpha
                        else: # natural
                            log_mu = log_mu1_for_nll[:, :, d]
                            log_r_diag = (2.0 * log_mu - d_raw)
                            r = torch.exp(log_r_diag).mean().item()
                            alpha = 1.0 / (r + 1e-6)
                        disp_str = f"NB_disp={alpha:.4f} (count={r:.1f})"
                    else: # lognormal
                        if self.parameterization == "mean":
                            var_ln = (F.softplus(d_raw) + 0.01).mean().item()
                            std_val = var_ln**0.5
                        else: # natural
                            std_val = (torch.exp(d_raw) + 0.1).mean().item()
                        disp_str = f"LN_std={std_val:.4f}"
                    
                    print(f"  > {col_name} ({dist_type}): Loss={col_loss:.4f} | E_pred={m1_raw:.2f} vs Tar={t_raw:.2f} | {disp_str}")
        
        if self.training:
            self._train_step_count += 1
        
        return total_loss, per_burden_nll

    def _vfm_velocity(self, t, x_t, augmented_context):
        """
        Velocity field derived from Two-Sided endpoint predictions.
        
        Formula: v_t = mu1(x_t, t, z) - mu0(x_t, t, z)
        Justification: For Rectified Flow (alpha=1-t, beta=t), v_t = beta_dot*x1 + alpha_dot*x0.
        By replacing x0/x1 with their conditional expectations from f_theta, we get mu1 - mu0.
        """
        context = augmented_context[:, :-self.latent_dim]
        z = augmented_context[:, -self.latent_dim:]
        
        # Predict endpoints conditioned on fixed latent z
        mu0, mu1, _ = self.flow_model(x_t, t, z, context)
        
        return mu1 - mu0

    def unscale_samples(self, x1_flat):
        """Standardised log-space -> raw counts."""
        z1_log_raw = x1_flat * self.log_stds.view(1, -1) + self.log_means.view(1, -1)
        return torch.expm1(z1_log_raw).clamp(min=0)

    @torch.no_grad()
    def sample(self, context, num_steps=50, device=None):
        """
        Inference Loop.
        1. Sample z ~ N(0, I) from the prior.
        2. Fix z for the duration of the ODE integration (t=0 -> 1).
        3. Velocity is derived as mu1 - mu0 at each step.
        """
        if device is None:
            device = context.device

        is_3d = (context.ndim == 3)
        if is_3d:
            B, M, C = context.shape
            context_flat = context.reshape(B * M, -1)
        else:
            N, C = context.shape
            context_flat = context
            
        D = self.output_dim
        num_samples_total = context_flat.size(0)
        
        # Initial noise: x0 ~ N(0, sigma^2 * I)
        initial_x0 = torch.randn(num_samples_total, D, device=device) * self.sigma
        
        # Sample the latent variable z from the prior (Normal(0, 1))
        # Note: z captures trajectory characteristics and stays constant during integration
        z = torch.randn(num_samples_total, self.latent_dim, device=device)
        
        # Bag the context and z together for the ODE solver
        augmented_context = torch.cat([context_flat, z], dim=-1)
        
        class VFMOdeWrapper(nn.Module):
            def __init__(self, vfm_model):
                super().__init__()
                self.vfm_model = vfm_model
            def forward(self, t, x, context):
                return self.vfm_model._vfm_velocity(t, x, context)
                
        model_wrapped = ModelWrapper(VFMOdeWrapper(self))
        t_span = torch.linspace(0, 1, num_steps + 1, device=device)
        solver = ODESolver(velocity_model=model_wrapped)
        
        # Integrate dx/dt = v_t from t=0 to t=1
        x1_scaled_flat = solver.sample(
            x_init=initial_x0,
            step_size=None,
            method='euler',
            time_grid=t_span,
            return_intermediates=False,
            context=augmented_context
        )
        
        # Transform back to count space
        x1_unscaled = self.unscale_samples(x1_scaled_flat)
        
        # Quantise NB columns to integers
        x1_quantized = torch.where(
            self.nb_mask.view(1, -1),
            torch.round(x1_unscaled),
            x1_unscaled 
        ).clamp(min=0)
        
        if is_3d:
            return x1_quantized.view(B, M, D)
        return x1_quantized

    def forward(self, context, num_steps=50):
        if self.training:
            raise RuntimeError("Use model.training_step(context, targets) for training, not model(context)")
        return self.sample(context, num_steps=num_steps, device=context.device)
