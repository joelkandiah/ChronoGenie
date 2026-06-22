import torch
import torch.distributions as D
import torch.nn.functional as F

def MixedLoss(mu_pred: torch.Tensor, var_pred: torch.Tensor, y_true: torch.Tensor, types, parameterization="natural", transform="softplus", use_log_nb=False) -> torch.Tensor:
    """
    MIXED DISTRIBUTION NEGATIVE LOG-LIKELIHOOD (NB + LOGNORMAL)

    This loss lets us train a single model that predicts multiple burdens where:
        - some targets are COUNT data (modelled with a Negative Binomial)
        - some targets are CONTINUOUS positive data (modelled with a LogNormal)

    Inputs:
        mu_pred: [B, M, V] Predicted location/mean parameters. 
                 If use_log_nb=True, for NB channels this is expected to be log(mu).
        var_pred: [B, M, V] Predicted dispersion/variance parameters.
        y_true: [B, M, V] Ground truth targets.
        types: List of length V, ["nb" or "lognormal"].
        parameterization: "natural" or "mean".
        transform: "softplus" or "exp".
        use_log_nb: If True, uses a stable log-domain calculation for NB NLL to prevent 
                    exponential gradient explosion. Compatible with VFM.
    """
    B, M, V = y_true.shape
    device = mu_pred.device
    
    # Per-entry NLL container
    nll = torch.zeros_like(y_true)
    
    # Mask for distribution types
    ln_mask = torch.tensor([t.lower() == 'lognormal' for t in types], device=device)
    nb_mask = ~ln_mask

    # 1. Negative Binomial channels
    if nb_mask.any():
        nb_idx = nb_mask.nonzero(as_tuple=False).squeeze(-1)
        mu_nb_raw = mu_pred.index_select(dim=-1, index=nb_idx)
        var_nb_raw = var_pred.index_select(dim=-1, index=nb_idx)
        y_nb = y_true.index_select(dim=-1, index=nb_idx)

        if use_log_nb:
            # === STABLE LOG-DOMAIN NB NLL ===
            # mu_nb_raw is expected to be log(mu)
            log_mu = mu_nb_raw
            
            if parameterization == "mean":
                # var_nb_raw is raw dispersion parameter
                # softplus always has gradient (sigmoid), unlike clamp which kills gradient below floor
                alpha = F.softplus(var_nb_raw) + 0.01  # min 0.01, gradient always flows
                r = 1.0 / alpha
                log_r = torch.log(r)
            else:
                # parameterization == "natural"
                # log_r = 2*log_mu - log_var_resid
                log_var_resid = var_nb_raw
                log_r = (2.0 * log_mu - log_var_resid) 
                r = torch.exp(log_r)

            # Stable log(r + mu)
            log_r_plus_mu = torch.logaddexp(log_r, log_mu)
            
            # NB NLL
            nll_val = -(
                torch.lgamma(y_nb + r) - 
                torch.lgamma(y_nb + 1.0) - 
                torch.lgamma(r) + 
                r * log_r + 
                y_nb * log_mu - 
                (y_nb + r) * log_r_plus_mu
            )
            nll[..., nb_idx] = nll_val
        else:
            # === ORIGINAL AUTO-DIFF NB NLL (Standard MLPs) ===
            if parameterization == "mean":
                mu_nb = (torch.exp(mu_nb_raw) if transform == "exp" else F.softplus(mu_nb_raw)) + 1e-3
                disp_nb = F.softplus(var_nb_raw) + 0.01 # Added floor
                total_count = 1.0 / disp_nb
                probs = total_count / (total_count + mu_nb)
            else:
                mu_nb = (torch.exp(mu_nb_raw) if transform == "exp" else F.softplus(mu_nb_raw)) + 1e-3
                var_resid = (torch.exp(var_nb_raw) if transform == "exp" else F.softplus(var_nb_raw)) + 1.0
                var_nb = mu_nb + var_resid
                total_count = (mu_nb**2) / (var_nb - mu_nb + 1e-6)
                probs = total_count / (total_count + mu_nb)

            dist = D.NegativeBinomial(total_count=total_count, probs=probs)
            nll[..., nb_idx] = -dist.log_prob(y_nb)

    # 2. LogNormal channels
    if ln_mask.any():
        ln_idx = ln_mask.nonzero(as_tuple=False).squeeze(-1)
        mu_ln_raw = mu_pred.index_select(dim=-1, index=ln_idx)
        var_ln_raw = var_pred.index_select(dim=-1, index=ln_idx)
        y_ln = y_true.index_select(dim=-1, index=ln_idx)

        if parameterization == "mean":
            mean_ln = (torch.exp(mu_ln_raw) if transform == "exp" else F.softplus(mu_ln_raw)) + 1e-6
            var_ln = F.softplus(var_ln_raw) + 0.01 # Consistent floor (std=0.1)
            loc = torch.log(mean_ln) - 0.5 * var_ln
            scale = torch.sqrt(var_ln)
        else:
            loc = mu_ln_raw
            scale = (torch.exp(var_ln_raw) if transform == "exp" else F.softplus(var_ln_raw)) + 0.1 # Consistent floor
        
        y_ln_clamped = y_ln.clamp_min(1e-8)
        dist = D.LogNormal(loc=loc, scale=scale)
        nll[..., ln_idx] = -dist.log_prob(y_ln_clamped)

    # Average over batch and MSOA
    per_burden_nll = nll.mean(dim=[0, 1])
    total_loss = per_burden_nll.mean()

    return total_loss, per_burden_nll

def sample_mixed(mu_pred: torch.Tensor, var_pred: torch.Tensor, types, parameterization="natural", transform="softplus"):
    """
    SAMPLE FROM A MIXED OUTPUT DISTRIBUTION (NB + LOGNORMAL)
    """
    B, M, V = mu_pred.shape
    samples = torch.zeros_like(mu_pred)
    device = mu_pred.device
    
    ln_mask = torch.tensor([t.lower() == 'lognormal' for t in types], device=device)
    nb_mask = ~ln_mask

    # 1. Negative Binomial channels
    if nb_mask.any():
        nb_idx = nb_mask.nonzero(as_tuple=False).squeeze(-1)
        mu_nb_raw = mu_pred.index_select(dim=-1, index=nb_idx)
        var_nb_raw = var_pred.index_select(dim=-1, index=nb_idx)

        if parameterization == "mean":
            if transform == "exp":
                mu_nb = torch.exp(mu_nb_raw) + 1e-3
            else:
                mu_nb = F.softplus(mu_nb_raw) + 1e-3
            disp_nb = F.softplus(var_nb_raw) + 1e-3
            total_count = 1.0 / disp_nb
            probs = total_count / (total_count + mu_nb)
        else:
            if transform == "exp":
                mu_nb = torch.exp(mu_nb_raw) + 1e-3
                var_resid = torch.exp(var_nb_raw) + 1.0
            else:
                mu_nb = F.softplus(mu_nb_raw) + 1e-3
                var_resid = F.softplus(var_nb_raw) + 1.0
            
            var_nb = mu_nb + var_resid
            total_count = (mu_nb**2) / (var_nb - mu_nb)
            probs = total_count / (total_count + mu_nb)

        dist = D.NegativeBinomial(total_count=total_count, probs=probs)
        samples[..., nb_idx] = dist.sample()
    
    # 2. LogNormal channels
    if ln_mask.any():
        ln_idx = ln_mask.nonzero(as_tuple=False).squeeze(-1)
        mu_ln_raw = mu_pred.index_select(dim=-1, index=ln_idx)
        var_ln_raw = var_pred.index_select(dim=-1, index=ln_idx)

        if parameterization == "mean":
            if transform == "exp":
                mean_ln = torch.exp(mu_ln_raw) + 1e-6
            else:
                mean_ln = F.softplus(mu_ln_raw) + 1e-6
            var_ln = F.softplus(var_ln_raw) + 1e-6
            
            loc = torch.log(mean_ln) - 0.5 * var_ln
            scale = torch.sqrt(var_ln)
        else:
            loc = mu_ln_raw
            if transform == "exp":
                scale = torch.exp(var_ln_raw) + 1e-6
            else:
                scale = F.softplus(var_ln_raw) + 1e-6
        
        dist = D.LogNormal(loc=loc, scale=scale)
        samples[..., ln_idx] = dist.sample()

    return samples