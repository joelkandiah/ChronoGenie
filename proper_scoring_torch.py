import torch

def interval_score_torch(y, lower, upper, alpha=0.05):
    """
    Compute the interval score for a prediction interval.
    
    Args:
        y: Ground truth tensor
        lower: Lower bound tensor
        upper: Upper bound tensor
        alpha: Level of importance
        
    Returns:
        is_score: The interval score (same shape as y)
    """
    width = upper - lower
    under = torch.maximum(lower - y, torch.zeros_like(y))
    over = torch.maximum(y - upper, torch.zeros_like(y))
    return width + (2.0 / alpha) * under + (2.0 / alpha) * over

def crps_torch(y, samples):
    """
    Compute the Continuous Ranked Probability Score (CRPS).
    
    Args:
        y: Ground truth tensor [..., T, M, V] (without sample dimension)
        samples: Predictive sample tensor [S, ..., T, M, V]
        
    Returns:
        crps_score: The CRPS score [..., T, M, V]
    """
    # term1 = E|X-y|
    term1 = torch.mean(torch.abs(samples - y), dim=0)

    # term2 = 0.5 * E|X-X'|
    S = samples.size(0)
    # Reshape to [S, N] where N is everything else
    original_shape = samples.shape[1:]
    samples_flat = samples.reshape(S, -1)
    
    # We compute pairwise differences for term2.
    # To reduce memory overhead, we can utilize vectorization over N but compute Mean over S*S.
    # diffs: [S, S, N]
    diffs = torch.abs(samples_flat.unsqueeze(1) - samples_flat.unsqueeze(0))
    term2_flat = 0.5 * torch.mean(diffs, dim=(0, 1))
    
    term2 = term2_flat.reshape(original_shape)
    return term1 - term2

def energy_score_torch(y_true, samples):
    """
    Compute the Energy Score (multivariate generalization of CRPS).
    
    Args:
        y_true: Ground truth tensor [T, M]
        samples: Predictive sample tensor [S, T, M]
        
    Returns:
        es_score: The energy score [T]
    """
    # term1: mean ||X-y|| across samples S
    # y_true is broadcastable to [S, T, M]
    term1 = torch.mean(torch.linalg.norm(samples - y_true, dim=-1), dim=0)

    # term2: 0.5 * E||X-X'||
    # Samples are [S, T, M]
    S = samples.size(0)
    T = samples.size(1)
    M = samples.size(2)
    
    # Pairwise differences: [S, S, T, M]
    # Linaly.norm over M (last dim)
    diff = samples.unsqueeze(1) - samples.unsqueeze(0)
    term2 = 0.5 * torch.mean(torch.linalg.norm(diff, dim=-1), dim=(0, 1))
    
    return term1 - term2

def variogram_torch(y_true, samples, p=0.5):
    """
    Compute the Variogram score (multivariate scoring rule emphasizing spatial correlation).
    
    Args:
        y_true: Ground truth tensor [T, M]
        samples: Predictive sample tensor [S, T, M]
        p: Power parameter (usually 0.5)
        
    Returns:
        vs_score: The variogram score [T]
    """
    T, M = y_true.shape
    S = samples.size(0)
    
    # Pairwise differences in ground truth: [T, M, M]
    y_diff = torch.abs(y_true.unsqueeze(-1) - y_true.unsqueeze(-2)) ** p
    
    # Pairwise differences in samples: [S, T, M, M]
    x_diff = torch.abs(samples.unsqueeze(-1) - samples.unsqueeze(-2)) ** p
    x_diff_mean = torch.mean(x_diff, dim=0) # [T, M, M]
    
    vs = torch.sum((y_diff - x_diff_mean) ** 2, dim=(-1, -2))
    return vs


def compute_scores_torch(forecast_sample, y, alpha=0.05, score_type="all", weighted=False):
    """
    Compute the supported proper scoring rules for torch tensors.
    """
    del weighted

    if score_type not in {"all", "interval", "crps", "energy", "variogram"}:
        raise ValueError(f"Unsupported score_type: {score_type}")

    if forecast_sample.dim() < 3:
        raise ValueError("forecast_sample must have at least 3 dimensions: [S, ..., T, M]")

    # 1. Standardize dimensions so the sample dimension is ALWAYS at dim=0
    # Current input: [Batch (0), Variables (1), Samples (2), Nodes (3)]
    # Target for metrics: [Samples (2), Batch (0), Variables (1), Nodes (3)]
    samples_first = forecast_sample.permute(2, 0, 1, 3) 
    
    # 2. Add a matching singleton sample dimension to y for safe right-to-left broadcasting
    # y changes from [Batch, Variables, Nodes] -> [1, Batch, Variables, Nodes]
    y_broadcastable = y.unsqueeze(0)

    # 3. Compute interval bounds along dim=0 now that it's permuted
    lower = torch.quantile(samples_first, alpha / 2, dim=0)
    upper = torch.quantile(samples_first, 1 - alpha / 2, dim=0)

    # Interval score expects matching dimensions without sample dimension (both [4, 2, 84])
    interval_score = interval_score_torch(y, lower, upper, alpha=alpha)

    # CRPS functions now receive properly permuted matching layouts
    crps = crps_torch(y_broadcastable, samples_first)

    # 4. Prepare inputs for multivariate scores (Energy & Variogram)
    # They expect 3D arrays: [Samples, Time/Batch, Channels/Nodes]
    # We collapse Batch & Variables together to form a uniform multivariate space
    S = samples_first.size(0)
    M = samples_first.size(-1) # Nodes (84)
    
    energy_input = samples_first.reshape(S, -1, M)
    energy_target = y_broadcastable.reshape(-1, M)

    energy_score = energy_score_torch(energy_target, energy_input)
    variogram_score = variogram_torch(energy_target, energy_input)

    if score_type == "interval":
        return interval_score
    if score_type == "crps":
        return crps
    if score_type == "energy":
        return energy_score
    if score_type == "variogram":
        return variogram_score

    return interval_score, crps, energy_score, variogram_score
