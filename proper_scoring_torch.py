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
    samples_flat = samples.view(S, -1)
    
    # We compute pairwise differences for term2.
    # To reduce memory overhead, we can utilize vectorization over N but compute Mean over S*S.
    # diffs: [S, S, N]
    diffs = torch.abs(samples_flat.unsqueeze(1) - samples_flat.unsqueeze(0))
    term2_flat = 0.5 * torch.mean(diffs, dim=(0, 1))
    
    term2 = term2_flat.view(original_shape)
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

    Args:
        forecast_sample: Predictive samples with shape [S, ..., T, M]
        y: Ground truth tensor broadcastable to the non-sample dimensions of forecast_sample
        alpha: Interval score level
        score_type: "all" or a single score name
        weighted: Present for API compatibility; currently unused

    Returns:
        A tuple of requested scores in the order:
        interval_score, crps, energy_score, variogram_score
    """
    del weighted

    if score_type not in {"all", "interval", "crps", "energy", "variogram"}:
        raise ValueError(f"Unsupported score_type: {score_type}")

    if forecast_sample.dim() < 3:
        raise ValueError("forecast_sample must have at least 3 dimensions: [S, ..., T, M]")

    lower = torch.quantile(forecast_sample, 0.05, dim=0)
    upper = torch.quantile(forecast_sample, 0.95, dim=0)
    interval_score = interval_score_torch(y, lower, upper, alpha=alpha)

    crps = crps_torch(y, forecast_sample)

    if forecast_sample.dim() == 3:
        energy_input = forecast_sample
        energy_target = y
        variogram_input = forecast_sample
        variogram_target = y
    else:
        energy_input = forecast_sample.reshape(forecast_sample.shape[0], -1, forecast_sample.shape[-1])
        energy_target = y.reshape(-1, y.shape[-1])
        variogram_input = energy_input
        variogram_target = energy_target

    energy_score = energy_score_torch(energy_target, energy_input)
    variogram_score = variogram_torch(variogram_target, variogram_input)

    if score_type == "interval":
        return interval_score
    if score_type == "crps":
        return crps
    if score_type == "energy":
        return energy_score
    if score_type == "variogram":
        return variogram_score

    return interval_score, crps, energy_score, variogram_score
