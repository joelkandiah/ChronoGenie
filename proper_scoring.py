import numpy as np

def energy_score(y_true, samples):
    # ES_beta(D,y) = E||X-Y|| - (1/2)E||X-X'||, with beta=2
    # Compute the energy score for a multivariate predictive distribution
    # y_true: shape (num_msoas,)
    # samples: shape (num_samples, num_msoas)

    # First term: mean distance between samples and observation
    term1 = np.mean(np.linalg.norm(samples - y_true, axis=1))

    # Second term: mean pairwise distance between samples
    diff = samples[:, None, :] - samples[None, :, :]
    term2 = np.mean(np.linalg.norm(diff, axis=2))

    es_score = term1 - 0.5 * term2
    return es_score

def variogram(y_true: np.ndarray, samples: np.ndarray, p: float = 0.5):
    # vario(D,y) = sum_ij wij (|yi-yj|^p - E(|xi-xj|^p))^2, with p=0.5
    # Compute the variogram for a multivariate predictive distribution
    # y_true: shape (num_msoas,)
    # samples: shape (num_samples, num_msoas)

    # Pairwise differences in observation
    y_diff = np.abs(y_true[:, None] - y_true[None, :]) ** p

    # Pairwise differences in samples (mean over all samples)
    x_diff = np.abs(samples[:, :, None] - samples[:, None, :]) ** p
    x_diff_mean = np.mean(x_diff, axis=0)

    # Variogram score
    w = 1
    vs = np.sum(w * (y_diff - x_diff_mean) ** 2)
    return vs

def interval_score(y, lower, upper, alpha=0.05):
    # IS_a(F,y) = (u-l) + (2/a)(l-y) 1(y<l) + (2/a)(y-u) 1(y>u)
    # l and u are the upper/lower bounds of a prediction interval
    # First term is for sharpness
    # Second term is for penalising under and over prediction when the observations fall outside the interval

    width = upper - lower # --> Calculates the sharpness term (u-l)
    
    # Calculating (l-y)
    # If y falls below l then l-y > 0, np.maximum keeps the positive value. 
    # The amount by which y is under the interval.
    under = np.maximum(lower - y, 0.0)

    # Calculating (y-u)
    # If y is above u, then y-u > 0, np.maximum keeps the postiive value.
    # The amount by which y is over the interval.  
    over = np.maximum(y - upper, 0.0)
    
    # Adding the sharpness term and the two penalty terms scaled by 2/a.
    is_score = width + (2.0/alpha) * under + (2.0/alpha) * over
    
    return is_score

def crps(y, samples):
    # Check Page 367 https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf?
    
    # CRPS can be written as: E|X-y| - (1/2)E|X-X'|
    
    # Getting the Monte-Carlo estimate of E|X-y| using the sample average
    term1 = np.mean(np.abs(samples - y))

    # Creating the nxn matrix |x_i - x_j| for all pairs (i,j)
    # If samples is a 1D array:
    # samples[:, None] reshapes to column vector
    # samples[:, None] reshapes to row vector
    diffs = np.abs(samples[:, None] - samples[None, :]) 
    
    # Estimation of (1/2)E|X-X'|
    term2 = 0.5 * np.mean(diffs)

    return term1 - term2