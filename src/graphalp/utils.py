import numpy as np

def compute_risk(probabilities: np.ndarray) -> float:
    """
    Computes the estimated risk, defined as the sum of the uncertainty of each prediction.
    Uncertainty for a single prediction `p` is `min(p, 1-p)`.
    """
    uncertainty = np.minimum(probabilities, 1 - probabilities)
    return float(np.sum(uncertainty))