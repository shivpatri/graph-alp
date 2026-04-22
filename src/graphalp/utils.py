import networkx as nx
import numpy as np
from typing import Tuple

def compute_laplacian(graph: nx.Graph) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Laplacian matrix of a graph.
    """
    W = nx.adjacency_matrix(graph).toarray().astype(float)
    D = np.diag(W.sum(axis=1))
    L = D - W
    return W, D, L

def compute_risk(probabilities: np.ndarray) -> float:
    """
    Computes the estimated risk, defined as the sum of the uncertainty of each prediction.
    Uncertainty for a single prediction `p` is `min(p, 1-p)`.
    """
    uncertainty = np.minimum(probabilities, 1 - probabilities)
    return float(np.sum(uncertainty))