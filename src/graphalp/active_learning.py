import networkx as nx
import numpy as np
from typing import Any, Dict, List
from .label_propagation import HarmonicLabelPropagator
from .utils import compute_risk

class HarmonicGreedySampler:
    """
    Harmonic sampler for graph-based semi-supervised learning.
    """
    def __init__(self, graph: nx.Graph) -> None:
        """
        Initialize the sampler with the underlying graph structure.
        """
        self.propagator = HarmonicLabelPropagator(graph)

    def fit(self, X: Any, y: Any) -> "HarmonicGreedySampler":
        """
        Fit the harmonic model by parsing the provided labels.
        """
        self.propagator.fit(X, y)
        return self

    def sample(self) -> Any:
        """
        Sample one node using a greedy strategy that minimizes future expected risk.

        For each unlabeled node, this method calculates the expected risk if that node
        were to be labeled, and selects the node that yields the greatest risk reduction.

        Returns:
            The ID of the best node to sample, or None if no unlabeled nodes exist.
        """
        if not hasattr(self.propagator, 'labels'):
            raise RuntimeError("You must call .fit(X, y) on the sampler before sampling.")

        if not self.propagator.u_idx:
            return None

        def _get_unlabeled_scores(prop: HarmonicLabelPropagator) -> np.ndarray:
            """Helper to get scores for all currently unlabeled nodes as a numpy array."""
            return np.array([prop.f[prop.nodes[idx]] for idx in prop.u_idx])
        
        best_node_to_sample = None
        min_expected_risk = float('inf')
        label_propagator = HarmonicLabelPropagator(self.propagator.graph)
        
        # Iterate through candidate nodes to find the one that minimizes future risk
        for i, u_node_idx in enumerate(self.propagator.u_idx):
            node_id = self.propagator.nodes[u_node_idx]
            p1 = self.propagator.f[node_id]
            p0 = 1 - p1
            
            # Current labeled set
            X_curr = list(self.propagator.labels.keys())
            y_curr = list(self.propagator.labels.values())
            
            # Candidate node
            X_curr.append(node_id)
            
            # Case 1: Node label is 1
            y_curr.append(1)
            label_propagator.fit(X_curr, y_curr)
            risk1 = compute_risk(_get_unlabeled_scores(label_propagator))
            
            # Case 0: Node label is 0
            y_curr[-1] = 0
            label_propagator.fit(X_curr, y_curr)
            risk0 = compute_risk(_get_unlabeled_scores(label_propagator))
            
            expected_risk = p1 * risk1 + p0 * risk0
            if expected_risk < min_expected_risk:
                min_expected_risk = expected_risk
                best_node_to_sample = node_id

        return best_node_to_sample
