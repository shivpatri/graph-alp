import networkx as nx
import numpy as np
from typing import Any, Dict
from .parametric_graph_models import GaussianRandomFieldModel
from .utils import compute_risk
class HarmonicGreedySampler:
    """
    Harmonic sampler for graph-based semi-supervised learning.
    """
    def __init__(self, graph: nx.Graph) -> None:
        """
        Initialize the sampler with the underlying graph structure.
        """
        self.model = GaussianRandomFieldModel(graph)

    def fit(self, labels: Dict[Any, int]) -> "HarmonicGreedySampler":
        """
        Fit the harmonic model by parsing the provided labels.
        """
        self.model.fit(labels)
        return self

    def sample(self) -> Any:
        """
        Sample one node using a greedy strategy that minimizes future expected risk.

        For each unlabeled node, this method calculates the expected risk if that node
        were to be labeled, and selects the node that yields the greatest risk reduction.

        Returns:
            The ID of the best node to sample, or None if no unlabeled nodes exist.
        """
        if not hasattr(self.model, 'labels'):
            raise RuntimeError("You must call .fit(labels) on the sampler before sampling.")

        if not self.model.u_idx:
            return None

        
        # Get current probabilities for unlabeled nodes to calculate the expectation
        base_probs_arr = self.model.f_u
        unlabeled_nodes = [self.model.nodes[i] for i in self.model.u_idx]
        base_probs = dict(zip(unlabeled_nodes, base_probs_arr))

        best_node_to_sample = None
        min_expected_risk = float('inf')

        # Iterate through candidate nodes to find the one that minimizes future risk
        for u_node in unlabeled_nodes:
            prob_u = np.clip(base_probs[u_node], 0.0, 1.0)

            # --- Calculate risk if we add (u_node, 1) ---
            labels_with_1 = {**self.model.labels, u_node: 1.0}
            try:
                self.model.fit(labels_with_1)
                risk_if_1 = compute_risk(self.model.f_u)
            except ValueError: # This happens if all nodes become labeled
                risk_if_1 = 0.0

            # --- Calculate risk if we add (u_node, 0) ---
            labels_with_0 = {**self.model.labels, u_node: 0.0}
            try:
                self.model.fit(labels_with_0)
                risk_if_0 = compute_risk(self.model.f_u)
            except ValueError: # This happens if all nodes become labeled
                risk_if_0 = 0.0

            # Calculate expected future risk for this candidate node
            expected_risk = prob_u * risk_if_1 + (1 - prob_u) * risk_if_0

            if expected_risk < min_expected_risk:
                min_expected_risk = expected_risk
                best_node_to_sample = u_node
        
        return best_node_to_sample