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

        
        best_node_to_sample = None
        min_expected_risk = float('inf')
        label_propagator = HarmonicLabelPropagator(self.model.graph)
        # Iterate through candidate nodes to find the one that minimizes future risk
        for u_node in self.model.u_idx:
            p1 = self.model.f_u[u_node]
            p0 = 1 - p1
            labels = self.model.labels.copy()
            labels[self.model.nodes[u_node]] = 1
            label_propagator.fit(labels)
            risk0 = compute_risk(list(label_propagator.predict_probabilities().values()))
            labels[self.model.nodes[u_node]] = 0
            label_propagator.fit(labels)
            risk1 = compute_risk(list(label_propagator.predict_probabilities().values()))
            expected_risk = p1 * risk0 + p0 * risk1
            if expected_risk < min_expected_risk:
                min_expected_risk = expected_risk
                best_node_to_sample = self.model.nodes[u_node]

        return best_node_to_sample
