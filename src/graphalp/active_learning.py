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

class RandomSampler:
    """
    Standard Random Active Learning Sampler.
    """
    def __init__(self, graph: nx.Graph) -> None:
        """
        Initialize the sampler with the underlying graph structure.
        """
        self.nodes = list(graph.nodes())
        self.labels = {}

    def fit(self, X: Any, y: Any) -> "RandomSampler":
        """
        Store the current labels for the graph nodes.
        """
        self.labels = dict(zip(X, y))
        return self

    def sample(self) -> Any:
        """
        Sample a node uniformly at random from the unlabeled set.
        """
        unlabeled = [n for n in self.nodes if n not in self.labels]
        return np.random.choice(unlabeled) if unlabeled else None

class S2Sampler:
    """
    S2 Active Learning Sampler based on Dasarathy et al. (2015).
    
    Algorithm logic:
    1. Prune edges between nodes with different labels.
    2. Find the shortest path between a Label 0 node and a Label 1 node.
    3. Query the middle node of that path.
    4. Fallback to a secondary sampler if no path exists.
    """
    def __init__(self, graph: nx.Graph, fallback_sampler: Any = None) -> None:
        """
        Initialize the sampler with a graph and an optional fallback sampler.
        """
        self.graph = graph.copy()
        self.nodes = list(graph.nodes())
        self.labels = {}
        self.fallback_sampler = fallback_sampler or RandomSampler(graph)

    def fit(self, X: Any, y: Any) -> "S2Sampler":
        """
        Fit both the S2 logic and the fallback sampler.
        """
        self.labels = dict(zip(X, y))
        self.fallback_sampler.fit(X, y)
        return self

    def sample(self) -> Any:
        """
        Find the shortest path between different label sets and sample the middle node.
        """
        l0_nodes = [n for n, l in self.labels.items() if l == 0]
        l1_nodes = [n for n, l in self.labels.items() if l == 1]
        
        # If we don't have representatives from both classes, use fallback
        if not l0_nodes or not l1_nodes:
            return self.fallback_sampler.sample()

        # 1. Prune edges between nodes with different labels
        G_prime = self.graph.copy()
        edges_to_remove = []
        for u, v in G_prime.edges():
            lab_u = self.labels.get(u)
            lab_v = self.labels.get(v)
            if (lab_u == 0 and lab_v == 1) or (lab_u == 1 and lab_v == 0):
                edges_to_remove.append((u, v))
        G_prime.remove_edges_from(edges_to_remove)

        # 2. Find the shortest path between any L0 node and any L1 node
        min_dist = float('inf')
        best_path = []

        for n0 in l0_nodes:
            try:
                paths = nx.single_source_shortest_path(G_prime, n0)
                for n1 in l1_nodes:
                    if n1 in paths:
                        path = paths[n1]
                        if len(path) < min_dist:
                            min_dist = len(path)
                            best_path = path
                            if min_dist == 3: break
                if min_dist == 3: break
            except Exception:
                continue

        # 3. If no path exists, use fallback
        if not best_path:
            return self.fallback_sampler.sample()

        # 4. Return the middle node of the path
        return best_path[len(best_path) // 2]
