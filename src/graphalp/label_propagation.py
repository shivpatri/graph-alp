import networkx as nx
import numpy as np
from typing import Dict, Any, List

class HarmonicLabelPropagator:
    """
    Harmonic predictor for graph-based semi-supervised learning using Gaussian Random Fields.
    """
    def __init__(self, graph: nx.Graph) -> None:
        """
        Initialize the propagator with the underlying graph structure.
        """
        self.graph = graph.copy()
        self.nodes = list(graph.nodes())
        
        # Direct computation using networkx and numpy
        self.W = nx.adjacency_matrix(self.graph).toarray().astype(float)
        self.L = nx.laplacian_matrix(self.graph).toarray().astype(float)
    
    def fit(self, X: Any, y: Any) -> "HarmonicLabelPropagator":
        """
        Fit the harmonic model by parsing the provided labels.
        
        Args:
            X: Iterable of labeled node identifiers.
            y: Iterable of corresponding labels.
        """
        # Identify indices for labeled (l) and unlabeled (u) nodes
        self.labels = dict(zip(X, y))
        self.l_idx = [i for i, n in enumerate(self.nodes) if n in self.labels]
        self.u_idx = [i for i, n in enumerate(self.nodes) if n not in self.labels]

        if not self.l_idx:
            raise ValueError("At least one node must be labeled.")
        if not self.u_idx:
            raise ValueError("At least one node must be unlabeled. Nothing to compute here")

        # Harmonic solution logic
        W_ul = self.W[np.ix_(self.u_idx, self.l_idx)]
        L_uu = self.L[np.ix_(self.u_idx, self.u_idx)]
        f_l = np.array([self.labels[self.nodes[i]] for i in self.l_idx])

        # Solve L_uu * f_u = W_ul * f_l (since L_ul = -W_ul)
        f_u = np.linalg.solve(L_uu, W_ul @ f_l)
        
        # Create a dictionary of nodes and corresponding prediction scores (f)
        self.f = {}
        # Labeled nodes
        for i, idx in enumerate(self.l_idx):
            self.f[self.nodes[idx]] = f_l[i]
        # Unlabeled nodes
        for i, idx in enumerate(self.u_idx):
            self.f[self.nodes[idx]] = f_u[i]
            
        return self
    
    def predict_probabilities(self, X: List[Any]) -> List[float]:
        """
        Return the predicted probabilities for the provided nodes.
        
        Args:
            X: List of node identifiers to get probabilities for.
            
        Returns:
            List of predicted probabilities.
        """
        if not hasattr(self, 'f'):
            raise RuntimeError("The model must be fitted before calling predict().")
            
        probs = []
        for node in X:
            if node not in self.f:
                raise ValueError(f"Node {node} was not part of the graph used for fitting.")
            probs.append(float(np.clip(self.f[node], 0.0, 1.0)))
            
        return probs

    def predict(self, X: List[Any]) -> List[int]:
        """
        Predict labels for the provided nodes.
        
        Args:
            X: List of node identifiers to predict labels for.
            
        Returns:
            List of predicted labels (0 or 1).
        """
        probs = self.predict_probabilities(X)
        return [1 if p >= 0.5 else 0 for p in probs]


class MinCutLabelPropagator():
    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph.copy()
        self.nodes = list(graph.nodes())

    def fit(self, X: Any, y: Any) -> "MinCutLabelPropagator":
        """
        Fit the Min-Cut model by finding the minimum s-t cut directly on self.graph.
        """
        # Identify labeled nodes
        self.labels = dict(zip(X, y))
        pos_nodes = [n for n, l in self.labels.items() if l == 1]
        neg_nodes = [n for n, l in self.labels.items() if l == 0]

        if not pos_nodes or not neg_nodes:
            raise ValueError("At least one positive (1) and one negative (0) label are required for min-cut.")

        # Add source and sink nodes directly to self.graph
        source, sink = "_SOURCE_NODE_", "_SINK_NODE_"
        self.graph.add_node(source)
        self.graph.add_node(sink)

        # Connect source to positive nodes and sink to negative nodes with infinite capacity
        inf_capacity = float('inf')
        for node in pos_nodes:
            self.graph.add_edge(source, node, capacity=inf_capacity)
        for node in neg_nodes:
            self.graph.add_edge(node, sink, capacity=inf_capacity)

        # Ensure existing edges have capacity
        for u, v, d in self.graph.edges(data=True):
            if 'capacity' not in d:
                d['capacity'] = d.get('weight', 1.0)

        # Compute minimum s-t cut
        _, (self.reachable, self.non_reachable) = nx.minimum_cut(self.graph, source, sink)
        
        # Identify the min-cut wall (edges between the two sets, excluding source/sink edges)
        self.mincut_wall = []
        for u, v in self.graph.edges():
            if u in [source, sink] or v in [source, sink]:
                continue
            if (u in self.reachable and v in self.non_reachable) or \
               (u in self.non_reachable and v in self.reachable):
                self.mincut_wall.append((u, v))

        # Cleanup: remove source and sink nodes from self.graph
        self.graph.remove_node(source)
        self.graph.remove_node(sink)
                
        return self

    def predict(self, X: List[Any]) -> List[int]:
        """
        Predict hard labels (0 or 1) for the provided nodes based on the cut partition.
        """
        if not hasattr(self, 'reachable'):
            raise RuntimeError("The model must be fitted before calling predict().")
            
        predictions = []
        for node in X:
            if node not in self.graph:
                raise ValueError(f"Node {node} is not in the graph.")
            
            if node in self.reachable:
                predictions.append(1)
            else:
                predictions.append(0)
                
        return predictions