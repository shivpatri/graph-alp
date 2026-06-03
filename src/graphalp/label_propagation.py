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

class SpectralLabelPropagator:
    """
    Label propagator that uses SVM classification on spectral embeddings of the graph.
    """
    def __init__(self, graph: nx.Graph, n_components: int = 2) -> None:
        """
        Initialize the propagator with the graph structure and embedding dimensions.
        """
        self.graph = graph.copy()
        self.nodes = list(graph.nodes())
        self.n_components = n_components
        
        # Direct computation using networkx and numpy
        self.W = nx.adjacency_matrix(self.graph).toarray().astype(float)
        self.L = nx.laplacian_matrix(self.graph).toarray().astype(float)
        
        # Pre-compute spectral embedding for all nodes
        self.embeddings = self._compute_spectral_embedding()
    
    def _compute_spectral_embedding(self) -> np.ndarray:
        """
        Compute the spectral embedding (eigenvectors of the Laplacian).
        """
        # Compute the spectral embedding of the graph
        eigenvalues, eigenvectors = np.linalg.eigh(self.L)
        
        # Sort the eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Return the first n_components
        return eigenvectors[:, :self.n_components]

    def fit(self, X: Any, y: Any) -> "SpectralLabelPropagator":
        """
        Train an SVM on the spectral embeddings of the labeled nodes.
        """
        self.labels = dict(zip(X, y))
        self.l_idx = [i for i, n in enumerate(self.nodes) if n in self.labels]
        
        if not self.l_idx:
            raise ValueError("At least one node must be labeled.")

        # Prepare training data from embeddings
        X_train = self.embeddings[self.l_idx]
        y_train = np.array(y)
        
        # Train SVM (lazy import to avoid dependency if not used)
        from sklearn.svm import SVC
        self.svc = SVC(probability=True, kernel='rbf')
        self.svc.fit(X_train, y_train)
        
        return self

    def predict_probabilities(self, X: List[Any]) -> List[float]:
        """
        Return the predicted probabilities (confidence) for the provided nodes.
        """
        if not hasattr(self, 'svc'):
            raise RuntimeError("The model must be fitted before calling predict().")
            
        # Map node IDs to indices
        node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        indices = [node_to_idx[node] for node in X]
        
        X_test = self.embeddings[indices]
        # Get probability for class 1
        probs = self.svc.predict_proba(X_test)[:, 1]
        
        return probs.tolist()

    def predict(self, X: List[Any]) -> List[int]:
        """
        Predict labels for the provided nodes using the trained SVM.
        """
        if not hasattr(self, 'svc'):
            raise RuntimeError("The model must be fitted before calling predict().")
            
        node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        indices = [node_to_idx[node] for node in X]
        
        X_test = self.embeddings[indices]
        preds = self.svc.predict(X_test)
        
        return preds.astype(int).tolist()


class GCNLabelPropagator:
    """
    Graph Convolutional Label Propagator based on Kipf & Welling's normalized adjacency operator.
    Propagates labels using: H^(l+1) = D~^-1/2 * A~ * D~^-1/2 * H^(l)
    where A~ = A + I is the adjacency matrix with self-loops.
    """
    def __init__(self, graph: nx.Graph, n_layers: int = 2) -> None:
        """
        Initialize the propagator with the underlying graph structure.
        """
        self.graph = graph.copy()
        self.nodes = list(graph.nodes())
        self.n_layers = n_layers
        
        # Build adjacency matrix A
        A = nx.adjacency_matrix(self.graph).toarray().astype(float)
        # Add self-loops: A~ = A + I
        self.A_tilde = A + np.eye(len(self.nodes))
        
        # Build diagonal degree matrix D~
        degrees = self.A_tilde.sum(axis=1)
        D_tilde_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        
        # Build Kipf & Welling's Symmetric Normalized Adjacency Operator: D~^-1/2 * A~ * D~^-1/2
        self.A_hat = D_tilde_inv_sqrt @ self.A_tilde @ D_tilde_inv_sqrt
        
    def fit(self, X: Any, y: Any) -> "GCNLabelPropagator":
        """
        Propagate label indicators through multiple Graph Convolutional layers.
        """
        self.labels = dict(zip(X, y))
        N = len(self.nodes)
        
        # Compute dynamic geodesic coverage hops (L) based on seed set proximity
        shortest_paths = nx.multi_source_dijkstra_path_length(self.graph, list(self.labels.keys()))
        self.n_layers = max(shortest_paths.values()) if shortest_paths else 2
        if self.n_layers == 0:
            self.n_layers = 2
            
        # Initialize label indicator matrix H_0 of shape N x 2
        # Class 0 seed -> [1.0, 0.0]
        # Class 1 seed -> [0.0, 1.0]
        # Unlabeled node -> [0.0, 0.0] (no initial features to allow scale-invariant diffusion)
        self.H = np.zeros((N, 2))
        for i, node in enumerate(self.nodes):
            if node in self.labels:
                if self.labels[node] == 0:
                    self.H[i] = [1.0, 0.0]
                else:
                    self.H[i] = [0.0, 1.0]
                
        # Propagate through L layers of Kipf & Welling convolutions
        for _ in range(self.n_layers):
            self.H = self.A_hat @ self.H
            
        # Convert raw representation H to soft probabilities using scale-invariant L1 row normalization
        self.f = {}
        for i, node in enumerate(self.nodes):
            row = self.H[i]
            row_sum = row.sum()
            if row_sum > 0:
                prob_1 = row[1] / row_sum
            else:
                prob_1 = 0.5 # Neutral grey if no signal has reached
            self.f[node] = prob_1
            
        return self

    def predict_probabilities(self, X: List[Any]) -> List[float]:
        """
        Return the predicted probabilities for the requested nodes.
        """
        if not hasattr(self, 'f'):
            raise RuntimeError("The model must be fitted before calling predict().")
            
        probs = []
        for node in X:
            if node not in self.f:
                raise ValueError(f"Node {node} was not part of the graph used for fitting.")
            probs.append(float(self.f[node]))
            
        return probs

    def predict(self, X: List[Any]) -> List[int]:
        """
        Predict hard labels (0 or 1) based on GCN convolved probabilities.
        """
        probs = self.predict_probabilities(X)
        return [1 if p >= 0.5 else 0 for p in probs]
