from .parametric_graph_models import GaussianRandomFieldModel
import networkx as nx
import numpy as np
from typing import Dict, Any

class HarmonicLabelPropagator:
    """
    Harmonic predictor for graph-based semi-supervised learning using Gaussian Random Fields.
    """
    def __init__(self, graph: nx.Graph) -> None:
        """
        Initialize the propagator with the underlying graph structure.
        """
        self.model = GaussianRandomFieldModel(graph)
    
    def fit(self, labels: Dict[Any, float]) -> "HarmonicLabelPropagator":
        """
        Fit the harmonic model by parsing the provided labels.
        """
        self.model.fit(labels)
        return self
    
    def predict(self) -> Dict[Any, int]:
        """
        Predict labels for unlabeled nodes using the fitted harmonic solution.
        """
        if not hasattr(self.model, 'f_u'):
            raise RuntimeError("The model must be fitted with labels before calling predict().")
            
        self.prediction_scores = {}
        self.predicted_labels = {}
        self.risk = 0
        
        for i, idx in enumerate(self.model.u_idx):
            prob = np.clip(self.model.f_u[i], 0.0, 1.0)
            risk += min(prob, 1 - prob)
            node_id = self.model.nodes[idx]
            self.prediction_scores[node_id] = prob
            self.predicted_labels[node_id] = 1 if prob >= 0.5 else 0

        return self.predicted_labels

    def predict_probabilities(self) -> Dict[Any, float]:
        """
        Return the predicted probabilities for unlabeled nodes.
        """
        _ = self.predict()  # Ensure predictions are computed
        return self.prediction_scores