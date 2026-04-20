from .parametric_graph_models import GaussianRandomFieldModel
import networkx as nx
import numpy as np

class HarmonicLabelPropagator:
    """
    Harmonic predictor for graph-based semi-supervised learning using Gaussian Random Fields.
    """
    def __init__(self, graph):
        """
        Initialize the propagator with the underlying graph structure.
        """
        self.model = GaussianRandomFieldModel(graph)
    
    def fit(self, labels):
        """
        Fit the harmonic model by parsing the provided labels.
        """
        self.model.fit(labels)
        return self
    
    def predict(self):
        """
        Predict labels for unlabeled nodes using the fitted harmonic solution.
        """
        if not hasattr(self.model, 'f_u'):
            raise RuntimeError("The model must be fitted with labels before calling predict().")
            
        prediction_scores = {}
        predicted_labels = {}
        
        for i, idx in enumerate(self.model.u_idx):
            prob = np.clip(self.model.f_u[i], 0.0, 1.0)
            node_id = self.model.nodes[idx]
            prediction_scores[node_id] = prob
            predicted_labels[node_id] = 1 if prob >= 0.5 else 0

        return predicted_labels, prediction_scores