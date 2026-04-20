import networkx as nx
import numpy as np
from .utils import compute_laplacian
class GaussianRandomFieldModel:
    """
    Gaussian Random Field (GRF) model for graph-based semi-supervised learning.
    """
    def __init__(self, graph, labels):
        # Store the graph and its properties
        self.graph = graph.copy()
        self.nodes = list(graph.nodes())
        self.W, self.D, self.L = compute_laplacian(self.graph)


    def fit(self, labels):
        """
        Fit the Gaussian Random Field model.
        """
        # Identify indices for labeled (l) and unlabeled (u) nodes
        self.labels = labels
        self.l_idx = [i for i, n in enumerate(self.nodes) if n in self.labels]
        self.u_idx = [i for i, n in enumerate(self.nodes) if n not in self.labels]

        if not self.l_idx:
            raise ValueError("At least one node must be labeled.")
        if not self.u_idx:
            raise ValueError("At least one node must be unlabeled. Nothing to compute here")



        # Harmonic solution: f_u = (D_uu - W_uu)^-1 * W_ul * f_l
        W_ul = self.W[np.ix_(self.u_idx, self.l_idx)]
        L_uu = self.L[np.ix_(self.u_idx, self.u_idx)]
        f_l = np.array([self.labels[self.nodes[i]] for i in self.l_idx])

        # Solve (D_uu - W_uu) * f_u = W_ul * f_l
        self.f_u = np.linalg.solve(L_uu, W_ul @ f_l)