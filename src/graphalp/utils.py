import networkx as nx
import numpy as np

def compute_laplacian(graph):
    """
    Compute the Laplacian matrix of a graph.
    """
    W = nx.adjacency_matrix(graph).toarray().astype(float)
    D = np.diag(W.sum(axis=1))
    L = D - W
    return W, D, L