# src/graphalp/__init__.py

from .label_propagation import HarmonicLabelPropagator
from .parametric_graph_models import GaussianRandomFieldModel

__version__ = "0.1.0"

__all__ = [
    "HarmonicLabelPropagator",
    "GaussianRandomFieldModel"
]