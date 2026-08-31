"""Small, dependency-local numerical primitives used across Spateo.

The public Spateo API should not depend on a second analysis framework.  This
package contains the focused AnnData, graph, sampling, and vector-field
operations that Spateo needs internally.
"""

from .anndata import fetch_X_data, log1p, normalize_total
from .graph import cluster_graph, neighbors
from .sampling import sample
from .vectorfield import (
    SparseVectorField,
    integrate_vf,
    predict_fate,
    sparse_vector_field,
)

__all__ = [
    "SparseVectorField",
    "cluster_graph",
    "fetch_X_data",
    "integrate_vf",
    "log1p",
    "neighbors",
    "normalize_total",
    "predict_fate",
    "sample",
    "sparse_vector_field",
]
