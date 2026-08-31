"""Nearest-neighbor graph construction and community detection."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def neighbors(
    adata: AnnData,
    X_data: Optional[object] = None,
    n_neighbors: int = 30,
    result_prefix: Optional[str] = None,
    metric: str = "euclidean",
) -> None:
    """Build a symmetric k-nearest-neighbor graph in AnnData conventions."""

    X = adata.X if X_data is None else X_data
    if sparse.issparse(X):
        X = X.tocsr()
    else:
        X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] != adata.n_obs:
        raise ValueError("X_data must be a two-dimensional matrix with one row per observation.")
    if adata.n_obs < 2:
        raise ValueError("At least two observations are required to construct a neighbor graph.")

    k = min(max(int(n_neighbors), 1), adata.n_obs - 1)
    model = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    model.fit(X)
    distances, indices = model.kneighbors(X)
    distances, indices = distances[:, 1:], indices[:, 1:]

    rows = np.repeat(np.arange(adata.n_obs), k)
    cols = indices.ravel()
    values = distances.ravel()
    distance_graph = sparse.csr_matrix((values, (rows, cols)), shape=(adata.n_obs, adata.n_obs))
    distance_graph = distance_graph.maximum(distance_graph.T).tocsr()

    positive = values[values > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    weights = np.exp(-np.square(values / max(scale, np.finfo(float).eps)))
    connectivity_graph = sparse.csr_matrix((weights, (rows, cols)), shape=(adata.n_obs, adata.n_obs))
    connectivity_graph = connectivity_graph.maximum(connectivity_graph.T).tocsr()

    prefix = f"{result_prefix}_" if result_prefix else ""
    distance_key = f"{prefix}distances"
    connectivity_key = f"{prefix}connectivities"
    uns_key = f"{result_prefix}_neighbors" if result_prefix else "neighbors"
    adata.obsp[distance_key] = distance_graph
    adata.obsp[connectivity_key] = connectivity_graph
    adata.uns[uns_key] = {
        "connectivities_key": connectivity_key,
        "distances_key": distance_key,
        "indices": indices,
        "params": {"n_neighbors": k, "metric": metric, "method": "spateo"},
    }


def cluster_graph(
    adata: AnnData,
    resolution: float = 1.0,
    key_added: str = "louvain",
    method: str = "leiden",
    adjacency: Optional[object] = None,
    random_state: int = 0,
) -> None:
    """Cluster an AnnData neighbor graph with the Leiden community model.

    ``method='louvain'`` is retained as a public compatibility label.  The
    maintained implementation uses Leiden's resolution-aware RB objective for
    both labels, avoiding a second graph framework and giving deterministic
    behavior at a fixed seed.
    """

    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:  # pragma: no cover - declared installation requirements
        raise ImportError("Graph clustering requires python-igraph and leidenalg.") from exc

    if adjacency is None:
        if "connectivities" not in adata.obsp:
            neighbors(adata)
        adjacency = adata.obsp["connectivities"]
    graph_matrix = sparse.coo_matrix(adjacency)
    upper = graph_matrix.row < graph_matrix.col
    edges = list(zip(graph_matrix.row[upper].tolist(), graph_matrix.col[upper].tolist()))
    weights = graph_matrix.data[upper].astype(float).tolist()
    graph = ig.Graph(n=adata.n_obs, edges=edges, directed=False)
    if not edges:
        labels = np.arange(adata.n_obs, dtype=int)
    else:
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights,
            resolution_parameter=float(resolution),
            seed=int(random_state),
        )
        labels = np.asarray(partition.membership, dtype=int)
    adata.obs[key_added] = pd.Categorical(labels.astype(str))
    adata.uns[key_added] = {
        "method": "leiden-rb" if method.lower() == "leiden" else "leiden-rb-louvain-compatible",
        "resolution": float(resolution),
        "random_state": int(random_state),
    }
