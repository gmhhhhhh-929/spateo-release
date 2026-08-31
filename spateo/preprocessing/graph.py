"""Spatial and expression neighbor graphs."""

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
from anndata import AnnData
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from ..configuration import SKM
from ..spateo_logger import LoggerManager
from .utils import _record_step

logger = LoggerManager.get_main_logger()


def _symmetrize(dist: sparse.csr_matrix, conn: sparse.csr_matrix) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    conn = conn.maximum(conn.T).tocsr()
    dist = dist.maximum(dist.T).tocsr()
    conn_diagonal = conn.diagonal()
    dist_diagonal = dist.diagonal()
    if np.any(conn_diagonal):
        conn = (conn - sparse.diags(conn_diagonal)).tocsr()
    if np.any(dist_diagonal):
        dist = (dist - sparse.diags(dist_diagonal)).tocsr()
    conn.eliminate_zeros()
    dist.eliminate_zeros()
    return dist, conn


def spatial_neighbors(
    adata: AnnData,
    spatial_key: str = "spatial",
    library_key: Optional[str] = None,
    coord_type: Optional[Literal["grid", "generic"]] = None,
    n_neighbors: int = 6,
    radius: Optional[Union[float, tuple[float, float]]] = None,
    delaunay: bool = False,
    n_rings: int = 1,
    key_added: str = "spatial",
    inplace: bool = True,
) -> Optional[AnnData]:
    """Build a spatial neighbor graph from coordinates.

    Args:
        adata: Input AnnData object.
        spatial_key: Key in ``adata.obsm`` storing coordinates.
        library_key: Optional library/slice key; edges never cross libraries.
        coord_type: ``grid`` or ``generic`` coordinate type.
        n_neighbors: Number of nearest neighbors for kNN graph.
        radius: Optional radius or radius interval for radius graph.
        delaunay: Whether to use Delaunay triangulation for 2D coordinates.
        n_rings: Recorded for grid provenance.
        key_added: Namespace key for metadata.
        inplace: If ``True``, modify ``adata`` in place.

    Returns:
        Updated AnnData when ``inplace=False``; otherwise ``None``.
    """
    adata = adata if inplace else adata.copy()
    logger.info("Building spatial neighbor graph...")
    if n_neighbors < 1:
        raise ValueError("`n_neighbors` must be at least 1.")
    coords = SKM.ensure_spatial_key(adata, spatial_key=spatial_key)
    coord_type = coord_type or "generic"
    coords = np.asarray(coords[:, : min(coords.shape[1], 3)], dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2 or not np.isfinite(coords).all():
        raise ValueError("Spatial coordinates must be a finite numeric array with at least two columns.")
    if library_key is not None and library_key not in adata.obs:
        raise KeyError(f"`library_key={library_key!r}` is not present in `adata.obs`.")
    if radius is not None:
        limits = radius if isinstance(radius, tuple) else (0.0, radius)
        if len(limits) != 2 or limits[0] < 0 or limits[1] <= limits[0]:
            raise ValueError("`radius` must be positive, or a `(min_radius, max_radius)` pair with 0 <= min < max.")

    groups = np.asarray(adata.obs[library_key]) if library_key is not None and library_key in adata.obs else None
    group_values = np.unique(groups) if groups is not None else [None]
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    dist_vals: list[np.ndarray] = []

    for group in group_values:
        idx = np.arange(adata.n_obs) if group is None else np.where(groups == group)[0]
        if idx.size <= 1:
            continue
        group_coords = coords[idx]
        if delaunay:
            try:
                from scipy.spatial import Delaunay
            except ImportError as exc:
                raise ImportError("Delaunay spatial graph requires scipy.spatial.") from exc
            if group_coords.shape[1] < 2 or idx.size < 3:
                continue
            try:
                tri = Delaunay(group_coords[:, :2])
                edge_set: set[tuple[int, int]] = set()
                for simplex in tri.simplices:
                    for a in range(len(simplex)):
                        for b in range(a + 1, len(simplex)):
                            edge_set.add((simplex[a], simplex[b]))
                            edge_set.add((simplex[b], simplex[a]))
                if not edge_set:
                    continue
                local_rows, local_cols = np.array(list(edge_set)).T
                d = np.linalg.norm(group_coords[local_rows] - group_coords[local_cols], axis=1)
            except Exception as exc:
                logger.warning(f"Delaunay graph failed ({exc}); falling back to k-nearest neighbors.")
                k = min(n_neighbors + 1, idx.size)
                nn = NearestNeighbors(n_neighbors=k).fit(group_coords)
                dists, inds = nn.kneighbors(group_coords, return_distance=True)
                local_rows = np.repeat(np.arange(idx.size), k - 1)
                local_cols = inds[:, 1:].reshape(-1)
                d = dists[:, 1:].reshape(-1)
        elif radius is not None:
            rad = radius[1] if isinstance(radius, tuple) else radius
            min_rad = radius[0] if isinstance(radius, tuple) else 0
            nn = NearestNeighbors(radius=rad)
            nn.fit(group_coords)
            dists, inds = nn.radius_neighbors(group_coords, return_distance=True)
            local_rows_list = []
            local_cols_list = []
            d_list = []
            for i, (di, ii) in enumerate(zip(dists, inds)):
                keep = (ii != i) & (di >= min_rad)
                local_rows_list.append(np.full(np.sum(keep), i, dtype=int))
                local_cols_list.append(ii[keep])
                d_list.append(di[keep])
            if not local_rows_list:
                continue
            local_rows = np.concatenate(local_rows_list)
            local_cols = np.concatenate(local_cols_list)
            d = np.concatenate(d_list)
        else:
            k = min(n_neighbors + 1, idx.size)
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(group_coords)
            dists, inds = nn.kneighbors(group_coords, return_distance=True)
            local_rows = np.repeat(np.arange(idx.size), k - 1)
            local_cols = inds[:, 1:].reshape(-1)
            d = dists[:, 1:].reshape(-1)

        rows.append(idx[local_rows])
        cols.append(idx[local_cols])
        dist_vals.append(d)

    if rows:
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        d = np.concatenate(dist_vals)
        distances = sparse.csr_matrix((d, (row, col)), shape=(adata.n_obs, adata.n_obs))
        connectivities = sparse.csr_matrix((np.ones_like(d, dtype=float), (row, col)), shape=(adata.n_obs, adata.n_obs))
    else:
        distances = sparse.csr_matrix((adata.n_obs, adata.n_obs), dtype=float)
        connectivities = sparse.csr_matrix((adata.n_obs, adata.n_obs), dtype=float)
    distances, connectivities = _symmetrize(distances, connectivities)

    adata.obsp[SKM.OBSP_SPATIAL_CONNECTIVITIES_KEY] = connectivities
    adata.obsp[SKM.OBSP_SPATIAL_DISTANCES_KEY] = distances
    SKM.init_uns_spatial_namespace(adata)
    adata.uns[SKM.UNS_SPATIAL_KEY]["neighbors"] = {
        "spatial_key": spatial_key,
        "library_key": library_key,
        "coord_type": coord_type,
        "n_neighbors": n_neighbors,
        "radius": radius,
        "delaunay": delaunay,
        "n_rings": n_rings,
        "key_added": key_added,
    }
    _record_step(adata, "spatial_neighbors", adata.uns[SKM.UNS_SPATIAL_KEY]["neighbors"])
    return None if inplace else adata


def expression_neighbors(
    adata: AnnData,
    basis: str = "X_pca",
    n_neighbors: int = 15,
    key_added: str = "neighbors",
    inplace: bool = True,
) -> Optional[AnnData]:
    """Build an expression neighbor graph from an embedding.

    Args:
        adata: Input AnnData object.
        basis: Key in ``adata.obsm`` containing representation.
        n_neighbors: Number of expression neighbors.
        key_added: Metadata namespace.
        inplace: If ``True``, modify ``adata`` in place.

    Returns:
        Updated AnnData when ``inplace=False``; otherwise ``None``.
    """
    adata = adata if inplace else adata.copy()
    logger.info("Building expression neighbor graph...")
    if n_neighbors < 1:
        raise ValueError("`n_neighbors` must be at least 1.")
    if basis not in adata.obsm:
        raise KeyError(f"`adata.obsm[{basis!r}]` is required to build expression neighbors.")
    X = np.asarray(adata.obsm[basis])
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError(f"`adata.obsm[{basis!r}]` must be a finite two-dimensional array.")
    if adata.n_obs <= 1:
        distances = sparse.csr_matrix((adata.n_obs, adata.n_obs), dtype=float)
        connectivities = sparse.csr_matrix((adata.n_obs, adata.n_obs), dtype=float)
    else:
        k = min(n_neighbors + 1, adata.n_obs)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        dists, inds = nn.kneighbors(X, return_distance=True)
        rows = np.repeat(np.arange(adata.n_obs), k - 1)
        cols = inds[:, 1:].reshape(-1)
        d = dists[:, 1:].reshape(-1)
        distances = sparse.csr_matrix((d, (rows, cols)), shape=(adata.n_obs, adata.n_obs))
        connectivities = sparse.csr_matrix((np.ones_like(d), (rows, cols)), shape=(adata.n_obs, adata.n_obs))
        distances, connectivities = _symmetrize(distances, connectivities)

    adata.obsp[SKM.OBSP_CONNECTIVITIES_KEY] = connectivities
    adata.obsp[SKM.OBSP_DISTANCES_KEY] = distances
    adata.uns[key_added] = {"connectivities_key": SKM.OBSP_CONNECTIVITIES_KEY, "distances_key": SKM.OBSP_DISTANCES_KEY}
    _record_step(adata, "expression_neighbors", {"basis": basis, "n_neighbors": n_neighbors, "key_added": key_added})
    return None if inplace else adata
