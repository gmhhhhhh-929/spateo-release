"""Spatial feature selection for preprocessing."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from anndata import AnnData
from scipy import sparse
from sklearn.utils.sparsefuncs import mean_variance_axis

from ...configuration import SKM
from ...spateo_logger import LoggerManager
from .qc import _axis_sum
from .utils import _record_step

logger = LoggerManager.get_main_logger()


def _mean_var(X: object) -> tuple[np.ndarray, np.ndarray]:
    if sparse.issparse(X):
        means, variances = mean_variance_axis(X, axis=0)
        return np.asarray(means).ravel(), np.asarray(variances).ravel()
    X = np.asarray(X)
    return X.mean(axis=0), X.var(axis=0)


def _technical_gene_mask(var_names: object) -> np.ndarray:
    names = var_names.astype(str)
    prefixes = ("MT-", "mt-", "Mt-", "RPS", "RPL", "Rps", "Rpl", "HB", "Hb")
    mask = np.zeros(len(names), dtype=bool)
    for prefix in prefixes:
        mask |= np.asarray(names.str.startswith(prefix), dtype=bool)
    return mask


def _top_mask(scores: np.ndarray, n_top: int, allowed: Optional[np.ndarray] = None) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores[~np.isfinite(scores)] = -np.inf
    if allowed is None:
        allowed = np.ones(scores.size, dtype=bool)
    available = np.where(allowed)[0]
    n_top = min(max(1, n_top), available.size)
    mask = np.zeros(scores.size, dtype=bool)
    if available.size:
        order = available[np.argsort(scores[available])[::-1][:n_top]]
        mask[order] = True
    return mask


def _moran_i_scores(X: object, W: sparse.spmatrix, chunk_size: int = 256) -> np.ndarray:
    W = W.tocsr()
    W.setdiag(0)
    W.eliminate_zeros()
    s0 = float(W.sum())
    if s0 <= 0:
        return np.zeros(X.shape[1], dtype=float)
    n = X.shape[0]
    scores = np.zeros(X.shape[1], dtype=float)
    for start in range(0, X.shape[1], chunk_size):
        stop = min(start + chunk_size, X.shape[1])
        block = X[:, start:stop].toarray() if sparse.issparse(X) else np.asarray(X[:, start:stop])
        block = block.astype(float, copy=False)
        block -= block.mean(axis=0)
        denom = np.sum(block * block, axis=0)
        wx = W.dot(block)
        num = np.sum(block * wx, axis=0)
        valid = denom > 0
        chunk_scores = np.zeros(stop - start, dtype=float)
        chunk_scores[valid] = (n / s0) * (num[valid] / denom[valid])
        scores[start:stop] = chunk_scores
    return scores


def select_spatial_features(
    adata: AnnData,
    layer: str = "log1p_norm",
    method: Literal["hvg", "svg", "hvg_svg_union", "hvg_svg_intersection", "all"] = "hvg",
    n_top_genes: int = 3000,
    batch_key: Optional[str] = None,
    spatial_connectivities_key: str = "spatial_connectivities",
    feature_key: str = "use_for_pca",
    inplace: bool = True,
) -> Optional[AnnData]:
    """Select genes for PCA using HVG, SVG, or combined strategies.

    Args:
        adata: Input AnnData object.
        layer: Expression layer used for feature selection.
        method: Feature selection method.
        n_top_genes: Number of top features to select.
        batch_key: Optional batch key, recorded for provenance.
        spatial_connectivities_key: Spatial graph key used by SVG methods.
        feature_key: Output key in ``adata.var``.
        inplace: If ``True``, modify ``adata`` in place.

    Returns:
        Updated AnnData when ``inplace=False``; otherwise ``None``.
    """
    adata = adata if inplace else adata.copy()
    logger.info(f"Selecting spatial features with method `{method}`...")
    X = SKM.select_layer_data(adata, layer=layer, copy=False)
    means, variances = _mean_var(X)
    dispersions = variances / np.maximum(means, np.finfo(float).eps)
    finite = np.isfinite(dispersions)
    disp_norm = np.zeros_like(dispersions, dtype=float)
    if finite.any():
        disp_norm[finite] = (dispersions[finite] - np.mean(dispersions[finite])) / (
            np.std(dispersions[finite]) + np.finfo(float).eps
        )

    technical = _technical_gene_mask(adata.var_names)
    expressed = _axis_sum(X, axis=0) > 0
    allowed = expressed & ~technical
    if not allowed.any():
        allowed = expressed if expressed.any() else np.ones(adata.n_vars, dtype=bool)

    hvg = _top_mask(disp_norm, n_top_genes, allowed=allowed)
    adata.var[SKM.VAR_HIGHLY_VARIABLE_KEY] = hvg.astype(bool)
    adata.var["means"] = means
    adata.var["dispersions"] = dispersions
    adata.var["dispersions_norm"] = disp_norm

    svg = np.zeros(adata.n_vars, dtype=bool)
    moran = np.zeros(adata.n_vars, dtype=float)
    if method in {"svg", "hvg_svg_union", "hvg_svg_intersection"}:
        if spatial_connectivities_key not in adata.obsp:
            logger.warning("Spatial graph is missing; falling back to HVG scores for SVG selection.")
            moran = disp_norm.copy()
        else:
            moran = _moran_i_scores(X, adata.obsp[spatial_connectivities_key])
        svg = _top_mask(moran, n_top_genes, allowed=allowed)
    adata.var[SKM.VAR_SPATIALLY_VARIABLE_KEY] = svg.astype(bool)
    adata.var[SKM.VAR_MORAN_I_KEY] = moran
    adata.var[SKM.VAR_SPATIAL_SCORE_KEY] = moran

    if method == "hvg":
        selected = hvg
    elif method == "svg":
        selected = svg
    elif method == "hvg_svg_union":
        selected = hvg | svg
    elif method == "hvg_svg_intersection":
        selected = hvg & svg
    elif method == "all":
        selected = np.ones(adata.n_vars, dtype=bool)
    else:
        raise ValueError("Unknown feature selection method.")

    if not selected.any():
        logger.warning("No features selected; falling back to genes with the highest total expression.")
        selected = _top_mask(_axis_sum(X, axis=0), min(n_top_genes, adata.n_vars), allowed=expressed)

    adata.var[feature_key] = selected.astype(bool)
    SKM.init_uns_pp_namespace(adata)
    adata.uns[SKM.UNS_PP_KEY]["feature_selection"] = {
        "layer": layer,
        "method": method,
        "n_top_genes": n_top_genes,
        "batch_key": batch_key,
        "spatial_connectivities_key": spatial_connectivities_key,
        "feature_key": feature_key,
    }
    _record_step(adata, "select_spatial_features", adata.uns[SKM.UNS_PP_KEY]["feature_selection"])
    return None if inplace else adata
