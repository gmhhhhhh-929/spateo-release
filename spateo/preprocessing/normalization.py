"""Normalization routines for spatial transcriptomics preprocessing."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from anndata import AnnData
from scipy import sparse

from ..configuration import SKM
from ..spateo_logger import LoggerManager
from .qc import _axis_sum
from .utils import _record_step

logger = LoggerManager.get_main_logger()


def calculate_size_factors(
    adata: AnnData,
    layer: str = "counts",
    size_factor_key: str = "size_factor",
    library_key: Optional[str] = None,
    method: Literal["median", "mean", "target_sum"] = "median",
    target_sum: Optional[float] = 1e4,
    inplace: bool = True,
) -> Optional[AnnData]:
    """Calculate per-observation size factors from a count layer.

    Args:
        adata: Input AnnData object.
        layer: Count layer.
        size_factor_key: Output key in ``adata.obs``.
        library_key: Optional library key for per-library factors.
        method: Scaling center, one of ``median``, ``mean`` or ``target_sum``.
        target_sum: Target library size when ``method='target_sum'``.
        inplace: If ``True``, modify ``adata`` in place.

    Returns:
        Updated AnnData when ``inplace=False``; otherwise ``None``.
    """
    adata = adata if inplace else adata.copy()
    logger.info("Calculating size factors...")
    if method == "target_sum" and (target_sum is None or target_sum <= 0):
        raise ValueError("`target_sum` must be positive when `method='target_sum'`.")
    if library_key is not None and library_key not in adata.obs:
        raise KeyError(f"`library_key={library_key!r}` is not present in `adata.obs`.")
    X = SKM.select_layer_data(adata, layer=layer, copy=False)
    totals = _axis_sum(X, axis=1).astype(float)
    size_factors = np.ones(adata.n_obs, dtype=float)

    groups = np.asarray(adata.obs[library_key]) if library_key is not None and library_key in adata.obs else None
    group_values = np.unique(groups) if groups is not None else [None]
    for group in group_values:
        idx = np.arange(adata.n_obs) if group is None else np.where(groups == group)[0]
        group_totals = totals[idx]
        positive = group_totals[group_totals > 0]
        if method == "median":
            center = np.median(positive) if positive.size else 1.0
            sf = group_totals / center
        elif method == "mean":
            center = np.mean(positive) if positive.size else 1.0
            sf = group_totals / center
        elif method == "target_sum":
            assert target_sum is not None
            sf = group_totals / target_sum
        else:
            raise ValueError("`method` must be one of {'median', 'mean', 'target_sum'}.")
        sf[~np.isfinite(sf) | (sf <= 0)] = 1.0
        size_factors[idx] = sf

    adata.obs[size_factor_key] = size_factors
    SKM.init_uns_pp_namespace(adata)
    adata.uns[SKM.UNS_PP_KEY]["size_factors"] = {
        "layer": layer,
        "size_factor_key": size_factor_key,
        "library_key": library_key,
        "method": method,
        "target_sum": target_sum,
    }
    _record_step(adata, "calculate_size_factors", adata.uns[SKM.UNS_PP_KEY]["size_factors"])
    return None if inplace else adata


def normalize_total(
    adata: AnnData,
    layer: str = "counts",
    out_layer: Optional[str] = "norm",
    target_sum: Optional[float] = 1e4,
    size_factor_key: str = "size_factor",
    library_key: Optional[str] = None,
    inplace: bool = True,
) -> Optional[AnnData]:
    """Normalize counts per observation without densifying sparse input.

    ``target_sum=None`` scales each non-empty observation to the median positive
    library size.  This is the direct replacement for Dynamo's median size-factor
    normalization.  Set ``out_layer="X"`` (or ``None``) only when intentionally
    replacing ``adata.X``; the default preserves the source counts in a new layer.

    Args:
        adata: Input AnnData object.
        layer: Count layer.
        out_layer: Output normalized layer. ``"X"`` or ``None`` writes to
            ``adata.X``.
        target_sum: Target sum per observation. ``None`` uses the median positive
            total, independently within each ``library_key`` group when supplied.
        size_factor_key: Observation key for size factors, recorded for metadata.
        library_key: Optional observation key for per-library median targets.
        inplace: If ``True``, modify ``adata`` in place.

    Returns:
        Updated AnnData when ``inplace=False``; otherwise ``None``.
    """
    adata = adata if inplace else adata.copy()
    logger.info("Normalizing total counts...")
    if target_sum is not None and target_sum <= 0:
        raise ValueError("`target_sum` must be positive.")
    if library_key is not None and library_key not in adata.obs:
        raise KeyError(f"`library_key={library_key!r}` is not present in `adata.obs`.")
    X = SKM.select_layer_data(adata, layer=layer, copy=True)
    values = X.data if sparse.issparse(X) else np.asarray(X)
    if values.size and (not np.isfinite(values).all() or np.nanmin(values) < 0):
        raise ValueError("Total-count normalization requires finite, non-negative expression values.")
    totals = _axis_sum(X, axis=1).astype(float)
    scale = np.ones_like(totals, dtype=float)
    size_factors = np.ones_like(totals, dtype=float)
    groups = np.asarray(adata.obs[library_key]) if library_key is not None else None
    group_values = np.unique(groups) if groups is not None else [None]
    resolved_targets: dict[str, float] = {}
    for group in group_values:
        idx = np.arange(adata.n_obs) if group is None else np.flatnonzero(groups == group)
        group_totals = totals[idx]
        positive = group_totals[group_totals > 0]
        group_target = (
            float(target_sum) if target_sum is not None else float(np.median(positive)) if positive.size else 1.0
        )
        valid = group_totals > 0
        group_scale = np.ones(idx.size, dtype=float)
        group_factors = np.ones(idx.size, dtype=float)
        group_scale[valid] = group_target / group_totals[valid]
        group_factors[valid] = group_totals[valid] / group_target
        scale[idx] = group_scale
        size_factors[idx] = group_factors
        resolved_targets["all" if group is None else str(group)] = group_target

    if sparse.issparse(X):
        X = X.tocsr(copy=True).astype(float)
        X = sparse.diags(scale).dot(X).tocsr()
    else:
        X = np.asarray(X, dtype=float) * scale[:, None]

    if out_layer in {None, "X"}:
        adata.X = X
        output_key = "X"
    else:
        adata.layers[out_layer] = X
        output_key = out_layer
    adata.obs[size_factor_key] = size_factors
    SKM.init_uns_pp_namespace(adata)
    adata.uns[SKM.UNS_PP_KEY]["normalize_total"] = {
        "layer": layer,
        "out_layer": output_key,
        "target_sum": target_sum,
        "resolved_target_sum": resolved_targets,
        "size_factor_key": size_factor_key,
        "library_key": library_key,
    }
    _record_step(adata, "normalize_total", adata.uns[SKM.UNS_PP_KEY]["normalize_total"])
    return None if inplace else adata
