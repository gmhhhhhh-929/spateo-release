"""Normalization routines for spatial transcriptomics preprocessing."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from anndata import AnnData
from scipy import sparse

from ...configuration import SKM
from ...spateo_logger import LoggerManager
from .qc import _axis_sum
from .utils import _record_step

logger = LoggerManager.get_main_logger()


def calculate_size_factors(
    adata: AnnData,
    layer: str = "counts",
    size_factor_key: str = "size_factor",
    library_key: Optional[str] = None,
    method: Literal["median", "mean", "target_sum"] = "median",
    target_sum: float = 1e4,
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
    out_layer: str = "norm",
    target_sum: float = 1e4,
    size_factor_key: str = "size_factor",
    inplace: bool = True,
) -> Optional[AnnData]:
    """Normalize counts per observation into a new layer without changing counts.

    Args:
        adata: Input AnnData object.
        layer: Count layer.
        out_layer: Output normalized layer.
        target_sum: Target sum per observation.
        size_factor_key: Observation key for size factors, recorded for metadata.
        inplace: If ``True``, modify ``adata`` in place.

    Returns:
        Updated AnnData when ``inplace=False``; otherwise ``None``.
    """
    adata = adata if inplace else adata.copy()
    logger.info("Normalizing total counts...")
    X = SKM.select_layer_data(adata, layer=layer, copy=True)
    totals = _axis_sum(X, axis=1).astype(float)
    scale = np.ones_like(totals, dtype=float)
    valid = totals > 0
    scale[valid] = target_sum / totals[valid]

    if sparse.issparse(X):
        X = X.tocsr(copy=True).astype(float)
        X = sparse.diags(scale).dot(X).tocsr()
    else:
        X = np.asarray(X, dtype=float) * scale[:, None]

    adata.layers[out_layer] = X
    SKM.init_uns_pp_namespace(adata)
    adata.uns[SKM.UNS_PP_KEY]["normalize_total"] = {
        "layer": layer,
        "out_layer": out_layer,
        "target_sum": target_sum,
        "size_factor_key": size_factor_key,
    }
    _record_step(adata, "normalize_total", adata.uns[SKM.UNS_PP_KEY]["normalize_total"])
    return None if inplace else adata
