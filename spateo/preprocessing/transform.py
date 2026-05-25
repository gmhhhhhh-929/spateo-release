"""Transformations for spatial preprocessing layers."""

from __future__ import annotations

from typing import Optional

import numpy as np
from anndata import AnnData
from scipy import sparse
from sklearn.utils.sparsefuncs import inplace_column_scale, mean_variance_axis

from ..configuration import SKM
from ..spateo_logger import LoggerManager
from .utils import _record_step

logger = LoggerManager.get_main_logger()


def log1p_layer(
    adata: AnnData,
    layer: str = "norm",
    out_layer: str = "log1p_norm",
    set_X: bool = True,
    inplace: bool = True,
) -> Optional[AnnData]:
    """Apply ``log1p`` to a layer and store the result in another layer.

    Args:
        adata: Input AnnData object.
        layer: Input layer.
        out_layer: Output layer.
        set_X: Whether to also set ``adata.X`` to the log-transformed data.
        inplace: If ``True``, modify ``adata`` in place.

    Returns:
        Updated AnnData when ``inplace=False``; otherwise ``None``.
    """
    adata = adata if inplace else adata.copy()
    logger.info("Applying log1p transform...")
    X = SKM.select_layer_data(adata, layer=layer, copy=True)
    if sparse.issparse(X):
        X = X.tocsr(copy=True).astype(float)
        X.data = np.log1p(X.data)
    else:
        X = np.log1p(np.asarray(X, dtype=float))
    adata.layers[out_layer] = X
    if set_X:
        adata.X = X.copy()
    SKM.init_uns_pp_namespace(adata)
    adata.uns[SKM.UNS_PP_KEY]["log1p"] = {"layer": layer, "out_layer": out_layer, "set_X": set_X}
    _record_step(adata, "log1p_layer", adata.uns[SKM.UNS_PP_KEY]["log1p"])
    return None if inplace else adata


def scale_layer(
    adata: AnnData,
    layer: str = "log1p_norm",
    out_layer: str = "scale",
    zero_center: bool = True,
    max_value: Optional[float] = 10,
    inplace: bool = True,
) -> Optional[AnnData]:
    """Scale a layer to unit variance and optionally zero center it.

    Args:
        adata: Input AnnData object.
        layer: Input layer.
        out_layer: Output layer.
        zero_center: Whether to subtract feature means.
        max_value: Optional clipping value.
        inplace: If ``True``, modify ``adata`` in place.

    Returns:
        Updated AnnData when ``inplace=False``; otherwise ``None``.
    """
    adata = adata if inplace else adata.copy()
    logger.info("Scaling expression layer...")
    X = SKM.select_layer_data(adata, layer=layer, copy=True)
    if sparse.issparse(X):
        X = X.tocsr(copy=True).astype(float)
        means, vars_ = mean_variance_axis(X, axis=0)
        std = np.sqrt(vars_)
        std[std == 0] = 1
        if zero_center:
            logger.warning("`scale_layer(zero_center=True)` densifies sparse input.")
            X = X.toarray()
            X -= means
            X /= std
        else:
            inplace_column_scale(X, 1 / std)
    else:
        X = np.asarray(X, dtype=float)
        means = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1
        if zero_center:
            X = X - means
        X = X / std
    if max_value is not None:
        if sparse.issparse(X):
            X.data = np.clip(X.data, -max_value, max_value)
        else:
            X = np.clip(X, -max_value, max_value)
    adata.layers[out_layer] = X
    SKM.init_uns_pp_namespace(adata)
    adata.uns[SKM.UNS_PP_KEY]["scale"] = {
        "layer": layer,
        "out_layer": out_layer,
        "zero_center": zero_center,
        "max_value": max_value,
    }
    _record_step(adata, "scale_layer", adata.uns[SKM.UNS_PP_KEY]["scale"])
    return None if inplace else adata
