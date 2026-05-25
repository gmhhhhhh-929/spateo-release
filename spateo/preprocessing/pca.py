"""PCA for spatial preprocessing."""

from __future__ import annotations

from typing import Optional

import numpy as np
from anndata import AnnData
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD

from ..configuration import SKM
from ..spateo_logger import LoggerManager
from .utils import _record_step

logger = LoggerManager.get_main_logger()


def pca(
    adata: AnnData,
    layer: str = "log1p_norm",
    feature_key: str = "use_for_pca",
    n_pca_components: int = 50,
    pca_key: str = "X_pca",
    random_state: int = 0,
    inplace: bool = True,
) -> Optional[AnnData]:
    """Run PCA or sparse truncated SVD on selected spatial features.

    Args:
        adata: Input AnnData object.
        layer: Expression layer.
        feature_key: Boolean key in ``adata.var`` selecting features.
        n_pca_components: Requested number of components.
        pca_key: Output key in ``adata.obsm``.
        random_state: Random seed.
        inplace: If ``True``, modify ``adata`` in place.

    Returns:
        Updated AnnData when ``inplace=False``; otherwise ``None``.
    """
    adata = adata if inplace else adata.copy()
    logger.info("Running PCA for spatial preprocessing...")
    X = SKM.select_layer_data(adata, layer=layer, copy=False)
    if feature_key in adata.var:
        feature_mask = np.asarray(adata.var[feature_key], dtype=bool)
    else:
        logger.warning(f"`adata.var[{feature_key!r}]` is missing; using all genes for PCA.")
        feature_mask = np.ones(adata.n_vars, dtype=bool)
    if not feature_mask.any():
        logger.warning("No PCA features selected; using all genes.")
        feature_mask = np.ones(adata.n_vars, dtype=bool)

    X_use = X[:, feature_mask]
    max_components = min(n_pca_components, X_use.shape[0] - 1, X_use.shape[1])
    if sparse.issparse(X_use):
        max_components = min(max_components, max(1, X_use.shape[1] - 1))
    if max_components < 1:
        logger.warning("PCA skipped because there are too few observations or features.")
        adata.obsm[pca_key] = np.zeros((adata.n_obs, 0), dtype=float)
        adata.uns["PCs"] = np.zeros((adata.n_vars, 0), dtype=float)
        adata.uns["explained_variance_ratio_"] = np.array([], dtype=float)
        adata.uns["pca"] = {"params": {"layer": layer, "feature_key": feature_key, "n_pca_components": 0}}
        return None if inplace else adata

    if max_components < n_pca_components:
        logger.warning(f"Reducing PCA components from {n_pca_components} to {max_components}.")

    if sparse.issparse(X_use):
        model = TruncatedSVD(n_components=max_components, random_state=random_state)
        X_pca = model.fit_transform(X_use)
        components = model.components_
        explained = model.explained_variance_ratio_
    else:
        X_use = np.asarray(X_use, dtype=float)
        model = PCA(n_components=max_components, random_state=random_state)
        X_pca = model.fit_transform(X_use)
        components = model.components_
        explained = model.explained_variance_ratio_

    pcs = np.zeros((adata.n_vars, max_components), dtype=float)
    pcs[feature_mask, :] = components.T
    adata.obsm[pca_key] = X_pca
    adata.uns["PCs"] = pcs
    adata.uns["explained_variance_ratio_"] = explained
    adata.uns["pca"] = {
        "params": {
            "layer": layer,
            "feature_key": feature_key,
            "n_pca_components": int(max_components),
            "pca_key": pca_key,
            "random_state": random_state,
        },
        "variance_ratio": explained,
    }
    _record_step(adata, "pca", adata.uns["pca"]["params"])
    return None if inplace else adata
