"""Python-native SCTransform-style normalization for spatial preprocessing.

This module intentionally does not call R or rpy2.  It follows the Python
SCTransform/SCTransformPy idea used in ``spateo/ref/sctransform.py``: estimate a
sequencing-depth-aware negative-binomial model, regularize gene dispersions, and
store clipped Pearson residuals for downstream PCA.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union
from warnings import warn

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from ...configuration import SKM
from ...spateo_logger import LoggerManager
from ..utils import _record_step

logger = LoggerManager.get_main_logger()
_EPS = np.finfo(float).eps


def _as_csr_counts(X: object) -> sparse.csr_matrix:
    """Return a float CSR matrix without densifying input."""
    if sparse.issparse(X):
        return X.tocsr(copy=True).astype(float)
    return sparse.csr_matrix(np.asarray(X, dtype=float))


def _is_nonnegative_integer_matrix(X: sparse.csr_matrix, max_check: int = 100_000) -> bool:
    """Check whether a sparse matrix looks like raw count data."""
    data = X.data[:max_check]
    if data.size == 0:
        return True
    return bool(np.all(data >= 0) and np.all(np.equal(np.mod(data, 1), 0)))


def _top_mask(scores: np.ndarray, n_top: int) -> np.ndarray:
    """Return a boolean mask for the highest scoring entries."""
    scores = np.asarray(scores, dtype=float)
    scores[~np.isfinite(scores)] = -np.inf
    n_top = min(max(1, int(n_top)), scores.size)
    mask = np.zeros(scores.size, dtype=bool)
    if scores.size:
        mask[np.argsort(scores)[::-1][:n_top]] = True
    return mask


def gmean(X: sparse.spmatrix, axis: int = 0, eps: Union[int, float] = 1) -> np.ndarray:
    """Compute geometric means for sparse counts, matching the reference helper."""
    X_work = X.copy().asfptype()
    if X_work.nnz == 0:
        return np.zeros(X_work.shape[1 if axis == 0 else 0], dtype=float)
    X_work.data[:] = np.log(X_work.data + eps)
    return np.exp(np.asarray(X_work.mean(axis)).ravel()) - eps


def _regularized_theta(
    means: np.ndarray,
    variances: np.ndarray,
    log10_gmean: np.ndarray,
    n_bins: int = 20,
    min_theta: float = 1e-7,
) -> np.ndarray:
    """Estimate and smooth gene-wise NB theta values using expression bins."""
    raw_theta = np.full(means.shape, 1e6, dtype=float)
    overdispersed = variances > means + _EPS
    raw_theta[overdispersed] = means[overdispersed] ** 2 / np.maximum(
        variances[overdispersed] - means[overdispersed], _EPS
    )
    raw_theta[~np.isfinite(raw_theta) | (raw_theta < min_theta)] = min_theta

    finite = np.isfinite(log10_gmean)
    if finite.sum() < 3:
        return raw_theta
    order = np.argsort(log10_gmean[finite])
    finite_idx = np.where(finite)[0][order]
    bins = np.array_split(finite_idx, min(n_bins, finite_idx.size))
    smooth = raw_theta.copy()
    for idx in bins:
        if idx.size:
            smooth[idx] = np.median(raw_theta[idx])
    smooth[~np.isfinite(smooth) | (smooth < min_theta)] = min_theta
    return smooth


def _cell_attributes(X: sparse.csr_matrix, obs_names: pd.Index) -> pd.DataFrame:
    """Compute cell-level attributes stored by the Python SCT workflow."""
    umi = np.asarray(X.sum(axis=1)).ravel().astype(float)
    detected = np.asarray((X > 0).sum(axis=1)).ravel().astype(float)
    safe_umi = np.maximum(umi, 1.0)
    safe_detected = np.maximum(detected, 1.0)
    umi_per_gene = safe_umi / safe_detected
    return pd.DataFrame(
        {
            "umi_sct": umi,
            "log_umi_sct": np.log10(safe_umi),
            "gene_sct": detected,
            "log_gene_sct": np.log10(safe_detected),
            "umi_per_gene_sct": umi_per_gene,
            "log_umi_per_gene_sct": np.log10(np.maximum(umi_per_gene, _EPS)),
        },
        index=obs_names,
    )


def sctransform_core(
    adata: AnnData,
    layer: str = SKM.X_LAYER,
    out_layer: str = "sctransform",
    min_cells: int = 5,
    n_genes: int = 2000,
    theta: Optional[float] = None,
    clip: Optional[float] = None,
    set_X: bool = False,
    inplace: bool = True,
) -> Optional[AnnData]:
    """Run a Python-native SCTransform-style normalization.

    Args:
        adata: Input AnnData object.
        layer: Count layer. Defaults to ``"X"`` and reads ``adata.X``.
        out_layer: Layer where clipped Pearson residuals are stored.
        min_cells: Genes detected in fewer observations are excluded from theta regularization but kept in output.
        n_genes: Number of highest residual-variance genes marked in ``var['use_for_pca']``.
        theta: Optional fixed NB overdispersion. If ``None``, estimate and regularize gene-wise theta.
        clip: Positive residual clipping threshold. ``None`` uses ``sqrt(n_obs / 30)`` as in the reference workflow.
        set_X: Whether to also set ``adata.X`` to the SCT layer.
        inplace: If ``True``, modify ``adata`` in place.

    Returns:
        Updated AnnData when ``inplace=False``; otherwise ``None``.
    """
    target = adata if inplace else adata.copy()
    logger.info("Running Python SCTransform-style normalization...")

    X = _as_csr_counts(SKM.select_layer_data(target, layer=layer, copy=False))
    X.eliminate_zeros()
    if not _is_nonnegative_integer_matrix(X):
        warn("SCTransform expects raw non-negative integer count data.", UserWarning, stacklevel=2)

    n_obs, n_vars = X.shape
    if n_obs == 0 or n_vars == 0:
        target.layers[out_layer] = sparse.csr_matrix(X.shape, dtype=float)
        return None if inplace else target

    cell_attrs = _cell_attributes(X, target.obs_names)
    for column in cell_attrs.columns:
        target.obs[column] = cell_attrs[column].values

    gene_counts = np.asarray((X > 0).sum(axis=0)).ravel().astype(int)
    gene_totals = np.asarray(X.sum(axis=0)).ravel().astype(float)
    total = float(gene_totals.sum())
    means = gene_totals / max(n_obs, 1)

    X_sq = X.copy()
    X_sq.data **= 2
    second_moment = np.asarray(X_sq.mean(axis=0)).ravel()
    variances = np.maximum(second_moment - means**2, 0)

    log10_gmean = np.zeros(n_vars, dtype=float)
    eligible = gene_counts >= min_cells
    if eligible.any():
        log_gmean_values = gmean(X[:, eligible], axis=0, eps=1)
        log10_gmean[eligible] = np.log10(np.maximum(log_gmean_values, _EPS))

    if theta is not None:
        if theta <= 0:
            raise ValueError("SCTransform requires `theta > 0` when a fixed theta is provided.")
        theta_values = np.full(n_vars, float(theta), dtype=float)
    else:
        theta_values = _regularized_theta(means, variances, log10_gmean)

    if clip is None:
        clip = float(np.sqrt(max(n_obs / 30, 1)))
    if clip < 0:
        raise ValueError("SCTransform requires `clip >= 0` or `clip=None`.")

    if total <= 0:
        residuals = sparse.csr_matrix(X.shape, dtype=float)
    else:
        residuals = X.copy().astype(float)
        rows, cols = residuals.nonzero()
        expected = np.asarray(cell_attrs["umi_sct"])[rows] * gene_totals[cols] / total
        var = expected + (expected**2 / theta_values[cols])
        var[var <= 0] = 1
        residuals.data = (residuals.data - expected) / np.sqrt(var)
        residuals.data[residuals.data < 0] = 0
        if np.isfinite(clip):
            residuals.data[residuals.data > clip] = clip
        residuals.eliminate_zeros()
        residuals = residuals.tocsr()

    target.layers[out_layer] = residuals
    if set_X:
        target.X = residuals.copy()

    residual_sq = residuals.copy()
    residual_sq.data **= 2
    residual_variance = np.asarray(residual_sq.mean(axis=0)).ravel() - np.square(
        np.asarray(residuals.mean(axis=0)).ravel()
    )
    residual_variance = np.maximum(residual_variance, 0)
    use_for_pca = _top_mask(residual_variance, min(n_genes, n_vars))

    target.var["theta_sct"] = theta_values
    target.var["log10_gmean_sct"] = log10_gmean
    target.var["sct_residual_variance"] = residual_variance
    target.var["sct_score"] = residual_variance
    target.var[SKM.VAR_USE_FOR_PCA_KEY] = use_for_pca.astype(bool)
    target.var[SKM.VAR_HIGHLY_VARIABLE_KEY] = use_for_pca.astype(bool)

    SKM.init_uns_pp_namespace(target)
    target.uns[SKM.UNS_PP_KEY]["sctransform"] = {
        "layer": layer,
        "out_layer": out_layer,
        "min_cells": min_cells,
        "n_genes": int(min(n_genes, n_vars)),
        "theta": theta,
        "clip": clip,
        "set_X": set_X,
        "implementation": "python_offset_nb_pearson_residuals",
    }
    _record_step(target, "sctransform", target.uns[SKM.UNS_PP_KEY]["sctransform"])
    return None if inplace else target


def sctransform(
    adata: AnnData,
    layer: str = SKM.X_LAYER,
    out_layer: str = "sctransform",
    min_cells: int = 5,
    n_top_genes: int = 2000,
    theta: Optional[float] = None,
    clip: Optional[float] = None,
    set_X: bool = False,
    inplace: bool = True,
    layers: Optional[Sequence[str]] = None,
    output_layer: Optional[str] = None,
    **kwargs: object,
) -> Optional[AnnData]:
    """Run Python-native SCTransform-style normalization.

    Args:
        adata: Input AnnData object.
        layer: Count layer used when ``layers`` is not provided. Defaults to ``"X"``.
        out_layer: Output layer used when ``layers`` is not provided.
        min_cells: Minimum observations detecting a gene for theta regularization.
        n_top_genes: Number of genes marked in ``var['use_for_pca']``.
        theta: Optional fixed NB overdispersion.
        clip: Optional positive clipping threshold.
        set_X: Whether to set ``adata.X`` to the SCT result.
        inplace: If ``True``, modify ``adata`` in place.
        layers: Backward-compatible list of layers to transform.
        output_layer: Backward-compatible output layer name for a single input layer.
        **kwargs: Accepted for API compatibility; currently ignored.

    Returns:
        Updated AnnData when ``inplace=False``; otherwise ``None``.
    """
    if kwargs:
        logger.warning(f"Ignoring unsupported SCTransform keyword arguments: {sorted(kwargs)}")
    target = adata if inplace else adata.copy()
    if layers is None:
        sctransform_core(
            target,
            layer=layer,
            out_layer=output_layer or out_layer,
            min_cells=min_cells,
            n_genes=n_top_genes,
            theta=theta,
            clip=clip,
            set_X=set_X,
            inplace=True,
        )
    else:
        for i, current_layer in enumerate(layers):
            if output_layer is not None and len(layers) == 1:
                current_out = output_layer
            else:
                current_out = f"{current_layer}_sctransform"
            if current_layer == SKM.X_LAYER:
                current_out = output_layer or out_layer if i == 0 else current_out
            sctransform_core(
                target,
                layer=current_layer,
                out_layer=current_out,
                min_cells=min_cells,
                n_genes=n_top_genes,
                theta=theta,
                clip=clip,
                set_X=set_X and i == 0,
                inplace=True,
            )
    return None if inplace else target
