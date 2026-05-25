"""Python-native Pearson residual recipes for spatial preprocessing.

The implementation follows the analytic Pearson residual recipe of Lause,
Berens and Kobak, adapted for Spateo's spatial preprocessing layers.  It does
not depend on dynamo velocity layers and never overwrites the raw count layer.
"""

from __future__ import annotations

from typing import Iterable, Optional
from warnings import warn

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from sklearn.utils.sparsefuncs import mean_variance_axis

from ...configuration import SKM
from ...spateo_logger import LoggerManager
from ..utils import _record_step

logger = LoggerManager.get_main_logger()


def _as_array(values: object) -> np.ndarray:
    """Return a one-dimensional NumPy array."""
    return np.asarray(values).ravel()


def _is_nonnegative_integer_matrix(X: object, max_check: int = 100_000) -> bool:
    """Check whether a matrix looks like raw UMI counts without densifying sparse input."""
    if sparse.issparse(X):
        data = X.data
        if data.size > max_check:
            data = data[:max_check]
    else:
        data = np.asarray(X).ravel()
        if data.size > max_check:
            data = data[:max_check]
    if data.size == 0:
        return True
    return bool(np.all(data >= 0) and np.all(np.equal(np.mod(data, 1), 0)))


def _mean_var(X: object) -> tuple[np.ndarray, np.ndarray]:
    """Compute feature means and variances for dense or sparse matrices."""
    if sparse.issparse(X):
        means, variances = mean_variance_axis(X, axis=0)
        return _as_array(means), _as_array(variances)
    X_arr = np.asarray(X)
    return X_arr.mean(axis=0), X_arr.var(axis=0)


def _rank_residual_variances(
    residual_gene_vars_by_batch: np.ndarray,
    n_top_genes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank genes by residual variance across batches."""
    n_top = min(max(1, n_top_genes), residual_gene_vars_by_batch.shape[1])
    ranks = np.argsort(np.argsort(-residual_gene_vars_by_batch, axis=1), axis=1).astype(float)
    highly_variable_nbatches = np.sum(ranks < n_top, axis=0).astype(int)
    ranks[ranks >= n_top] = np.nan
    median_rank = np.ma.median(np.ma.masked_invalid(ranks), axis=0).filled(np.nan)

    order = np.lexsort((np.nan_to_num(median_rank, nan=np.inf), -highly_variable_nbatches))
    highly_variable = np.zeros(residual_gene_vars_by_batch.shape[1], dtype=bool)
    highly_variable[order[:n_top]] = True
    return highly_variable, median_rank, highly_variable_nbatches


def compute_pearson_residuals(
    X: object,
    theta: float = 100,
    clip: Optional[float] = None,
    check_values: bool = True,
    copy: bool = False,
    chunksize: int = 512,
) -> np.ndarray:
    """Compute analytic Pearson residuals from raw count data.

    Args:
        X: Count matrix with observations in rows and genes in columns.
        theta: Shared negative-binomial overdispersion parameter. Must be positive.
        clip: Absolute clipping threshold. ``None`` uses ``sqrt(n_obs)``; ``np.inf`` disables clipping.
        check_values: Warn when the matrix does not look like non-negative integer counts.
        copy: Whether to copy ``X`` before computation.
        chunksize: Number of genes processed per chunk.

    Returns:
        Dense Pearson residual matrix with the same shape as ``X``.
    """
    if theta <= 0:
        raise ValueError("Pearson residuals require `theta > 0`.")
    if clip is None:
        clip = float(np.sqrt(X.shape[0]))
    if clip < 0:
        raise ValueError("Pearson residuals require `clip >= 0` or `clip=None`.")

    X_work = X.copy() if copy else X
    if check_values and not _is_nonnegative_integer_matrix(X_work):
        warn("Pearson residuals expect raw non-negative integer count data.", UserWarning, stacklevel=2)

    sums_genes = _as_array(X_work.sum(axis=0)).astype(float)
    sums_cells = _as_array(X_work.sum(axis=1)).astype(float)
    total = float(sums_genes.sum())
    residuals = np.zeros(X_work.shape, dtype=float)
    if total <= 0:
        return residuals

    for start in range(0, X_work.shape[1], chunksize):
        stop = min(start + chunksize, X_work.shape[1])
        if sparse.issparse(X_work):
            block = X_work[:, start:stop].toarray()
        else:
            block = np.asarray(X_work[:, start:stop])
        mu = np.outer(sums_cells, sums_genes[start:stop]) / total
        denom = np.sqrt(mu + (mu**2 / theta))
        denom[denom == 0] = 1
        chunk = (block - mu) / denom
        if np.isfinite(clip):
            chunk = np.clip(chunk, -clip, clip)
        residuals[:, start:stop] = chunk
    return residuals


def _residual_variance_by_batch(
    adata: AnnData,
    layer: Optional[str],
    theta: float,
    clip: Optional[float],
    batch_key: Optional[str],
    chunksize: int,
    check_values: bool,
) -> np.ndarray:
    """Compute Pearson residual variances for each gene in each batch."""
    if batch_key is None:
        batches: Iterable[object] = [None]
        batch_labels = None
    else:
        if batch_key not in adata.obs:
            raise KeyError(
                f"`adata.obs[{batch_key!r}]` is required for batch-aware Pearson residual gene selection."
            )
        batch_labels = np.asarray(adata.obs[batch_key])
        batches = pd.unique(adata.obs[batch_key])

    residual_vars: list[np.ndarray] = []
    for batch in batches:
        idx = np.arange(adata.n_obs) if batch is None else np.where(batch_labels == batch)[0]
        X_batch = SKM.select_layer_data(adata[idx, :], layer=layer, copy=False)
        if check_values and not _is_nonnegative_integer_matrix(X_batch):
            warn(
                "Pearson residual HVG selection expects raw non-negative integer count data.",
                UserWarning,
                stacklevel=2,
            )
        residual_var = np.zeros(adata.n_vars, dtype=float)
        gene_totals = _as_array(X_batch.sum(axis=0)).astype(float)
        nonzero = gene_totals > 0
        if nonzero.any():
            residuals = compute_pearson_residuals(
                X_batch[:, nonzero],
                theta=theta,
                clip=clip,
                check_values=False,
                chunksize=chunksize,
            )
            residual_var[nonzero] = residuals.var(axis=0)
        residual_vars.append(residual_var)
    return np.vstack(residual_vars)


def compute_highly_variable_genes(
    adata: AnnData,
    *,
    theta: float = 100,
    clip: Optional[float] = None,
    n_top_genes: Optional[int] = 3000,
    batch_key: Optional[str] = None,
    chunksize: int = 512,
    recipe: str = "pearson_residuals",
    check_values: bool = True,
    layer: Optional[str] = "counts",
    subset: bool = False,
    inplace: bool = True,
) -> Optional[pd.DataFrame]:
    """Select highly variable genes using Pearson residual variance.

    Args:
        adata: Input AnnData object.
        theta: Shared negative-binomial overdispersion parameter.
        clip: Pearson residual clipping threshold.
        n_top_genes: Number of genes to select.
        batch_key: Optional batch key. Genes are ranked by the number of batches in which they are selected, with median
            rank used to break ties.
        chunksize: Number of genes processed per chunk.
        recipe: Currently only ``"pearson_residuals"`` is supported.
        check_values: Warn when counts do not look like raw non-negative integers.
        layer: Count layer used for selection. ``None`` means ``adata.X``.
        subset: Whether to subset genes to selected HVGs.
        inplace: If ``True``, update ``adata.var``; otherwise return a DataFrame.

    Returns:
        A DataFrame when ``inplace=False``; otherwise ``None``.
    """
    if recipe != "pearson_residuals":
        raise ValueError("Only `recipe='pearson_residuals'` is supported.")
    if n_top_genes is None:
        raise ValueError("`n_top_genes` is required for Pearson residual HVG selection.")
    n_top = min(max(1, int(n_top_genes)), adata.n_vars)

    logger.info("Selecting highly variable genes with Pearson residual variance...")
    X = SKM.select_layer_data(adata, layer=layer, copy=False)
    means, variances = _mean_var(X)
    residual_vars_by_batch = _residual_variance_by_batch(
        adata,
        layer=layer,
        theta=theta,
        clip=clip,
        batch_key=batch_key,
        chunksize=chunksize,
        check_values=check_values,
    )
    highly_variable, median_rank, nbatches = _rank_residual_variances(residual_vars_by_batch, n_top)
    n_batches = residual_vars_by_batch.shape[0]

    df = pd.DataFrame(
        {
            "means": means,
            "variances": variances,
            "residual_variances": residual_vars_by_batch.mean(axis=0),
            "highly_variable_rank": median_rank,
            "highly_variable_nbatches": nbatches,
            "highly_variable_intersection": nbatches == n_batches,
            SKM.VAR_HIGHLY_VARIABLE_KEY: highly_variable,
            SKM.VAR_USE_FOR_PCA_KEY: highly_variable,
        },
        index=adata.var_names,
    )

    if not inplace:
        if subset:
            return df.loc[df[SKM.VAR_HIGHLY_VARIABLE_KEY]].copy()
        return df

    for col in df.columns:
        adata.var[col] = df[col].values
    SKM.init_uns_pp_namespace(adata)
    adata.uns[SKM.UNS_PP_KEY]["hvg"] = {
        "flavor": "pearson_residuals",
        "layer": layer,
        "theta": theta,
        "clip": clip,
        "n_top_genes": n_top,
        "batch_key": batch_key,
    }
    if subset:
        adata._inplace_subset_var(highly_variable)
    _record_step(adata, "select_genes_by_pearson_residuals", adata.uns[SKM.UNS_PP_KEY]["hvg"])
    return None


def pearson_residuals(
    adata: AnnData,
    layer: str = "counts",
    out_layer: str = "pearson_residuals",
    theta: float = 100,
    clip: Optional[float] = None,
    check_values: bool = True,
    chunksize: int = 512,
    inplace: bool = True,
) -> Optional[AnnData]:
    """Compute analytic Pearson residuals from a count layer and store them in a new layer.

    Args:
        adata: Input AnnData object.
        layer: Count layer. Use ``"X"`` to read ``adata.X``.
        out_layer: Output layer for residuals.
        theta: Shared negative-binomial overdispersion parameter.
        clip: Residual clipping threshold. ``None`` uses ``sqrt(n_obs)``.
        check_values: Warn when input does not look like raw counts.
        chunksize: Number of genes processed per chunk.
        inplace: If ``True``, modify ``adata`` in place.

    Returns:
        Updated AnnData when ``inplace=False``; otherwise ``None``.
    """
    target = adata if inplace else adata.copy()
    logger.info("Computing Pearson residuals for spatial preprocessing...")
    X = SKM.select_layer_data(target, layer=layer, copy=False)
    target.layers[out_layer] = compute_pearson_residuals(
        X,
        theta=theta,
        clip=clip,
        check_values=check_values,
        chunksize=chunksize,
    )
    SKM.init_uns_pp_namespace(target)
    target.uns[SKM.UNS_PP_KEY]["pearson_residuals"] = {
        "layer": layer,
        "out_layer": out_layer,
        "theta": theta,
        "clip": clip,
        "check_values": check_values,
        "chunksize": chunksize,
    }
    _record_step(target, "pearson_residuals", target.uns[SKM.UNS_PP_KEY]["pearson_residuals"])
    return None if inplace else target


def preprocess_pearson_residuals(
    adata: AnnData,
    spatial_key: str = "spatial",
    counts_layer: str = "counts",
    n_top_genes: int = 3000,
    run_pca: bool = True,
    inplace: bool = True,
) -> Optional[AnnData]:
    """Run the spatial preprocessing recipe that uses Pearson residuals as the PCA input."""
    from ..preprocessor import SpatialPreprocessor

    preprocessor = SpatialPreprocessor(
        recipe="pearson_residuals",
        spatial_key=spatial_key,
        counts_layer=counts_layer,
    )
    return preprocessor.preprocess_adata(
        adata,
        recipe="pearson_residuals",
        spatial_key=spatial_key,
        counts_layer=counts_layer,
        n_top_genes=n_top_genes,
        run_pca=run_pca,
        inplace=inplace,
    )


def normalize_layers_pearson_residuals(
    adata: AnnData,
    layers: Optional[list[str]] = None,
    out_layers: Optional[list[str]] = None,
    theta: float = 100,
    clip: Optional[float] = None,
    check_values: bool = True,
    chunksize: int = 512,
    copy: bool = False,
) -> Optional[AnnData]:
    """Apply Pearson residual normalization to one or more layers.

    This is a backward-compatible Python-native replacement for the old dynamo-derived helper.  It writes residuals to
    new layers and never overwrites raw count layers.
    """
    target = adata.copy() if copy else adata
    layers = [SKM.X_LAYER] if layers is None else layers
    out_layers = out_layers or [
        "pearson_residuals" if layer == SKM.X_LAYER else f"{layer}_pearson_residuals" for layer in layers
    ]
    if len(out_layers) != len(layers):
        raise ValueError("`out_layers` must have the same length as `layers`.")
    for layer, out_layer in zip(layers, out_layers):
        pearson_residuals(
            target,
            layer=layer,
            out_layer=out_layer,
            theta=theta,
            clip=clip,
            check_values=check_values,
            chunksize=chunksize,
            inplace=True,
        )
    return target if copy else None


def select_genes_by_pearson_residuals(
    adata: AnnData,
    layer: str = "counts",
    theta: float = 100,
    clip: Optional[float] = None,
    n_top_genes: int = 3000,
    batch_key: Optional[str] = None,
    chunksize: int = 512,
    check_values: bool = True,
    subset: bool = False,
    inplace: bool = True,
) -> Optional[pd.DataFrame]:
    """Select PCA genes using Pearson residual variance."""
    return compute_highly_variable_genes(
        adata,
        layer=layer,
        theta=theta,
        clip=clip,
        n_top_genes=n_top_genes,
        batch_key=batch_key,
        chunksize=chunksize,
        check_values=check_values,
        subset=subset,
        inplace=inplace,
    )
