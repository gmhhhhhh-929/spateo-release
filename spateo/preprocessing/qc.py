"""Quality control helpers for spatial transcriptomics data."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from ..configuration import SKM
from ..spateo_logger import LoggerManager
from .utils import _record_step

logger = LoggerManager.get_main_logger()


def _axis_sum(matrix: object, axis: int) -> np.ndarray:
    result = matrix.sum(axis=axis)
    return np.asarray(result).ravel()


def _nnz_axis(matrix: object, axis: int) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.getnnz(axis=axis)).ravel()
    return np.count_nonzero(matrix, axis=axis)


def _gene_prefix_mask(var_names: pd.Index, prefixes: tuple[str, ...]) -> np.ndarray:
    names = var_names.astype(str)
    mask = np.zeros(len(names), dtype=bool)
    for prefix in prefixes:
        mask |= np.asarray(names.str.startswith(prefix), dtype=bool)
    return mask


def _percent_for_mask(matrix: object, total_counts: np.ndarray, gene_mask: np.ndarray) -> np.ndarray:
    if gene_mask.sum() == 0:
        return np.zeros(matrix.shape[0], dtype=float)
    sub_counts = _axis_sum(matrix[:, gene_mask], axis=1)
    pct = np.zeros_like(total_counts, dtype=float)
    valid = total_counts > 0
    pct[valid] = sub_counts[valid] / total_counts[valid] * 100
    return pct


def calculate_spatial_qc(
    adata: AnnData,
    layer: str = "counts",
    spatial_key: str = "spatial",
    mt_prefix: tuple[str, ...] = ("MT-", "mt-", "Mt-"),
    ribo_prefix: tuple[str, ...] = ("RPS", "RPL", "Rps", "Rpl"),
    hb_prefix: tuple[str, ...] = ("HB", "Hb"),
    inplace: bool = True,
) -> Optional[AnnData]:
    """Calculate spatial QC metrics and write them to ``obs`` and ``var``.

    Args:
        adata: Input AnnData object.
        layer: Count layer used for QC.
        spatial_key: Coordinate key in ``adata.obsm``.
        mt_prefix: Mitochondrial gene prefixes.
        ribo_prefix: Ribosomal gene prefixes.
        hb_prefix: Hemoglobin gene prefixes.
        inplace: If ``True``, modify ``adata`` in place.

    Returns:
        Updated AnnData when ``inplace=False``; otherwise ``None``.
    """
    adata = adata if inplace else adata.copy()
    logger.info("Calculating spatial QC metrics...")
    coords = SKM.ensure_spatial_key(adata, spatial_key=spatial_key)
    X = SKM.select_layer_data(adata, layer=layer, copy=False)

    total_counts = _axis_sum(X, axis=1).astype(float)
    n_genes = _nnz_axis(X, axis=1).astype(int)
    gene_total = _axis_sum(X, axis=0).astype(float)
    n_cells = _nnz_axis(X, axis=0).astype(int)
    mean_counts = gene_total / max(adata.n_obs, 1)

    mt = _gene_prefix_mask(adata.var_names, mt_prefix)
    ribo = _gene_prefix_mask(adata.var_names, ribo_prefix)
    hb = _gene_prefix_mask(adata.var_names, hb_prefix)

    adata.obs[SKM.OBS_TOTAL_COUNTS_KEY] = total_counts
    adata.obs[SKM.OBS_N_GENES_BY_COUNTS_KEY] = n_genes
    adata.obs[SKM.OBS_PCT_COUNTS_MT_KEY] = _percent_for_mask(X, total_counts, mt)
    adata.obs[SKM.OBS_PCT_COUNTS_RIBO_KEY] = _percent_for_mask(X, total_counts, ribo)
    adata.obs[SKM.OBS_PCT_COUNTS_HB_KEY] = _percent_for_mask(X, total_counts, hb)
    adata.obs["x_coord"] = coords[:, 0]
    adata.obs["y_coord"] = coords[:, 1]
    if coords.shape[1] >= 3:
        adata.obs["z_coord"] = coords[:, 2]
    adata.obs["nCounts"] = adata.obs[SKM.OBS_TOTAL_COUNTS_KEY]
    adata.obs["nGenes"] = adata.obs[SKM.OBS_N_GENES_BY_COUNTS_KEY]
    adata.obs["pMito"] = adata.obs[SKM.OBS_PCT_COUNTS_MT_KEY] / 100

    adata.var[SKM.VAR_N_CELLS_BY_COUNTS_KEY] = n_cells
    adata.var[SKM.VAR_TOTAL_COUNTS_KEY] = gene_total
    adata.var[SKM.VAR_MEAN_COUNTS_KEY] = mean_counts

    SKM.init_uns_spatial_namespace(adata)
    adata.uns[SKM.UNS_SPATIAL_KEY][SKM.UNS_SPATIAL_QC_KEY] = {
        "metrics": [
            SKM.OBS_TOTAL_COUNTS_KEY,
            SKM.OBS_N_GENES_BY_COUNTS_KEY,
            SKM.OBS_PCT_COUNTS_MT_KEY,
            SKM.OBS_PCT_COUNTS_RIBO_KEY,
            SKM.OBS_PCT_COUNTS_HB_KEY,
        ],
        "filter_key": SKM.OBS_PASS_SPATIAL_QC_KEY,
        "local_qc": bool(adata.obs.get(SKM.OBS_LOCAL_QC_OUTLIER_KEY, pd.Series(False, index=adata.obs_names)).any()),
    }
    _record_step(adata, "calculate_spatial_qc", {"layer": layer, "spatial_key": spatial_key})
    return None if inplace else adata


def _adaptive_mask(values: np.ndarray, lower: bool, upper: bool, nmads: float) -> np.ndarray:
    if values.size == 0:
        return np.ones(0, dtype=bool)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if mad == 0 or not np.isfinite(mad):
        return np.ones(values.size, dtype=bool)
    mask = np.ones(values.size, dtype=bool)
    if lower:
        mask &= values >= median - nmads * mad
    if upper:
        mask &= values <= median + nmads * mad
    return mask


def filter_spots(
    adata: AnnData,
    min_counts: Optional[int] = None,
    max_counts: Optional[int] = None,
    min_genes: Optional[int] = None,
    max_genes: Optional[int] = None,
    max_pct_mt: Optional[float] = None,
    use_in_tissue: bool = False,
    in_tissue_key: str = "in_tissue",
    library_key: Optional[str] = None,
    adaptive: bool = True,
    nmads: float = 3.0,
    keep_filtered: bool = False,
    inplace: bool = True,
) -> Optional[AnnData]:
    """Filter low-quality spots or cells using QC metrics.

    Args:
        adata: Input AnnData object with QC metrics.
        min_counts: Minimum total counts.
        max_counts: Maximum total counts.
        min_genes: Minimum detected genes.
        max_genes: Maximum detected genes.
        max_pct_mt: Maximum mitochondrial percentage.
        use_in_tissue: Whether to require ``obs[in_tissue_key] == 1``.
        in_tissue_key: Visium tissue flag key.
        library_key: Optional grouping key for adaptive thresholds.
        adaptive: Whether to use MAD-based adaptive thresholds.
        nmads: Number of MADs for adaptive thresholds.
        keep_filtered: If ``True``, only annotate filtered spots.
        inplace: If ``True``, modify ``adata`` in place.

    Returns:
        Filtered copy when ``inplace=False``; otherwise ``None``.
    """
    adata = adata if inplace else adata.copy()
    logger.info("Filtering spots by spatial QC...")
    total = np.asarray(adata.obs[SKM.OBS_TOTAL_COUNTS_KEY], dtype=float)
    genes = np.asarray(adata.obs[SKM.OBS_N_GENES_BY_COUNTS_KEY], dtype=float)
    mask = np.ones(adata.n_obs, dtype=bool)

    if min_counts is not None:
        mask &= total >= min_counts
    if max_counts is not None:
        mask &= total <= max_counts
    if min_genes is not None:
        mask &= genes >= min_genes
    if max_genes is not None:
        mask &= genes <= max_genes
    if max_pct_mt is not None:
        mask &= np.asarray(adata.obs[SKM.OBS_PCT_COUNTS_MT_KEY], dtype=float) <= max_pct_mt
    if use_in_tissue:
        if in_tissue_key not in adata.obs:
            logger.warning(f"`use_in_tissue=True` but `adata.obs[{in_tissue_key!r}]` is missing; skipping tissue filter.")
        else:
            mask &= np.asarray(adata.obs[in_tissue_key]) == 1

    if adaptive:
        groups = np.asarray(adata.obs[library_key]) if library_key is not None and library_key in adata.obs else None
        group_values = np.unique(groups) if groups is not None else [None]
        adaptive_mask = np.ones(adata.n_obs, dtype=bool)
        for group in group_values:
            idx = np.arange(adata.n_obs) if group is None else np.where(groups == group)[0]
            adaptive_mask[idx] &= _adaptive_mask(total[idx], lower=True, upper=True, nmads=nmads)
            adaptive_mask[idx] &= _adaptive_mask(genes[idx], lower=True, upper=False, nmads=nmads)
        mask &= adaptive_mask

    if not mask.any():
        logger.warning("All spots would be filtered.")
        if not keep_filtered:
            logger.warning("Keeping all spots to avoid creating an empty AnnData object.")
            mask = np.ones(adata.n_obs, dtype=bool)

    adata.obs[SKM.OBS_PASS_SPATIAL_QC_KEY] = mask.astype(bool)
    if not keep_filtered:
        adata._inplace_subset_obs(mask)

    _record_step(
        adata,
        "filter_spots",
        {
            "min_counts": min_counts,
            "max_counts": max_counts,
            "min_genes": min_genes,
            "max_genes": max_genes,
            "max_pct_mt": max_pct_mt,
            "use_in_tissue": use_in_tissue,
            "library_key": library_key,
            "keep_filtered": keep_filtered,
        },
    )
    return None if inplace else adata


def filter_genes_by_spatial_qc(
    adata: AnnData,
    layer: str = "counts",
    min_cells: int = 3,
    min_counts: Optional[int] = None,
    library_key: Optional[str] = None,
    keep_filtered: bool = False,
    inplace: bool = True,
) -> Optional[AnnData]:
    """Filter genes by detection across spatial observations.

    Args:
        adata: Input AnnData object.
        layer: Count layer.
        min_cells: Minimum cells/spots expressing a gene.
        min_counts: Minimum total gene counts.
        library_key: Optional library key, recorded for provenance.
        keep_filtered: If ``True``, only annotate filtered genes.
        inplace: If ``True``, modify ``adata`` in place.

    Returns:
        Filtered copy when ``inplace=False``; otherwise ``None``.
    """
    adata = adata if inplace else adata.copy()
    logger.info("Filtering genes by spatial QC...")
    X = SKM.select_layer_data(adata, layer=layer, copy=False)
    min_cells_eff = min(min_cells, max(1, adata.n_obs))
    n_cells = _nnz_axis(X, axis=0).astype(int)
    total = _axis_sum(X, axis=0).astype(float)
    mask = n_cells >= min_cells_eff
    if min_counts is not None:
        mask &= total >= min_counts

    if not mask.any():
        logger.warning("All genes would be filtered; keeping original genes and marking pass_basic_filter=True.")
        mask = np.ones(adata.n_vars, dtype=bool)

    adata.var[SKM.VAR_PASS_BASIC_FILTER_KEY] = mask.astype(bool)
    if not keep_filtered:
        adata._inplace_subset_var(mask)

    _record_step(
        adata,
        "filter_genes_by_spatial_qc",
        {"layer": layer, "min_cells": min_cells, "min_counts": min_counts, "library_key": library_key},
    )
    return None if inplace else adata


def flag_local_qc_outliers(
    adata: AnnData,
    spatial_connectivities_key: str = "spatial_connectivities",
    metrics: tuple[str, ...] = ("total_counts", "n_genes_by_counts", "pct_counts_mt"),
    nmads: float = 3.0,
    obs_store_key: str = "local_qc_outlier",
) -> None:
    """Flag observations whose QC metrics deviate from local spatial neighbors.

    Args:
        adata: Input AnnData object.
        spatial_connectivities_key: Key in ``adata.obsp`` containing spatial graph.
        metrics: Observation QC metrics to inspect.
        nmads: Number of MADs for outlier calls.
        obs_store_key: Output key in ``adata.obs``.
    """
    logger.info("Flagging local spatial QC outliers...")
    if spatial_connectivities_key not in adata.obsp:
        raise KeyError(f"`adata.obsp[{spatial_connectivities_key!r}]` is required for local QC.")
    W = adata.obsp[spatial_connectivities_key].tocsr()
    outlier = np.zeros(adata.n_obs, dtype=bool)
    for metric in metrics:
        if metric not in adata.obs:
            continue
        values = np.asarray(adata.obs[metric], dtype=float)
        degree = np.asarray(W.sum(axis=1)).ravel()
        local_mean = np.zeros(adata.n_obs, dtype=float)
        valid = degree > 0
        local_mean[valid] = np.asarray(W[valid].dot(values)).ravel() / degree[valid]
        delta = values - local_mean
        mad = np.nanmedian(np.abs(delta - np.nanmedian(delta)))
        if mad > 0 and np.isfinite(mad):
            outlier |= np.abs(delta - np.nanmedian(delta)) > nmads * mad
    adata.obs[obs_store_key] = outlier.astype(bool)
    SKM.init_uns_spatial_namespace(adata)
    adata.uns[SKM.UNS_SPATIAL_KEY].setdefault(SKM.UNS_SPATIAL_QC_KEY, {})
    adata.uns[SKM.UNS_SPATIAL_KEY][SKM.UNS_SPATIAL_QC_KEY]["local_qc"] = True
    _record_step(adata, "flag_local_qc_outliers", {"metrics": list(metrics), "nmads": nmads})
