"""AnnData matrix helpers used by multiple Spateo subsystems."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
from anndata import AnnData
from scipy import sparse


def fetch_X_data(
    adata: AnnData,
    genes: Optional[Iterable[str]] = None,
    layer: Optional[str] = None,
    basis: Optional[str] = None,
) -> Tuple[Optional[list[str]], object]:
    """Return an AnnData representation and its matching gene names.

    Gene requests preserve the caller's order and silently discard names that
    are not present, matching the historical Spateo behavior.  A request with
    no overlapping genes is rejected explicitly.
    """

    if basis is not None:
        key = basis if basis in adata.obsm else f"X_{basis}"
        if key not in adata.obsm:
            raise KeyError(f"Embedding {basis!r} is not present in adata.obsm.")
        return None, adata.obsm[key]

    if layer in (None, "X"):
        matrix = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(f"Layer {layer!r} is not present in adata.layers.")
        matrix = adata.layers[layer]

    if genes is None:
        if "use_for_dynamics" in adata.var and bool(np.asarray(adata.var["use_for_dynamics"]).any()):
            selected = np.asarray(adata.var["use_for_dynamics"], dtype=bool)
            return adata.var_names[selected].tolist(), matrix[:, selected]
        return adata.var_names.tolist(), matrix

    genes = [genes] if isinstance(genes, str) else genes
    requested = list(dict.fromkeys(str(gene) for gene in genes))
    positions = adata.var_names.get_indexer(requested)
    present = positions >= 0
    if not np.any(present):
        raise ValueError("None of the requested genes are present in the AnnData object.")
    selected_genes = [gene for gene, keep in zip(requested, present) if keep]
    return selected_genes, matrix[:, positions[present]]


def normalize_total(
    adata: AnnData,
    target_sum: Optional[float] = 10_000.0,
    layer: Optional[str] = None,
    inplace: bool = True,
) -> Optional[AnnData]:
    """Normalize every observation to a common non-zero library size."""

    result = adata if inplace else adata.copy()
    matrix = result.X if layer in (None, "X") else result.layers[layer]
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    positive = totals > 0
    if target_sum is None:
        target_sum = float(np.median(totals[positive])) if np.any(positive) else 1.0
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("target_sum must be a finite positive value.")
    factors = np.zeros_like(totals, dtype=float)
    factors[positive] = float(target_sum) / totals[positive]

    if sparse.issparse(matrix):
        normalized = sparse.diags(factors).dot(matrix).tocsr()
    else:
        normalized = np.asarray(matrix, dtype=float) * factors[:, None]

    if layer in (None, "X"):
        result.X = normalized
    else:
        result.layers[layer] = normalized
    result.obs["size_factor"] = np.divide(
        totals,
        float(target_sum),
        out=np.zeros_like(totals, dtype=float),
        where=float(target_sum) != 0,
    )
    return None if inplace else result


def log1p(adata: AnnData, layer: Optional[str] = None, inplace: bool = True) -> Optional[AnnData]:
    """Apply a sparse-safe natural ``log(1+x)`` transformation."""

    result = adata if inplace else adata.copy()
    matrix = result.X if layer in (None, "X") else result.layers[layer]
    if sparse.issparse(matrix):
        transformed = matrix.copy().astype(float)
        transformed.data = np.log1p(transformed.data)
    else:
        transformed = np.log1p(np.asarray(matrix, dtype=float))
    if layer in (None, "X"):
        result.X = transformed
    else:
        result.layers[layer] = transformed
    return None if inplace else result
