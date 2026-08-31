"""Utilities for spatial transcriptomics preprocessing."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from anndata import AnnData
from scipy import sparse

from ..configuration import SKM
from ..spateo_logger import LoggerManager

logger = LoggerManager.get_main_logger()


def _ensure_csr_if_sparse(matrix: object) -> object:
    """Return CSR for sparse matrices without densifying."""
    return matrix.tocsr() if sparse.issparse(matrix) and not sparse.isspmatrix_csr(matrix) else matrix


def _record_step(adata: AnnData, name: str, params: Optional[dict[str, object]] = None) -> None:
    """Append a preprocessing step to ``adata.uns['pp']['steps']``."""
    SKM.init_uns_pp_namespace(adata)
    adata.uns[SKM.UNS_PP_KEY].setdefault("steps", [])
    adata.uns[SKM.UNS_PP_KEY]["steps"].append({"name": name, "params": params or {}})


def _validate_count_matrix(matrix: object, *, mode: Literal["warn", "error", "ignore"]) -> None:
    """Check the count-layer contract on a bounded sample of stored values."""
    if mode == "ignore":
        return
    if mode not in {"warn", "error"}:
        raise ValueError("`validate_counts` must be one of {'warn', 'error', 'ignore'}.")
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    if values.size > 100_000:
        indices = np.linspace(0, values.size - 1, 100_000, dtype=int)
        values = values[indices]
    values = np.asarray(values)
    valid = (
        np.issubdtype(values.dtype, np.number)
        and np.isfinite(values).all()
        and (values >= 0).all()
        and np.allclose(values, np.rint(values), rtol=0, atol=1e-6)
    )
    if valid:
        return
    message = (
        "The selected counts matrix is not finite, non-negative, integer-like data. "
        "Normalization and count-based QC may be invalid; pass the raw count layer "
        "or set `validate_counts='ignore'` deliberately."
    )
    if mode == "error":
        raise ValueError(message)
    logger.warning(message)


def standardize_spatial_adata(
    adata: AnnData,
    spatial_key: str = "spatial",
    layer: Optional[str] = None,
    counts_layer: str = "counts",
    library_key: Optional[str] = None,
    sample_key: Optional[str] = None,
    copy_raw_counts: bool = True,
    validate_counts: Literal["warn", "error", "ignore"] = "warn",
    inplace: bool = True,
) -> Optional[AnnData]:
    """Standardize an AnnData object for spatial preprocessing.

    Examples:
        >>> import spateo as st
        >>> adata = st.read_h5ad("sample.h5ad")
        >>> st.pp.preprocess_spatial(adata, recipe="standard", spatial_key="spatial")

    Args:
        adata: Input AnnData object.
        spatial_key: Key in ``adata.obsm`` containing spatial coordinates.
        layer: Input layer to copy into ``counts_layer``. ``None`` means ``adata.X``.
        counts_layer: Layer used to store raw counts.
        library_key: Optional observation key identifying slices/libraries.
        sample_key: Optional observation key identifying samples.
        copy_raw_counts: Whether to create ``counts_layer`` if absent.
        inplace: If ``True``, modify ``adata`` in place.

    Returns:
        A standardized copy when ``inplace=False``; otherwise ``None``.
    """
    adata = adata if inplace else adata.copy()
    logger.info("Standardizing spatial AnnData...")
    if adata.is_view:
        logger.warning("Received a view of an AnnData object; making a copy.")
        if inplace:
            adata._init_as_actual(adata.copy())
        else:
            adata = adata.copy()

    if not adata.obs_names.is_unique:
        logger.warning(
            "`adata.obs_names` is not unique. Calling `obs_names_make_unique()` and storing original names "
            "in `adata.obs['original_obs_names']`."
        )
        adata.obs["original_obs_names"] = adata.obs_names.astype(str)
        adata.obs_names_make_unique()
    if not adata.var_names.is_unique:
        logger.warning(
            "`adata.var_names` is not unique. Calling `var_names_make_unique()` and storing original names "
            "in `adata.var['original_var_names']`."
        )
        adata.var["original_var_names"] = adata.var_names.astype(str)
        adata.var_names_make_unique()

    coords = SKM.ensure_spatial_key(adata, spatial_key=spatial_key)
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"`adata.obsm[{spatial_key!r}]` must have shape (n_obs, 2+) but has {coords.shape}.")
    if not np.issubdtype(coords.dtype, np.number) or not np.isfinite(coords[:, :3]).all():
        raise ValueError(f"`adata.obsm[{spatial_key!r}]` must contain finite numeric coordinates.")
    if coords.shape[1] not in (2, 3):
        logger.warning(f"`adata.obsm[{spatial_key!r}]` has {coords.shape[1]} columns; using the first three at most.")

    if copy_raw_counts and counts_layer not in adata.layers:
        counts = SKM.select_layer_data(adata, layer=layer, copy=True)
        adata.layers[counts_layer] = _ensure_csr_if_sparse(counts)
        logger.info(f"Copied raw counts to `adata.layers[{counts_layer!r}]`.")
    elif counts_layer in adata.layers:
        adata.layers[counts_layer] = _ensure_csr_if_sparse(adata.layers[counts_layer])
    elif not copy_raw_counts:
        raise KeyError(f"`copy_raw_counts=False` but `adata.layers[{counts_layer!r}]` does not exist.")

    _validate_count_matrix(adata.layers[counts_layer], mode=validate_counts)

    if sparse.issparse(adata.X):
        adata.X = _ensure_csr_if_sparse(adata.X)

    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)

    for key_name, key in (("library_key", library_key), ("sample_key", sample_key)):
        if key is not None and key not in adata.obs:
            raise KeyError(f"`{key_name}={key!r}` is not present in `adata.obs`.")

    adata.obs["x_coord"] = coords[:, 0]
    adata.obs["y_coord"] = coords[:, 1]
    if coords.shape[1] >= 3:
        adata.obs["z_coord"] = coords[:, 2]

    adata.uns[SKM.UNS_SPATIAL_KEY].setdefault("metadata", {})
    adata.uns[SKM.UNS_SPATIAL_KEY]["metadata"].update(
        {
            "spatial_key": spatial_key,
            "library_key": library_key,
            "sample_key": sample_key,
            "n_spatial_dims": int(min(coords.shape[1], 3)),
        }
    )
    _record_step(
        adata,
        "standardize_spatial_adata",
        {
            "spatial_key": spatial_key,
            "layer": layer,
            "counts_layer": counts_layer,
            "library_key": library_key,
            "sample_key": sample_key,
            "validate_counts": validate_counts,
        },
    )
    return None if inplace else adata
