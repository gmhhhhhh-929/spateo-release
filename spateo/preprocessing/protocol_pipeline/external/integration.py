"""Python-native spatial dataset integration helpers.

The module keeps the lightweight AnnData concatenation helper from the reference
implementation and optional Python backends for embedding correction.  Optional
packages are imported lazily so ``import spateo`` never fails because an
integration backend is missing.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import numpy as np
from anndata import AnnData
from scipy import sparse

from ....configuration import SKM
from ....spateo_logger import LoggerManager

logger = LoggerManager.get_main_logger()


def to_dense_matrix(X: object) -> np.ndarray:
    """Convert sparse or dense array-like input to a dense NumPy array."""
    return np.asarray(X.toarray() if sparse.issparse(X) else X)


def concatenate_adatas(
    adatas: Sequence[AnnData],
    batch_key: str = "slices",
    fill_value: Union[int, float] = 0,
    batch_categories: Optional[Sequence[str]] = None,
    join: Literal["inner", "outer"] = "outer",
) -> AnnData:
    """Concatenate spatial AnnData objects while preserving shared ``obsm`` and per-batch ``uns``.

    Args:
        adatas: AnnData objects to concatenate.
        batch_key: Observation key used to store batch labels in the concatenated object.
        fill_value: Value used for missing entries in an outer gene join.
        batch_categories: Optional batch labels. If omitted, labels are taken from ``obs[batch_key]`` when available.
        join: Gene join strategy passed to AnnData concatenation.

    Returns:
        Concatenated AnnData object.
    """
    if len(adatas) == 0:
        raise ValueError("`adatas` must contain at least one AnnData object.")
    logger.info("Concatenating spatial AnnData objects...")
    copies = [adata.copy() for adata in adatas]
    if batch_categories is None:
        labels = []
        for i, adata in enumerate(copies):
            if batch_key in adata.obs and adata.n_obs > 0:
                labels.append(str(adata.obs[batch_key].iloc[0]))
            else:
                labels.append(str(i))
    else:
        labels = [str(v) for v in batch_categories]
    if len(labels) != len(copies):
        raise ValueError("`batch_categories` must have the same length as `adatas`.")

    common_obsm = set(copies[0].obsm.keys())
    for adata in copies[1:]:
        common_obsm &= set(adata.obsm.keys())
    obsm_values = {
        key: np.concatenate([to_dense_matrix(adata.obsm[key]) for adata in copies], axis=0)
        for key in common_obsm
    }

    uns_values: dict[str, object] = {}
    uns_keys: set[str] = set()
    for adata in copies:
        uns_keys |= set(adata.uns.keys())
    for key in uns_keys:
        if key == SKM.ADATA_TYPE_KEY and key in copies[0].uns:
            uns_values[key] = copies[0].uns[key]
        else:
            uns_values[key] = {
                label: adata.uns[key] if key in adata.uns else None for label, adata in zip(labels, copies)
            }

    integrated = copies[0].concatenate(
        *copies[1:],
        batch_key=batch_key,
        batch_categories=labels,
        join=join,
        fill_value=fill_value,
        uns_merge=None,
    )
    for key, value in obsm_values.items():
        if value.shape[0] == integrated.n_obs:
            integrated.obsm[key] = value
    for key, value in uns_values.items():
        integrated.uns[key] = value
    return integrated


def harmony_debatch(
    adata: AnnData,
    key: str,
    basis: str = "X_pca",
    adjusted_basis: str = "X_pca_harmony",
    max_iter_harmony: int = 10,
    copy: bool = False,
    **kwargs: object,
) -> Optional[AnnData]:
    """Use harmonypy to remove batch effects from an existing embedding."""
    try:
        import harmonypy
    except ImportError as exc:
        raise ImportError("Harmony integration requires `harmonypy` (`pip install harmonypy`).") from exc
    target = adata.copy() if copy else adata
    if key not in target.obs:
        raise KeyError(f"`adata.obs[{key!r}]` is required for Harmony integration.")
    if basis not in target.obsm:
        raise KeyError(f"`adata.obsm[{basis!r}]` is required for Harmony integration.")
    matrix = to_dense_matrix(target.obsm[basis])
    output = harmonypy.run_harmony(matrix, target.obs, key, max_iter_harmony=max_iter_harmony, **kwargs)
    adjusted = output.Z_corr.T
    if sparse.issparse(target.obsm[basis]):
        adjusted = sparse.csr_matrix(adjusted)
    target.obsm[adjusted_basis] = adjusted
    SKM.init_uns_pp_namespace(target)
    target.uns[SKM.UNS_PP_KEY]["harmony"] = {
        "key": key,
        "basis": basis,
        "adjusted_basis": adjusted_basis,
        "max_iter_harmony": max_iter_harmony,
    }
    return target if copy else None


def _scanorama_integrate(
    adata: AnnData,
    key: str,
    adjusted_basis: str,
    **kwargs: object,
) -> None:
    """Run scanorama lazily."""
    try:
        import scanorama
    except ImportError as exc:
        raise ImportError("Scanorama integration requires `scanorama` (`pip install scanorama`).") from exc
    groups = [adata[adata.obs[key] == value].copy() for value in adata.obs[key].unique()]
    corrected = scanorama.correct_scanpy(groups, return_dimred=True, **kwargs)
    adata.obsm[adjusted_basis] = np.concatenate([a.obsm["X_scanorama"] for a in corrected], axis=0)


def _bbknn_integrate(adata: AnnData, key: str, basis: str, **kwargs: object) -> None:
    """Run BBKNN lazily."""
    try:
        import bbknn
    except ImportError as exc:
        raise ImportError("BBKNN integration requires `bbknn` (`pip install bbknn`).") from exc
    bbknn.bbknn(adata, batch_key=key, use_rep=basis, **kwargs)


def integrate(
    adata: Union[AnnData, Sequence[AnnData]],
    batch_key: Optional[str] = "slices",
    library_key: Optional[str] = None,
    method: Optional[Literal["harmony", "scanorama", "bbknn"]] = None,
    basis: str = "X_pca",
    adjusted_basis: str = "X_pca_integrated",
    fill_value: Union[int, float] = 0,
    inplace: bool = True,
    **kwargs: object,
) -> Optional[AnnData]:
    """Integrate spatial datasets or correct an embedding with an optional backend.

    Args:
        adata: Either a sequence of AnnData objects to concatenate, or a single AnnData object to batch-correct.
        batch_key: Batch key. For concatenation this stores labels; for correction it must exist in ``adata.obs``.
        library_key: Alternative batch key for single-object correction.
        method: Correction backend. If ``adata`` is a sequence, ``None`` means concatenate only. If ``adata`` is a single
            object, ``None`` defaults to ``"harmony"``.
        basis: Embedding key used by correction backends.
        adjusted_basis: Output embedding key for corrected PCs.
        fill_value: Fill value used for multi-object concatenation.
        inplace: For single-object correction, whether to update in place. Concatenation always returns a new AnnData.
        **kwargs: Passed to the selected backend.

    Returns:
        Concatenated AnnData for sequence input; corrected copy when ``inplace=False``; otherwise ``None``.
    """
    if isinstance(adata, Sequence) and not isinstance(adata, AnnData):
        return concatenate_adatas(adata, batch_key=batch_key or "slices", fill_value=fill_value, **kwargs)

    key = batch_key or library_key
    if key is None:
        raise ValueError("`batch_key` or `library_key` is required for integration.")
    target = adata if inplace else adata.copy()
    method = "harmony" if method is None else method
    if method == "harmony":
        harmony_kwargs = dict(kwargs)
        max_iter = int(harmony_kwargs.pop("max_iter_harmony", 10))
        harmony_debatch(
            target,
            key=key,
            basis=basis,
            adjusted_basis=adjusted_basis,
            max_iter_harmony=max_iter,
            copy=False,
            **harmony_kwargs,
        )
    elif method == "scanorama":
        if key not in target.obs:
            raise KeyError(f"`adata.obs[{key!r}]` is required for Scanorama integration.")
        _scanorama_integrate(target, key=key, adjusted_basis=adjusted_basis, **kwargs)
    elif method == "bbknn":
        if key not in target.obs:
            raise KeyError(f"`adata.obs[{key!r}]` is required for BBKNN integration.")
        _bbknn_integrate(target, key=key, basis=basis, **kwargs)
    else:
        raise ValueError("`method` must be one of {'harmony', 'scanorama', 'bbknn'}.")

    SKM.init_uns_pp_namespace(target)
    target.uns[SKM.UNS_PP_KEY]["integration"] = {
        "batch_key": key,
        "method": method,
        "basis": basis,
        "adjusted_basis": adjusted_basis,
    }
    return None if inplace else target
