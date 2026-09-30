"""Seq-Scope spatial transcriptomics reader."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.io
from anndata import AnnData
from scipy.sparse import coo_matrix, csr_matrix, issparse

from ...configuration import SKM
from ._matrix import _ids, _numeric, integer_counts
from ._utils import bin_indices, get_bin_props

PathLike = Union[str, "Path"]


def _resolve_file(root: Path, *names: str) -> Path:
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"None of {names!r} was found in {root}.")


def _read_seqscope_matrix(matrix_dir: PathLike) -> AnnData:
    root = Path(matrix_dir).expanduser().resolve()
    barcodes_path = _resolve_file(root, "barcodes.tsv.gz", "barcodes.tsv")
    features_path = _resolve_file(root, "features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv")
    matrix_path = _resolve_file(root, "matrix.mtx.gz", "matrix.mtx")

    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None, dtype=str).iloc[:, 0]
    features = pd.read_csv(features_path, sep="\t", header=None, dtype=str)
    if features.shape[1] < 2:
        raise ValueError("Seq-Scope features.tsv must contain at least gene ID and gene name columns.")
    feature_columns = ["gene_id", "gene_name", "feature_type"][: features.shape[1]]
    features = features.iloc[:, : len(feature_columns)]
    features.columns = feature_columns

    matrix = scipy.io.mmread(matrix_path)
    if issparse(matrix):
        matrix.data = integer_counts(matrix.data, "Seq-Scope")
        matrix = matrix.tocsr()
    else:
        dense = np.asarray(matrix)
        matrix = csr_matrix(integer_counts(dense.ravel(), "Seq-Scope").reshape(dense.shape))
    expected = (len(features), len(barcodes))
    if matrix.shape == expected:
        matrix = matrix.T.tocsr()
    elif matrix.shape != expected[::-1]:
        raise ValueError(f"Seq-Scope matrix has shape {matrix.shape}; expected {expected} or {expected[::-1]}.")
    adata = AnnData(
        X=matrix,
        obs=pd.DataFrame(index=_ids(barcodes, "Seq-Scope matrix barcodes").rename("barcode")),
        var=features.set_index("gene_id"),
    )
    _ids(adata.var_names, "Seq-Scope gene IDs")
    return adata


def _read_seqscope_positions(path: PathLike) -> pd.DataFrame:
    positions = pd.read_csv(
        Path(path).expanduser().resolve(), sep=r"\s+", header=None, dtype=str, keep_default_na=False
    )
    if positions.shape[1] != 5 or positions.empty:
        raise ValueError("Seq-Scope positions require five fields: barcode, lane, tile, x, y")
    positions.columns = ["barcode", "lane", "tile", "x", "y"]
    positions.index = _ids(positions.pop("barcode"), "Seq-Scope position barcodes")
    for column in positions:
        positions[column] = integer_counts(_numeric(positions[column], f"Seq-Scope {column}"), "Seq-Scope positions")
    return positions


def _initialize_seqscope_metadata(adata: AnnData, binsize: Optional[int]) -> None:
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_BINSIZE_KEY, binsize)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, 1.0)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, None)


def read_seqscope(
    matrix_dir: PathLike,
    positions_path: PathLike,
    binsize: Optional[int] = 1,
    add_props: bool = True,
) -> AnnData:
    """Read a Seq-Scope matrix and barcode-coordinate file.

    Set ``binsize=None`` to preserve barcode-level observations. Positive
    integer bin sizes aggregate barcodes into spatial bins.
    """

    if binsize is not None and (not isinstance(binsize, (int, np.integer)) or int(binsize) < 1):
        raise ValueError("binsize must be a positive integer or None.")
    adata = _read_seqscope_matrix(matrix_dir)
    positions = _read_seqscope_positions(positions_path)
    available = adata.obs_names.isin(positions.index)
    if not np.all(available):
        raise ValueError(f"Missing Seq-Scope positions for {int((~available).sum())} matrix barcodes")
    adata.obs = adata.obs.join(positions, how="left")

    if binsize is None:
        adata.obsm[SKM.OBSM_SPATIAL_KEY] = adata.obs[["x", "y"]].to_numpy(dtype=float)
        _initialize_seqscope_metadata(adata, binsize=None)
        return adata

    binsize = int(binsize)
    adata.obs["x"] = bin_indices(adata.obs["x"].to_numpy(), 0, binsize)
    adata.obs["y"] = bin_indices(adata.obs["y"].to_numpy(), 0, binsize)
    labels = adata.obs["x"].astype(str).str.cat(adata.obs["y"].astype(str), sep="-")
    categories = pd.Index(sorted(labels.unique()), dtype=str)
    categorical = pd.Categorical(labels, categories=categories)
    indicator = coo_matrix(
        (np.ones(adata.n_obs, dtype=np.uint8), (categorical.codes, np.arange(adata.n_obs))),
        shape=(len(categories), adata.n_obs),
    ).tocsr()
    grouped_obs = adata.obs.assign(label=labels).drop_duplicates("label").set_index("label").reindex(categories)
    grouped = AnnData(X=indicator @ adata.X, obs=grouped_obs, var=adata.var.copy())

    if add_props:
        props = get_bin_props(grouped.obs[["x", "y"]].assign(label=grouped.obs_names), binsize).reindex(
            grouped.obs_names
        )
        grouped.obs["area"] = props["area"].to_numpy()
        grouped.obsm[SKM.OBSM_SPATIAL_KEY] = props.filter(regex="centroid-").to_numpy()
        grouped.obsm["bbox"] = props.filter(regex="bbox-").to_numpy()
        grouped.obsm["contour"] = props["contour"].to_numpy()
    else:
        grouped.obsm[SKM.OBSM_SPATIAL_KEY] = np.column_stack(
            (
                (grouped.obs["x"].to_numpy() + 0.5) * binsize,
                (grouped.obs["y"].to_numpy() + 0.5) * binsize,
            )
        )
    _initialize_seqscope_metadata(grouped, binsize=binsize)
    return grouped


__all__ = ["read_seqscope"]
