"""Compatibility API for STARmap and the maintained STARmap PLUS reader."""

from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from ..configuration import SKM
from .spatial._starmap_plus import read_starmap_plus
from .utils import get_points_props


def read_starmap_as_anndata(data_dir: str) -> AnnData:
    """Read the historical STARmap count and gene CSV pair."""
    root = Path(data_dir)
    counts = pd.read_csv(root / "cell_barcode_count.csv", header=None)
    genes = pd.read_csv(root / "cell_barcode_names.csv", header=None)
    return AnnData(
        X=csr_matrix(counts.to_numpy(dtype=np.uint16)),
        obs=pd.DataFrame(index=[f"Cell_{index}" for index in range(len(counts))]),
        var=pd.DataFrame(index=genes.iloc[:, 2].astype(str)),
    )


def read_starmap_positions_as_dataframe(path: str) -> pd.DataFrame:
    """Read the historical STARmap label NPZ as point coordinates."""
    labels = csr_matrix(np.load(path)["labels"]).tocoo()
    points = pd.DataFrame({"x": labels.row, "y": labels.col, "label": labels.data})
    unique_labels, areas = np.unique(points["label"], return_counts=True)
    valid = unique_labels[(areas > 1000) & (areas < 100000)]
    points = points[points["label"].isin(valid)]
    if not points.empty:
        points = points[points["label"] != points["label"].max()]
    return points[["x", "y", "label"]]


def read_starmap(data_dir: str) -> AnnData:
    """Read the historical STARmap directory layout."""
    root = Path(data_dir)
    adata = read_starmap_as_anndata(data_dir)
    properties = get_points_props(read_starmap_positions_as_dataframe(str(root / "labels.npz")))
    if len(properties) != adata.n_obs:
        raise ValueError("STARmap expression cells and filtered label objects do not have matching lengths.")
    properties.index = adata.obs_names
    adata.obs["area"] = properties["area"].to_numpy()
    adata.obsm[SKM.OBSM_SPATIAL_KEY] = properties.filter(regex="centroid-").to_numpy()
    adata.obsm["contour"] = properties["contour"].to_numpy()
    adata.obsm["bbox"] = properties.filter(regex="bbox-").to_numpy()
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, 1.0)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, None)
    return adata


__all__ = [
    "read_starmap",
    "read_starmap_plus",
    "read_starmap_as_anndata",
    "read_starmap_positions_as_dataframe",
]
