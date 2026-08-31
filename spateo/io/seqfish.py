"""Compatibility API for the maintained seqFISH reader."""

from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from ..configuration import SKM
from .spatial._seqfish import read_seqfish as _read_seqfish


def read_seqfish_meta_as_dataframe(
    path: str,
    fov_offset: Optional[pd.DataFrame] = None,
    accumulate_x: bool = False,
    accumulate_y: bool = False,
) -> pd.DataFrame:
    """Read and standardize the historical seqFISH centroid table."""
    frame = pd.read_csv(path).rename(
        columns={
            "Field of View": "fov",
            "Cell ID": "cell_id",
            "X": "x",
            "Y": "y",
            "Region": "region",
        }
    )
    required = {"fov", "cell_id", "x", "y"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"seqFISH metadata is missing required columns: {missing}")
    if fov_offset is not None:
        offsets = fov_offset.copy()
        required_offsets = {"fov", "x_offset", "y_offset"}
        if missing := sorted(required_offsets - set(offsets.columns)):
            raise ValueError(f"seqFISH FOV offsets are missing columns: {missing}")
        if accumulate_x:
            offsets["x_offset"] = offsets["x_offset"].cumsum()
        if accumulate_y:
            offsets["y_offset"] = offsets["y_offset"].cumsum()
        frame = frame.merge(offsets[list(required_offsets)], on="fov", how="left")
        frame["x"] += frame.pop("x_offset").fillna(0)
        frame["y"] += frame.pop("y_offset").fillna(0)
    frame["spatial"] = list(frame[["x", "y"]].astype(int).itertuples(index=False, name=None))
    return frame


def read_seqfish(
    path: str,
    meta_path: Optional[str] = None,
    fov_offset: Optional[pd.DataFrame] = None,
    accumulate_x: bool = False,
    accumulate_y: bool = False,
    **kwargs,
) -> AnnData:
    """Read modern directory outputs or dispatch the historical two-file API."""
    if meta_path is None:
        return _read_seqfish(path, **kwargs)
    counts = pd.read_csv(path)
    metadata = read_seqfish_meta_as_dataframe(meta_path, fov_offset, accumulate_x, accumulate_y)
    if len(metadata) != len(counts):
        raise ValueError("seqFISH counts and metadata must contain the same number of cells.")
    adata = AnnData(
        X=csr_matrix(counts.to_numpy(dtype=np.uint16)),
        obs=metadata.drop(columns=["spatial"]).copy(),
        var=pd.DataFrame(index=counts.columns.astype(str)),
    )
    adata.obsm[SKM.OBSM_SPATIAL_KEY] = np.asarray(metadata["spatial"].tolist(), dtype=float)
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, 1.0)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, None)
    return adata


__all__ = ["read_seqfish", "read_seqfish_meta_as_dataframe"]
