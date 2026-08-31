"""Compatibility API for the maintained MERFISH/MERSCOPE reader."""

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from ..configuration import SKM
from .spatial._merfish import read_merfish as _read_merfish


def read_merfish_as_anndata(path: str) -> AnnData:
    """Read the historical gene-by-cell MERFISH CSV representation."""
    frame = pd.read_csv(path, index_col=0).transpose()
    return AnnData(
        X=csr_matrix(frame.to_numpy(dtype=np.uint16)),
        obs=pd.DataFrame(index=frame.index.astype(str)),
        var=pd.DataFrame(index=frame.columns.astype(str)),
    )


def read_merfish_positions_as_dataframe(path: str) -> pd.DataFrame:
    """Read the historical MERFISH Excel centroid table."""
    positions = pd.read_excel(path, names=["x", "y"], index_col=0, dtype=np.float32)
    return positions - min(positions["x"].min(), positions["y"].min())


def read_merfish(path: str, positions_path: str | None = None, **kwargs) -> AnnData:
    """Read modern directory outputs or dispatch the historical two-file API."""
    if positions_path is None:
        return _read_merfish(path, **kwargs)
    adata = read_merfish_as_anndata(path)
    positions = read_merfish_positions_as_dataframe(positions_path)
    keep = adata.obs_names.intersection(positions.index.astype(str))
    adata = adata[keep].copy()
    adata.obsm[SKM.OBSM_SPATIAL_KEY] = positions.reindex(keep)[["x", "y"]].to_numpy()
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, 1.0)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, None)
    return adata


__all__ = ["read_merfish", "read_merfish_as_anndata", "read_merfish_positions_as_dataframe"]
