r"""
Input/Output utilities for Spateo datasets.

Subpackages:
    general: Shared I/O helpers for tabular data and serialization.
    bulk: I/O helpers for bulk omics resources.
    single: I/O helpers for single-cell data.
    spatial: I/O helpers for spatial omics data.

Compatibility shortcuts:
    - ``st.io.read(...)``
    - ``st.io.read_h5ad(...)``
    - ``st.io.read_10x_h5(...)``
    - ``st.io.read_10x_mtx(...)``
    - ``st.io.read_visium_hd(...)``
    - ``st.io.read_csv(...)``, ``st.io.save(...)``, ``st.io.load(...``
"""

from . import single, spatial, general
from .single import read, read_10x_h5, read_10x_mtx, read_h5ad
from .general import read_csv, save, load
from .spatial import (
    SpatialReadMatch,
    detect_spatial_technologies,
    detect_spatial_technology,
    read_auto_spatial,
    read_bgi,
    read_merfish,
    read_nanostring,
    read_seqfish,
    read_slideseq,
    read_spatial_auto,
    read_starmap_plus,
    read_visium,
    read_visium_hd,
    read_xenium,
)



__all__ = [
    "single",
    "general",
    "spatial",
    # top-level compatibility exports
    "read",
    "read_h5ad",
    "read_10x_h5",
    "read_10x_mtx",
    
    "read_seqfish",
    "read_merfish",
    "read_slideseq",
    "read_visium",
    "read_visium_hd",
    "read_xenium",
    "read_starmap_plus",
    "read_nanostring",
    "read_bgi",
    "SpatialReadMatch",
    "detect_spatial_technologies",
    "detect_spatial_technology",
    "read_auto_spatial",
    "read_spatial_auto",

    "read_csv",
    "save",
    "load",
]
