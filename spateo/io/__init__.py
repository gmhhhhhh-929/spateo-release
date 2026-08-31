"""Unified input/output API for Spateo.

The ``general``, ``single`` and ``spatial`` packages are the primary
implementations. Historical flat modules remain importable for compatibility,
but top-level functions resolve to the maintained readers below.
"""

from . import general, single, spatial
from .bbs import alpha_shape, get_concave_hull
from .general import load, read_csv, save
from .image import read_image
from .single import read, read_10x_h5, read_10x_mtx, read_h5ad
from .spatial import (
    SpatialReadMatch,
    detect_spatial_technologies,
    detect_spatial_technology,
    read_atera,
    read_auto_spatial,
    read_bgi,
    read_bgi_agg,
    read_merfish,
    read_nanostring,
    read_seqfish,
    read_slideseq,
    read_spatial_auto,
    read_starmap_plus,
    read_visium,
    read_visium_hd,
    read_visium_hd_bin,
    read_visium_hd_seg,
    read_xenium,
    spatial_file_manifest,
    write_visium_hd_cellseg,
)
from .tenx import read_10x, read_10x_as_anndata

__all__ = [
    "general",
    "single",
    "spatial",
    "read",
    "read_h5ad",
    "read_10x_h5",
    "read_10x_mtx",
    "read_csv",
    "save",
    "load",
    "read_auto_spatial",
    "read_spatial_auto",
    "detect_spatial_technology",
    "detect_spatial_technologies",
    "SpatialReadMatch",
    "spatial_file_manifest",
    "read_atera",
    "read_xenium",
    "read_visium",
    "read_visium_hd",
    "read_visium_hd_bin",
    "read_visium_hd_seg",
    "write_visium_hd_cellseg",
    "read_merfish",
    "read_nanostring",
    "read_seqfish",
    "read_slideseq",
    "read_starmap_plus",
    "read_bgi",
    "read_bgi_agg",
    "read_image",
    "read_10x",
    "read_10x_as_anndata",
    "alpha_shape",
    "get_concave_hull",
]
