"""Unified input/output API for Spateo.

The ``general``, ``single`` and ``spatial`` packages are the primary
implementations. The former flat technology modules have been removed; the
top-level functions below are direct exports of the maintained subpackages.
"""

from . import general, single, spatial
from .general import load, read_csv, save
from .single import read, read_10x_h5, read_10x_mtx, read_h5ad
from .spatial import (
    SpatialReadMatch,
    alpha_shape,
    detect_spatial_technologies,
    detect_spatial_technology,
    get_concave_hull,
    read_atera,
    read_auto_spatial,
    read_bgi,
    read_bgi_agg,
    read_image,
    read_merfish,
    read_nanostring,
    read_seqfish,
    read_seqscope,
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
    "read_seqscope",
    "read_slideseq",
    "read_starmap_plus",
    "read_bgi",
    "read_bgi_agg",
    "read_image",
    "alpha_shape",
    "get_concave_hull",
]
