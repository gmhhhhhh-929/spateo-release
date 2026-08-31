r"""I/O utilities for spatial omics datasets."""

from ._atera import read_atera
from ._merfish import read_merfish
from ._nanostring import read_nanostring
from ._provenance import spatial_file_manifest
from ._seqfish import read_seqfish
from ._slideseq import read_slideseq
from ._starmap_plus import read_starmap_plus
from ._stereoseq import read_bgi, read_bgi_agg
from ._visium import read_visium
from ._visium_hd import (
    read_visium_hd,
    read_visium_hd_bin,
    read_visium_hd_seg,
    write_visium_hd_cellseg,
)
from ._xenium import read_xenium
from .auto import (
    SpatialReadMatch,
    detect_spatial_technologies,
    detect_spatial_technology,
    read_auto_spatial,
    read_spatial_auto,
)

__all__ = [
    "read_visium",
    "read_xenium",
    "read_atera",
    "read_slideseq",
    "read_merfish",
    "read_starmap_plus",
    "read_bgi",
    "read_bgi_agg",
    "read_nanostring",
    "read_visium_hd",
    "read_visium_hd_bin",
    "read_visium_hd_seg",
    "write_visium_hd_cellseg",
    "read_seqfish",
    "SpatialReadMatch",
    "detect_spatial_technologies",
    "detect_spatial_technology",
    "read_auto_spatial",
    "read_spatial_auto",
    "spatial_file_manifest",
]
