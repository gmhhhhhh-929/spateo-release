"""Explicit readers for documented native domestic spatial-transcriptomics exports."""

from pathlib import Path
from typing import Union

from ..._registry import register_function


def _read(path, technology, load_images, max_memory_bytes, return_result):
    from .auto._automatic import read_spatial

    result = read_spatial(path, technology=technology, load_images=load_images, max_memory_bytes=max_memory_bytes)
    return result if return_result else result.adata


@register_function(
    aliases=["read_seekspace", "SeekSpace native spatial input"],
    category="io",
    description="Read native SeekSpace matrix and spatial barcode coordinates",
)
def read_seekspace(
    path: Union[str, Path], *, load_images: bool = True, max_memory_bytes: int = 1024**3, return_result: bool = False
):
    """Read SeekSpace MEX and Cell_Barcode/X/Y coordinates, joined by barcode.

    Args:
        path: Matrix directory or parent containing native SeekSpace exports.
        load_images: Load supported optional single-frame images within the image budget.
        max_memory_bytes: Conservative allocation budget, not an OS memory cap.
        return_result: Return all named outcomes and recovery advice instead of a unique AnnData.

    Returns:
        AnnData, or SpatialReadResult when ``return_result=True``.
    """
    return _read(path, "seekspace", load_images, max_memory_bytes, return_result)


@register_function(
    aliases=["read_bmkmanu", "BMKMANU S1000 aggregated spatial input"],
    category="io",
    description="Read BMKMANU aggregated spatial matrix exports",
)
def read_bmkmanu(
    path: Union[str, Path], *, load_images: bool = True, max_memory_bytes: int = 1024**3, return_result: bool = False
):
    """Read BSTMatrix aggregated MEX and headerless barcodes_pos.tsv(.gz).

    Raw five-field chip indices require upstream BSTMatrix aggregation and are
    reported explicitly as unsupported; no geometry or unit conversion is guessed.
    Parameters and return contract are the same as :func:`read_seekspace`.
    """
    return _read(path, "bmkmanu", load_images, max_memory_bytes, return_result)


@register_function(
    aliases=["read_salus", "Salus STS native spatial input"],
    category="io",
    description="Read Salus STS workflow spatial exports",
)
def read_salus(
    path: Union[str, Path], *, load_images: bool = True, max_memory_bytes: int = 1024**3, return_result: bool = False
):
    """Read Salus STS workflow MEX plus headerless barcode/x/y spatial.txt(.gz).

    Coordinates retain native pixels. Parameters and return contract are the same
    as :func:`read_seekspace`; no binning or image registration is performed.
    """
    return _read(path, "salus", load_images, max_memory_bytes, return_result)


@register_function(
    aliases=["read_singleron", "CeleScope space native spatial input"],
    category="io",
    description="Read identified CeleScope space exports",
)
def read_singleron(
    path: Union[str, Path], *, load_images: bool = True, max_memory_bytes: int = 1024**3, return_result: bool = False
):
    """Read identifiable CeleScope space H5 plus spatial/positions_list.csv.

    Requires the published H5 Spatial3 chemistry attribute. Generic unmarked
    Visium-compatible exports are not relabelled as Singleron. Raw and filtered
    matrices are separate named outcomes. Use ``return_result=True`` for bundles
    containing both; other parameters match :func:`read_seekspace`.
    """
    return _read(path, "singleron", load_images, max_memory_bytes, return_result)
