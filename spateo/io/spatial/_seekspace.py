"""Native SeekSpace matrix and barcode-coordinate reader.

The published SeekSpaceTools export is a MEX directory containing
``cell_locations.tsv(.gz)`` with ``Cell_Barcode``, ``X`` and ``Y`` fields.
Coordinate values are retained without an inferred physical-unit conversion.
"""

import sys
from pathlib import Path
from typing import Union

from ..._registry import register_function
from ._errors import ContractError
from ._native_common import (
    _ids,
    _numeric,
    discover_mex,
    probe_native,
    read_direct,
    read_native,
    table,
)

TECHNOLOGY = "seekspace"


def discover(files, requested):
    """Identify SeekSpace exports and restrict optional images to the same sample."""
    candidates = discover_mex(files, requested, TECHNOLOGY, ("cell_locations.tsv.gz", "cell_locations.tsv"), "cells")
    scope = requested.parent if requested.is_file() else requested
    suffix = "_filtered_feature_bc_matrix"
    for candidate in candidates:
        matrix = candidate.counts
        if matrix.name.endswith(suffix) and matrix.parent.is_relative_to(scope):
            candidate.root = matrix.parent
            candidate.options["image_prefix"] = matrix.name[: -len(suffix)] + "_aligned_"
            candidate.evidence = [candidate.metadata.relative_to(candidate.root).as_posix()]
    return candidates


def metadata(candidate, *, full=False, budget=512 * 1024**2):
    """Validate native cell coordinates; barcode identity defines matrix alignment."""
    frame = table(candidate.metadata, full=full, budget=budget)
    required = {"Cell_Barcode", "X", "Y"}
    if not required.issubset(frame.columns):
        raise ContractError(f"SeekSpace cell_locations requires Cell_Barcode, X, Y; found {list(frame.columns)}")
    ids = _ids(frame["Cell_Barcode"], str(candidate.metadata))
    xy = _numeric(frame[["X", "Y"]], "seekspace spatial coordinates")
    frame = frame.copy()
    frame.index = ids
    return frame, xy, "native chip/image pixels (X,Y); no implicit micrometer conversion"


def probe(candidate, budget):
    """Check the native MEX and coordinate contract without loading the matrix."""
    return probe_native(candidate, budget, metadata)


def read_core(candidate, budget):
    """Read all native counts and join validated cell coordinates by barcode."""
    return read_native(candidate, budget, metadata, "cell")


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
        path: Native matrix directory or parent containing SeekSpace exports.
        load_images: Load supported optional single-frame images within the image budget.
        max_memory_bytes: Conservative allocation budget, not an OS memory cap.
        return_result: Return named outcomes and recovery advice instead of a unique AnnData.

    Returns:
        AnnData, or SpatialReadResult when ``return_result=True``.

    This platform reader performs its own native discovery and parsing; it does
    not call the automatic platform dispatcher.
    """
    return read_direct(path, sys.modules[__name__], load_images, max_memory_bytes, return_result)
