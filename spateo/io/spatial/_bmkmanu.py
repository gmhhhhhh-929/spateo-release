"""Native BMKMANU S1000 aggregated spatial matrix reader.

BSTMatrix aggregated ``barcodes_pos.tsv(.gz)`` contains headerless barcode,
pos_w, pos_h values. Raw singular ``barcode_pos.tsv`` contains chip indices;
its geometry must be resolved by the vendor workflow before importing.
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
    headerless_coordinates,
    probe_native,
    read_direct,
    read_native,
)

TECHNOLOGY = "bmkmanu"


def discover(files, requested):
    """Keep both aggregated and unsupported raw-chip layouts visible in reports."""
    candidates = discover_mex(
        files,
        requested,
        TECHNOLOGY,
        ("barcodes_pos.tsv.gz", "barcodes_pos.tsv", "barcode_pos.tsv", "barcode_pos.tsv.gz"),
        "aggregated_bins",
    )
    for candidate in candidates:
        if candidate.metadata.name.startswith("barcode_pos."):
            candidate.representation = "raw_chip_indices"
    return candidates


def metadata(candidate, *, full=False, budget=512 * 1024**2):
    """Read aggregated display coordinates without guessing raw chip geometry."""
    if candidate.representation == "raw_chip_indices":
        raise ContractError(
            "BMK raw chip-index coordinates require BSTMatrix aggregated export (barcodes_pos.tsv.gz); "
            "no physical-coordinate conversion was guessed"
        )
    frame = headerless_coordinates(candidate.metadata, TECHNOLOGY, full=full, budget=budget)
    ids = _ids(frame["barcode"], str(candidate.metadata))
    xy = _numeric(frame[["x", "y"]], "bmkmanu spatial coordinates")
    frame = frame.copy()
    frame.index = ids
    return frame, xy, "native BSTMatrix display coordinates (pos_w,pos_h); physical units not established"


def probe(candidate, budget):
    """Check the native aggregated export, explicitly rejecting raw chip indices."""
    if candidate.representation == "raw_chip_indices":
        metadata(candidate, budget=budget)
    return probe_native(candidate, budget, metadata)


def read_core(candidate, budget):
    """Read integer aggregated counts and join display coordinates by barcode."""
    return read_native(candidate, budget, metadata, "aggregated_bin")


@register_function(
    aliases=["read_bmkmanu", "BMKMANU S1000 aggregated spatial input"],
    category="io",
    description="Read BMKMANU aggregated spatial matrix exports",
)
def read_bmkmanu(
    path: Union[str, Path], *, load_images: bool = True, max_memory_bytes: int = 1024**3, return_result: bool = False
):
    """Read BSTMatrix aggregated MEX and headerless barcodes_pos.tsv(.gz).

    ``path`` may name a native matrix directory or a parent with multiple
    exports. Counts and coordinates are matched by barcode. Raw five-field
    chip indices require upstream BSTMatrix aggregation; no geometry or unit
    conversion is guessed. Optional images are loaded within the image budget
    when ``load_images=True``. ``max_memory_bytes`` is a conservative allocation
    budget, not an OS memory cap. Return a unique AnnData, or named outcomes and
    recovery advice with ``return_result=True``.

    Parsing is independent of the automatic platform dispatcher.
    """
    return read_direct(path, sys.modules[__name__], load_images, max_memory_bytes, return_result)
