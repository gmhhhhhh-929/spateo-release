"""Native Salus STS workflow matrix and spatial-barcode reader."""

import sys
from pathlib import Path
from typing import Union

from ..._registry import register_function
from ._native_common import (
    _ids,
    _numeric,
    discover_mex,
    headerless_coordinates,
    probe_native,
    read_direct,
    read_native,
)

TECHNOLOGY = "salus"


def discover(files, requested):
    """Identify the published MEX plus spatial.txt(.gz) workflow export."""
    return discover_mex(files, requested, TECHNOLOGY, ("spatial.txt.gz", "spatial.txt"), "spatial_barcodes")


def metadata(candidate, *, full=False, budget=512 * 1024**2):
    """Read headerless barcode/x/y coordinates without registration or binning."""
    frame = headerless_coordinates(candidate.metadata, TECHNOLOGY, full=full, budget=budget)
    ids = _ids(frame["barcode"], str(candidate.metadata))
    xy = _numeric(frame[["x", "y"]], "salus spatial coordinates")
    frame = frame.copy()
    frame.index = ids
    return frame, xy, "native Salus workflow pixel coordinates (x,y); no implicit registration"


def probe(candidate, budget):
    """Check the native MEX and coordinate contract without loading the matrix."""
    return probe_native(candidate, budget, metadata)


def read_core(candidate, budget):
    """Read integer counts and join the supplied pixel coordinates by barcode."""
    return read_native(candidate, budget, metadata, "spatial_barcode")


@register_function(
    aliases=["read_salus", "Salus STS native spatial input"],
    category="io",
    description="Read Salus STS workflow spatial exports",
)
def read_salus(
    path: Union[str, Path], *, load_images: bool = True, max_memory_bytes: int = 1024**3, return_result: bool = False
):
    """Read Salus STS workflow MEX plus headerless barcode/x/y spatial.txt(.gz).

    ``path`` may name a native matrix directory or a parent with multiple
    exports. Coordinates retain native pixels; no binning or image registration
    is performed. Optional images are loaded within the image budget when
    ``load_images=True``. ``max_memory_bytes`` is a conservative allocation
    budget, not an OS memory cap. Return a unique AnnData, or named outcomes and
    recovery advice with ``return_result=True``.

    Parsing is independent of the automatic platform dispatcher.
    """
    return read_direct(path, sys.modules[__name__], load_images, max_memory_bytes, return_result)
