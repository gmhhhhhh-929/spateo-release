"""Native identifiable Singleron/CeleScope space export reader.

Published CeleScope space H5 outputs carry ``chemistry_description=Spatial3``
and pair with headerless ``spatial/positions_list.csv``. Generic unmarked 10x
files are not assigned a vendor identity by this adapter.
"""

import sys
from pathlib import Path
from typing import Union

import h5py
import numpy as np

from ..._registry import register_function
from ._errors import ContractError
from ._layout import Candidate
from ._native_common import (
    _ids,
    _numeric,
    probe_native,
    read_direct,
    read_native,
    table,
)

TECHNOLOGY = "singleron"


def discover(files, requested):
    """Identify chemistry-marked H5 exports, keeping raw/filtered independently."""
    present = set(files)
    candidates = []
    for root in sorted({p.parent for p in files}):
        positions = root / "spatial" / "positions_list.csv"
        for name, population in (("filtered_feature_bc_matrix.h5", "filtered"), ("raw_feature_bc_matrix.h5", "raw")):
            counts = root / name
            if counts not in present or (requested.is_file() and requested not in (counts, positions)):
                continue
            try:
                with h5py.File(counts, "r") as handle:
                    value = np.atleast_1d(handle.attrs.get("chemistry_description", ""))
                    marker = [v.decode() if isinstance(v, bytes) else str(v) for v in value]
                    identified = marker == ["Spatial3"]
            except (OSError, ValueError):
                identified = False
            if identified:
                candidates.append(
                    Candidate(
                        TECHNOLOGY,
                        root,
                        counts,
                        positions,
                        "spots/" + population,
                        {"identity": "CeleScope space: H5 Spatial3 attribute plus spatial/positions_list.csv"},
                        [name, "spatial/positions_list.csv", "H5 chemistry_description=Spatial3"],
                    )
                )
    return candidates


def metadata(candidate, *, full=False, budget=512 * 1024**2):
    """Validate the six-column table; XY means full-resolution column and row."""
    frame = table(candidate.metadata, full=full, budget=budget, positions=True)
    axes = ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    required = {"barcode", "in_tissue", "array_row", "array_col", *axes}
    if not required.issubset(frame.columns):
        raise ContractError(f"CeleScope space positions lack fields: {sorted(required-set(frame.columns))}")
    ids = _ids(frame["barcode"], str(candidate.metadata))
    xy = _numeric(frame[axes], "singleron spatial coordinates")
    frame = frame.copy()
    frame.index = ids
    return frame, xy, "full-resolution image pixels (x=column,y=row)"


def probe(candidate, budget):
    """Check the identified native H5 and coordinate contract."""
    return probe_native(candidate, budget, metadata)


def read_core(candidate, budget):
    """Read integer H5 counts and join the source spot coordinates by barcode."""
    return read_native(candidate, budget, metadata, "spot")


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
    matrices are separate named outcomes; use ``return_result=True`` for bundles
    containing both. Optional images are loaded within the image budget when
    ``load_images=True``. ``max_memory_bytes`` is a conservative allocation
    budget, not an OS memory cap. Return a unique AnnData, or named outcomes and
    recovery advice with ``return_result=True``.

    Parsing is independent of the automatic platform dispatcher.
    """
    return read_direct(path, sys.modules[__name__], load_images, max_memory_bytes, return_result)
