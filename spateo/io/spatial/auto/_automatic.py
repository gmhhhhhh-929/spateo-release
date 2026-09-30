"""Automatic layout selection delegates parsing to platform-native readers."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

from ...._registry import register_function
from .._assets import load_assets
from .._native_readers import DOMESTIC, get_reader
from .._read_engine import run_reading
from .._read_result import SpatialReadResult
from ._contracts import probe, read_core
from ._discovery import discover
from ._formats import _canonical_technologies

_DEFAULT_MEMORY = 1024**3
_IMAGE_BUDGET = 32 * 1024**2


def _resolve(candidates):
    """Only explicit specialization relations can remove overlapping claims."""
    if not candidates:
        return None, "No candidate passed the required format contract"
    # Exact MERFISH file prefixes are a specialization of generic seqFISH tables.
    merfish = [c for c in candidates if c.technology == "merfish"]
    if merfish:
        candidates = [
            c
            for c in candidates
            if not (
                c.technology == "seqfish" and any(c.counts == m.counts and c.metadata == m.metadata for m in merfish)
            )
        ]
    if len(candidates) == 1:
        return candidates[0], "Unique validated core layout; no score comparison"
    return None, "Multiple validated readers or companion encodings claim the same logical input"


def _group_candidates(candidates):
    groups = defaultdict(list)
    domestic_counts = {c.counts for c in candidates if c.technology in DOMESTIC}
    for c in candidates:
        identity = c.identity
        if c.counts in domestic_counts and c.technology in DOMESTIC | {"visium", "visium_hd_bin"}:
            identity = str(c.counts), "domestic_native_matrix"
        elif c.technology in ("visium", "visium_hd_bin"):
            identity = str(c.root), c.representation
        groups[identity].append(c)
    return groups


def _reader_name(candidate):
    if candidate.technology in DOMESTIC:
        return get_reader(candidate.technology).__name__ + ".read_core"
    return "spateo.io.spatial.auto._contracts.read_core"


def _assets(adata, candidate, enabled, budget, diagnostics):
    return load_assets(adata, candidate, enabled, budget, diagnostics, image_budget=_IMAGE_BUDGET)


@register_function(
    aliases=["read_spatial", "read_auto_spatial", "read_spatial_auto", "automatic spatial reading without thresholds"],
    category="io",
    description="Discover spatial inputs, validate core format contracts and return named results without score thresholds.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix="none",
    examples=["result = st.io.read_spatial('dataset_dir')", "adata = result.adata", "print(result.report)"],
    related=["io.read_visium", "io.read_slideseq"],
)
def read_spatial(
    path: Union[str, Path],
    *,
    technology: Optional[str] = None,
    load: bool = True,
    lazy: bool = False,
    load_images: bool = True,
    max_memory_bytes: int = _DEFAULT_MEMORY,
    max_files: int = 10000,
    max_depth: int = 4,
    stereoseq_bin_size: Optional[int] = None,
    stereoseq_chemistry: Optional[str] = None,
) -> SpatialReadResult:
    """Automatically read supported spatial layouts without confidence thresholds.

    A path is the only required argument. All discovered samples/representations
    are retained as named entries; no matrix is selected by a platform score.
    Core-format or ID failures stay visible in ``result.report``.
    ``read_auto_spatial`` and ``read_spatial_auto`` are aliases of this function
    and also return ``SpatialReadResult``. Direct platform readers are unchanged.

    Parameters beyond ``path`` are optional: ``technology`` restricts discovery;
    ``load=False`` discovers/probes but defers content loading. ``lazy=True``
    also defers core loading, but permits a unique ``result.adata`` access or an
    explicit ``entry.materialize()`` to load on demand. These modes cannot be
    combined. Lazy mode still reads bounded headers/metadata to resolve the
    format; it is not backed AnnData or out-of-core matrix slicing. Failed or
    resource-deferred lazy access is not retried implicitly. Use
    ``entry.load(max_memory_bytes=..., retry=True)`` for an explicit retry.
    Memory, inventory
    and depth limits are resource bounds, never statistical matching thresholds.
    ``max_memory_bytes`` is a conservative allocation budget, not an OS RSS cap.
    Entries exceeding it remain explicitly deferred and can be loaded later.
    ``stereoseq_bin_size`` optionally aggregates GEM records into square bins
    or selects an existing GEF resolution. Omitted: preserve GEM resolution,
    or select the smallest stored GEF bin. CellBin stays cells.
    ``stereoseq_chemistry`` records a user-declared V1/V2 label; matrix schema
    versions never determine chemistry. All input features are preserved.
    See ``docs/technicals/automatic_spatial_reading.md`` for supported contracts.
    """
    if stereoseq_chemistry not in (None, "V1", "V2"):
        raise ValueError("stereoseq_chemistry must be V1, V2 or None")
    if stereoseq_bin_size is not None and (
        isinstance(stereoseq_bin_size, bool) or not isinstance(stereoseq_bin_size, int) or stereoseq_bin_size < 1
    ):
        raise ValueError("stereoseq_bin_size must be a positive integer")
    allowed = _canonical_technologies(technology)

    def prepare(candidate):
        if candidate.technology == "bgi":
            candidate.options.update(stereoseq_bin_size=stereoseq_bin_size, stereoseq_chemistry=stereoseq_chemistry)

    return run_reading(
        path,
        discover=discover,
        probe=probe,
        read_core=read_core,
        reader_name=_reader_name,
        technology=technology,
        allowed=allowed,
        load=load,
        lazy=lazy,
        load_images=load_images,
        max_memory_bytes=max_memory_bytes,
        max_files=max_files,
        max_depth=max_depth,
        prepare_candidate=prepare,
        group_candidates=_group_candidates,
        resolve=_resolve,
        asset_loader=_assets,
    )
