"""Portable file manifests and provenance for spatial readers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Union

from anndata import AnnData

PathLike = Union[str, Path]


def _asset_role(path: Path) -> str:
    """Return a coarse, format-independent role for a spatial output file."""
    name = path.name.lower()
    suffixes = "".join(path.suffixes).lower()
    if "matrix" in name or suffixes.endswith(".mtx.gz") or suffixes.endswith(".mtx"):
        return "expression_matrix"
    if "transcript" in name or "molecule" in name:
        return "molecules"
    if "boundar" in name or "segment" in name or suffixes.endswith(".geojson"):
        return "segmentation"
    if "position" in name or "centroid" in name or "coordinate" in name:
        return "coordinates"
    if "scale" in name or "align" in name or name.endswith("experiment.xenium"):
        return "metadata"
    if "metric" in name or "summary" in name:
        return "quality_control"
    if suffixes.endswith((".ome.tif", ".ome.tiff", ".tif", ".tiff", ".png", ".jpg", ".jpeg")):
        return "image"
    if suffixes.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz", ".parquet", ".json")):
        return "table"
    return "other"


def spatial_file_manifest(path: PathLike, *, max_files: int = 10_000) -> dict[str, Any]:
    """Inventory files below a spatial dataset without loading their contents.

    The manifest uses parallel arrays instead of a dictionary keyed by relative
    paths because HDF5-backed ``AnnData.uns`` keys cannot safely contain ``/``.
    File paths are relative to the detected dataset root, making cached H5AD
    files easier to move between machines.
    """
    requested = Path(path).expanduser().resolve()
    root = requested.parent if requested.is_file() else requested
    if not root.exists():
        return {
            "root": str(root),
            "paths": [],
            "sizes_bytes": [],
            "roles": [],
            "truncated": False,
        }

    candidates = [requested] if requested.is_file() else root.rglob("*")
    records: list[tuple[str, int, str]] = []
    truncated = False
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            relative = candidate.relative_to(root).as_posix()
            size = int(candidate.stat().st_size)
        except (OSError, ValueError):
            continue
        records.append((relative, size, _asset_role(candidate)))
        if len(records) >= max_files:
            truncated = True
            break
    records.sort(key=lambda record: record[0].lower())
    return {
        "root": str(root),
        "paths": [record[0] for record in records],
        "sizes_bytes": [record[1] for record in records],
        "roles": [record[2] for record in records],
        "truncated": truncated,
    }


def record_spatial_io(
    adata: AnnData,
    *,
    technology: str,
    source: PathLike,
    reader: str,
    confidence: Optional[float] = None,
    evidence: tuple[str, ...] = (),
    reader_kwargs: Optional[Mapping[str, Any]] = None,
    manifest: Optional[dict[str, Any]] = None,
    format_status: str = "stable",
) -> None:
    """Write a serializable, shared provenance block to ``adata.uns``."""
    source_path = Path(source).expanduser().resolve()
    current = dict(adata.uns.get("spateo_io", {}))
    current.update(
        {
            "technology": str(technology),
            "source": str(source_path),
            "reader": str(reader),
            "format_status": str(format_status),
            "evidence": list(evidence),
            "reader_kwargs": {str(key): repr(value) for key, value in dict(reader_kwargs or {}).items()},
            "manifest": manifest or spatial_file_manifest(source_path),
        }
    )
    if confidence is not None:
        current["confidence"] = float(confidence)
    adata.uns["spateo_io"] = current


__all__ = ["record_spatial_io", "spatial_file_manifest"]
