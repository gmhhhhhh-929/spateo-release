"""Documented native spatial exports from Chinese vendors/research pipelines.

File contracts identify compatible exports, not manufacturing provenance. A
generic 10x matrix alone is deliberately insufficient for these adapters.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ._errors import ContractError, ResourceDeferred

DOMESTIC = frozenset({"seekspace", "bmkmanu", "salus", "singleron"})
_COORDINATES = {
    "seekspace": ("cell_locations.tsv.gz", "cell_locations.tsv"),
    "bmkmanu": ("barcodes_pos.tsv.gz", "barcodes_pos.tsv", "barcode_pos.tsv", "barcode_pos.tsv.gz"),
    "salus": ("spatial.txt.gz", "spatial.txt"),
}


def discover_domestic(files, requested):
    """Propose native layouts, retaining missing matrix components for diagnostics."""
    from ._discovery import Candidate

    present = set(files)
    scope = requested.parent if requested.is_file() else requested
    result = []
    for tech, names in _COORDINATES.items():
        for meta in sorted(p for p in files if p.name in names):
            matrix = meta.parent
            root = matrix
            options = {}
            representation = {"seekspace": "cells", "bmkmanu": "aggregated_bins", "salus": "spatial_barcodes"}[tech]
            if tech == "seekspace" and matrix.name.endswith("_filtered_feature_bc_matrix"):
                # Only inspect companions in the explicitly requested scope.
                if matrix.parent.is_relative_to(scope):
                    root = matrix.parent
                    options["image_prefix"] = matrix.name[: -len("_filtered_feature_bc_matrix")] + "_aligned_"
            if tech == "bmkmanu" and meta.name.startswith("barcode_pos."):
                representation = "raw_chip_indices"
            if requested.is_file() and requested != meta:
                component_names = {
                    "matrix.mtx",
                    "matrix.mtx.gz",
                    "features.tsv",
                    "features.tsv.gz",
                    "genes.tsv",
                    "genes.tsv.gz",
                    "barcodes.tsv",
                    "barcodes.tsv.gz",
                }
                if requested.parent != matrix or requested.name not in component_names:
                    continue
            result.append(
                Candidate(tech, root, matrix, meta, representation, options, [meta.relative_to(root).as_posix()])
            )

    for root in sorted({p.parent for p in files}):
        positions = root / "spatial" / "positions_list.csv"
        for name, population in (("filtered_feature_bc_matrix.h5", "filtered"), ("raw_feature_bc_matrix.h5", "raw")):
            counts = root / name
            if counts not in present:
                continue
            try:
                with h5py.File(counts, "r") as handle:
                    value = np.atleast_1d(handle.attrs.get("chemistry_description", ""))
                    marker = [v.decode() if isinstance(v, bytes) else str(v) for v in value]
                    identified = marker == ["Spatial3"]
            except (OSError, ValueError):
                identified = False
            if not identified or (requested.is_file() and requested not in (counts, positions)):
                continue
            result.append(
                Candidate(
                    "singleron",
                    root,
                    counts,
                    positions,
                    "spots/" + population,
                    {"identity": "CeleScope space: H5 Spatial3 attribute plus spatial/positions_list.csv"},
                    [name, "spatial/positions_list.csv", "H5 chemistry_description=Spatial3"],
                )
            )
    return result


def metadata(candidate, *, full=False, budget=512 * 1024**2):
    """Read native coordinates, preserving IDs and units without row-order joins."""
    from ._contracts import _ids, _numeric, table

    tech, path = candidate.technology, candidate.metadata
    if candidate.representation == "raw_chip_indices":
        raise ContractError(
            "BMK raw chip-index coordinates require BSTMatrix aggregated export (barcodes_pos.tsv.gz); "
            "no physical-coordinate conversion was guessed"
        )
    if tech == "seekspace":
        frame = table(path, full=full, budget=budget)
        required = {"Cell_Barcode", "X", "Y"}
        if not required.issubset(frame.columns):
            raise ContractError(f"SeekSpace cell_locations requires Cell_Barcode, X, Y; found {list(frame.columns)}")
        key, axes = "Cell_Barcode", ["X", "Y"]
        units = "native chip/image pixels (X,Y); no implicit micrometer conversion"
    elif tech == "singleron":
        frame = table(path, full=full, budget=budget, positions=True)
        key, axes = "barcode", ["pxl_col_in_fullres", "pxl_row_in_fullres"]
        required = {key, "in_tissue", "array_row", "array_col", *axes}
        if not required.issubset(frame.columns):
            raise ContractError(f"CeleScope space positions lack fields: {sorted(required-set(frame.columns))}")
        units = "full-resolution image pixels (x=column,y=row)"
    else:
        if not path.is_file():
            raise ContractError(f"Missing required coordinate file: {path}")
        args = dict(sep=r"\s+", header=None, dtype=str, keep_default_na=False)
        if full:
            chunks, used = [], 0
            with pd.read_csv(path, chunksize=min(16384, max(1, budget // 4096)), **args) as reader:
                for chunk in reader:
                    used += int(chunk.memory_usage(deep=True).sum()) * 4
                    if used > budget:
                        raise ResourceDeferred(f"Decoded coordinate table exceeds memory budget: {path}")
                    if chunk.shape[1] != 3:
                        raise ContractError(f"{tech} native coordinate table requires exactly 3 headerless fields")
                    chunks.append(chunk)
            frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        else:
            frame = pd.read_csv(path, nrows=min(128, max(1, budget // 4096)), **args)
        if frame.empty or frame.shape[1] != 3:
            raise ContractError(f"{tech} native coordinate table requires exactly 3 headerless fields")
        frame.columns = ["barcode", "x", "y"]
        key, axes = "barcode", ["x", "y"]
        units = (
            "native BSTMatrix display coordinates (pos_w,pos_h); physical units not established"
            if tech == "bmkmanu"
            else "native Salus workflow pixel coordinates (x,y); no implicit registration"
        )
    ids = _ids(frame[key], str(path))
    xy = _numeric(frame[axes], f"{tech} spatial coordinates")
    frame = frame.copy()
    frame.index = ids
    return frame, xy, units


def integer_counts(values, technology):
    """Validate raw counts before sparse coalescing; prevent silent integer overflow."""
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all() or np.any(values < 0):
        raise ContractError(f"{technology} requires finite nonnegative real counts")
    if values.dtype.kind == "f":
        if np.any(values != np.floor(values)):
            raise ContractError(f"{technology} raw count matrix contains non-integer values")
        if np.any(values >= 2**53):
            raise ContractError(
                "Floating-point native counts exceed exact integer precision; export integer Matrix Market"
            )
    if values.dtype.kind in "iu" and np.any(values > np.iinfo(np.int64).max):
        raise ContractError("Native counts exceed int64 range")
    if sum(map(int, values)) > np.iinfo(np.int64).max:
        raise ContractError("Aggregated native counts exceed int64 range")
    return values.astype(np.int64, copy=False)


def finish(adata, candidate):
    """Retain stable feature identities and annotate the observed export contract."""
    from ._contracts import _ids

    adata.X.data = integer_counts(adata.X.data, candidate.technology)
    if "gene_name" not in adata.var:
        adata.var["gene_name"] = adata.var_names.to_numpy()
    adata.var_names = _ids(adata.var["gene_ids"], "native feature IDs")
    adata.uns["native_spatial_export"] = {
        "technology": candidate.technology,
        "identity_evidence": "documented export layout; not independent experimental-provenance verification",
        "observation_type": {
            "seekspace": "cell",
            "bmkmanu": "aggregated_bin",
            "salus": "spatial_barcode",
            "singleron": "spot",
        }[candidate.technology],
        "counts_policy": "retain all supplied matrix entries and features; no normalization or tissue filtering",
    }
