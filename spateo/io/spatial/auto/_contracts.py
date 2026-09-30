"""Strict core-data adapters. Never repair values or align observations by row order.

These adapters are deliberately separate from permissive legacy platform readers.
Probes check structure only; ``read_core`` validates all core identifiers and values.
"""

from __future__ import annotations

import json
import re
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from ....configuration import SKM
from .._matrix import _POS, _h5, _ids, _mex, _numeric, table, table_values
from .._native_readers import DOMESTIC, get_reader
from .._slideseq_matrix import _slideseq_counts
from ._discovery import Candidate
from ._errors import ContractError, ResourceDeferred
from ._stereo import probe_stereo, read_stereo_core


def _token(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _column(frame, aliases, *, first_index=False):
    columns = {_token(c): c for c in frame.columns}
    for alias in aliases:
        if sum(_token(c) == _token(alias) for c in frame.columns) > 1:
            raise ContractError(f"Ambiguous column aliases for {alias}")
    for name in aliases:
        if _token(name) in columns:
            return columns[_token(name)]
    first = str(frame.columns[0]) if len(frame.columns) else ""
    if first_index and (_token(first) in ("", "unnamed0", "index")):
        return first
    raise ContractError(f"Missing identifier/coordinate column; expected {aliases}; found {list(frame.columns)[:15]}")


_ID = ("cell_id", "cell", "cellid", "barcode", "barcodes", "bead_barcode", "bead", "name", "id", "entity_id", "spot_id")
_X = (
    "center_x",
    "x_centroid",
    "centroid_x",
    "global_x",
    "x_global",
    "xcoord",
    "x_coord",
    "x_location",
    "x",
    "coordx",
    "positionx",
    "pixelx",
)
_Y = (
    "center_y",
    "y_centroid",
    "centroid_y",
    "global_y",
    "y_global",
    "ycoord",
    "y_coord",
    "y_location",
    "y",
    "coordy",
    "positiony",
    "pixely",
)


def _meta(candidate, *, full=False, budget=512 * 1024**2):
    tech = candidate.technology
    if tech in DOMESTIC:
        return get_reader(tech).metadata(candidate, full=full, budget=budget)
    if tech == "visium_hd_cellseg":
        from shapely.geometry import shape

        if not candidate.metadata.is_file():
            raise ContractError(f"Missing segmentation file: {candidate.metadata}")
        if candidate.metadata.stat().st_size * 12 > budget:
            raise ResourceDeferred("GeoJSON requires more memory than the probe/read budget")
        data = json.loads(candidate.metadata.read_text())
        if data.get("type") != "FeatureCollection" or not data.get("features"):
            raise ContractError("Expected a nonempty GeoJSON FeatureCollection")
        ids, geometries, xy = [], [], []
        # Geometry must be checked in full; a sampled GeoJSON is not marked validated.
        for feature in data["features"]:
            prop = feature.get("properties", {})
            if "cellid" in prop:
                key = str(prop["cellid"])
            elif "cell_id" in prop:
                key = f"cellid_{str(prop['cell_id']).zfill(9)}-1"
            else:
                raise ContractError("GeoJSON lacks cell_id/cellid")
            polygon = shape(feature.get("geometry"))
            if polygon.geom_type not in ("Polygon", "MultiPolygon") or polygon.is_empty or not polygon.is_valid:
                raise ContractError(f"Invalid cell geometry: {key}")
            ids.append(key)
            geometries.append(polygon.wkt)
            xy.append([polygon.centroid.x, polygon.centroid.y])
        frame = pd.DataFrame({"geometry": geometries}, index=_ids(ids, "segmentation"))
        return frame, _numeric(xy, "centroids"), "geometry coordinate units (not declared)"
    frame = table(candidate.metadata, full=full, budget=budget, positions=tech in ("visium", "visium_hd_bin"))
    if tech == "starmap_plus" and "NAME" in frame and frame.iloc[0]["NAME"] == "TYPE":
        # STARmap exports may contain one schema row, not an observation.
        axes = [c for c in ("X", "Y", "Z") if c in frame]
        if len(axes) < 2 or any(frame.iloc[0][c] != "numeric" for c in axes):
            raise ContractError("Malformed STARmap TYPE coordinate declaration")
        if any(v not in ("numeric", "group", "string") for v in frame.iloc[0].drop("NAME")):
            raise ContractError("Unknown STARmap TYPE field declaration")
        frame = frame.iloc[1:].copy()
        if frame.empty:
            raise ContractError("STARmap spatial table contains only a TYPE declaration")
    if tech in ("visium", "visium_hd_bin"):
        missing = set(_POS) - set(frame.columns)
        if missing:
            raise ContractError(f"Missing positions fields: {sorted(missing)}")
        key = "barcode"
        xy = ("pxl_col_in_fullres", "pxl_row_in_fullres")
        units = "full-resolution image pixels (x=column,y=row)"
    elif tech in ("xenium", "atera"):
        key = _column(frame, ("cell_id",))
        xy = (_column(frame, ("x_centroid",)), _column(frame, ("y_centroid",)))
        units = "micrometers (x,y)"
    elif tech == "nanostring":
        key = _column(frame, ("cell_ID", "cellid"))
        fov = _column(frame, ("fov", "fov_id"))
        frame[key] = frame[key].astype(str) + "_" + frame[fov].astype(str)
        xy = (
            _column(frame, ("CenterX_local_px", "center_x_local_px", "x_local_px")),
            _column(frame, ("CenterY_local_px", "center_y_local_px", "y_local_px")),
        )
        units = "FOV-local pixels; FOVs are not implicitly aligned"
    else:
        aliases = (*_ID, "label", "celllabel") if tech == "seqfish" else _ID
        key = _column(frame, aliases, first_index=True)
        xy = (_column(frame, _X), _column(frame, _Y))
        try:
            z = _column(frame, ("center_z", "global_z", "z_centroid", "centroid_z", "zcoord", "z_coord", "z"))
            xy = (*xy, z)
        except ContractError:
            pass
        units = "source coordinate units (not declared)"
    ids = _ids(frame[key], str(candidate.metadata))
    values = _numeric(frame[list(xy)], "spatial coordinates")
    frame = frame.copy()
    frame.index = ids
    return frame, values, units


def probe(candidate: Candidate, budget):
    """Bounded format checks; no ranking, inference of missing values, or IO writes."""
    tech = candidate.technology
    if tech in DOMESTIC:
        return get_reader(tech).probe(candidate, budget)
    for file in (candidate.counts, candidate.metadata):
        for part in [file, *file.parents]:
            if part == candidate.root:
                break
            if part.is_symlink():
                raise ContractError(f"Symlink input is not followed: {part}")
    if candidate.options.get("identity", "").startswith("unresolved"):
        raise ContractError(candidate.options["identity"])
    if tech == "visium_hd_bin" and not candidate.options.get("binsize"):
        raise ContractError("Cannot infer bin size from the supported square_NNN um directory layout")
    if tech == "bgi":
        result = probe_stereo(candidate.counts, budget, candidate.options.get("stereoseq_bin_size"))
    elif tech == "slideseq":
        result = _slideseq_counts(candidate.counts, budget=budget)
    elif candidate.counts.suffix == ".h5":
        result = _h5(candidate.counts)
    elif candidate.counts.is_dir():
        result = _mex(candidate.counts, technology=tech if tech in DOMESTIC else None)
    else:
        head = table(candidate.counts, budget=budget)
        if head.shape[1] < 2:
            raise ContractError("Expression table must contain identifiers and expression columns")
        result = dict(
            estimated_bytes=max(
                candidate.counts.stat().st_size * (80 if candidate.counts.name.endswith(".gz") else 12), 1024
            ),
            storage="table",
        )
    if tech != "bgi":
        _meta(candidate, budget=budget)
        result["estimated_bytes"] += candidate.metadata.stat().st_size * 12
    result.update(structure="passed", content="not_loaded")
    return result


def _table_counts(candidate, metadata_ids, budget):
    if candidate.technology == "slideseq":
        return _slideseq_counts(candidate.counts, full=True, budget=budget)
    frame = table(candidate.counts, full=True, budget=budget)
    tech = candidate.technology
    if tech == "nanostring":
        key = _column(frame, ("cell_ID", "cellid"))
        fov = _column(frame, ("fov", "fov_id"))
        ids = _ids(frame[key].astype(str) + "_" + frame[fov].astype(str), "CosMx expression IDs")
        numeric = frame.drop(columns=[key, fov])
        genes = list(numeric.columns)
        values = table_values(numeric, "CosMx counts", raw=True)
    else:
        first = frame.columns[0]
        rows = _ids(frame[first], "expression row IDs")
        cols = _ids(frame.columns[1:], "expression column IDs")
        row_match, col_match = rows.isin(metadata_ids).all(), cols.isin(metadata_ids).all()
        transpose = tech == "slideseq"
        if tech != "slideseq":
            if row_match == col_match:
                raise ContractError("Cannot uniquely align expression axis to metadata IDs; no row-order fallback")
            transpose = bool(col_match)
        ids, genes = (cols, list(rows)) if transpose else (rows, list(cols))
        values = table_values(frame.iloc[:, 1:], "expression values", raw=candidate.representation != "processed")
        if tech == "merfish" and np.any(values != np.floor(values)):
            raise ContractError("MERFISH raw count table contains non-integer values")
        if transpose:
            values = values.T
    if values.nbytes * 4 > budget:
        raise ResourceDeferred("Dense table-to-sparse conversion exceeds memory budget")
    return AnnData(sparse.csr_matrix(values), obs=pd.DataFrame(index=ids), var=pd.DataFrame(index=genes))


def read_core(candidate: Candidate, budget):
    """Read one resolved layout under strict, explicit platform contracts."""
    if candidate.technology in DOMESTIC:
        return get_reader(candidate.technology).read_core(candidate, budget)

    if candidate.technology == "bgi":
        adata = read_stereo_core(
            candidate.counts,
            budget,
            candidate.options.get("stereoseq_bin_size"),
            candidate.options.get("stereoseq_chemistry"),
        )
        units = adata.uns["stereoseq"]["coordinate_system"]
    else:
        meta, xy, units = _meta(candidate, full=True, budget=budget)
        if candidate.counts.suffix == ".h5":
            adata = _h5(
                candidate.counts,
                full=True,
                budget=budget,
                technology=candidate.technology if candidate.technology in DOMESTIC else None,
            )
        elif candidate.counts.is_dir():
            adata = _mex(
                candidate.counts,
                full=True,
                budget=budget,
                technology=candidate.technology if candidate.technology in DOMESTIC else None,
            )
        else:
            adata = _table_counts(candidate, meta.index, budget)
        missing = adata.obs_names[~adata.obs_names.isin(meta.index)]
        if len(missing):
            raise ContractError(f"{len(missing)} matrix IDs lack metadata/coordinates; examples: {list(missing[:5])}")
        order = meta.index.get_indexer(adata.obs_names)
        adata.obs = meta.iloc[order].copy()
        adata.obsm["spatial"] = xy[order]
        if candidate.technology == "nanostring":
            try:
                gx = _column(adata.obs, ("CenterX_global_px", "center_x_global_px", "x_global_px"))
                gy = _column(adata.obs, ("CenterY_global_px", "center_y_global_px", "y_global_px"))
                adata.obsm["spatial_fov"] = _numeric(adata.obs[[gx, gy]], "global FOV coordinates")
                adata.uns.setdefault("spateo_io", {})["optional_global_coordinates"] = "loaded"
            except ContractError as exc:
                present = any(
                    _token(c) in {"centerxglobalpx", "centeryglobalpx", "xglobalpx", "yglobalpx"} for c in adata.obs
                )
                adata.uns.setdefault("spateo_io", {})["optional_global_coordinates"] = (
                    "invalid" if present else "absent"
                )
                if present:
                    warnings.warn(f"Optional global FOV coordinates are invalid; local coordinates retained: {exc}")
    if not adata.obs_names.is_unique or adata.n_obs == 0 or adata.n_vars == 0:
        raise ContractError("Empty or duplicate observation axis")
    if (
        adata.obsm["spatial"].shape[0] != adata.n_obs or adata.obsm["spatial"].shape[1] not in (2, 3)
    ) or not np.isfinite(adata.obsm["spatial"]).all():
        raise ContractError("Incomplete spatial coordinate mapping")
    _numeric(adata.X.data, "output expression values", nonnegative=candidate.representation != "processed")
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    metadata = adata.uns.pop("_source_h5_metadata", {})
    libraries = metadata.get("library_ids", [])
    library = libraries[0] if len(libraries) == 1 else candidate.root.name
    metadata.update(coordinate_system=units, representation=candidate.representation)
    if candidate.technology == "bgi":
        metadata.update(adata.uns["stereoseq"])
    adata.uns["spatial"] = {library: {"images": {}, "scalefactors": {}, "metadata": metadata}}
    return adata
