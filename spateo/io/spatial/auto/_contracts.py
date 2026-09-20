"""Strict core-data adapters. Never repair values or align observations by row order.

These adapters are deliberately separate from permissive legacy platform readers.
Probes check structure only; ``read_core`` validates all core identifiers and values.
"""

from __future__ import annotations

import csv
from array import array
import gzip
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from scipy.io import mminfo, mmread

from ....configuration import SKM
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
_POS = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]


def _open(path):
    return (
        gzip.open(path, "rt", encoding="utf-8-sig", newline="")
        if path.name.endswith(".gz")
        else path.open(encoding="utf-8-sig", newline="")
    )


def table(path, *, full=False, budget=512 * 1024**2, positions=False):
    if not path.is_file():
        raise ContractError(f"Missing required file: {path}")
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        if len(set(pf.schema.names)) != len(pf.schema.names):
            raise ContractError(f"Duplicate table fields: {path}")
        batches, used = [], 0
        for batch in pf.iter_batches(batch_size=16384 if full else 128):
            chunk = batch.to_pandas()
            used += int(chunk.memory_usage(deep=True).sum()) * 4
            if used > budget:
                raise ResourceDeferred(f"Table exceeds memory budget: {path}")
            batches.append(chunk)
            if not full:
                break
        frame = pd.concat(batches, ignore_index=True) if batches else pd.DataFrame(columns=pf.schema.names)
    else:
        with _open(path) as handle:
            line = ""
            for line in handle:
                if line.strip() and not line.startswith("#"):
                    break
            if not line.strip():
                raise ContractError(f"Empty table: {path}")
            sep = "\t" if line.count("\t") > line.count(",") else ","
            header = next(csv.reader([line], delimiter=sep))
            headerless = positions and (path.name == "tissue_positions_list.csv" or "barcode" not in header)
            if not headerless and len(set(header)) != len(header):
                raise ContractError(f"Duplicate column names: {path}")
        args = dict(sep=sep, comment="#", dtype=str, keep_default_na=False)
        if headerless:
            if len(header) != 6:
                raise ContractError(f"Legacy positions must have six columns: {path}")
            args.update(header=None, names=_POS)
        if not full:
            frame = pd.read_csv(path, nrows=min(128, max(1, budget // max(512, len(header) * 512))), **args)
        else:
            chunks, used = [], 0
            with pd.read_csv(
                path, chunksize=min(16384, max(1, budget // max(2048, len(header) * 2048))), **args
            ) as reader:
                for chunk in reader:
                    used += int(chunk.memory_usage(deep=True).sum()) * 4
                    if used > budget:
                        raise ResourceDeferred(f"Decoded table exceeds memory budget: {path}")
                    chunks.append(chunk)
            frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if frame.empty:
        raise ContractError(f"Empty table: {path}")
    return frame


def _ids(values, context):
    arr = pd.Index([v.decode() if isinstance(v, bytes) else str(v) for v in values])
    if arr.has_duplicates or any(not v.strip() or v.lower() in ("nan", "none", "<na>") for v in arr):
        raise ContractError(f"Missing or duplicated identifiers in {context}")
    return arr


def _numeric(frame, context, *, nonnegative=False):
    try:
        values = np.asarray(frame, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"Non-numeric values in {context}") from exc
    if not np.isfinite(values).all() or (nonnegative and (values < 0).any()):
        raise ContractError(f"Non-finite or invalid negative values in {context}")
    return values


def _h5(path, *, full=False, budget=512 * 1024**2):
    if not path.is_file():
        raise ContractError(f"Missing count matrix: {path}")
    with h5py.File(path, "r") as f:
        if "matrix" not in f:
            raise ContractError("Expected a supported 10x v3 matrix group; legacy H5 requires a direct reader")
        g = f["matrix"]
        required = ("data", "indices", "indptr", "shape", "barcodes", "features/id", "features/name")
        missing = [key for key in required if key not in g or not isinstance(g[key], h5py.Dataset)]
        if missing:
            raise ContractError(f"Missing H5 datasets: {missing}")
        shape = np.asarray(g["shape"])
        if shape.shape != (2,) or shape.dtype.kind not in "iu" or np.any(shape <= 0):
            raise ContractError("Invalid H5 matrix dimensions")
        n_genes, n_obs = map(int, shape)
        nnz = len(g["data"])
        if len(g["indices"]) != nnz or len(g["indptr"]) != n_obs + 1 or len(g["barcodes"]) != n_obs:
            raise ContractError("H5 sparse arrays and barcode dimensions disagree")
        if len(g["features/name"]) != n_genes or len(g["features/id"]) != n_genes:
            raise ContractError("H5 feature dimensions disagree")
        if g["indices"].dtype.kind not in "iu" or g["indptr"].dtype.kind not in "iu":
            raise ContractError("Sparse indices must be integers")
        estimated = nnz * 32 + (n_obs + n_genes) * 1024
        if not full:
            return dict(estimated_bytes=estimated, n_obs=n_obs, n_vars=n_genes, storage="10x_h5")
        if estimated > budget:
            raise ResourceDeferred("Sparse matrix exceeds estimated memory budget")
        ptr, indices, data = g["indptr"][:], g["indices"][:], g["data"][:]
        if (
            ptr[0] != 0
            or ptr[-1] != nnz
            or np.any(ptr[1:] < ptr[:-1])
            or np.any(indices >= n_genes)
            or np.any(indices < 0)
        ):
            raise ContractError("Invalid sparse matrix indexing")
        _numeric(data, "H5 counts", nonnegative=True)
        obs = _ids(g["barcodes"][:], "matrix barcodes")
        ids = _ids(g["features/id"][:], "feature IDs")
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in g["features/name"][:]]
        X = sparse.csc_matrix((data, indices, ptr), shape=(n_genes, n_obs)).T.tocsr()
        adata = AnnData(X, obs=pd.DataFrame(index=obs), var=pd.DataFrame({"gene_ids": ids}, index=names))
        for src, dest in [("feature_type", "feature_types"), ("genome", "genome")]:
            if src in g["features"]:
                if len(g["features"][src]) != n_genes:
                    raise ContractError(f"Invalid {src} feature length")
                adata.var[dest] = g["features"][src].asstr()[:]
        metadata = {}
        for name in ("library_ids", "chemistry_description", "software_version"):
            if name in f.attrs:
                value = np.atleast_1d(f.attrs[name])
                if value.size <= 100:
                    metadata[name] = [v.decode() if isinstance(v, bytes) else str(v) for v in value]
        adata.uns["_source_h5_metadata"] = metadata
        return adata


def _mex(path, *, full=False, budget=512 * 1024**2):
    def find(stems):
        for name in stems:
            if (path / name).is_file():
                return path / name
        raise ContractError(f"Missing MEX component {stems} under {path}")

    matrix = find(("matrix.mtx.gz", "matrix.mtx"))
    barcodes = find(("barcodes.tsv.gz", "barcodes.tsv"))
    features = find(("features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"))
    with gzip.open(matrix, "rb") if matrix.name.endswith(".gz") else matrix.open("rb") as f:
        rows, cols, nnz, fmt, field, symmetry = mminfo(f)
    if rows <= 0 or cols <= 0 or fmt != "coordinate" or symmetry != "general":
        raise ContractError("Unsupported Matrix Market structure")
    estimated = int(nnz * 40 + (rows + cols) * 1024)
    if not full:
        return dict(estimated_bytes=estimated, n_obs=int(cols), n_vars=int(rows), storage="10x_mex")
    if estimated > budget:
        raise ResourceDeferred("MEX matrix exceeds memory budget")
    with _open(barcodes) as f:
        obs = _ids([line.rstrip("\n\r").split("\t")[0] for line in f], "MEX barcodes")
    with _open(features) as f:
        genes = list(csv.reader(f, delimiter="\t"))
    if len(obs) != cols or len(genes) != rows or any(len(g) < 2 for g in genes):
        raise ContractError("MEX matrix axes disagree with barcodes/features")
    ids = _ids([g[0] for g in genes], "MEX feature IDs")
    with gzip.open(matrix, "rb") if matrix.name.endswith(".gz") else matrix.open("rb") as f:
        X = mmread(f).T.tocsr()
    _numeric(X.data, "MEX counts", nonnegative=True)
    return AnnData(X, obs=pd.DataFrame(index=obs), var=pd.DataFrame({"gene_ids": ids}, index=[g[1] for g in genes]))


def _meta(candidate, *, full=False, budget=512 * 1024**2):
    tech = candidate.technology
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


def _slideseq_counts(path, *, full=False, budget=512 * 1024**2):
    """Stream gene-by-bead CSV into sparse storage without a dense table.

    Probe estimates only the bounded row workspace; the full scan enforces a
    growing sparse-storage budget. All values, row widths and IDs are checked.
    """
    with _open(path) as handle:
        lines = (line for line in handle if line.strip() and not line.startswith("#"))
        first = next(lines, None)
        if first is None:
            raise ContractError(f"Empty table: {path}")
        sep = "\t" if first.count("\t") > first.count(",") else ","
        header = next(csv.reader([first], delimiter=sep))
        if len(header) < 2 or len(set(header)) != len(header):
            raise ContractError("Slide-seq requires unique barcode columns and a gene column")
        ids = _ids(header[1:], "Slide-seq expression barcodes")
        workspace = len(header) * 1024
        if workspace > budget:
            raise ResourceDeferred("Slide-seq row workspace exceeds memory budget")
        reader = csv.reader(lines, delimiter=sep)
        values, indices, indptr = array("d"), array("q"), array("q", [0])
        genes = []
        for row in reader:
            if len(row) != len(header):
                raise ContractError(f"Slide-seq row {len(genes) + 2} has inconsistent field count")
            gene = row[0]
            if not gene.strip() or gene.lower() in ("nan", "none", "<na>"):
                raise ContractError("Missing Slide-seq gene identifier")
            numeric = _numeric(row[1:], "Slide-seq counts", nonnegative=True)
            if not full:
                return dict(estimated_bytes=workspace, storage="streamed_slideseq_csv", n_obs=len(ids))
            nz = np.flatnonzero(numeric)
            # Reserve space for buffers, final CSR conversion, ID tables and one row.
            estimated = workspace + (len(genes) + 1) * 1024 + (len(values) + len(nz)) * 64
            if estimated > budget:
                raise ResourceDeferred("Slide-seq sparse storage exceeds memory budget")
            values.frombytes(numeric[nz].astype(np.float64, copy=False).tobytes())
            indices.frombytes(nz.astype(np.int64, copy=False).tobytes())
            indptr.append(len(values))
            genes.append(gene)
        if not genes:
            raise ContractError(f"Empty expression table: {path}")
        gene_ids = _ids(genes, "Slide-seq expression genes")
        matrix = sparse.csr_matrix(
            (
                np.frombuffer(values, dtype=np.float64),
                np.frombuffer(indices, dtype=np.int64),
                np.frombuffer(indptr, dtype=np.int64),
            ),
            shape=(len(genes), len(ids)),
        ).T.tocsr()
        return AnnData(matrix, obs=pd.DataFrame(index=ids), var=pd.DataFrame(index=gene_ids))


def probe(candidate: Candidate, budget):
    """Bounded format checks; no ranking, inference of missing values, or IO writes."""
    tech = candidate.technology
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
        result = _mex(candidate.counts)
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
        values = _numeric(numeric, "counts", nonnegative=True)
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
        values = _numeric(frame.iloc[:, 1:], "expression values", nonnegative=candidate.representation != "processed")
        if tech == "merfish" and np.any(values != np.floor(values)):
            raise ContractError("MERFISH raw count table contains non-integer values")
        if transpose:
            values = values.T
    if values.nbytes * 4 > budget:
        raise ResourceDeferred("Dense table-to-sparse conversion exceeds memory budget")
    return AnnData(sparse.csr_matrix(values), obs=pd.DataFrame(index=ids), var=pd.DataFrame(index=genes))


def read_core(candidate: Candidate, budget):
    """Read one resolved layout under strict, explicit platform contracts."""
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
            adata = _h5(candidate.counts, full=True, budget=budget)
        elif candidate.counts.is_dir():
            adata = _mex(candidate.counts, full=True, budget=budget)
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
            except ContractError:
                pass  # Optional global coordinates do not replace required local coordinates.
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
