"""Strict matrix/table primitives shared by explicit and automatic spatial readers."""

import csv
import gzip

import h5py
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from scipy.io import mminfo, mmread

from ._errors import ContractError, ResourceDeferred

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


def _h5(path, *, full=False, budget=512 * 1024**2, technology=None):
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
        data = integer_counts(data, technology or "10x")
        obs = _ids(g["barcodes"][:], "matrix barcodes")
        ids = _ids(g["features/id"][:], "feature IDs")
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in g["features/name"][:]]
        X = sparse.csc_matrix((data, indices, ptr), shape=(n_genes, n_obs)).T.tocsr()
        var = pd.DataFrame({"gene_ids": ids}, index=ids if technology else names)
        if technology:
            var["gene_name"] = names
        adata = AnnData(X, obs=pd.DataFrame(index=obs), var=var)
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


def _mex(path, *, full=False, budget=512 * 1024**2, technology=None):
    def find(stems):
        if technology is not None:
            found = [path / name for name in stems if (path / name).is_file()]
            if len(found) > 1:
                raise ContractError(f"Multiple alternative MEX files require an explicit choice: {found}")
        for name in stems:
            if (path / name).is_file():
                return path / name
        raise ContractError(f"Missing MEX component {stems} under {path}")

    matrix = find(("matrix.mtx.gz", "matrix.mtx"))
    barcodes = find(("barcodes.tsv.gz", "barcodes.tsv"))
    features = find(("features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"))
    rows, cols, nnz, fmt, field, symmetry = mminfo(matrix)
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
    raw = mmread(matrix)
    # Validate before duplicate coordinates are coalesced: a negative entry must
    # not be hidden by a positive entry at the same matrix location.
    _numeric(raw.data, "raw MEX counts", nonnegative=True)
    raw.data = integer_counts(raw.data, technology or "10x MEX")
    X = raw.T.tocsr()
    _numeric(X.data, "MEX counts", nonnegative=True)
    var = pd.DataFrame({"gene_ids": ids}, index=ids if technology else [g[1] for g in genes])
    if technology:
        var["gene_name"] = [g[1] for g in genes]
    if all(len(g) >= 3 for g in genes):
        var["feature_types"] = [g[2] for g in genes]
    return AnnData(X, obs=pd.DataFrame(index=obs), var=var)


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


def table_values(frame, context, *, raw=True):
    """Parse table values without float32 truncation or invalid-value coercion.

    Raw exports must be nonnegative integer counts. Processed expression keeps
    finite signed values at float64 precision; integer-only columns stay exact.
    """
    frame = pd.DataFrame(frame)
    columns = []
    total = 0
    for column in frame:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy()
        except (ValueError, TypeError) as exc:
            raise ContractError(f"Non-numeric values in {context}") from exc
        _numeric(values, context, nonnegative=raw)
        if raw:
            values = integer_counts(values, context)
            total += sum(map(int, values))
        columns.append(values)
    if not columns:
        raise ContractError(f"No expression columns in {context}")
    if raw and total > np.iinfo(np.int64).max:
        raise ContractError(f"Aggregated counts exceed int64 range in {context}")
    result = np.column_stack(columns)
    if not raw and result.dtype.kind == "f":
        for i, original in enumerate(columns):
            if original.dtype.kind in "iu" and any(int(a) != int(b) for a, b in zip(original, result[:, i])):
                raise ContractError(f"Mixed numeric columns exceed exact float64 precision in {context}")
    return result
