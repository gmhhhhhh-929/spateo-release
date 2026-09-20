"""Native SAW GEM and GEF contracts, shared by Stereo-seq V1 and V2.

Chemistry is not encoded by GEM/GEF format versions. No chemistry, gene class,
physical pitch, segmentation or image registration is inferred from a filename.
"""

from __future__ import annotations

import csv
import gzip
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from ._errors import ContractError, ResourceDeferred

COUNTS = ("MIDCount", "MIDCounts", "UMICount", "UMICounts", "count", "total")
LAYERS = {"ExonCount": "exon", "EXONIC": "spliced", "INTRONIC": "unspliced"}


def _text(value):
    return value.decode("utf-8").rstrip("\x00") if isinstance(value, bytes) else str(value)


def _integers(values, what):
    """Reject fractional/negative/overflowed source values before narrowing."""
    arr = np.asarray(values)
    if arr.dtype.kind in "iu":
        if (arr < 0).any() or (arr > np.iinfo(np.int64).max).any():
            raise ContractError(f"Invalid {what}: outside nonnegative int64 range")
        return arr.astype(np.int64)
    strings = np.asarray(values, dtype=str)
    if not all(re.fullmatch(r"\+?\d+(?:\.0+)?", v) for v in strings.flat):
        raise ContractError(f"Invalid {what}: expected nonnegative integers")
    try:
        return np.asarray([v.split(".", 1)[0] for v in strings.flat], dtype=np.int64).reshape(strings.shape)
    except (ValueError, OverflowError) as exc:
        raise ContractError(f"Invalid {what}: outside int64 range") from exc


def _identifiers(values, what, unique=True):
    result = pd.Index([_text(v) for v in values])
    if any(not x.strip() for x in result) or (unique and not result.is_unique):
        raise ContractError(f"Empty or duplicated {what}")
    return result


def gem_header(path):
    """Read bounded comment metadata and the actual tab-separated header."""
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open
    metadata = {}
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        for _ in range(128):
            line = handle.readline(65537)
            if len(line) > 65536:
                raise ContractError("GEM header line exceeds 64 KiB")
            if line.startswith("#"):
                key, sep, value = line[1:].strip().partition("=")
                if sep:
                    if key in metadata and metadata[key] != value:
                        raise ContractError(f"Conflicting GEM header: {key}")
                    metadata[key] = value
            elif line.strip():
                columns = next(csv.reader([line], delimiter="\t"))
                if len(set(columns)) != len(columns):
                    raise ContractError("Duplicate GEM column names")
                if not {"geneID", "x", "y"}.issubset(columns):
                    raise ContractError("GEM requires geneID, x, y")
                present = [x for x in COUNTS if x in columns]
                if len(present) != 1:
                    raise ContractError("GEM requires exactly one unambiguous total-count column")
                fmt = metadata.get("FileFormat", "legacy_unspecified")
                if fmt not in ("GEMv0.1", "GEMv0.2", "legacy_unspecified"):
                    raise ContractError(f"Unsupported GEM format version: {fmt}")
                if metadata.get("Omics", "Transcriptomics").lower() not in ("transcriptomics", "rna"):
                    raise ContractError("This Stereo-seq adapter requires transcriptomic data")
                cell = metadata.get("BinType", "").lower() in ("cell", "cellbin") or "CellID" in columns
                if cell and "CellID" not in columns:
                    raise ContractError("CellBin GEM requires CellID")
                native = 1 if cell else int(_integers([metadata.get("BinSize", "1")], "BinSize")[0])
                if native < 1:
                    raise ContractError("GEM BinSize must be positive")
                for k in ("OffsetX", "OffsetY", "Resolution"):
                    if k in metadata:
                        _integers([metadata[k]], k)
                return dict(columns=columns, header=metadata, count=present[0], cell=cell, native_bin=native)
    raise ContractError("Missing GEM header within 128 lines")


def _bin_size(requested, native, cell):
    if requested is not None and (isinstance(requested, bool) or not isinstance(requested, int) or requested < 1):
        raise ValueError("Stereo-seq bin size must be a positive integer")
    if cell and requested is not None:
        raise ContractError("CellBin data cannot be silently converted to square bins")
    target = requested or native
    if not cell and (target < native or target % native):
        raise ContractError("Requested bin size must be a multiple of the stored bin size; no upsampling")
    return target


def _metadata(fmt, header, bin_size, cell, chemistry):
    if chemistry not in (None, "V1", "V2"):
        raise ValueError("Stereo-seq chemistry must be V1, V2 or None")
    return dict(
        platform="Stereo-seq",
        chemistry=chemistry or "unspecified",
        chemistry_evidence="user_declared" if chemistry else "not_encoded_by_matrix_schema",
        file_format=fmt,
        source_metadata=header,
        bin_size=bin_size,
        observation_type="cell" if cell else "square_bin",
        coordinate_system="stored x/y coordinates in native chip units; no implicit image registration",
        total_rna_policy="retain every input feature; no poly(A), host or protein-coding filter",
    )


def probe_stereo(path, budget, bin_size=None):
    path = Path(path)
    if path.suffix.lower() == ".gef":
        with h5py.File(path, "r") as f:
            g, cell, size = _gef_group(f, bin_size)
            _gef_schema(g, cell)
            exp = g["cellExp" if cell else "expression"]
            n = len(exp)
            estimated = n * 192 + len(g["gene"]) * 1024
            if cell:
                estimated += len(g["cell"]) * 1024
            return dict(estimated_bytes=estimated, storage="cellbin_gef" if cell else "bin_gef", bin_size=size)
    header = gem_header(path)
    size = _bin_size(bin_size, header["native_bin"], header["cell"])
    # Compressed byte size is not a prediction of decoded output size. The
    # streaming reader checks each chunk and growing sparse allocations instead.
    return dict(estimated_bytes=1024**2, storage="streamed_gem", bin_size=size)


def read_gem(path, budget, bin_size=None, chemistry=None):
    """Stream all native records; aggregate counts without dense gene matrices."""
    path = Path(path)
    spec = gem_header(path)
    size = _bin_size(bin_size, spec["native_bin"], spec["cell"])
    layer_columns = {k: v for k, v in LAYERS.items() if k in spec["columns"]}
    columns = {spec["count"]: "X", **layer_columns}
    genes, observations, symbols, seen_dnb, centers = {}, {}, {}, {}, {}
    matrices = {v: sparse.csr_matrix((0, 0), dtype=np.int64) for v in columns.values()}
    totals = {v: 0 for v in columns.values()}
    rows_read = 0
    chunk_rows = min(100000, max(1, budget // 4096))
    for frame in pd.read_csv(path, sep="\t", comment="#", dtype=str, keep_default_na=False, chunksize=chunk_rows):
        used = int(frame.memory_usage(deep=True).sum()) * 3
        used += sum(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes for m in matrices.values()) * 4
        used += (len(genes) + len(observations) + len(seen_dnb)) * 384
        if used > budget:
            raise ResourceDeferred("Streaming GEM exceeds allocation budget; select coarser bins or increase budget")
        rows_read += len(frame)
        ids = _identifiers(frame["geneID"], "gene IDs", unique=False)
        xy = _integers(frame[["x", "y"]].to_numpy(), "GEM coordinates")
        if not spec["cell"] and np.any(xy % spec["native_bin"]):
            raise ContractError("Stored GEM coordinates disagree with BinSize grid")
        for gene in pd.unique(ids):
            genes.setdefault(gene, len(genes))
        if "geneName" in frame:
            for gene, symbol in frame[["geneID", "geneName"]].drop_duplicates().itertuples(index=False, name=None):
                if gene in symbols and symbols[gene] != symbol:
                    raise ContractError(f"Conflicting geneName for stable ID {gene}")
                symbols[gene] = symbol
        if spec["cell"]:
            labels = _integers(frame["CellID"], "CellID")
            keys = [str(x) for x in labels]
            for (x, y), key in zip(xy, keys):
                point = (int(x), int(y))
                if point in seen_dnb:
                    if seen_dnb[point] != key:
                        raise ContractError("One DNB is assigned to multiple CellIDs")
                else:
                    seen_dnb[point] = key
                    a = centers.setdefault(key, [0, 0, 0])
                    a[0] += int(x)
                    a[1] += int(y)
                    a[2] += 1
        else:
            xy = (xy // size) * size
            keys = list(zip(xy[:, 0].tolist(), xy[:, 1].tolist()))
        for key in dict.fromkeys(keys):
            observations.setdefault(key, len(observations))
        ri = np.fromiter((observations[k] for k in keys), dtype=np.int64, count=len(keys))
        ci = np.fromiter((genes[k] for k in ids), dtype=np.int64, count=len(ids))
        values = {out: _integers(frame[col], col) for col, out in columns.items()}
        for key in layer_columns.values():
            if np.any(values[key] > values["X"]):
                raise ContractError(f"{key} counts exceed total MID counts")
        shape = (len(observations), len(genes))
        for name, val in values.items():
            # Python sum prevents int64 wraparound even with repeated records.
            totals[name] += sum(map(int, val))
            if totals[name] > np.iinfo(np.int64).max:
                raise ContractError("Aggregated GEM counts exceed int64 range")
            matrices[name].resize(shape)
            matrices[name] = matrices[name] + sparse.coo_matrix((val, (ri, ci)), shape=shape).tocsr()
    if not rows_read:
        raise ContractError("Empty GEM")
    order = sorted(observations, key=lambda k: int(k) if spec["cell"] else k)
    ix = np.array([observations[k] for k in order])
    names = [str(k) if spec["cell"] else f"{k[0]}_{k[1]}" for k in order]
    coords = (
        np.array([[centers[k][0] / centers[k][2], centers[k][1] / centers[k][2]] for k in order])
        if spec["cell"]
        else np.asarray(order, dtype=np.int64)
    )
    var = pd.DataFrame({"gene_ids": list(genes)}, index=pd.Index(genes))
    if symbols:
        var["gene_name"] = [symbols[g] for g in genes]
    a = AnnData(
        matrices.pop("X")[ix], obs=pd.DataFrame(index=names), var=var, layers={k: m[ix] for k, m in matrices.items()}
    )
    a.obsm["spatial"] = coords
    meta = _metadata(spec["header"].get("FileFormat", "legacy_gem"), spec["header"], size, spec["cell"], chemistry)
    meta.update(
        input_records=rows_read,
        total_molecules=totals["X"],
        coordinate_reference=(
            "mean of distinct captured DNB positions per CellID"
            if spec["cell"]
            else "bin origin in stored coordinate axes"
        ),
    )
    if {"OffsetX", "OffsetY"}.issubset(spec["header"]):
        offsets = _integers([spec["header"]["OffsetX"], spec["header"]["OffsetY"]], "offsets")
        if np.any(coords.max(axis=0) > np.iinfo(np.int64).max - offsets):
            raise ContractError("Global coordinate overflow")
        a.obsm["spatial_global"] = coords + offsets
    a.uns["stereoseq"] = meta
    return a


def _gef_group(f, requested):
    if ("cellBin" in f) == ("geneExp" in f):
        raise ContractError("GEF requires exactly one of geneExp or cellBin")
    if "cellBin" in f:
        _bin_size(requested, 1, True)
        return f["cellBin"], True, 1
    bins = sorted(int(k[3:]) for k in f["geneExp"] if re.fullmatch(r"bin[1-9]\d*", k))
    if not bins:
        raise ContractError("No supported geneExp/binN group")
    size = requested if requested is not None else bins[0]
    _bin_size(size, 1, False)
    if size not in bins:
        raise ContractError(f"Requested bin{size} is absent; available bins: {bins}. No implicit GEF rebinning")
    return f[f"geneExp/bin{size}"], False, size


def _dataset(g, key, fields=()):
    if key not in g or not isinstance(g[key], h5py.Dataset) or g[key].ndim != 1:
        raise ContractError(f"Missing or invalid GEF dataset {g.name}/{key}")
    if not set(fields).issubset(g[key].dtype.names or ()):
        raise ContractError(f"Invalid GEF compound fields in {g.name}/{key}: require {fields}")
    return g[key]


def _gef_schema(g, cell):
    gene = _dataset(g, "gene")
    fields = gene.dtype.names or ()
    if not any(k in fields for k in ("geneID", "gene", "geneName")):
        raise ContractError("GEF gene identifiers missing")
    if cell:
        _dataset(g, "cell", ("id", "x", "y", "offset", "geneCount"))
        _dataset(g, "cellExp", ("geneID", "count"))
    else:
        _dataset(g, "gene", ("offset", "count"))
        _dataset(g, "expression", ("x", "y", "count"))


def _index_blocks(records, offset_key, length_key, n):
    offsets = _integers(records[offset_key], "GEF offsets")
    lengths = _integers(records[length_key], "GEF block lengths")
    if sum(map(int, lengths)) != n:
        raise ContractError("GEF block lengths do not cover the expression dataset")
    expected = np.r_[0, np.cumsum(lengths[:-1])]
    if not np.array_equal(offsets, expected):
        raise ContractError("GEF blocks overlap, have gaps, or are out of order")
    return np.repeat(np.arange(len(records)), lengths)


def read_gef(path, budget, bin_size=None, chemistry=None):
    """Read native square-bin or cell-bin compound datasets with h5py."""
    checks = probe_stereo(path, budget, bin_size)
    if checks["estimated_bytes"] > budget:
        raise ResourceDeferred("GEF sparse allocation exceeds memory budget")
    with h5py.File(path, "r") as f:
        g, cell, size = _gef_group(f, bin_size)
        _gef_schema(g, cell)
        gene = g["gene"][:]
        names = gene.dtype.names
        key = next(k for k in ("geneID", "gene", "geneName") if k in names)
        ids = _identifiers(gene[key], "GEF gene IDs")
        var = pd.DataFrame({"gene_ids": ids}, index=ids)
        if "geneName" in names and key != "geneName":
            var["gene_name"] = [_text(v) for v in gene["geneName"]]
        exp = g["cellExp" if cell else "expression"][:]
        counts = _integers(exp["count"], "GEF counts")
        if sum(map(int, counts)) > np.iinfo(np.int64).max:
            raise ContractError("GEF count total exceeds int64 range")
        if cell:
            cells = g["cell"][:]
            rows = _index_blocks(cells, "offset", "geneCount", len(exp))
            cols = _integers(exp["geneID"], "GEF gene indices")
            if (cols >= len(ids)).any():
                raise ContractError("GEF gene index out of range")
            obs_ids = _identifiers(_integers(cells["id"], "CellIDs"), "GEF cell IDs")
            obs = pd.DataFrame(index=obs_ids)
            for k in cells.dtype.names:
                if k not in ("offset", "x", "y"):
                    obs[k] = _integers(cells[k], f"GEF cell {k}")
            coords = _integers(np.column_stack([cells["x"], cells["y"]]), "GEF cell coordinates")
            exon_key = "cellExpExon"
        else:
            cols = _index_blocks(gene, "offset", "count", len(exp))
            raw_xy = _integers(np.column_stack([exp["x"], exp["y"]]), "GEF coordinates")
            coords, rows = np.unique(raw_xy, axis=0, return_inverse=True)
            obs = pd.DataFrame(index=[f"{x}_{y}" for x, y in coords])
            exon_key = "exon"
        shape = (len(obs), len(ids))
        X = sparse.coo_matrix((counts, (rows, cols)), shape=shape).tocsr()
        if not X.shape[0] or not X.shape[1]:
            raise ContractError("Empty GEF matrix")
        a = AnnData(X, obs=obs, var=var)
        if exon_key in g:
            dataset = _dataset(g, exon_key)
            if len(dataset) != len(exp):
                raise ContractError("GEF exon and expression lengths differ")
            exon = dataset[:]
            if exon.dtype.names:
                if "count" not in exon.dtype.names:
                    raise ContractError("GEF exon compound requires count")
                exon = exon["count"]
            exon = _integers(exon, "GEF exon counts")
            if (exon > counts).any():
                raise ContractError("GEF exon counts exceed total counts")
            a.layers["exon"] = sparse.coo_matrix((exon, (rows, cols)), shape=shape).tocsr()
        if cell and "expCount" in obs and not np.array_equal(np.asarray(X.sum(axis=1)).ravel(), obs["expCount"]):
            raise ContractError("GEF cell expCount disagrees with expression total")
        a.obsm["spatial"] = coords
        header = {}
        for prefix, attrs in (("root", f.attrs), ("expression", g["cell" if cell else "expression"].attrs)):
            for key, value in attrs.items():
                if np.asarray(value).size <= 32:
                    header[f"{prefix}:{key}"] = ",".join(_text(x) for x in np.atleast_1d(value).flat)
        omics = header.get("root:omics", "Transcriptomics").lower()
        if omics not in ("transcriptomics", "rna"):
            raise ContractError("GEF contains non-transcriptomic omics")
        meta = _metadata("GEF", header, size, cell, chemistry)
        meta.update(
            gef_schema_version=header.get("root:version", "unspecified"),
            source_group=g.name,
            total_molecules=sum(map(int, counts)),
        )
        resolution = f.attrs.get("resolution") if cell else g["expression"].attrs.get("resolution")
        if resolution is not None:
            pitch = float(np.asarray(resolution).item())
            if not np.isfinite(pitch) or pitch <= 0:
                raise ContractError("Invalid GEF physical resolution")
            meta["pitch_nm"] = pitch
        a.uns["stereoseq"] = meta
        return a


def read_stereo_core(path, budget, bin_size=None, chemistry=None):
    reader = read_gef if Path(path).suffix.lower() == ".gef" else read_gem
    return reader(path, budget, bin_size, chemistry)
