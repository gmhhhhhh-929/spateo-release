"""Storage-independent helpers for platform-native spatial readers.

This layer deliberately has no dependency on the automatic detector. Platform
modules own their filename/field contracts and pass operations to these helpers.
"""

import numpy as np
import pandas as pd

from ...configuration import SKM
from ._errors import ContractError, ResourceDeferred
from ._layout import Candidate
from ._matrix import _h5, _ids, _mex, _numeric, integer_counts, table

_MEX_COMPONENTS = frozenset(
    {
        "matrix.mtx",
        "matrix.mtx.gz",
        "features.tsv",
        "features.tsv.gz",
        "genes.tsv",
        "genes.tsv.gz",
        "barcodes.tsv",
        "barcodes.tsv.gz",
    }
)


def discover_mex(files, requested, technology, metadata_names, representation):
    """Describe MEX inputs selected by a platform's own coordinate filenames."""
    candidates = []
    for metadata in sorted(p for p in files if p.name in metadata_names):
        matrix = metadata.parent
        if requested.is_file() and requested != metadata:
            if requested.parent != matrix or requested.name not in _MEX_COMPONENTS:
                continue
        candidates.append(Candidate(technology, matrix, matrix, metadata, representation, {}, [metadata.name]))
    return candidates


def headerless_coordinates(path, technology, *, full=False, budget=512 * 1024**2):
    """Read a bounded three-column table; interpretation belongs to its caller."""
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
                    raise ContractError(f"{technology} native coordinate table requires exactly 3 headerless fields")
                chunks.append(chunk)
        frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    else:
        frame = pd.read_csv(path, nrows=min(128, max(1, budget // 4096)), **args)
    if frame.empty or frame.shape[1] != 3:
        raise ContractError(f"{technology} native coordinate table requires exactly 3 headerless fields")
    frame.columns = ["barcode", "x", "y"]
    return frame


def _check_paths(candidate):
    paths = [candidate.counts, candidate.metadata]
    if candidate.counts.is_dir():
        paths.extend(candidate.counts / name for name in _MEX_COMPONENTS)
    for path in paths:
        for part in [path, *path.parents]:
            if part == candidate.root:
                break
            if part.is_symlink():
                raise ContractError(f"Symlink input is not followed: {part}")


def probe_native(candidate, budget, metadata_reader):
    """Probe the platform's coordinates and supported sparse storage structure."""
    _check_paths(candidate)
    metadata_reader(candidate, budget=budget)
    read_matrix = _h5 if candidate.counts.suffix == ".h5" else _mex
    result = read_matrix(candidate.counts, technology=candidate.technology)
    result["estimated_bytes"] += candidate.metadata.stat().st_size * 12
    result.update(structure="passed", content="not_loaded")
    return result


def read_native(candidate, budget, metadata_reader, observation_type):
    """Parse native storage and align the platform's full coordinates by ID."""
    _check_paths(candidate)
    meta, xy, units = metadata_reader(candidate, full=True, budget=budget)
    read_matrix = _h5 if candidate.counts.suffix == ".h5" else _mex
    adata = read_matrix(candidate.counts, full=True, budget=budget, technology=candidate.technology)
    missing = adata.obs_names[~adata.obs_names.isin(meta.index)]
    if len(missing):
        raise ContractError(f"{len(missing)} matrix IDs lack metadata/coordinates; examples: {list(missing[:5])}")
    order = meta.index.get_indexer(adata.obs_names)
    adata.obs = meta.iloc[order].copy()
    adata.obsm["spatial"] = xy[order]
    adata.X.data = integer_counts(adata.X.data, candidate.technology)
    if "gene_name" not in adata.var:
        adata.var["gene_name"] = adata.var_names.to_numpy()
    adata.var_names = _ids(adata.var["gene_ids"], "native feature IDs")
    if not adata.obs_names.is_unique or adata.n_obs == 0 or adata.n_vars == 0:
        raise ContractError("Empty or duplicate observation axis")
    if adata.obsm["spatial"].shape != (adata.n_obs, 2) or not np.isfinite(adata.obsm["spatial"]).all():
        raise ContractError("Incomplete spatial coordinate mapping")
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    source_metadata = adata.uns.pop("_source_h5_metadata", {})
    libraries = source_metadata.get("library_ids", [])
    library = libraries[0] if len(libraries) == 1 else candidate.root.name
    source_metadata.update(coordinate_system=units, representation=candidate.representation)
    adata.uns["spatial"] = {library: {"images": {}, "scalefactors": {}, "metadata": source_metadata}}
    adata.uns["native_spatial_export"] = {
        "technology": candidate.technology,
        "identity_evidence": "documented export layout; not independent experimental-provenance verification",
        "observation_type": observation_type,
        "counts_policy": "retain all supplied matrix entries and features; no normalization or tissue filtering",
    }
    return adata


def read_direct(path, module, load_images, max_memory_bytes, return_result):
    """Execute a known platform's own operations without automatic detection."""
    from ._read_engine import run_reading

    result = run_reading(
        path,
        discover=lambda files, requested, diagnostics=None: module.discover(files, requested),
        probe=module.probe,
        read_core=module.read_core,
        reader_name=lambda candidate: module.__name__ + ".read_core",
        technology=module.TECHNOLOGY,
        allowed={module.TECHNOLOGY},
        explicit_platform=True,
        load_images=load_images,
        max_memory_bytes=max_memory_bytes,
    )
    return result if return_result else result.adata
