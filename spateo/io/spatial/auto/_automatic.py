"""Automatic spatial reading driven by format contracts, without confidence scores."""

from __future__ import annotations

import hashlib
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from ...._registry import register_function
from .._provenance import record_spatial_io
from ._formats import _canonical_technologies
from ._contracts import ContractError, ResourceDeferred, probe, read_core
from ._discovery import discover, inventory
from ._result import POLICY_VERSION, SpatialDataset, SpatialReadResult

_DEFAULT_MEMORY = 1024**3
_IMAGE_BUDGET = 32 * 1024**2


def _diagnostic(code, message, severity="error", **extra):
    return dict(code=code, message=str(message), severity=severity, **extra)


def _assets(adata, candidate, enabled, budget, diagnostics):
    """Optional raster failures never alter the core result or platform identity."""
    slot = next(iter(adata.uns["spatial"].values()))
    root = candidate.root
    scales = root / "spatial/scalefactors_json.json"
    if scales.is_file() and not scales.is_symlink():
        try:
            if scales.stat().st_size > 1024**2:
                raise ValueError("Scale metadata exceeds size limit")
            value = json.loads(scales.read_text())
            if not isinstance(value, dict):
                raise ValueError("Scale metadata is not an object")
            for key, v in value.items():
                if not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(v) or v <= 0:
                    raise ValueError(f"Invalid scale factor {key}")
            slot["scalefactors"] = value
        except (OSError, ValueError) as exc:
            diagnostics.append(_diagnostic("optional_scale_error", exc, "warning", path=str(scales)))
    files = []
    # Bounded optional inventory; a directory is never recursively expanded here.
    for folder in (root, root / "spatial", root / "images", root / "morphology_focus"):
        if not folder.is_dir() or folder.is_symlink():
            continue
        for i, p in enumerate(folder.iterdir()):
            if i >= 1000:
                diagnostics.append(
                    _diagnostic("asset_inventory_limit", f"Optional inventory truncated at {folder}", "warning")
                )
                break
            if (
                not p.is_symlink()
                and p.is_file()
                and p.name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
            ):
                files.append(p)
    slot["image_files"] = {str(i): str(p.relative_to(root)) for i, p in enumerate(sorted(set(files)))}
    slot["asset_status"] = {}
    used = 0
    image_priority = {"tissue_hires_image.png": 0, "tissue_lowres_image.png": 1}
    for p in sorted(set(files), key=lambda p: (image_priority.get(p.name, 2), str(p))):
        relative = p.relative_to(root).as_posix()
        status = "not_requested"
        if enabled:
            try:
                with Image.open(p) as im:
                    estimated = (
                        im.width
                        * im.height
                        * max(len(im.getbands()), 1)
                        * (4 if im.mode in ("I", "F") else 2 if "16" in im.mode else 1)
                    )
                    if getattr(im, "n_frames", 1) > 1:
                        status = "deferred_multiframe"
                    elif estimated > min(_IMAGE_BUDGET, budget) - used:
                        status = "deferred_resource"
                    else:
                        key = {"tissue_hires_image.png": "hires", "tissue_lowres_image.png": "lowres"}.get(
                            p.name, relative.replace("/", "__")
                        )
                        arr = np.asarray(im).copy()
                        slot["images"][key] = arr
                        used += arr.nbytes
                        status = "loaded"
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                status = "unreadable"
                diagnostics.append(_diagnostic("optional_image_error", exc, "warning", path=relative))
        slot["asset_status"][relative.replace("/", "__")] = status
    if not files:
        diagnostics.append(_diagnostic("optional_images_missing", "No optional raster assets found", "warning"))
    # Images alone do not establish coordinate registration.
    slot["metadata"]["image_registration"] = "scale_metadata_present" if slot["scalefactors"] else "not_established"
    return used


def _key(candidate, scope):
    relative = candidate.root.relative_to(scope).as_posix()
    base = f"{relative}::{candidate.representation}::{candidate.counts.name}"
    return base


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


def _memory_used(result):
    total = 0
    for entry in result.datasets.values():
        if entry.adata is not None:
            a = entry.adata
            total += a.X.data.nbytes + a.X.indices.nbytes + a.X.indptr.nbytes
            total += int(a.obs.memory_usage(deep=True).sum() + a.var.memory_usage(deep=True).sum())
            total += sum(np.asarray(x).nbytes for x in a.obsm.values())
            for slot in a.uns.get("spatial", {}).values():
                total += sum(np.asarray(x).nbytes for x in slot.get("images", {}).values())
    return total


def _signature(candidate):
    paths = [candidate.counts, candidate.metadata, candidate.root / "experiment.xenium"]
    if candidate.counts.is_dir():
        paths.extend(
            candidate.counts / name
            for name in (
                "matrix.mtx",
                "matrix.mtx.gz",
                "features.tsv",
                "features.tsv.gz",
                "genes.tsv",
                "genes.tsv.gz",
                "barcodes.tsv",
                "barcodes.tsv.gz",
            )
        )
    return tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(set(paths)) if p.is_file())


def _load(entry, candidate, result, files, budget, load_images, reason, signature):
    remaining = budget - _memory_used(result)
    try:
        if _signature(candidate) != signature:
            raise ContractError("Source changed since discovery; call read_spatial again")
        checks = probe(candidate, remaining)
        entry.validation.update(checks)
        entry.estimated_bytes = checks["estimated_bytes"]
        if remaining <= 0 or entry.estimated_bytes > remaining:
            raise ResourceDeferred("Estimated core allocation exceeds remaining collection memory budget")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adata = read_core(candidate, remaining)
            if _signature(candidate) != signature:
                raise ContractError("Source changed during reading; no object returned")
        entry.diagnostics.extend(_diagnostic("reader_warning", w.message, "warning") for w in caught)
        entry.validation.update(content="passed", identifiers="complete", coordinates="finite", values="finite")
        _assets(adata, candidate, load_images, max(0, remaining - entry.estimated_bytes), entry.diagnostics)
        slot = next(iter(adata.uns["spatial"].values()))
        asset_paths = [candidate.root / name for name in slot.get("image_files", {}).values()]
        paths = sorted(set([p for p in files if p.is_relative_to(candidate.root)] + asset_paths))
        manifest = {
            "root": str(candidate.root),
            "paths": [p.relative_to(candidate.root).as_posix() for p in paths],
            "sizes_bytes": [p.stat().st_size for p in paths],
            "truncated": not result.discovery["complete"],
            "scope": "bounded core discovery plus optional raster inventory",
        }
        record_spatial_io(
            adata,
            technology=candidate.technology,
            source=candidate.root,
            reader="spateo.io.spatial.auto._contracts.read_core",
            evidence=tuple(entry.evidence),
            reader_kwargs={"representation": candidate.representation, "load_images": load_images},
            manifest=manifest,
            format_status="preview-xenium-v4" if candidate.technology == "atera" else "validated_core",
        )
        adata.uns["spateo_io"].update(
            policy_version=POLICY_VERSION,
            resolution_reason=reason,
            validation={k: v for k, v in entry.validation.items() if isinstance(v, (str, int, float, bool))},
            warnings=[d["message"] for d in entry.diagnostics if d["severity"] == "warning"],
        )
        entry.adata, entry.status = adata, "ready"
    except (ResourceDeferred, MemoryError) as exc:
        entry.status = "deferred"
        entry.diagnostics.append(_diagnostic("resource_deferred", exc, "warning"))
    except Exception as exc:
        # Do not swallow KeyboardInterrupt/SystemExit; isolate ordinary failures only.
        entry.adata, entry.status = None, "failed"
        entry.validation["content"] = "failed"
        code = (
            "dependency_missing"
            if isinstance(exc, ImportError)
            else "contract_error" if isinstance(exc, ContractError) else "read_error"
        )
        entry.diagnostics.append(_diagnostic(code, exc, exception_type=type(exc).__name__))


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
    load_images: bool = True,
    max_memory_bytes: int = _DEFAULT_MEMORY,
    max_files: int = 10000,
    max_depth: int = 4,
) -> SpatialReadResult:
    """Automatically read supported spatial layouts without confidence thresholds.

    A path is the only required argument. All discovered samples/representations
    are retained as named entries; no matrix is selected by a platform score.
    Core-format or ID failures stay visible in ``result.report``.
    ``read_auto_spatial`` and ``read_spatial_auto`` are aliases of this function
    and also return ``SpatialReadResult``. Direct platform readers are unchanged.

    Parameters beyond ``path`` are optional: ``technology`` restricts discovery;
    ``load=False`` discovers/probes but defers content loading. Memory, inventory
    and depth limits are resource bounds, never statistical matching thresholds.
    ``max_memory_bytes`` is a conservative allocation budget, not an OS RSS cap.
    Entries exceeding it remain explicitly deferred and can be loaded later.
    See ``docs/technicals/automatic_spatial_reading.md`` for supported contracts.
    """
    if not isinstance(max_memory_bytes, int) or max_memory_bytes <= 0 or max_files <= 0 or max_depth < 0:
        raise ValueError("Invalid memory/inventory/depth resource limits")
    allowed = _canonical_technologies(technology)
    requested = Path(path).expanduser().resolve()
    result = SpatialReadResult(str(requested))
    if not requested.exists():
        result.diagnostics.append(_diagnostic("path_missing", f"Input path does not exist: {requested}"))
        return result
    files, roots, diagnostics = inventory(requested, max_files, max_depth)
    result.diagnostics.extend(diagnostics)
    result.discovery = {
        "files_inspected": len(files),
        "directories_inspected": len(roots),
        "max_depth": max_depth,
        "complete": not any(d["severity"] == "error" for d in diagnostics),
        "symlinks_followed": False,
    }
    all_candidates = discover(files, requested)
    candidates = [c for c in all_candidates if allowed is None or c.technology in allowed]
    result.discovery["technology_filter"] = technology
    result.discovery["excluded_by_technology"] = len(all_candidates) - len(candidates)
    if not candidates:
        result.diagnostics.append(
            _diagnostic("unsupported_layout", "No supported spatial core layout found; no generic reader was guessed.")
        )
        return result
    known_roots = {c.root for c in all_candidates}
    for directory in sorted({p.parent for p in files}):
        if any(directory.is_relative_to(root) for root in known_roots):
            continue
        if any(
            p.parent == directory
            and p.name.lower().endswith((".csv", ".csv.gz", ".parquet", ".h5", ".h5ad", ".gem", ".tsv"))
            for p in files
        ):
            result.diagnostics.append(
                _diagnostic(
                    "unrecognized_input_directory",
                    "Data-like files outside recognized inputs; no reader guessed",
                    path=str(directory),
                )
            )
    groups = defaultdict(list)
    for c in candidates:
        identity = c.identity
        if c.technology in ("visium", "visium_hd_bin"):
            # Alternate matrix storage encodings are not assumed equivalent.
            identity = str(c.root), c.representation
        groups[identity].append(c)
    scope = requested.parent if requested.is_file() else requested
    for identity, alternatives in sorted(groups.items()):
        first = alternatives[0]
        key = _key(first, scope)
        if key in result.datasets:
            key += "::" + hashlib.sha256(repr(identity).encode()).hexdigest()[:8]
        entry = SpatialDataset(key, first.technology, str(first.root), first.representation)
        result.datasets[key] = entry
        valid, deferred, failures, probes = [], [], [], {}
        for c in alternatives:
            try:
                checks = probe(c, max_memory_bytes)
                valid.append(c)
                probes[id(c)] = checks
            except (ResourceDeferred, MemoryError) as exc:
                deferred.append(c)
                failures.append(_diagnostic("probe_deferred", exc, "warning", technology=c.technology))
            except Exception as exc:
                failures.append(
                    _diagnostic("probe_failed", exc, technology=c.technology, exception_type=type(exc).__name__)
                )
        candidate, reason = (
            _resolve(valid) if not deferred else (None, "Unprobed alternatives prevent a unique resolution")
        )
        entry.validation["candidate_diagnostics"] = failures
        if candidate is None:
            if len(alternatives) == 1 and deferred:
                candidate = deferred[0]
                reason = "Unique discovered layout; resource-bounded validation is still required"
            else:
                entry.status = "unresolved" if valid or deferred else "failed"
                entry.evidence = [f"{c.technology}: {c.counts.name} + {c.metadata.name}" for c in alternatives]
                entry.diagnostics = failures + [
                    _diagnostic("unresolved_layout" if valid or deferred else "no_valid_contract", reason)
                ]
                continue
        entry.technology = candidate.technology
        if id(candidate) in probes:
            entry.validation.update(probes[id(candidate)])
            entry.estimated_bytes = probes[id(candidate)]["estimated_bytes"]
        entry.evidence = candidate.evidence + (
            [candidate.options["identity"]] if "identity" in candidate.options else []
        )
        entry.status = "deferred"
        try:
            signature = _signature(candidate)
        except OSError as exc:
            entry.status = "failed"
            entry.diagnostics.append(_diagnostic("source_unavailable", exc))
            continue
        entry._loader = lambda e, limit, c=candidate, why=reason, sig=signature: _load(
            e, c, result, files, limit or max_memory_bytes, load_images, why, sig
        )
        if load:
            entry.load()
        else:
            entry.validation.update(structure="passed" if candidate in valid else "deferred", content="not_loaded")
            entry.diagnostics.append(_diagnostic("load_deferred", "Content loading was not requested", "info"))
    return result
