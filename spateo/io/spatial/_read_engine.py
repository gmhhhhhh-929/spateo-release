"""Shared execution, materialization and reporting; no automatic platform imports.

Detection and platform parsing are injected by the calling reader. Both explicit
and automatic routes use these same source guards, budgets and outcome rules.
"""

from __future__ import annotations

import hashlib
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from ._assets import load_assets
from ._errors import ContractError, ResourceDeferred
from ._layout import inventory
from ._provenance import record_spatial_io
from ._read_result import POLICY_VERSION, SpatialDataset, SpatialReadResult
from ._recovery import required_paths


def _diagnostic(code, message, severity="error", **extra):
    return dict(code=code, message=str(message), severity=severity, **extra)


def _key(candidate, scope):
    relative = candidate.root.relative_to(scope).as_posix()
    base = f"{relative}::{candidate.representation}::{candidate.counts.name}"
    return base


def _resolve(candidates):
    """Require one validated interpretation for a known platform."""
    if not candidates:
        return None, "No candidate passed the required format contract"
    if len(candidates) == 1:
        return candidates[0], "Unique validated core layout; no score comparison"
    return None, "Multiple validated readers or companion encodings claim the same logical input"


def _memory_used(result):
    total = 0
    for entry in result.datasets.values():
        if entry.adata is not None:
            a = entry.adata
            total += a.X.data.nbytes + a.X.indices.nbytes + a.X.indptr.nbytes
            total += sum(x.data.nbytes + x.indices.nbytes + x.indptr.nbytes for x in a.layers.values())
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
                "barcode.tsv",
                "barcode.tsv.gz",
                "matrix.tsv",
                "matrix.tsv.gz",
                "feature.tsv",
                "feature.tsv.gz",
            )
        )
    return tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(set(paths)) if p.is_file())


def _load(
    entry,
    candidate,
    result,
    files,
    budget,
    load_images,
    reason,
    signature,
    *,
    probe,
    read_core,
    reader_name,
    asset_loader,
):
    remaining = budget - _memory_used(result)
    try:
        if _signature(candidate) != signature:
            raise ContractError("Source changed since discovery; rerun the original reader")
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
        asset_loader(adata, candidate, load_images, max(0, remaining - entry.estimated_bytes), entry.diagnostics)
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
            reader=reader_name(candidate),
            evidence=tuple(entry.evidence),
            reader_kwargs={
                "representation": candidate.representation,
                "load_images": load_images,
                **{k: v for k, v in candidate.options.items() if k.startswith("stereoseq_") and v is not None},
            },
            manifest=manifest,
            format_status="preview-xenium-v4" if candidate.technology == "atera" else "validated_core",
        )
        adata.uns["spateo_io"].update(
            policy_version=POLICY_VERSION,
            resolution_reason=reason,
            validation={k: v for k, v in entry.validation.items() if isinstance(v, (str, int, float, bool))},
            warnings=[d["message"] for d in entry.diagnostics if d["severity"] == "warning"],
        )
        if result.discovery.get("unclassified_directories"):
            adata.uns["spateo_io"]["unclassified_directories"] = list(result.discovery["unclassified_directories"])
        entry.adata, entry.status = adata, "ready"
    except (ResourceDeferred, MemoryError) as exc:
        entry.status = "deferred"
        entry.diagnostics.append(_diagnostic("resource_deferred", exc, "warning"))
    except Exception as exc:
        # Do not swallow KeyboardInterrupt/SystemExit; isolate ordinary failures only.
        entry.adata, entry.status = None, "failed"
        entry.validation["content"] = "failed"
        code = (
            "source_changed"
            if "Source changed" in str(exc)
            else (
                "dependency_missing"
                if isinstance(exc, ImportError)
                else (
                    "permission_denied"
                    if isinstance(exc, PermissionError)
                    else "contract_error" if isinstance(exc, ContractError) else "read_error"
                )
            )
        )
        entry.diagnostics.append(_diagnostic(code, exc, exception_type=type(exc).__name__))


def _group_by_identity(candidates):
    groups = defaultdict(list)
    for candidate in candidates:
        groups[candidate.identity].append(candidate)
    return groups


def run_reading(
    path,
    *,
    discover,
    probe,
    read_core,
    reader_name,
    technology=None,
    allowed=None,
    load=True,
    lazy=False,
    load_images=True,
    max_memory_bytes=1024**3,
    max_files=10000,
    max_depth=4,
    prepare_candidate=None,
    group_candidates=None,
    resolve=_resolve,
    asset_loader=load_assets,
    explicit_platform=False,
):
    """Execute injected reader operations and retain every named outcome."""
    if not isinstance(load, bool) or not isinstance(lazy, bool):
        raise ValueError("load and lazy must be booleans")
    if lazy and not load:
        raise ValueError("Use either lazy=True or load=False, not both")
    if not isinstance(max_memory_bytes, int) or max_memory_bytes <= 0 or max_files <= 0 or max_depth < 0:
        raise ValueError("Invalid memory/inventory/depth resource limits")
    try:
        requested = Path(path).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        result = SpatialReadResult(str(path))
        result.diagnostics.append(
            _diagnostic("source_unavailable", exc, path=str(path), exception_type=type(exc).__name__)
        )
        return result
    result = SpatialReadResult(str(requested))
    try:
        exists = requested.exists()
    except OSError as exc:
        result.diagnostics.append(
            _diagnostic("source_unavailable", exc, path=str(requested), exception_type=type(exc).__name__)
        )
        return result
    if not exists:
        result.diagnostics.append(_diagnostic("path_missing", f"Input path does not exist: {requested}"))
        return result
    files, roots, diagnostics = inventory(requested, max_files, max_depth)
    result.diagnostics.extend(diagnostics)
    result.discovery = {
        "load_mode": "lazy" if lazy else "eager" if load else "inspect",
        "files_inspected": len(files),
        "directories_inspected": len(roots),
        "max_depth": max_depth,
        "complete": not any(d["severity"] == "error" for d in diagnostics),
        "symlinks_followed": False,
        "scope": "explicit_platform" if explicit_platform else "all_discovered_inputs",
        "unclassified_directories": [],
    }
    try:
        all_candidates = discover(files, requested, diagnostics=result.diagnostics)
    except (OSError, EOFError, UnicodeError, ValueError) as exc:
        result.discovery["complete"] = False
        result.diagnostics.append(
            _diagnostic("discovery_error", exc, path=str(requested), exception_type=type(exc).__name__)
        )
        return result
    result.discovery["complete"] = not any(d["severity"] == "error" for d in result.diagnostics)
    if prepare_candidate is not None:
        for c in all_candidates:
            prepare_candidate(c)
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
            and p.name.lower().endswith(
                (".csv", ".csv.gz", ".parquet", ".h5", ".h5ad", ".gem", ".tsv")
                + ((".tsv.gz", ".txt", ".txt.gz", ".mtx", ".mtx.gz", ".gem.gz", ".gef") if explicit_platform else ())
            )
            for p in files
        ):
            if explicit_platform:
                result.discovery["unclassified_directories"].append(str(directory))
            result.diagnostics.append(
                _diagnostic(
                    "unclassified_directory" if explicit_platform else "unrecognized_input_directory",
                    (
                        "Outside the requested platform's matched layouts; not classified or read"
                        if explicit_platform
                        else "Data-like files outside recognized inputs; no reader guessed"
                    ),
                    "info" if explicit_platform else "error",
                    path=str(directory),
                )
            )
    groups = group_candidates(candidates) if group_candidates is not None else _group_by_identity(candidates)
    scope = requested.parent if requested.is_file() else requested
    for identity, alternatives in sorted(groups.items()):
        first = alternatives[0]
        key = _key(first, scope)
        if key in result.datasets:
            key += "::" + hashlib.sha256(repr(identity).encode()).hexdigest()[:8]
        entry = SpatialDataset(key, first.technology, str(first.root), first.representation, lazy=lazy)
        entry._memory_budget = max_memory_bytes
        entry.required_files = sorted({p for c in alternatives for p in required_paths(c)})
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
                    _diagnostic(
                        "probe_failed",
                        exc,
                        technology=c.technology,
                        exception_type=type(exc).__name__,
                        required_files=required_paths(c),
                        missing_files=[p for p in required_paths(c) if not Path(p).is_file()],
                    )
                )
        candidate, reason = (
            resolve(valid) if not deferred else (None, "Unprobed alternatives prevent a unique resolution")
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
        entry.representation = candidate.representation
        entry.source = str(candidate.root)
        entry.required_files = required_paths(candidate)
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
            e,
            c,
            result,
            files,
            limit or max_memory_bytes,
            load_images,
            why,
            sig,
            probe=probe,
            read_core=read_core,
            reader_name=reader_name,
            asset_loader=asset_loader,
        )
        if load and not lazy:
            entry.load()
        else:
            entry.validation.update(structure="passed" if candidate in valid else "deferred", content="not_loaded")
            entry.diagnostics.append(
                _diagnostic(
                    "lazy_deferred" if lazy else "load_deferred",
                    (
                        "Core loading will occur on explicit materialization or unique result.adata access"
                        if lazy
                        else "Content loading was not requested"
                    ),
                    "info",
                )
            )
    return result
