"""Structured recovery advice; never downloads data or changes scientific values.

Links are documentation or data portals, not inferred sample download URLs.
Required paths come from resolved/discovered format contracts. Users must select
files for the same sample and pipeline run; names alone do not establish identity.
"""

from __future__ import annotations

_DOCUMENTATION = {
    "visium": (
        "Space Ranger output files",
        "https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/output-overview",
    ),
    "visium_hd_bin": (
        "Space Ranger output files",
        "https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/output-overview",
    ),
    "visium_hd_cellseg": (
        "Space Ranger output files",
        "https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/output-overview",
    ),
    "xenium": (
        "Xenium output files",
        "https://www.10xgenomics.com/support/software/xenium-onboard-analysis/3.1/analysis/xoa-output-understanding-outputs",
    ),
    "atera": (
        "Xenium output files",
        "https://www.10xgenomics.com/support/software/xenium-onboard-analysis/3.1/analysis/xoa-output-understanding-outputs",
    ),
    "bgi": (
        "STOmics SAW native matrices",
        "https://www.stomics.tech/service/saw_8_1/docs/analysis/outputs/matrices.html",
    ),
    "seekspace": ("SeekSpaceTools native inputs", "https://github.com/seekgene/SeekSpaceTools"),
    "bmkmanu": ("BMKMANU ST analysis", "https://www.biomarker.com.cn/archives/28318"),
    "singleron": ("CeleScope spatial pipeline", "https://github.com/singleron-RD/CeleScope"),
    "salus": ("Salus STS native data workflow", "https://github.com/xuzaoxu/SalusSTS"),
}


def required_paths(candidate):
    """Concrete discovered/expected paths, including missing MEX components."""
    paths = [candidate.metadata]
    if candidate.counts.is_dir():
        # This inventory is diagnostic only; readers retain their own contracts.
        for alternatives in (
            ("matrix.mtx", "matrix.mtx.gz"),
            ("features.tsv", "features.tsv.gz", "genes.tsv", "genes.tsv.gz"),
            ("barcodes.tsv", "barcodes.tsv.gz"),
        ):
            present = [candidate.counts / name for name in alternatives if (candidate.counts / name).is_file()]
            paths.extend(present or [candidate.counts / alternatives[0]])
    else:
        paths.append(candidate.counts)
    return sorted({str(p) for p in paths})


def diagnostic_report(diagnostic, *, technology=None, source=None, required_files=()):
    """Copy one diagnostic and attach conservative, JSON-safe recovery actions."""
    item = dict(diagnostic)
    code, message = item.get("code", ""), str(item.get("message", ""))
    lower = message.lower()
    context_paths = list(item.get("required_files", required_files))
    technology = item.get("technology", technology)
    docs = _DOCUMENTATION.get(technology)
    links = [{"label": docs[0], "url": docs[1]}] if docs else []
    actions = list(item.get("recovery", []))

    def add(action, text, paths=(), help_links=()):
        actions.append(dict(action=action, message=text, paths=list(paths), links=list(help_links), automatic=False))

    missing = list(item.get("missing_files", []))
    corrupt = any(
        s in lower
        for s in (
            "gzip",
            "crc check",
            "compressed file",
            "file signature",
            "truncated file",
            "unable to open file",
            "badgzip",
            "eoferror",
        )
    ) or item.get("exception_type") in ("BadGzipFile", "EOFError")
    if code == "path_missing":
        add(
            "locate_input",
            "Check the path, mounted volume and archive extraction. Pass the existing native output directory, not a website URL or an undownloaded cloud placeholder.",
            [source] if source else [],
        )
    elif missing or "missing required file" in lower or "missing count matrix" in lower:
        add(
            "restore_required_files",
            "Restore the listed native files from the same sample and pipeline run. Extract the complete output bundle, or obtain missing files from its original data portal/provider; do not create empty placeholders or substitute another sample.",
            missing or context_paths,
            links,
        )
        add(
            "download_from_verified_source",
            "At the original study/provider page, select the exact sample and download its native matrix plus coordinate metadata. Use the portal's download control or its documented CLI (including authentication if required), verify published checksum/size, then rerun the original reader. A sample-specific download URL cannot be inferred from local filenames.",
            missing or context_paths,
            links,
        )
    elif code in ("resource_deferred", "probe_deferred"):
        add(
            "increase_budget_or_select_input",
            "Inspect estimated_bytes and available RAM. Load one entry with entry.load(max_memory_bytes=..., retry=True), or use a machine with more memory. The budget covers the retained collection; lazy=True postpones full loading and does not provide out-of-core slicing. Rebinning/subsampling changes the data and is never performed automatically.",
        )
    elif code == "lazy_deferred":
        add(
            "materialize_when_needed",
            "Use result.adata for one unambiguous lazy input, entry.materialize(), or result.load(key). Only selected entries are loaded. Full matrix allocation and validation occur at that time; this is not disk-backed slicing.",
        )
    elif code == "load_deferred":
        add(
            "load_explicitly",
            "Inspection used load=False. Review the report, then call entry.load() or result.load(key); report inspection and result.adata do not implicitly load this mode.",
        )
    elif code == "source_changed" or "source changed" in lower:
        add(
            "rediscover_changed_source",
            "The native source changed after discovery. Wait for download/export completion, then rerun the original reader. The existing loader will not reuse a stale contract or silently switch source files.",
            context_paths,
        )
    elif code in ("unresolved_layout",):
        add(
            "select_unambiguous_input",
            "Multiple compatible readers or companion encodings remain. Inspect candidate_diagnostics and evidence, pass a specific native matrix file or a directory containing the intended bundle, and use technology= only when the experiment platform is known. Do not choose by a score or rename files to force a match.",
            context_paths,
            links,
        )
    elif code in ("unsupported_layout", "unrecognized_input_directory"):
        add(
            "inspect_native_layout",
            "Pass the extracted native output directory and inspect its file inventory. FASTQ/BAM, isolated images and prebuilt H5AD do not establish a supported spatial matrix contract. Obtain native counts together with matching coordinates from the provider; consult supported layouts before selecting a direct reader.",
            [item.get("path", source)] if item.get("path", source) else [],
            links,
        )
    elif code in ("discovery_limit", "depth_limit", "asset_inventory_limit"):
        add(
            "complete_inventory",
            "Pass a narrower sample/output directory or explicitly raise max_files/max_depth after checking its size. Incomplete discovery cannot establish that all inputs were found.",
            [item["path"]] if item.get("path") else [],
        )
    elif code == "symlink_skipped":
        add(
            "use_real_directory",
            "Pass the real target data directory directly. Automatic discovery does not follow symlinks; make required companion files available within the intended native bundle.",
            [item["path"]] if item.get("path") else [],
        )
    elif code == "dependency_missing" or item.get("exception_type") in ("ImportError", "ModuleNotFoundError"):
        add(
            "install_reader_dependency",
            "Install the module named in the error into the same Python environment as Spateo, following the repository installation instructions. Parquet requires pyarrow, HDF5 requires h5py. Retry explicitly after installation; packages are never installed automatically.",
            (),
            links,
        )
    elif corrupt and not code.startswith("optional_"):
        add(
            "verify_source_integrity",
            "Check downloaded bytes against the publisher's size/checksum and wait for transfer completion. Re-download the exact original file if truncated/corrupt, preserve its compression format, then rerun the original reader. Changing a filename extension does not convert a file.",
            context_paths or ([source] if source else []),
            links,
        )
    elif code in ("discovery_error", "source_unavailable", "permission_denied") or "permission denied" in lower:
        add(
            "restore_source_access",
            "Check read permission, macOS Files and Folders access, mounted/network volumes and cloud sync completion. Restore access without making files globally writable, then rediscover the input.",
            [item.get("path", source)] if item.get("path", source) else [],
        )
    elif code == "optional_images_missing":
        add(
            "optional_assets",
            "Counts and coordinates remain usable. If morphology is needed, obtain the matching image and scale/registration metadata from the same sample output bundle, then rerun with load_images=True. An image from another section must not be substituted.",
            [source] if source else [],
            links,
        )
    elif code in ("optional_image_resource_limit", "optional_image_multiframe"):
        add(
            "inspect_image_separately",
            "The core matrix remains usable. Open this large or multiframe image with a suitable image reader; export a documented display preview separately if desired. The fixed optional-image budget is not the matrix memory budget; raising max_memory_bytes alone may not load it. Retain the original image and coordinate transformation metadata.",
            [item["path"]] if item.get("path") else [],
            links,
        )
    elif code in ("optional_image_error", "optional_scale_error"):
        add(
            "restore_optional_asset",
            "Check the optional image/scale file against its original download or pipeline output and restore it if corrupt. Core counts remain usable. Do not invent scale factors or infer image registration from a filename.",
            [item["path"]] if item.get("path") else [],
            links,
        )
    elif "bmk raw chip-index" in lower:
        add(
            "export_bstmatrix_coordinates",
            "Run the provider's BSTMatrix workflow with the matching chip geometry and obtain its aggregated native matrix plus barcodes_pos.tsv.gz (barcode, pos_w, pos_h in the published display frame; physical units are not established). The five-column raw barcode_pos.tsv encodes chip/subarea indices; treating its last two columns as global coordinates is unsafe.",
            context_paths,
            links,
        )
    elif code in ("probe_failed", "contract_error", "no_valid_contract", "read_error"):
        add(
            "repair_upstream_contract",
            "Inspect this error and candidate_diagnostics for the failing field or ID. Match counts and coordinates by the original unique observation IDs; keep numeric finite values and valid sparse dimensions. Restore/re-export the original native output if needed. Do not silently zero-fill counts, drop duplicates, fabricate coordinates or align tables by row order. Rediscover after editing source files.",
            context_paths,
            links,
        )
    item["recovery"] = actions
    return item
