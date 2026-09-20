"""Bounded file discovery and deterministic layout proposals (no scores)."""

from __future__ import annotations

import gzip
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from . import _canonical_technologies, _merfish_group, _normalize_token, _seqfish_role

_TABLE_SUFFIXES = (".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt", ".txt.gz", ".parquet")
_SKIP_DIRS = {".git", "analysis", "images", "morphology_focus", "CellComposite", "CellLabels", "cell_boundaries"}


@dataclass
class Candidate:
    technology: str
    root: Path
    counts: Path
    metadata: Path
    representation: str
    options: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)

    @property
    def identity(self):
        # Representations of one source are grouped independently from platform claims.
        return str(self.counts), self.representation


def inventory(path: Path, max_files: int, max_depth: int):
    """Inspect known roots and bounded sample containers without following symlinks."""
    files, roots, diagnostics = [], set(), []
    if path.is_file():
        root = path.parent
    else:
        root = path
    stack = [(root, 0)]
    visited = 0
    while stack:
        directory, depth = stack.pop()
        roots.add(directory)
        try:
            with os.scandir(directory) as iterator:
                children = []
                for item in iterator:
                    visited += 1
                    if visited > max_files:
                        diagnostics.append(
                            dict(
                                code="discovery_limit",
                                severity="error",
                                path=str(directory),
                                message="Inventory limit reached; discovery is incomplete.",
                            )
                        )
                        return sorted(files), sorted(roots), diagnostics
                    if item.is_symlink():
                        diagnostics.append(
                            dict(
                                code="symlink_skipped",
                                severity="error",
                                path=item.path,
                                message="Symlinks are not followed; pass the intended data directory directly.",
                            )
                        )
                        continue
                    p = Path(item.path)
                    if item.is_file():
                        files.append(p)
                    elif item.is_dir() and not item.name.startswith("."):
                        if item.name in _SKIP_DIRS:
                            continue
                        if depth < max_depth:
                            children.append((p, depth + 1))
                        else:
                            diagnostics.append(
                                dict(
                                    code="depth_limit",
                                    severity="error",
                                    path=str(p),
                                    message="Directory outside discovery depth; pass it directly.",
                                )
                            )
                stack.extend(sorted(children, reverse=True))
        except OSError as exc:
            diagnostics.append(dict(code="discovery_error", severity="error", path=str(directory), message=str(exc)))
    return sorted(files), sorted(roots), diagnostics


def _one_or_expected(paths, expected):
    return sorted(paths) or [expected]


def discover(files: List[Path], requested: Path, technology=None):
    """Keep incomplete layouts as candidates so missing core files are reported."""
    present = set(files)
    parents = sorted(
        {p.parent for p in files}
        | {p.parent.parent for p in files if p.parent.name in ("spatial", "filtered_feature_bc_matrix")}
    )
    allowed = _canonical_technologies(technology)
    out = []

    def add(tech, root, counts, meta, rep, **options):
        if allowed is not None and tech not in allowed:
            return
        if requested.is_file() and requested not in (counts, meta):
            return
        out.append(
            Candidate(
                tech, root, counts, meta, rep, options, [str(counts.relative_to(root)), str(meta.relative_to(root))]
            )
        )

    for root in parents:
        local = [p for p in files if p.parent == root]
        names = {p.name: p for p in local}
        matrix_names = [n for n in ("filtered_feature_bc_matrix.h5", "raw_feature_bc_matrix.h5") if n in names]
        mex = root / "filtered_feature_bc_matrix"
        has_mex = any(p.parent == mex for p in files)
        pos = [
            root / "spatial" / n
            for n in ("tissue_positions.parquet", "tissue_positions.csv", "tissue_positions_list.csv")
            if root / "spatial" / n in present
        ]
        if matrix_names or has_mex or (root / "spatial" in parents and pos):
            bin_match = re.fullmatch(r"square[_-](\d+)um", root.name, re.I)
            is_bin = bin_match is not None or root.parent.name == "binned_outputs"
            tech = "visium_hd_bin" if is_bin else "visium"
            for matrix in ([root / n for n in matrix_names] + ([mex] if has_mex else [])) or [
                root / "filtered_feature_bc_matrix.h5"
            ]:
                population = "raw" if matrix.name.startswith("raw") else "filtered"
                rep = f"bin:{int(bin_match[1])}um/{population}" if bin_match else f"spots/{population}"
                for positions in _one_or_expected(pos, root / "spatial/tissue_positions.csv"):
                    add(tech, root, matrix, positions, rep, binsize=int(bin_match[1]) if bin_match else None)
        segs = [
            p
            for p in local
            if p.name
            in (
                "graphclust_annotated_cell_segmentations.geojson",
                "cell_segmentations.geojson",
                "cell_segmentations_annotated.geojson",
                "annotated_cell_segmentations.geojson",
            )
        ]
        if "filtered_feature_cell_matrix.h5" in names or segs:
            for seg in _one_or_expected(segs, root / "cell_segmentations.geojson"):
                add("visium_hd_cellseg", root, root / "filtered_feature_cell_matrix.h5", seg, "cells")
        cells = [p for p in local if p.name in ("cells.csv", "cells.csv.gz", "cells.parquet")]
        if "cell_feature_matrix.h5" in names or cells:
            tech, evidence = "xenium", "Xenium-compatible cell matrix and centroid table"
            experiment = root / "experiment.xenium"
            if experiment in present:
                try:
                    if experiment.stat().st_size > 1024 * 1024:
                        raise ValueError("Experiment metadata exceeds probe limit")
                    data = json.loads(experiment.read_text())
                    if not isinstance(data, dict):
                        raise ValueError("Experiment metadata must be an object")
                    # Explicit recognized metadata fields, not arbitrary JSON text.
                    identity = " ".join(
                        str(data.get(k, "")) for k in ("platform", "product", "technology", "run_name", "assay")
                    )
                    panel = data.get("panel", {})
                    identity += " " + str(panel.get("type", "") if isinstance(panel, dict) else panel)
                    if re.search(r"\batera\b|whole transcriptome|human wta", identity, re.I):
                        tech, evidence = "atera", "Explicit Atera/WTA metadata plus compatible core schema"
                except (OSError, ValueError) as exc:
                    evidence = f"unresolved identity metadata: {exc}"
            for cell in _one_or_expected(cells, root / "cells.parquet"):
                add(tech, root, root / "cell_feature_matrix.h5", cell, "cells", identity=evidence)
        tables = [p for p in local if p.name.lower().endswith(_TABLE_SUFFIXES)]
        for tech in ("merfish", "seqfish"):
            groups = {}
            for p in tables:
                if tech == "merfish":
                    role, group = None, None
                    for r, prefix in (("counts", "cell_by_gene"), ("meta", "cell_metadata")):
                        group = _merfish_group(p, prefix)
                        if group is not None:
                            role = r
                            break
                else:
                    role, group = _seqfish_role(p)
                if role in ("counts", "meta"):
                    groups.setdefault(group or "", {}).setdefault(role, []).append(p)
            for group, bundle in groups.items():
                if not bundle.get("counts"):
                    continue
                for counts in _one_or_expected(bundle.get("counts", []), root / f"counts_{group}.csv"):
                    for meta in _one_or_expected(bundle.get("meta", []), root / f"metadata_{group}.csv"):
                        add(tech, root, counts, meta, "cells", group=group)
        slide_counts = [p for p in tables if any(t in _normalize_token(p.name) for t in ("mappeddge", "dgeforr"))]
        slide_meta = [
            p for p in tables if any(t in _normalize_token(p.name) for t in ("beadlocations", "beadloacations"))
        ]
        if slide_counts or slide_meta:
            for counts in _one_or_expected(slide_counts, root / "MappedDGEForR.csv"):
                for meta in _one_or_expected(slide_meta, root / "BeadLocationsForR.csv"):
                    add("slideseq", root, counts, meta, "beads")
        star = {}
        for p in tables:
            name = p.name.removesuffix(".gz")
            for suffix, role in (
                ("processed_expression_pd.csv", "processed"),
                ("raw_expression_pd.csv", "raw"),
                ("spatial.csv", "meta"),
            ):
                if name.endswith(suffix):
                    star.setdefault(name[: -len(suffix)].rstrip("_"), {}).setdefault(role, []).append(p)
        for prefix, bundle in star.items():
            for role in ("processed", "raw"):
                if role not in bundle:
                    continue
                for counts in bundle[role]:
                    for meta in _one_or_expected(bundle.get("meta", []), root / f"{prefix}_spatial.csv"):
                        add("starmap_plus", root, counts, meta, role, group=prefix)
        cosmx = {}
        for p in tables:
            token = _normalize_token(p.name)
            if any(t in token for t in ("exprmat", "expressionmatrix", "expressionmat", "countmatrix")):
                cosmx.setdefault("counts", []).append(p)
            elif any(t in token for t in ("metadata", "cellmeta")) and "fovposition" not in token:
                cosmx.setdefault("meta", []).append(p)
        if cosmx.get("counts"):

            def sample_prefix(p, role):
                pattern = (
                    r"exprmat|expressionmatrix|expressionmat|countmatrix"
                    if role == "counts"
                    else r"cellmetadata|metadata|cellmeta"
                )
                hit = re.search(pattern, p.name, re.I)
                return p.name[: hit.start()].strip("_-. ").lower() if hit else ""

            for counts in cosmx["counts"]:
                metas = [
                    m for m in cosmx.get("meta", []) if sample_prefix(m, "meta") == sample_prefix(counts, "counts")
                ]
                if not metas and len(cosmx["counts"]) == 1 and len(cosmx.get("meta", [])) == 1:
                    metas = cosmx["meta"]
                for meta in _one_or_expected(metas, root / (sample_prefix(counts, "counts") + "_metadata.csv")):
                    add("nanostring", root, counts, meta, "cells_by_fov")
        for p in local:
            is_gem = p.name.endswith((".gem", ".gem.gz"))
            if not is_gem and p.name.endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")):
                try:
                    opener = gzip.open if p.name.endswith(".gz") else open
                    with opener(p, "rt", encoding="utf-8-sig") as handle:
                        for _ in range(100):
                            line = handle.readline(65536)
                            if not line:
                                break
                            if line.strip() and not line.startswith("#"):
                                columns = set(line.strip().split("\t"))
                                is_gem = {"geneID", "x", "y"}.issubset(columns) and bool(
                                    columns & {"MIDCount", "MIDCounts", "UMICount", "UMICounts", "count", "total"}
                                )
                                break
                except (OSError, UnicodeError):
                    pass
            if is_gem:
                add("bgi", root, p, p, "native_xy_bins", binsize=1)
    # Files in two valid storage encodings or companion variants remain visible.
    unique = {(c.technology, str(c.counts), str(c.metadata), c.representation): c for c in out}
    return list(unique.values())
