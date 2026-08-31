"""Automatic spatial transcriptomics reader dispatch.

This submodule detects common spatial transcriptomics output layouts and calls
the matching reader from :mod:`spateo.io.spatial`.

It is intentionally packaged as ``spateo.io.spatial.auto`` so existing spatial
reader files do not need to be edited.
"""

from __future__ import annotations

import gzip
import importlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from ...._registry import register_function
from .._provenance import record_spatial_io, spatial_file_manifest

PathLike = Union[str, Path]

_MISSING_METADATA_FILE = "__spateo_auto_missing_metadata__.csv"
_TABLE_ENDINGS = (".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt", ".txt.gz", ".parquet")
_IMAGE_ENDINGS = (".ome.tif", ".ome.tiff", ".tif", ".tiff", ".png", ".jpg", ".jpeg")
_BGI_ENDINGS = (".gem", ".gem.gz", ".tsv", ".tsv.gz", ".txt", ".txt.gz")
_AMBIGUITY_MARGIN = 0.05


@dataclass(frozen=True)
class SpatialReadMatch:
    """A detected spatial dataset layout and the reader call it maps to."""

    technology: str
    reader: str
    path: Path
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    evidence: Tuple[str, ...] = field(default_factory=tuple)

    def load_reader(self) -> Callable[..., Any]:
        """Import and return the reader callable for this match."""
        module_name, func_name = self.reader.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, func_name)

    def read(self, **overrides: Any) -> Any:
        """Read this match, with ``overrides`` taking precedence."""
        kwargs = dict(self.kwargs)
        kwargs.update(overrides)
        return self.load_reader()(self.path, **kwargs)


def _as_dir(path: PathLike) -> Path:
    p = Path(path).expanduser().resolve()
    return p.parent if p.is_file() else p


def _normalize_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _natural_sort_key(value: object) -> List[Union[int, str]]:
    parts = re.split(r"(\d+)", str(value))
    out: List[Union[int, str]] = []
    for part in parts:
        if part.isdigit():
            out.append(int(part))
        elif part:
            out.append(part.lower())
    return out


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _name_endswith(path: Path, endings: Sequence[str]) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(ending) for ending in endings)


def _shallow_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    if not root.exists() or not root.is_dir():
        return []
    return sorted([p for p in root.iterdir() if p.is_file()], key=_natural_sort_key)


def _shallow_dirs(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()], key=_natural_sort_key)


def _table_files(root: Path) -> List[Path]:
    return [p for p in _shallow_files(root) if _name_endswith(p, _TABLE_ENDINGS)]


def _first_file(root: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        p = root / name
        if p.exists() and p.is_file():
            return p
    return None


def _find_by_tokens(
    files: Sequence[Path],
    include_any: Sequence[str],
    *,
    exclude_any: Sequence[str] = (),
) -> Optional[Path]:
    include = tuple(_normalize_token(x) for x in include_any)
    exclude = tuple(_normalize_token(x) for x in exclude_any)
    hits = []
    for path in files:
        token = _normalize_token(path.name)
        if include and not any(x in token for x in include):
            continue
        if exclude and any(x in token for x in exclude):
            continue
        hits.append(path)
    return sorted(hits, key=_natural_sort_key)[0] if hits else None


def _all_existing(root: Path, rels: Sequence[str]) -> bool:
    return all((root / rel).exists() for rel in rels)


def _make_match(
    technology: str,
    reader: str,
    path: Path,
    kwargs: Optional[Mapping[str, Any]],
    confidence: float,
    evidence: Iterable[str],
) -> SpatialReadMatch:
    return SpatialReadMatch(
        technology=technology,
        reader=reader,
        path=path.resolve(),
        kwargs=dict(kwargs or {}),
        confidence=float(confidence),
        evidence=tuple(str(x) for x in evidence),
    )


def _detect_xenium(path: Path) -> List[SpatialReadMatch]:
    root = _as_dir(path)
    mat = _first_file(root, ("cell_feature_matrix.h5",))
    cells = _first_file(root, ("cells.parquet", "cells.csv.gz", "cells.csv"))
    if mat is None or cells is None:
        return []
    # The public Atera preview intentionally uses a Xenium-compatible core.
    # Defer to the dedicated Atera detector when the bundle has explicit Atera
    # metadata or its named multi-stain morphology layout.
    from .._atera import _atera_evidence

    if _atera_evidence(root):
        return []
    evidence = [_rel(mat, root), _rel(cells, root)]
    if (root / "experiment.xenium").exists():
        evidence.append("experiment.xenium")
    return [
        _make_match(
            "xenium",
            "spateo.io.spatial._xenium.read_xenium",
            root,
            {},
            0.98,
            evidence,
        )
    ]


def _detect_atera(path: Path) -> List[SpatialReadMatch]:
    root = _as_dir(path)
    if not (root / "cell_feature_matrix.h5").is_file() and (root / "outs").is_dir():
        root = root / "outs"
    matrix = _first_file(root, ("cell_feature_matrix.h5",))
    cells = _first_file(root, ("cells.parquet", "cells.csv.gz", "cells.csv"))
    if matrix is None or cells is None:
        return []
    from .._atera import _atera_evidence

    atera_evidence = _atera_evidence(root)
    if not atera_evidence:
        return []
    evidence = [_rel(matrix, root), _rel(cells, root), *atera_evidence]
    return [
        _make_match(
            "atera",
            "spateo.io.spatial._atera.read_atera",
            root,
            {},
            0.995,
            evidence,
        )
    ]


def _detect_visium_hd_cellseg(path: Path) -> List[SpatialReadMatch]:
    root = _as_dir(path)
    candidates = [root]
    seg_dir = root / "segmented_outputs"
    if seg_dir.is_dir():
        candidates.insert(0, seg_dir)

    out = []
    seg_names = (
        "graphclust_annotated_cell_segmentations.geojson",
        "cell_segmentations.geojson",
        "cell_segmentations_annotated.geojson",
        "annotated_cell_segmentations.geojson",
    )
    for candidate in candidates:
        seg = _first_file(candidate, seg_names)
        matrix = _first_file(candidate, ("filtered_feature_cell_matrix.h5",))
        if seg is None or matrix is None:
            continue
        out.append(
            _make_match(
                "visium_hd_cellseg",
                "spateo.io.spatial._visium_hd.read_visium_hd",
                candidate,
                {
                    "data_type": "cellseg",
                    "cell_segmentations_path": seg.name,
                    "cell_matrix_h5_path": matrix.name,
                },
                0.99,
                [_rel(seg, candidate), _rel(matrix, candidate)],
            )
        )
    return out


def _parse_square_binsize(path: Path) -> Optional[int]:
    m = re.search(r"square[_-](\d+)um", path.name.lower())
    return int(m.group(1)) if m else None


def _detect_visium_hd_bin(path: Path) -> List[SpatialReadMatch]:
    root = _as_dir(path)
    candidates = []
    if _normalize_token(root.parent.name) == "binnedoutputs" or root.name.lower().startswith("square_"):
        candidates.append(root)
    binned = root / "binned_outputs"
    if binned.is_dir():
        candidates.extend([p for p in _shallow_dirs(binned) if p.name.lower().startswith("square_")])

    out = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not _all_existing(candidate, ("spatial",)):
            continue
        has_matrix = (candidate / "filtered_feature_bc_matrix.h5").exists() or (
            candidate / "filtered_feature_bc_matrix"
        ).exists()
        tissue = _first_file(
            candidate / "spatial",
            ("tissue_positions.parquet", "tissue_positions.csv"),
        )
        if not has_matrix or tissue is None:
            continue
        binsize = _parse_square_binsize(candidate) or 16
        evidence = ["filtered_feature_bc_matrix(.h5|/)", _rel(tissue, candidate)]
        out.append(
            _make_match(
                "visium_hd_bin",
                "spateo.io.spatial._visium_hd.read_visium_hd",
                candidate,
                {"data_type": "bin", "binsize": binsize},
                0.96,
                evidence,
            )
        )
    return out


def _detect_visium(path: Path) -> List[SpatialReadMatch]:
    root = _as_dir(path)
    if root.name.lower().startswith("square_") or (root / "binned_outputs").is_dir():
        return []
    count = _first_file(root, ("filtered_feature_bc_matrix.h5", "raw_feature_bc_matrix.h5"))
    tissue = _first_file(
        root / "spatial",
        ("tissue_positions.parquet", "tissue_positions.csv", "tissue_positions_list.csv"),
    )
    if count is None or tissue is None:
        return []
    kwargs = {}
    if count.name != "filtered_feature_bc_matrix.h5":
        kwargs["count_file"] = count.name
    return [
        _make_match(
            "visium",
            "spateo.io.spatial._visium.read_visium",
            root,
            kwargs,
            0.92,
            [_rel(count, root), _rel(tissue, root)],
        )
    ]


def _detect_slideseq(path: Path) -> List[SpatialReadMatch]:
    root = _as_dir(path)
    files = _shallow_files(root)
    counts = _first_file(
        root,
        (
            "MappedDGEForR.csv",
            "MappedDGEForR.csv.gz",
            "mapped_dge_for_r.csv",
            "mapped_dge_for_r.csv.gz",
        ),
    )
    bead = _first_file(
        root,
        (
            "BeadLocationsForR.csv",
            "BeadLocationsForR.csv.gz",
            "BeadLoacationsForR.csv",
            "BeadLoacationsForR.csv.gz",
            "bead_locations_for_r.csv",
            "bead_locations_for_r.csv.gz",
        ),
    )
    if counts is None:
        counts = _find_by_tokens(files, ("mappeddge", "dgeforr"), exclude_any=("bead", "location"))
    if bead is None:
        bead = _find_by_tokens(files, ("beadlocations", "beadloacations", "beadlocation"))
    if counts is None or bead is None:
        return []
    return [
        _make_match(
            "slideseq",
            "spateo.io.spatial._slideseq.read_slideseq",
            root,
            {"counts_file": counts.name, "bead_file": bead.name},
            0.94,
            [_rel(counts, root), _rel(bead, root)],
        )
    ]


def _detect_nanostring(path: Path) -> List[SpatialReadMatch]:
    root = _as_dir(path)
    files = _table_files(root)
    if not files:
        return []

    counts = _find_by_tokens(
        files,
        ("exprmat", "expressionmatrix", "expressionmat", "countmatrix"),
        exclude_any=("metadata", "fov", "position"),
    )
    meta = _find_by_tokens(files, ("metadata", "cellmetadata", "cellmeta"), exclude_any=("fovposition",))
    fov = _find_by_tokens(files, ("fovpositions", "fovposition", "fovfile", "fov"))
    composite = root / "CellComposite"
    labels = root / "CellLabels"

    if counts is None or meta is None:
        return []

    score = 0.78
    evidence = [_rel(counts, root), _rel(meta, root)]
    if fov is not None:
        score += 0.06
        evidence.append(_rel(fov, root))
    if composite.is_dir():
        score += 0.08
        evidence.append("CellComposite/")
    if labels.is_dir():
        score += 0.08
        evidence.append("CellLabels/")

    kwargs: Dict[str, Any] = {"counts_file": counts.name, "meta_file": meta.name}
    if fov is not None:
        kwargs["fov_file"] = fov.name
    return [
        _make_match(
            "nanostring",
            "spateo.io.spatial._nanostring.read_nanostring",
            root,
            kwargs,
            min(score, 0.96),
            evidence,
        )
    ]


def _seqfish_role(path: Path) -> Tuple[Optional[str], str]:
    stem = re.sub(
        r"(\.csv\.gz|\.tsv\.gz|\.txt\.gz|\.ome\.tiff|\.ome\.tif|\.csv|\.tsv|\.txt|\.tiff|\.tif|\.png|\.jpg|\.jpeg)$",
        "",
        path.name,
        flags=re.IGNORECASE,
    )
    lower = stem.lower()
    patterns: Sequence[Tuple[str, Sequence[str]]] = (
        ("meta", (r"cell[\s_-]*coordinates?", r"cell[\s_-]*coords?", r"metadata", r"meta")),
        ("counts", (r"c[\s_-]*x[\s_-]*g", r"cell[\s_-]*by[\s_-]*gene", r"counts?")),
        ("cell_mask", (r"cell[\s_-]*mask", r"cell[\s_-]*labels?", r"seg(?:mentation)?")),
        ("dapi", (r"dapi",)),
    )
    for role, regexes in patterns:
        for regex in regexes:
            m = re.search(regex, lower, flags=re.IGNORECASE)
            if m is None:
                continue
            group = re.sub(r"^[\s_.-]+|[\s_.-]+$", "", stem[m.end() :])
            return role, group
    return None, ""


def _detect_seqfish(path: Path) -> List[SpatialReadMatch]:
    root = _as_dir(path)
    files = [p for p in _table_files(root) if not p.name.lower().endswith(".parquet")]
    images_dir = root / "images"
    if images_dir.is_dir():
        files.extend([p for p in _shallow_files(images_dir) if _name_endswith(p, _IMAGE_ENDINGS)])

    groups: Dict[str, Dict[str, Any]] = {}
    for file in files:
        role, group = _seqfish_role(file)
        if role is None:
            continue
        bucket = groups.setdefault(group, {"images": {}})
        if role in ("counts", "meta"):
            bucket[role] = file
        else:
            bucket["images"][role] = file

    out = []
    for group, bundle in groups.items():
        counts = bundle.get("counts")
        meta = bundle.get("meta")
        if counts is None or meta is None:
            continue
        evidence = [_rel(counts, root), _rel(meta, root)]
        images = bundle.get("images", {})
        evidence.extend(_rel(p, root) for p in images.values())
        score = 0.84 + (0.04 if images else 0.0)
        out.append(
            _make_match(
                "seqfish",
                "spateo.io.spatial._seqfish.read_seqfish",
                root,
                {"counts_file": _rel(counts, root), "meta_file": _rel(meta, root)},
                score,
                evidence,
            )
        )
    return out


def _merfish_group(path: Path, prefix: str) -> Optional[str]:
    name = path.name
    lower = name.lower()
    for suffix in sorted(_TABLE_ENDINGS + (".vzg", ".hdf5", ".h5"), key=len, reverse=True):
        if lower.endswith(suffix):
            name = name[: -len(suffix)]
            break
    norm_name = name.lower()
    norm_prefix = prefix.lower()
    if norm_name == norm_prefix:
        return ""
    if norm_name.startswith(norm_prefix + "_") or norm_name.startswith(norm_prefix + "-"):
        return name[len(prefix) + 1 :]
    return None


def _detect_merfish(path: Path) -> List[SpatialReadMatch]:
    root = _as_dir(path)
    files = _table_files(root)
    counts_map: Dict[str, Path] = {}
    meta_map: Dict[str, Path] = {}
    detected_map: Dict[str, Path] = {}
    for file in files:
        group = _merfish_group(file, "cell_by_gene")
        if group is not None:
            counts_map[group] = file
            continue
        group = _merfish_group(file, "cell_metadata")
        if group is not None:
            meta_map[group] = file
            continue
        group = _merfish_group(file, "detected_transcripts")
        if group is not None:
            detected_map[group] = file

    groups = sorted(set(counts_map) | set(meta_map), key=_natural_sort_key)
    out = []
    for group in groups:
        counts = counts_map.get(group)
        meta = meta_map.get(group)
        if counts is None or meta is None:
            continue
        evidence = [_rel(counts, root), _rel(meta, root)]
        score = 0.92
        if group in detected_map:
            score += 0.04
            evidence.append(_rel(detected_map[group], root))
        if (root / "cell_boundaries").is_dir():
            score += 0.02
            evidence.append("cell_boundaries/")
        if (root / "images").is_dir():
            score += 0.02
            evidence.append("images/")
        out.append(
            _make_match(
                "merfish",
                "spateo.io.spatial._merfish.read_merfish",
                root,
                {"counts_file": counts.name, "meta_file": meta.name},
                min(score, 0.98),
                evidence,
            )
        )
    return out


def _starmap_prefixes(root: Path) -> Dict[str, Dict[str, Path]]:
    suffix_roles = {
        "processed_expression_pd.csv": "processed",
        "raw_expression_pd.csv": "raw",
        "spatial.csv": "spatial",
        "spot_meta.csv": "spot_meta",
    }
    groups: Dict[str, Dict[str, Path]] = {}
    for file in _shallow_files(root):
        name = file.name
        if name.endswith(".gz"):
            name = name[:-3]
        lower = name.lower()
        for suffix, role in suffix_roles.items():
            if not lower.endswith(suffix):
                continue
            prefix = name[: -len(suffix)].rstrip("_")
            groups.setdefault(prefix, {})[role] = file
            break
    return groups


def _detect_starmap_plus(path: Path) -> List[SpatialReadMatch]:
    root = _as_dir(path)
    groups = _starmap_prefixes(root)
    out = []
    for prefix, bundle in sorted(groups.items(), key=lambda item: _natural_sort_key(item[0])):
        spatial = bundle.get("spatial")
        counts = bundle.get("processed") or bundle.get("raw")
        if spatial is None or counts is None:
            continue
        meta = _find_by_tokens(_table_files(root), ("metadata", "metatable", "samplemeta"))
        evidence = [_rel(counts, root), _rel(spatial, root)]
        kwargs = {
            "counts_file": counts.name,
            "spatial_file": spatial.name,
            "meta_file": meta.name if meta is not None else _MISSING_METADATA_FILE,
        }
        if meta is not None:
            evidence.append(_rel(meta, root))
        if bundle.get("spot_meta") is not None:
            evidence.append(_rel(bundle["spot_meta"], root))
        out.append(
            _make_match(
                "starmap_plus",
                "spateo.io.spatial._starmap_plus.read_starmap_plus",
                root,
                kwargs,
                0.9,
                evidence,
            )
        )
    return out


def _read_header_line(path: Path) -> Optional[str]:
    opener: Callable[..., Any] = gzip.open if path.name.lower().endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    return stripped
    except OSError:
        return None
    return None


def _looks_like_bgi_file(path: Path) -> bool:
    if not path.is_file() or not _name_endswith(path, _BGI_ENDINGS):
        return False
    header = _read_header_line(path)
    if header is None:
        return False
    columns = {_normalize_token(c) for c in re.split(r"\t|,", header)}
    has_gene = "geneid" in columns
    has_xy = "x" in columns and "y" in columns
    has_count = any(c in columns for c in ("midcount", "midcounts", "umicount", "umicounts"))
    return has_gene and has_xy and has_count


def _detect_bgi(path: Path) -> List[SpatialReadMatch]:
    p = Path(path).expanduser().resolve()
    if p.is_file():
        candidates = [p]
        root = p.parent
    else:
        root = p
        candidates = [q for q in _shallow_files(root) if _name_endswith(q, _BGI_ENDINGS)]
    hits = [q for q in candidates if _looks_like_bgi_file(q)]
    if not hits:
        return []
    out = []
    for hit in sorted(hits, key=_natural_sort_key):
        out.append(
            _make_match(
                "bgi",
                "spateo.io.spatial._stereoseq.read_bgi",
                hit,
                {"binsize": 1},
                0.86,
                [_rel(hit, root), "header: geneID/x/y/count"],
            )
        )
    return out


_DETECTORS: Sequence[Callable[[Path], List[SpatialReadMatch]]] = (
    _detect_atera,
    _detect_xenium,
    _detect_visium_hd_cellseg,
    _detect_visium_hd_bin,
    _detect_visium,
    _detect_slideseq,
    _detect_merfish,
    _detect_nanostring,
    _detect_seqfish,
    _detect_starmap_plus,
    _detect_bgi,
)

_TECH_ALIASES = {
    "10xvisium": {"visium"},
    "visium": {"visium"},
    "visiumhd": {"visium_hd_bin", "visium_hd_cellseg"},
    "visiumhdbin": {"visium_hd_bin"},
    "visiumhdcellseg": {"visium_hd_cellseg"},
    "cellseg": {"visium_hd_cellseg"},
    "xenium": {"xenium"},
    "atera": {"atera"},
    "aterainsitu": {"atera"},
    "aterawta": {"atera"},
    "wta": {"atera"},
    "slideseq": {"slideseq"},
    "slideseqv2": {"slideseq"},
    "cosmx": {"nanostring"},
    "nanostring": {"nanostring"},
    "smi": {"nanostring"},
    "seqfish": {"seqfish"},
    "merfish": {"merfish"},
    "merscope": {"merfish"},
    "vizgen": {"merfish"},
    "starmapplus": {"starmap_plus"},
    "starmap": {"starmap_plus"},
    "bgi": {"bgi"},
    "stereoseq": {"bgi"},
    "stereo": {"bgi"},
}


def _canonical_technologies(technology: Optional[str]) -> Optional[set]:
    if technology is None:
        return None
    key = _normalize_token(technology)
    if key in _TECH_ALIASES:
        return set(_TECH_ALIASES[key])
    canonical = {
        "visium",
        "visium_hd_bin",
        "visium_hd_cellseg",
        "xenium",
        "atera",
        "slideseq",
        "nanostring",
        "seqfish",
        "merfish",
        "starmap_plus",
        "bgi",
    }
    if technology in canonical:
        return {technology}
    raise ValueError(f"Unknown spatial technology override: {technology!r}")


def _rank_matches(matches: Sequence[SpatialReadMatch]) -> List[SpatialReadMatch]:
    return sorted(
        matches,
        key=lambda m: (m.confidence, len(m.evidence), str(m.path)),
        reverse=True,
    )


def detect_spatial_technologies(
    path: PathLike,
    *,
    technology: Optional[str] = None,
    min_confidence: float = 0.0,
) -> List[SpatialReadMatch]:
    """Return all matching spatial reader candidates for ``path``.

    Parameters
    ----------
    path
        Dataset directory, or a BGI/Stereo-seq read-level file.
    technology
        Optional technology filter or alias, for example ``"atera"``, ``"xenium"``,
        ``"visium_hd"``, ``"cosmx"``, or ``"merscope"``.
    min_confidence
        Drop matches below this score.
    """
    p = Path(path).expanduser().resolve()
    allowed = _canonical_technologies(technology)
    matches: List[SpatialReadMatch] = []
    for detector in _DETECTORS:
        for match in detector(p):
            if allowed is not None and match.technology not in allowed:
                continue
            if match.confidence >= min_confidence:
                matches.append(match)
    return _rank_matches(matches)


def _format_matches(matches: Sequence[SpatialReadMatch]) -> str:
    lines = []
    for match in matches:
        kwargs = ", ".join(f"{k}={v!r}" for k, v in match.kwargs.items())
        evidence = "; ".join(match.evidence)
        lines.append(
            f"- {match.technology} ({match.confidence:.2f}) via {match.reader}"
            f" at {match.path} kwargs={{ {kwargs} }} evidence=[{evidence}]"
        )
    return "\n".join(lines)


def detect_spatial_technology(
    path: PathLike,
    *,
    technology: Optional[str] = None,
    min_confidence: float = 0.5,
    strict: bool = True,
) -> SpatialReadMatch:
    """Return the best spatial reader candidate for ``path``.

    Set ``strict=False`` to take the top-ranked match when multiple layouts have
    very similar confidence.
    """
    matches = detect_spatial_technologies(path, technology=technology, min_confidence=min_confidence)
    if not matches:
        suffix = f" for technology {technology!r}" if technology is not None else ""
        raise FileNotFoundError(f"Could not detect a supported spatial dataset layout at {path!r}{suffix}.")

    best = matches[0]
    if strict and len(matches) > 1:
        tied = [m for m in matches[1:] if abs(best.confidence - m.confidence) <= _AMBIGUITY_MARGIN]
        if tied:
            raise ValueError(
                "Ambiguous spatial dataset layout. Please pass `technology=` or inspect candidates:\n"
                + _format_matches([best] + tied)
            )
    return best


@register_function(
    aliases=["read_auto_spatial", "auto spatial reader", "detect spatial reader"],
    category="io",
    description="Detect a spatial transcriptomics output layout and dispatch to the matching reader.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "adata = read_auto_spatial('Atera_outs')",
        "adata = read_auto_spatial('Xenium_outs')",
        "adata = read_auto_spatial('outs/binned_outputs/square_016um')",
        "adata = read_auto_spatial('merscope_dir', load_images=False)",
        "match = detect_spatial_technology('dataset_dir')",
    ],
    related=[
        "io.spatial.read_atera",
        "io.spatial.read_visium",
        "io.spatial.read_visium_hd",
        "io.spatial.read_xenium",
        "io.spatial.read_merfish",
    ],
)
def read_auto_spatial(
    path: PathLike,
    *,
    technology: Optional[str] = None,
    min_confidence: float = 0.5,
    strict: bool = True,
    return_match: bool = False,
    **reader_kwargs: Any,
) -> Any:
    """Detect and read a spatial transcriptomics dataset.

    ``reader_kwargs`` are forwarded to the selected reader and override inferred
    values such as ``counts_file`` or ``meta_file``.
    """
    match = detect_spatial_technology(
        path,
        technology=technology,
        min_confidence=min_confidence,
        strict=strict,
    )
    adata = match.read(**reader_kwargs)
    combined_kwargs = dict(match.kwargs)
    combined_kwargs.update(reader_kwargs)
    record_spatial_io(
        adata,
        technology=match.technology,
        source=match.path,
        reader=match.reader,
        confidence=match.confidence,
        evidence=match.evidence,
        reader_kwargs=combined_kwargs,
        manifest=spatial_file_manifest(match.path),
        format_status="preview-xenium-v4" if match.technology == "atera" else "stable",
    )
    if return_match:
        return adata, match
    return adata


read_spatial_auto = read_auto_spatial

__all__ = [
    "SpatialReadMatch",
    "detect_spatial_technologies",
    "detect_spatial_technology",
    "read_auto_spatial",
    "read_spatial_auto",
]
