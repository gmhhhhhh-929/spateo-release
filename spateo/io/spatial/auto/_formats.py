"""Shared format names and filename grouping; no ranking or score policy."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

_TABLE_ENDINGS = (".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt", ".txt.gz", ".parquet")


def _normalize_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


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
