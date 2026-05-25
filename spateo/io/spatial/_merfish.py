"""
Data reading functions for MERFISH / Vizgen MERSCOPE.

This module provides a reader for MERFISH dataset folders organized like::

    Merfish_dataset/
    ├── detected_transcripts_S1R1.csv
    ├── cell_by_gene_S1R1.csv
    ├── cell_metadata_S1R1.csv
    ├── cell_boundaries/
    │   └── feature_data_##.hdf5
    └── images/
        ├── mosaic_DAPI_z0.tif
        ├── ...
        └── micron_to_mosaic_pixel_transform.csv

The minimal useful input is typically either ``cell_by_gene*.csv`` (+ optional
``cell_metadata*.csv``) or ``detected_transcripts*.csv``. Mosaic images,
cell-boundary HDF5 files, and ``.vzg`` bundles are optional.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import coo_matrix, csr_matrix
from ...configuration import SKM
try:
    import tifffile
except Exception:  # pragma: no cover
    tifffile = None

from ..._registry import register_function

try:
    from ..._settings import Colors
except Exception:
    class Colors:
        HEADER = "\033[95m"
        BLUE = "\033[94m"
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        WARNING = "\033[93m"
        FAIL = "\033[91m"
        ENDC = "\033[0m"
        BOLD = "\033[1m"
        UNDERLINE = "\033[4m"


_IMAGE_SUFFIXES = (".ome.tif", ".ome.tiff", ".tif", ".tiff", ".png", ".jpg", ".jpeg")
_MOSAIC_RE = re.compile(r"^mosaic_(?P<channel>.+?)_z(?P<z>\d+)$", flags=re.IGNORECASE)
_CSV_EXTENSIONS = (".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt", ".txt.gz")


def _progress(message: str, level: str = "info") -> None:
    level_key = (level or "info").lower()
    if level_key == "info":
        msg = message.lower()
        if msg.startswith("reading"):
            level_key = "start"
        elif msg.startswith("loading"):
            level_key = "step"
        elif msg.startswith("done"):
            level_key = "success"
        elif "error" in msg or "failed" in msg:
            level_key = "error"

    color_map = {
        "start": Colors.HEADER,
        "step": Colors.BLUE,
        "info": Colors.CYAN,
        "success": Colors.GREEN,
        "warn": Colors.WARNING,
        "warning": Colors.WARNING,
        "error": Colors.FAIL,
        "fail": Colors.FAIL,
    }
    tag_map = {
        "start": "[START]",
        "step": "[STEP]",
        "info": "[INFO]",
        "success": "[OK]",
        "warn": "[WARN]",
        "warning": "[WARN]",
        "error": "[ERR]",
        "fail": "[ERR]",
    }

    color = color_map.get(level_key, Colors.CYAN)
    tag = tag_map.get(level_key, "[INFO]")
    text = f"[MERFISH]{tag} {message}"

    force_color = os.environ.get("FORCE_COLOR", "").strip() not in ("", "0", "false", "False")
    no_color = os.environ.get("NO_COLOR", "").strip() != ""
    supports_color = force_color or (hasattr(sys.stdout, "isatty") and sys.stdout.isatty())

    if no_color or not supports_color:
        print(text)
    else:
        print(f"{color}{text}{Colors.ENDC}")


@contextmanager
def _silence_tifffile_logger(level: int = logging.ERROR):
    logger = logging.getLogger("tifffile")
    old_level = logger.level
    old_propagate = logger.propagate
    try:
        logger.setLevel(level)
        logger.propagate = False
        yield
    finally:
        logger.setLevel(old_level)
        logger.propagate = old_propagate


def _normalize_token(text: Union[str, Path, None]) -> str:
    if text is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _natural_sort_key(text: Union[str, Path]) -> List[Union[int, str]]:
    parts = re.split(r"(\d+)", str(text))
    out: List[Union[int, str]] = []
    for part in parts:
        if part.isdigit():
            out.append(int(part))
        elif part:
            out.append(part.lower())
    return out


def _extract_last_int(text: Union[str, Path, None]) -> Optional[int]:
    if text is None:
        return None
    vals = re.findall(r"(\d+)", str(text))
    return int(vals[-1]) if vals else None


def _find_children(root: Path, want_files: bool) -> List[Path]:
    items = []
    for p in root.iterdir():
        if want_files and p.is_file():
            items.append(p)
        elif (not want_files) and p.is_dir():
            items.append(p)
    return sorted(items, key=_natural_sort_key)


def _looks_like_csv(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(ext) for ext in _CSV_EXTENSIONS)


def _read_table_with_auto_sep(path: Path, **kwargs) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return pd.read_csv(path, sep=",", **kwargs)
    if name.endswith(".tsv") or name.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t", **kwargs)
    if name.endswith(".txt") or name.endswith(".txt.gz"):
        try:
            return pd.read_csv(path, sep="\t", **kwargs)
        except Exception:
            return pd.read_csv(path, sep=",", **kwargs)
    try:
        return pd.read_csv(path, sep=",", **kwargs)
    except Exception:
        try:
            return pd.read_csv(path, sep="\t", **kwargs)
        except Exception:
            return pd.read_csv(path, sep=None, engine="python", **kwargs)


def _strip_known_suffixes(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for suffix in sorted(_CSV_EXTENSIONS + _IMAGE_SUFFIXES + (".vzg", ".hdf5", ".h5"), key=len, reverse=True):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _parse_group_suffix(path: Path, prefix: str) -> Optional[str]:
    stem = _strip_known_suffixes(path)
    stem_lower = stem.lower()
    prefix_lower = prefix.lower()
    if stem_lower == prefix_lower:
        return ""
    if stem_lower.startswith(prefix_lower + "_"):
        return stem[len(prefix) + 1 :]
    if stem_lower.startswith(prefix_lower + "-"):
        return stem[len(prefix) + 1 :]
    return None


def _discover_merfish_assets(root: Path) -> Dict[str, Any]:
    if root.is_file():
        root = root.parent
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"MERFISH reader expects a dataset directory: {root}")

    files = _find_children(root, want_files=True)
    dirs = _find_children(root, want_files=False)

    detected_map: Dict[str, Path] = {}
    cell_by_gene_map: Dict[str, Path] = {}
    cell_metadata_map: Dict[str, Path] = {}
    vzg_map: Dict[str, Path] = {}

    for p in files:
        if not _looks_like_csv(p) and p.suffix.lower() != ".vzg":
            continue
        if p.suffix.lower() == ".vzg":
            vzg_map[p.stem] = p
            continue

        tag = _parse_group_suffix(p, "detected_transcripts")
        if tag is not None:
            detected_map[tag] = p
            continue

        tag = _parse_group_suffix(p, "cell_by_gene")
        if tag is not None:
            cell_by_gene_map[tag] = p
            continue

        tag = _parse_group_suffix(p, "cell_metadata")
        if tag is not None:
            cell_metadata_map[tag] = p
            continue

    images_dir = None
    boundaries_dir = None
    for d in dirs:
        token = _normalize_token(d.name)
        if images_dir is None and "image" in token:
            images_dir = d
        if boundaries_dir is None and ("cellboundar" in token or token == "cellboundaries"):
            boundaries_dir = d

    transform_path = None
    if images_dir is not None:
        candidates = sorted(
            [p for p in images_dir.rglob("*") if p.is_file() and "microntomosaicpixeltransform" in _normalize_token(p.name)],
            key=_natural_sort_key,
        )
        transform_path = candidates[0] if candidates else None

    boundary_files: List[Path] = []
    if boundaries_dir is not None:
        boundary_files = sorted(
            [p for p in boundaries_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".hdf5", ".h5")],
            key=_natural_sort_key,
        )

    return {
        "root": root,
        "detected_map": detected_map,
        "cell_by_gene_map": cell_by_gene_map,
        "cell_metadata_map": cell_metadata_map,
        "vzg_map": vzg_map,
        "images_dir": images_dir,
        "boundaries_dir": boundaries_dir,
        "boundary_files": boundary_files,
        "transform_path": transform_path,
    }


def _pick_best_key(mapping: Mapping[str, Path], requested: Optional[str]) -> Tuple[Optional[str], Optional[Path]]:
    if not mapping:
        return None, None
    keys = list(mapping.keys())
    if requested is None:
        key = sorted(keys, key=_natural_sort_key)[0]
        return key, mapping[key]

    want = _normalize_token(requested)
    exact = [k for k in keys if _normalize_token(k) == want]
    if exact:
        key = sorted(exact, key=_natural_sort_key)[0]
        return key, mapping[key]

    partial = [k for k in keys if want and (want in _normalize_token(k) or _normalize_token(k) in want)]
    if partial:
        key = sorted(partial, key=_natural_sort_key)[0]
        return key, mapping[key]

    return None, None


def _resolve_dataset_group(assets: Mapping[str, Any], region_name: Optional[str]) -> Dict[str, Any]:
    detected_map = assets["detected_map"]
    cell_by_gene_map = assets["cell_by_gene_map"]
    cell_metadata_map = assets["cell_metadata_map"]
    vzg_map = assets["vzg_map"]

    available = sorted(set(detected_map) | set(cell_by_gene_map) | set(cell_metadata_map), key=_natural_sort_key)
    selected_key: Optional[str] = None

    if region_name is not None:
        k, _ = _pick_best_key({k: Path(k or ".") for k in available}, region_name)
        if k is not None:
            selected_key = k

    if selected_key is None:
        if len(available) == 1:
            selected_key = available[0]
        elif "" in available:
            selected_key = ""
        elif detected_map:
            selected_key = sorted(detected_map.keys(), key=_natural_sort_key)[0]
        elif cell_by_gene_map:
            selected_key = sorted(cell_by_gene_map.keys(), key=_natural_sort_key)[0]
        elif cell_metadata_map:
            selected_key = sorted(cell_metadata_map.keys(), key=_natural_sort_key)[0]

    if selected_key is None:
        raise FileNotFoundError(
            f"Could not find any MERFISH tables under {assets['root']}. Expected files like detected_transcripts_*.csv or cell_by_gene_*.csv."
        )

    detected_path = detected_map.get(selected_key)
    cell_by_gene_path = cell_by_gene_map.get(selected_key)
    cell_metadata_path = cell_metadata_map.get(selected_key)

    vzg_key, vzg_path = _pick_best_key(vzg_map, selected_key if selected_key else region_name)
    if vzg_path is None and len(vzg_map) == 1:
        vzg_key, vzg_path = next(iter(vzg_map.items()))

    return {
        "group_key": selected_key,
        "group_label": selected_key if selected_key not in (None, "") else assets["root"].name,
        "detected_transcripts_path": detected_path,
        "cell_by_gene_path": cell_by_gene_path,
        "cell_metadata_path": cell_metadata_path,
        "vzg_path": vzg_path,
        "vzg_key": vzg_key,
    }


def _guess_cell_id_column(df: pd.DataFrame) -> Optional[str]:
    alias_groups = [
        ["entityid", "entity_id", "cellid", "cell_id", "cell", "featureid", "feature_id", "id"],
    ]
    norm_to_col = {_normalize_token(c): c for c in df.columns}
    for aliases in alias_groups:
        for alias in aliases:
            if alias in norm_to_col:
                return norm_to_col[alias]
    return None


def _canonicalize_detected_transcripts(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    out = df.copy()
    col_map: Dict[str, str] = {}
    norm_to_col = {_normalize_token(c): c for c in out.columns}

    def choose(name: str, aliases: Sequence[str], required: bool = False) -> Optional[str]:
        for alias in aliases:
            if alias in norm_to_col:
                src = norm_to_col[alias]
                col_map[src] = name
                return src
        if required:
            raise KeyError(f"Could not find required transcript column for '{name}'. Available columns: {list(out.columns)}")
        return None

    gene_col = choose("gene", ["gene", "genename", "name"], required=True)
    fov_col = choose("fov", ["fov", "fieldofview", "fieldofviewindex"], required=False)
    gx_col = choose("global_x", ["globalx", "global_x", "centerx", "globalmicronx"], required=False)
    gy_col = choose("global_y", ["globaly", "global_y", "centery", "globalmicrony"], required=False)
    gz_col = choose("global_z", ["globalz", "global_z", "z", "zindex", "globalmicronz"], required=False)
    x_col = choose("x", ["x", "pixelx"], required=False)
    y_col = choose("y", ["y", "pixely"], required=False)
    transcript_id_col = choose("transcript_id", ["transcriptid", "transcript_id"], required=False)
    entity_col = choose(
        "EntityID",
        ["entityid", "entity_id", "cellid", "cell_id", "cell", "featureid", "feature_id"],
        required=False,
    )

    keep_cols = [c for c in [gene_col, fov_col, gx_col, gy_col, gz_col, x_col, y_col, transcript_id_col, entity_col] if c is not None]
    out = out[keep_cols].rename(columns={src: dst for src, dst in col_map.items()})

    out["gene"] = out["gene"].astype(str)
    if "fov" not in out.columns:
        out["fov"] = 0
    if "EntityID" in out.columns:
        out["EntityID"] = out["EntityID"].astype(str)

    for c in ["global_x", "global_y", "global_z", "x", "y"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "fov" in out.columns:
        out["fov"] = pd.to_numeric(out["fov"], errors="coerce").fillna(-1).astype(int)

    return out, {v: k for k, v in col_map.items()}


def _is_control_gene(name: str) -> bool:
    token = _normalize_token(name)
    return token.startswith("blank") or token.startswith("negcontrol") or token.startswith("antisense")


def _prepare_cell_by_gene(path: Path, dtype: str = "int32", sparse: bool = True) -> AnnData:
    df = _read_table_with_auto_sep(path, index_col=0)
    df = df.loc[:, [str(c).strip() != "" and not str(c).lower().startswith("unnamed") for c in df.columns]]
    df.index = df.index.astype(str)
    df.columns = pd.Index([str(c) for c in df.columns], dtype="object")

    arr = df.to_numpy(dtype=np.dtype(dtype), copy=False)
    X = csr_matrix(arr) if sparse else arr
    adata = ad.AnnData(X=X)
    adata.obs_names = pd.Index(df.index.astype(str), dtype="object")
    adata.var_names = pd.Index(df.columns.astype(str), dtype="object")
    adata.obs["cell_id"] = adata.obs_names.astype(str)
    adata.var["gene"] = adata.var_names.astype(str)
    return adata


def _prepare_cell_metadata(path: Path) -> pd.DataFrame:
    meta = _read_table_with_auto_sep(path)
    id_col = _guess_cell_id_column(meta)

    if id_col is None:
        # Common MERFISH export pattern: the cell/entity id is saved as the
        # first unnamed CSV column (pandas-written index), e.g. ``Unnamed: 0``.
        unnamed_candidates = [c for c in meta.columns if _normalize_token(c).startswith("unnamed")]
        ordered_candidates: List[str] = []
        ordered_candidates.extend(unnamed_candidates)
        if len(meta.columns) > 0:
            ordered_candidates.append(meta.columns[0])

        seen: set[str] = set()
        for cand in ordered_candidates:
            if cand in seen:
                continue
            seen.add(cand)
            series = meta[cand]
            if series.isna().any():
                continue
            values = series.astype(str)
            if values.nunique(dropna=True) == len(values):
                id_col = cand
                break

    if id_col is None:
        if meta.index.name is not None and _normalize_token(meta.index.name) not in ("", "index"):
            meta = meta.reset_index()
            id_col = meta.columns[0]
        else:
            # Final fallback: re-read with the first column as index. This helps
            # when a CSV was written with ``index=True`` and the id column was not
            # given an explicit header.
            try:
                meta_idx = _read_table_with_auto_sep(path, index_col=0)
                if len(meta_idx.index) > 0 and pd.Index(meta_idx.index.astype(str)).nunique() == len(meta_idx.index):
                    meta = meta_idx.reset_index()
                    id_col = meta.columns[0]
            except Exception:
                pass

    if id_col is None:
        raise KeyError(
            f"Could not identify cell/entity id column in {path.name}. Available columns: {list(meta.columns)}"
        )

    meta = meta.copy()
    meta[id_col] = meta[id_col].astype(str)
    meta = meta.drop_duplicates(subset=[id_col]).set_index(id_col)

    rename_map = {}
    norm_to_col = {_normalize_token(c): c for c in meta.columns}
    for aliases, target in [
        (["centerx", "center_x"], "center_x"),
        (["centery", "center_y"], "center_y"),
        (["centerz", "center_z"], "center_z"),
        (["volume"], "volume"),
        (["transcriptcount", "transcript_count"], "transcript_count"),
        (["fov"], "fov"),
        (["minx", "min_x"], "min_x"),
        (["maxx", "max_x"], "max_x"),
        (["miny", "min_y"], "min_y"),
        (["maxy", "max_y"], "max_y"),
    ]:
        for alias in aliases:
            if alias in norm_to_col:
                rename_map[norm_to_col[alias]] = target
                break
    meta = meta.rename(columns=rename_map)
    return meta


def _attach_cell_metadata(adata: AnnData, meta: Optional[pd.DataFrame], sample_key: str) -> None:
    if meta is None:
        adata.obs["sample"] = sample_key
        return

    adata.obs = adata.obs.join(meta, how="left")
    adata.obs["sample"] = sample_key
    adata.obs["cell_id"] = adata.obs_names.astype(str)

    if "fov" in adata.obs.columns:
        adata.obs["fov"] = adata.obs["fov"]

    if "center_x" in adata.obs.columns and "center_y" in adata.obs.columns:
        xy = np.c_[pd.to_numeric(adata.obs["center_x"], errors="coerce"), pd.to_numeric(adata.obs["center_y"], errors="coerce")]
        adata.obsm["spatial"] = xy.astype(float, copy=False)


def _aggregate_transcripts_to_adata(
    transcripts: pd.DataFrame,
    *,
    sample_key: str,
    sparse: bool = True,
    dtype: str = "int32",
) -> Tuple[AnnData, Dict[str, Any]]:
    if len(transcripts) == 0:
        adata = ad.AnnData(X=csr_matrix((0, 0), dtype=np.dtype(dtype)) if sparse else np.zeros((0, 0), dtype=np.dtype(dtype)))
        adata.obs["sample"] = []
        return adata, {"aggregation": "empty"}

    if "EntityID" in transcripts.columns:
        valid = transcripts["EntityID"].astype(str).str.lower() != "-1"
        if valid.any():
            unit_col = "EntityID"
            aggregation = "cell"
        else:
            unit_col = "fov"
            aggregation = "fov"
    elif "fov" in transcripts.columns:
        unit_col = "fov"
        aggregation = "fov"
    else:
        unit_col = None
        aggregation = "sample"

    if unit_col is None:
        grouped = transcripts.groupby(["gene"], observed=True).size().rename("count").reset_index()
        unit_index = pd.Index([sample_key], dtype="object")
        gene_index = pd.Index(sorted(grouped["gene"].astype(str).unique(), key=_natural_sort_key), dtype="object")
        gene_codes = gene_index.get_indexer(grouped["gene"].astype(str))
        X = np.zeros((1, len(gene_index)), dtype=np.dtype(dtype))
        X[0, gene_codes] = grouped["count"].to_numpy(dtype=np.dtype(dtype), copy=False)
        adata = ad.AnnData(X=csr_matrix(X) if sparse else X)
        adata.obs_names = unit_index
        adata.var_names = gene_index
        adata.obs["sample"] = sample_key
        return adata, {"aggregation": aggregation}

    work = transcripts.copy()
    work[unit_col] = work[unit_col].astype(str)
    grouped = work.groupby([unit_col, "gene"], observed=True).size().rename("count").reset_index()
    obs_index = pd.Index(sorted(grouped[unit_col].astype(str).unique(), key=_natural_sort_key), dtype="object")
    var_index = pd.Index(sorted(grouped["gene"].astype(str).unique(), key=_natural_sort_key), dtype="object")

    row_codes = obs_index.get_indexer(grouped[unit_col].astype(str))
    col_codes = var_index.get_indexer(grouped["gene"].astype(str))
    data = grouped["count"].to_numpy(dtype=np.dtype(dtype), copy=False)

    X_coo = coo_matrix((data, (row_codes, col_codes)), shape=(len(obs_index), len(var_index)), dtype=np.dtype(dtype))
    X = X_coo.tocsr() if sparse else X_coo.toarray()

    adata = ad.AnnData(X=X)
    adata.obs_names = obs_index
    adata.var_names = var_index
    adata.obs["sample"] = sample_key
    adata.var["gene"] = adata.var_names.astype(str)

    if aggregation == "cell":
        adata.obs["cell_id"] = adata.obs_names.astype(str)
    elif aggregation == "fov":
        adata.obs["fov"] = adata.obs_names.astype(str)

    if aggregation == "cell" and {"global_x", "global_y"}.issubset(transcripts.columns):
        centers = (
            transcripts.loc[transcripts[unit_col].astype(str).str.lower() != "-1"]
            .groupby(unit_col, observed=True)[["global_x", "global_y"]]
            .mean()
        )
        centers.index = centers.index.astype(str)
        adata.obs = adata.obs.join(centers.rename(columns={"global_x": "center_x", "global_y": "center_y"}), how="left")
        if "center_x" in adata.obs.columns and "center_y" in adata.obs.columns:
            adata.obsm["spatial"] = np.c_[adata.obs["center_x"].to_numpy(), adata.obs["center_y"].to_numpy()]

    return adata, {"aggregation": aggregation}


def _compact_transcripts_table(transcripts: pd.DataFrame, limit: Optional[int] = 2_000_000) -> pd.DataFrame:
    if limit is not None and len(transcripts) > limit:
        return transcripts.iloc[: int(limit)].copy()
    return transcripts.copy()


def _load_transform_matrix(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    try:
        df = _read_table_with_auto_sep(path, header=None)
        arr = df.to_numpy(dtype=float, copy=False)
        return arr
    except Exception:
        try:
            return np.loadtxt(path, delimiter=",")
        except Exception:
            return None


def _discover_mosaic_images(images_dir: Optional[Path]) -> Dict[str, List[Tuple[int, Path]]]:
    out: Dict[str, List[Tuple[int, Path]]] = {}
    if images_dir is None or not images_dir.exists():
        return out
    for p in sorted([p for p in images_dir.rglob("*") if p.is_file()], key=_natural_sort_key):
        lower = p.name.lower()
        if not any(lower.endswith(sfx) for sfx in _IMAGE_SUFFIXES):
            continue
        stem = _strip_known_suffixes(p)
        m = _MOSAIC_RE.match(stem)
        if not m:
            continue
        channel = m.group("channel")
        z = int(m.group("z"))
        out.setdefault(channel, []).append((z, p))
    for channel in list(out.keys()):
        out[channel] = sorted(out[channel], key=lambda x: (x[0], _natural_sort_key(x[1].name)))
    return out


def _resolve_selected_z(items: List[Tuple[int, Path]], z_layers: Optional[Union[int, Sequence[int]]]) -> List[Tuple[int, Path]]:
    if z_layers is None:
        return []
    if isinstance(z_layers, (int, np.integer)):
        wanted = [int(z_layers)]
    else:
        wanted = [int(z) for z in z_layers]

    available = [z for z, _ in items]
    selected = [(z, p) for z, p in items if z in set(wanted)]
    if selected:
        return selected

    if len(wanted) == 1 and available:
        chosen_z = sorted(available)[len(available) // 2]
        warnings.warn(
            f"Requested z-layer {wanted[0]} not found in available layers {available}. Falling back to middle available layer z{chosen_z}.",
            stacklevel=2,
        )
        return [(z, p) for z, p in items if z == chosen_z]

    raise ValueError(f"Requested z-layers {wanted} not found. Available layers: {available}")


def _read_one_image(path: Path):
    if tifffile is None:
        raise ImportError("tifffile is required to load MERFISH mosaic images.")
    with _silence_tifffile_logger():
        try:
            with tifffile.TiffFile(path, is_mmstack=False) as tif:
                return tif.series[0].asarray()
        except Exception:
            with _silence_tifffile_logger():
                return tifffile.imread(path)


def _attach_images(
    adata: AnnData,
    *,
    sample_key: str,
    images_dir: Optional[Path],
    transform_path: Optional[Path],
    mosaic_images: bool,
    z_layers: Optional[Union[int, Sequence[int]]],
) -> None:
    spatial_slot = adata.uns.setdefault("spatial", {}).setdefault(sample_key, {})
    spatial_slot.setdefault("metadata", {})
    spatial_slot["metadata"]["images_dir"] = str(images_dir) if images_dir is not None else None
    spatial_slot["metadata"]["micron_to_mosaic_pixel_transform"] = str(transform_path) if transform_path is not None else None

    transform = _load_transform_matrix(transform_path)
    if transform is not None:
        spatial_slot["transform_micron_to_mosaic_pixel"] = transform

    channel_map = _discover_mosaic_images(images_dir)
    if not channel_map:
        return

    spatial_slot.setdefault("image_files", {})
    if mosaic_images and z_layers is not None:
        spatial_slot.setdefault("images", {})

    for channel, items in channel_map.items():
        spatial_slot["image_files"][channel] = {f"z{z}": str(p) for z, p in items}
        if not mosaic_images or z_layers is None:
            continue
        chosen = _resolve_selected_z(items, z_layers=z_layers)
        if not chosen:
            continue
        arrays = []
        zs = []
        for z, p in chosen:
            try:
                arrays.append(_read_one_image(p))
                zs.append(z)
            except Exception as exc:
                warnings.warn(f"Failed to load image {p.name}: {exc}", stacklevel=2)
        if not arrays:
            continue
        if len(arrays) == 1:
            spatial_slot["images"][channel] = arrays[0]
            spatial_slot.setdefault("image_z", {})[channel] = zs
        else:
            same_shape = len({tuple(np.shape(a)) for a in arrays}) == 1
            spatial_slot["images"][channel] = np.stack(arrays, axis=0) if same_shape else {f"z{z}": a for z, a in zip(zs, arrays)}
            spatial_slot.setdefault("image_z", {})[channel] = zs


_ID_ALIASES = ["entityid", "entity_id", "cellid", "cell_id", "cell", "featureid", "feature_id", "id"]
_X_ALIASES = ["x", "vertexx", "coordx", "globalx", "centerx"]
_Y_ALIASES = ["y", "vertexy", "coordy", "globaly", "centery"]
_Z_ALIASES = ["z", "zindex", "z_index", "globalz", "plane", "slice"]
_FOV_ALIASES = ["fov", "fieldofview"]


def _pick_alias(norm_map: Mapping[str, str], aliases: Sequence[str]) -> Optional[str]:
    for alias in aliases:
        if alias in norm_map:
            return norm_map[alias]
    return None


def _maybe_decode_h5_value(ds: h5py.Dataset):
    try:
        value = ds[()]
    except Exception:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value
    return value


def _table_from_structured_array(arr: np.ndarray, source_file: str, source_group: str) -> Optional[pd.DataFrame]:
    if arr.dtype.names is None:
        return None
    norm_map = {_normalize_token(n): n for n in arr.dtype.names}
    x_col = _pick_alias(norm_map, _X_ALIASES)
    y_col = _pick_alias(norm_map, _Y_ALIASES)
    id_col = _pick_alias(norm_map, _ID_ALIASES)
    if x_col is None or y_col is None or id_col is None:
        return None

    data = {
        "x": np.asarray(arr[x_col]).reshape(-1),
        "y": np.asarray(arr[y_col]).reshape(-1),
        "cell_id": np.asarray(arr[id_col]).reshape(-1).astype(str),
        "source_file": source_file,
        "source_group": source_group,
    }

    z_col = _pick_alias(norm_map, _Z_ALIASES)
    if z_col is not None:
        data["z"] = np.asarray(arr[z_col]).reshape(-1)
    fov_col = _pick_alias(norm_map, _FOV_ALIASES)
    if fov_col is not None:
        data["fov"] = np.asarray(arr[fov_col]).reshape(-1)
    return pd.DataFrame(data)


def _table_from_dataset_group(group: h5py.Group, source_file: str, source_group: str) -> Optional[pd.DataFrame]:
    ds_map: Dict[str, h5py.Dataset] = {k: v for k, v in group.items() if isinstance(v, h5py.Dataset)}
    if not ds_map:
        return None
    norm_map = {_normalize_token(k): k for k in ds_map.keys()}
    x_name = _pick_alias(norm_map, _X_ALIASES)
    y_name = _pick_alias(norm_map, _Y_ALIASES)
    id_name = _pick_alias(norm_map, _ID_ALIASES)
    if x_name is None or y_name is None or id_name is None:
        return None

    try:
        x = np.asarray(ds_map[x_name][()]).reshape(-1)
        y = np.asarray(ds_map[y_name][()]).reshape(-1)
        cell_id = np.asarray(ds_map[id_name][()]).reshape(-1).astype(str)
    except Exception:
        return None
    if not (len(x) == len(y) == len(cell_id)):
        return None

    data: Dict[str, Any] = {
        "x": x,
        "y": y,
        "cell_id": cell_id,
        "source_file": source_file,
        "source_group": source_group,
    }
    z_name = _pick_alias(norm_map, _Z_ALIASES)
    if z_name is not None:
        try:
            z = np.asarray(ds_map[z_name][()]).reshape(-1)
            if len(z) == len(x):
                data["z"] = z
        except Exception:
            pass
    fov_name = _pick_alias(norm_map, _FOV_ALIASES)
    if fov_name is not None:
        try:
            fov = np.asarray(ds_map[fov_name][()]).reshape(-1)
            if len(fov) == len(x):
                data["fov"] = fov
        except Exception:
            pass
    return pd.DataFrame(data)


def _parse_boundary_hdf5_file(path: Path) -> List[pd.DataFrame]:
    tables: List[pd.DataFrame] = []
    with h5py.File(path, "r") as f:
        def visit(group: h5py.Group, gpath: str) -> None:
            table = _table_from_dataset_group(group, source_file=path.name, source_group=gpath)
            if table is not None and len(table) > 0:
                tables.append(table)

            for key, item in group.items():
                item_path = f"{gpath}/{key}" if gpath != "/" else f"/{key}"
                if isinstance(item, h5py.Dataset):
                    try:
                        arr = item[()]
                    except Exception:
                        continue
                    if isinstance(arr, np.ndarray) and arr.dtype.names is not None:
                        table = _table_from_structured_array(arr, source_file=path.name, source_group=item_path)
                        if table is not None and len(table) > 0:
                            tables.append(table)
                elif isinstance(item, h5py.Group):
                    visit(item, item_path)

        visit(f, "/")
    return tables


def _load_cell_boundaries(boundary_files: Sequence[Path]) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    if not boundary_files:
        return None, {"n_boundary_files": 0, "n_polygon_rows": 0, "files": []}

    tables: List[pd.DataFrame] = []
    for path in boundary_files:
        try:
            tables.extend(_parse_boundary_hdf5_file(path))
        except Exception as exc:
            warnings.warn(f"Failed to parse boundary file {path.name}: {exc}", stacklevel=2)

    if not tables:
        return None, {
            "n_boundary_files": len(boundary_files),
            "n_polygon_rows": 0,
            "files": [p.name for p in boundary_files],
            "parsed": False,
        }

    poly = pd.concat(tables, ignore_index=True)
    poly["cell_id"] = poly["cell_id"].astype(str)
    for c in ["x", "y", "z"]:
        if c in poly.columns:
            poly[c] = pd.to_numeric(poly[c], errors="coerce")

    meta = {
        "n_boundary_files": len(boundary_files),
        "n_polygon_rows": int(len(poly)),
        "files": [p.name for p in boundary_files],
        "parsed": True,
    }
    return poly, meta


def _maybe_add_centroids_from_polygons(adata: AnnData, polygons: Optional[pd.DataFrame]) -> None:
    if polygons is None or "spatial" in adata.obsm:
        return
    if not {"cell_id", "x", "y"}.issubset(polygons.columns):
        return
    centers = polygons.groupby("cell_id", observed=True)[["x", "y"]].mean()
    centers.index = centers.index.astype(str)
    tmp = adata.obs.join(centers.rename(columns={"x": "center_x", "y": "center_y"}), how="left")
    if tmp[["center_x", "center_y"]].notna().any().any():
        adata.obs["center_x"] = tmp["center_x"]
        adata.obs["center_y"] = tmp["center_y"]
        adata.obsm["spatial"] = np.c_[pd.to_numeric(adata.obs["center_x"], errors="coerce"), pd.to_numeric(adata.obs["center_y"], errors="coerce")]


def _empty_adata() -> AnnData:
    return ad.AnnData(X=csr_matrix((0, 0), dtype=np.int32))


@register_function(
    aliases=["read_merfish", "merfish", "MERSCOPE", "Vizgen MERFISH", "读取merfish", "read_merscope"],
    category="io",
    description="Read MERFISH / Vizgen MERSCOPE outputs from cell_by_gene, cell_metadata, detected_transcripts, cell_boundaries, and optional mosaic images.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "adata = st.io.spatial.read_merfish(",
        "    'Merfish_dataset',",
        "    counts_file='cell_by_gene_S1R1.csv',",
        "    meta_file='cell_metadata_S1R1.csv',",
        ")",
        "adata = st.io.spatial.read_merfish(",
        "    'Merfish_dataset',",
        "    counts_file='cell_by_gene_S1R1.csv',",
        "    meta_file='cell_metadata_S1R1.csv',",
        "    load_images=False,",
        ")",
    ],
    related=["io.spatial.read_seqfish_plus", "io.spatial.read_starmap_plus", "io.spatial.read_slideseq"],
)
def read_merfish(
    path: Union[str, Path],
    *,
    counts_file: str,
    meta_file: str,
    load_images: bool = True,
    z_layer: Optional[int] = 3,
    load_boundaries: bool = True,
) -> AnnData:
    """
    Read a MERFISH / Vizgen MERSCOPE dataset folder.

    Parameters
    ----------
    path : str or Path
        MERFISH dataset directory.
    counts_file : str
        Cell-by-gene count matrix filename relative to ``path``.
    meta_file : str
        Cell metadata filename relative to ``path``.
    load_images : bool, default True
        Whether to load mosaic image arrays into memory. Image file paths are
        still indexed even when this is ``False``.
    z_layer : int or None, default 3
        Z-layer of mosaic images to load. Set to ``None`` to skip loading image
        arrays while still keeping image file paths.
    load_boundaries : bool, default True
        Whether to parse cell boundary polygons from ``cell_boundaries/*.hdf5``.

    Returns
    -------
    anndata.AnnData
        A MERFISH AnnData object.

        ``counts_file`` is loaded as the main count matrix and ``meta_file`` is
        joined into ``obs``. Optional transcript table, boundaries, image
        paths, images, and transform matrix are stored under
        ``uns['spatial'][sample_key]``.
    """
    root = Path(path).resolve()
    assets = _discover_merfish_assets(root)
    counts_path = root / counts_file
    meta_path = root / meta_file
    if not counts_path.exists():
        raise FileNotFoundError(f"Counts file not found: {counts_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    def _match_group_key(mapping: Mapping[str, Path], target: Path, prefix: str) -> Optional[str]:
        for key, candidate in mapping.items():
            if candidate.resolve() == target.resolve():
                return key
        return _parse_group_suffix(target, prefix)

    counts_key = _match_group_key(assets["cell_by_gene_map"], counts_path, "cell_by_gene")
    meta_key = _match_group_key(assets["cell_metadata_map"], meta_path, "cell_metadata")
    if counts_key is not None and meta_key is not None and counts_key != meta_key:
        raise ValueError(
            f"`counts_file` and `meta_file` appear to belong to different MERFISH groups: "
            f"{counts_key!r} vs {meta_key!r}."
        )

    group_key = counts_key if counts_key is not None else meta_key
    group_label = group_key if group_key not in (None, "") else assets["root"].name
    sample_key = str(group_label or assets["root"].name)
    _progress(f"Reading MERFISH data from: {assets['root']}")
    _progress(f"Region key: {sample_key}")

    detected_path = assets["detected_map"].get(group_key)
    vzg_key, vzg_path = _pick_best_key(assets["vzg_map"], group_key)
    if vzg_path is None and len(assets["vzg_map"]) == 1:
        vzg_key, vzg_path = next(iter(assets["vzg_map"].items()))

    transcripts_df: Optional[pd.DataFrame] = None
    transcripts_meta: Dict[str, Any] = {}
    if detected_path is not None and detected_path.exists():
        _progress(f"Loading transcript detections: {detected_path.name}")
        raw = _read_table_with_auto_sep(detected_path)
        transcripts_df, colmap = _canonicalize_detected_transcripts(raw)
        transcripts_meta = {
            "path": str(detected_path),
            "columns": list(transcripts_df.columns),
            "column_mapping": colmap,
            "n_rows": int(len(transcripts_df)),
            "n_genes": int(transcripts_df["gene"].nunique()) if len(transcripts_df) else 0,
            "n_fovs": int(transcripts_df["fov"].nunique()) if "fov" in transcripts_df.columns and len(transcripts_df) else 0,
        }
    elif detected_path is None:
        warnings.warn(
            f"No detected_transcripts file was found for group '{sample_key}'.",
            stacklevel=2,
        )

    _progress(f"Loading cell-by-gene matrix: {counts_path.name}")
    adata = _prepare_cell_by_gene(counts_path, dtype="int32", sparse=True)
    aggregation = "cell_table"

    _progress(f"Loading cell metadata: {meta_path.name}")
    meta = _prepare_cell_metadata(meta_path)
    _attach_cell_metadata(adata, meta, sample_key=sample_key)

    #Set spateo keys
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    _progress(f"Set Spadeo-specific key values:adata.uns['__type'] and adata.uns['pp']",level='step')

    adata.uns.setdefault("spatial", {})
    spatial_slot = adata.uns["spatial"].setdefault(sample_key, {})
    spatial_slot["metadata"] = {
        "platform": "MERFISH",
        "vendor": "Vizgen MERSCOPE",
        "sample": sample_key,
        "root": str(assets["root"]),
        "group_key": group_key,
        "group_label": group_label,
        "detected_transcripts_file": detected_path.name if detected_path is not None else None,
        "cell_by_gene_file": counts_path.name,
        "cell_metadata_file": meta_path.name,
        "vzg_file": vzg_path.name if vzg_path is not None else None,
        "aggregation": aggregation,
        "images_dir": str(assets["images_dir"]) if assets["images_dir"] is not None else None,
        "cell_boundaries_dir": str(assets["boundaries_dir"]) if assets["boundaries_dir"] is not None else None,
        "z_layers_requested": None if z_layer is None else [int(z_layer)],
    }

    if transcripts_df is not None:
        spatial_slot["transcripts"] = _compact_transcripts_table(transcripts_df, limit=2_000_000)
        spatial_slot["metadata"]["transcripts_summary"] = transcripts_meta

    if load_boundaries:
        _progress("Loading cell boundaries")
        polygons, poly_meta = _load_cell_boundaries(assets["boundary_files"])
        spatial_slot["boundary_files"] = [str(p) for p in assets["boundary_files"]]
        spatial_slot["metadata"]["cell_boundaries_summary"] = poly_meta
        if polygons is not None:
            spatial_slot["polygons"] = polygons
            _maybe_add_centroids_from_polygons(adata, polygons)
    else:
        spatial_slot["boundary_files"] = [str(p) for p in assets["boundary_files"]]

    _attach_images(
        adata,
        sample_key=sample_key,
        images_dir=assets["images_dir"],
        transform_path=assets["transform_path"],
        mosaic_images=load_images,
        z_layers=z_layer,
    )

    adata.uns.setdefault("spateo_io", {})
    adata.uns["spateo_io"].update(
        {
            "type": "merfish",
            "sample": sample_key,
            "spatial_key": "spatial",
            "aggregation": aggregation,
        }
    )

    _progress(f"Done. Loaded MERFISH dataset with shape {adata.shape}", level="success")
    return adata
