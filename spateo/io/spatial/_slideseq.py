"""
Data reading functions for Spateo.

This module provides functions for reading Slide-seq / Slide-seqV2 style
outputs, including bead-level count matrices, bead coordinates, and
optional microscopy images.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from PIL import Image
from scipy.sparse import csr_matrix

from ..._registry import register_function
from ...configuration import SKM

try:
    from ..._settings import Colors
except Exception:
    class Colors:
        HEADER = '\033[95m'
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'


_DEFAULT_COUNTS_CANDIDATES = (
    "MappedDGEForR.csv",
    "MappedDGEForR.csv.gz",
    "mapped_dge_for_r.csv",
    "mapped_dge_for_r.csv.gz",
)

_DEFAULT_BEAD_CANDIDATES = (
    "BeadLocationsForR.csv",
    "BeadLocationsForR.csv.gz",
    "BeadLoacationsForR.csv",  # 常见拼写变体/笔误
    "BeadLoacationsForR.csv.gz",
    "bead_locations_for_r.csv",
    "bead_locations_for_r.csv.gz",
)

_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


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
    text = f"[SlideSeq]{tag} {message}"

    force_color = os.environ.get("FORCE_COLOR", "").strip() not in ("", "0", "false", "False")
    no_color = os.environ.get("NO_COLOR", "").strip() != ""
    supports_color = force_color or (hasattr(sys.stdout, "isatty") and sys.stdout.isatty())

    if no_color or not supports_color:
        print(text)
    else:
        print(f"{color}{text}{Colors.ENDC}")


def _infer_sample_name(path: Path) -> str:
    return path.name


def _read_table_with_auto_sep(path: Path, **kwargs) -> pd.DataFrame:
    suffixes = set(path.suffixes)

    if ".parquet" in suffixes:
        return pd.read_parquet(path, **kwargs)

    if ".csv" in suffixes or ".gz" in suffixes:
        try:
            return pd.read_csv(path, sep=",", **kwargs)
        except Exception:
            return pd.read_csv(path, sep="\t", **kwargs)

    try:
        return pd.read_csv(path, sep=",", **kwargs)
    except Exception:
        try:
            return pd.read_csv(path, sep="\t", **kwargs)
        except Exception:
            return pd.read_csv(path, sep=None, engine="python", **kwargs)


def _find_first_existing(root: Path, candidates: Sequence[str]) -> Optional[Path]:
    expanded = []
    for rel in candidates:
        if rel is None:
            continue
        rel = str(rel)
        expanded.append(rel)

        if rel.endswith(".csv"):
            expanded.append(rel + ".gz")
        elif rel.endswith(".csv.gz"):
            expanded.append(rel[:-3])

    seen = set()
    for rel in expanded:
        if rel in seen:
            continue
        seen.add(rel)
        p = root / rel
        if p.exists():
            return p
    return None


def _normalize_names(index_like) -> pd.Index:
    vals = []
    for x in index_like:
        if pd.isna(x):
            vals.append(None)
        else:
            vals.append(str(x).strip())
    return pd.Index(vals, dtype="object")


def _drop_empty_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = []
    for col in df.columns:
        col_str = str(col).strip()
        is_unnamed = col_str == "" or col_str.lower().startswith("unnamed:")
        series = df[col]
        if is_unnamed and series.isna().all():
            continue
        keep_cols.append(col)
    return df.loc[:, keep_cols].copy()


def _name_like_columns() -> List[str]:
    return [
        "barcode", "barcodes", "Barcode", "Barcodes",
        "bead_barcode", "bead_barcodes", "bead", "Bead",
        "NAME", "name", "Name",
    ]


def _guess_name_col(df: pd.DataFrame) -> Optional[str]:
    for col in _name_like_columns():
        if col in df.columns:
            return col
    return None


def _find_coord_cols(df: pd.DataFrame) -> List[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    x_candidates = ["xcoord", "x", "imagecol", "pxl_col", "pxl_col_in_fullres"]
    y_candidates = ["ycoord", "y", "imagerow", "pxl_row", "pxl_row_in_fullres"]

    x_col = next((lower_map[k] for k in x_candidates if k in lower_map), None)
    y_col = next((lower_map[k] for k in y_candidates if k in lower_map), None)

    if x_col is None or y_col is None:
        return []
    return [x_col, y_col]


def _set_name_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    name_col = _guess_name_col(df)
    if name_col is not None:
        df[name_col] = df[name_col].astype(str)
        if df[name_col].duplicated().any():
            warnings.warn(
                f"Found duplicated bead/barcode names in column '{name_col}'. "
                "Keeping the first occurrence."
            )
            df = df.drop_duplicates(subset=[name_col], keep="first")
        df = df.set_index(name_col)
    else:
        df.index = df.index.astype(str)
    return df


def _resolve_counts_file(root: Path, counts_file: Optional[str] = None) -> Path:
    if counts_file is not None:
        p = root / counts_file
        if not p.exists():
            raise FileNotFoundError(f"Counts file not found: {p}")
        return p

    p = _find_first_existing(root, _DEFAULT_COUNTS_CANDIDATES)
    if p is None:
        raise FileNotFoundError(
            f"Could not find Slide-seq counts file under {root}. "
            f"Expected one of: {list(_DEFAULT_COUNTS_CANDIDATES)}"
        )
    return p


def _resolve_bead_file(root: Path, bead_file: Optional[str] = None) -> Path:
    if bead_file is not None:
        p = root / bead_file
        if not p.exists():
            raise FileNotFoundError(f"Bead file not found: {p}")
        return p

    p = _find_first_existing(root, _DEFAULT_BEAD_CANDIDATES)
    if p is None:
        raise FileNotFoundError(
            f"Could not find Slide-seq bead file under {root}. "
            f"Expected one of: {list(_DEFAULT_BEAD_CANDIDATES)}"
        )
    return p


def _read_slideseq_counts(
    path: Path,
    dtype: str = "int32",
    make_sparse: bool = True,
) -> AnnData:
    df = _read_table_with_auto_sep(path)
    df = _drop_empty_unnamed_columns(df)

    if df.shape[1] < 2:
        raise ValueError(f"Counts file has too few columns: {path}")

    gene_names = pd.Index(df.iloc[:, 0].astype(str).str.strip(), dtype="object")
    expr = df.iloc[:, 1:].copy()

    # Drop empty trailing columns and unnamed columns introduced by CSV formatting.
    valid_cols = []
    for col in expr.columns:
        col_str = str(col).strip()
        if (col_str == "" or col_str.lower().startswith("unnamed:")) and expr[col].isna().all():
            continue
        valid_cols.append(col)
    expr = expr.loc[:, valid_cols]

    obs_names = _normalize_names(expr.columns)
    expr.columns = obs_names

    if obs_names.duplicated().any():
        dup_names = obs_names[obs_names.duplicated()].tolist()
        raise ValueError(
            "Duplicated barcode names detected in counts columns. "
            f"Examples: {dup_names[:10]}"
        )

    expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0)
    matrix = expr.to_numpy(dtype=dtype).T  # beads x genes

    if make_sparse:
        X = csr_matrix(matrix)
    else:
        X = matrix

    adata = AnnData(X=X)
    adata.obs_names = obs_names
    adata.var_names = gene_names
    adata.var["gene_symbol"] = adata.var_names.astype(str)
    adata.var_names_make_unique()
    return adata


def _read_slideseq_beads(path: Path) -> pd.DataFrame:
    df = _read_table_with_auto_sep(path)
    df = _drop_empty_unnamed_columns(df)
    df = _set_name_index(df)
    df.index = _normalize_names(df.index)

    coord_cols = _find_coord_cols(df)
    if len(coord_cols) < 2:
        raise ValueError(
            f"Could not detect coordinate columns in {path}. "
            "Expected columns like 'xcoord'/'ycoord' or 'x'/'y'."
        )

    df[coord_cols[0]] = pd.to_numeric(df[coord_cols[0]], errors="coerce")
    df[coord_cols[1]] = pd.to_numeric(df[coord_cols[1]], errors="coerce")

    if df.index.duplicated().any():
        warnings.warn("Found duplicated barcodes in bead table. Keeping the first occurrence.")
        df = df[~df.index.duplicated(keep="first")].copy()

    return df


def _discover_images(root: Path) -> List[Path]:
    image_paths = []
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
            image_paths.append(p)
    return sorted(image_paths)


def _infer_image_key(path: Path, seen: set, total_images: int) -> str:
    stem = path.stem.lower()

    lowres_tokens = ["lowres", "low_res", "thumbnail", "downsample", "small"]
    hires_tokens = ["hires", "highres", "high_res", "fullres", "full_res", "raw"]

    if any(tok in stem for tok in hires_tokens) and "hires" not in seen:
        return "hires"
    if any(tok in stem for tok in lowres_tokens) and "lowres" not in seen:
        return "lowres"
    if total_images == 1 and "hires" not in seen:
        return "hires"

    key = path.stem
    if key not in seen:
        return key

    i = 1
    while f"{key}_{i}" in seen:
        i += 1
    return f"{key}_{i}"


def _read_spatial_images(
    root: Path,
    image_paths: Optional[Sequence[Union[str, Path]]] = None,
    auto_discover_images: bool = True,
) -> Dict[str, np.ndarray]:
    resolved_paths: List[Path] = []

    if image_paths is not None:
        for p in image_paths:
            pp = Path(p)
            if not pp.is_absolute():
                pp = root / pp
            if pp.exists():
                resolved_paths.append(pp)
            else:
                warnings.warn(f"Image file does not exist and will be skipped: {pp}")
    elif auto_discover_images:
        resolved_paths = _discover_images(root)

    images: Dict[str, np.ndarray] = {}
    seen = set()
    total = len(resolved_paths)
    for p in resolved_paths:
        try:
            with Image.open(p) as img:
                arr = np.asarray(img)
            key = _infer_image_key(p, seen=seen, total_images=total)
            images[key] = arr
            seen.add(key)
        except Exception as exc:
            warnings.warn(f"Could not load image '{p}': {exc}")

    return images


def _init_spatial_slot(
    adata: AnnData,
    sample: str,
    images: Optional[Dict[str, np.ndarray]] = None,
    scalefactors: Optional[dict] = None,
) -> None:
    adata.uns.setdefault("spatial", {})
    adata.uns["spatial"][sample] = {}
    if images:
        adata.uns["spatial"][sample]["images"] = images
    adata.uns["spatial"][sample]["scalefactors"] = scalefactors or {}


@register_function(
    aliases=["read_slideseq", "slide-seq", "slide seq", "读取slideseq", "slide-seq reader"],
    category="io",
    description="Read Slide-seq / Slide-seqV2 bead-level outputs and attach bead coordinates plus optional images.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "adata = ov.io.spatial.read_slideseq('slideseq_dir')",
        "adata = ov.io.spatial.read_slideseq('slideseq_dir', load_images=False)",
        "adata = ov.io.spatial.read_slideseq(",
        "    'slideseq_dir',",
        "    counts_file='MappedDGEForR.csv',",
        "    bead_file='BeadLocationsForR.csv',",
        ")",
    ],
    related=["io.spatial.read_visium", "io.spatial.read_visium_hd", "io.spatial.read_starmap_plus"],
)
def read_slideseq(
    path: Union[str, Path],
    *,
    counts_file: str = "MappedDGEForR.csv",
    bead_file: str = "BeadLocationsForR.csv",
    load_images: bool = True,
) -> AnnData:
    """
    Read Slide-seq / Slide-seqV2 bead-level outputs.

    Expected core files under ``path``
    ----------------------------------
    - ``MappedDGEForR.csv``
        Gene-by-bead counts matrix where the first column stores gene names
        and remaining columns are bead barcodes.
    - ``BeadLocationsForR.csv``
        Bead table containing at least barcode + X/Y columns,
        for example ``barcodes,xcoord,ycoord``.

    Parameters
    ----------
    path
        Path to the Slide-seq data directory.
    counts_file
        Counts matrix filename relative to ``path``.
    bead_file
        Bead table filename relative to ``path``.
    load_images
        Whether to load microscopy images under the same folder.

    Returns
    -------
    anndata.AnnData
        Bead-level AnnData with counts matrix, bead coordinates, and optional image metadata.

        - ``X``: bead x gene count matrix
        - ``obs_names``: bead barcodes
        - ``var_names``: gene names
        - ``obs['barcode']``: barcode alias
        - ``obs['xcoord']`` / ``obs['ycoord']``: bead coordinates
        - ``obsm['spatial']``: ``(n_beads, 2)`` spatial coordinates in ``[x, y]`` order
        - ``uns['spatial'][sample]['images']``: optional microscopy images
        - ``uns['spatial'][sample]['metadata']``: file names, coordinate columns, platform info
    """
    root = Path(path).resolve()
    sample = _infer_sample_name(root)

    _progress(f"Reading Slide-seq data from: {root}")
    _progress(f"Sample key: {sample}")

    counts_path = _resolve_counts_file(root, counts_file=counts_file)
    bead_path = _resolve_bead_file(root, bead_file=bead_file)

    _progress(f"Loading counts matrix: {counts_path.name}")
    adata = _read_slideseq_counts(counts_path, dtype="int32", make_sparse=True)

    # Set spateo keys
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    _progress(f"Set Spadeo-specific key values:adata.uns['__type'] and adata.uns['pp']",level='step')

    _progress(f"Loading bead table: {bead_path.name}")
    bead_df = _read_slideseq_beads(bead_path)
    coord_cols = _find_coord_cols(bead_df)
    if len(coord_cols) < 2:
        raise ValueError(
            f"Could not find X/Y coordinate columns in {bead_path}. "
            f"Available columns: {list(bead_df.columns)}"
        )

    common_barcodes = adata.obs_names.intersection(bead_df.index)
    if len(common_barcodes) == 0:
        raise ValueError(
            "No overlapping barcodes between counts matrix and bead table. "
            f"Example matrix barcodes: {list(adata.obs_names[:5])}; "
            f"example bead barcodes: {list(bead_df.index[:5])}"
        )

    if len(common_barcodes) < adata.n_obs:
        warnings.warn(
            f"Only {len(common_barcodes)} / {adata.n_obs} barcodes are shared between counts and bead files. "
            "Subsetting to the intersection."
        )

    adata = adata[common_barcodes, :].copy()
    bead_df = bead_df.loc[common_barcodes].copy()

    adata.obs = adata.obs.join(bead_df, how="left")
    adata.obs["barcode"] = adata.obs_names.astype(str)
    adata.obs["sample"] = sample
    adata.obs["dataset"] = sample

    x_col, y_col = coord_cols[:2]
    x = pd.to_numeric(adata.obs[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(adata.obs[y_col], errors="coerce").to_numpy(dtype=float)

    adata.obs["xcoord"] = x
    adata.obs["ycoord"] = y
    adata.obs["imagecol"] = x
    adata.obs["imagerow"] = y
    adata.obsm["spatial"] = np.column_stack([x, y])

    images = {}
    if load_images:
        _progress("Loading spatial images")
        images = _read_spatial_images(root=root, image_paths=None, auto_discover_images=True)

    _init_spatial_slot(adata, sample=sample, images=images, scalefactors={})
    adata.uns["spatial"][sample]["metadata"] = {
        "platform": "Slide-seq",
        "sample": sample,
        "counts_file": counts_path.name,
        "bead_file": bead_path.name,
        "coord_columns": [x_col, y_col],
        "coordinate_unit": "pixel",
        "n_images": len(images),
        "image_keys": list(images.keys()),
        "spatial_key": "spatial",
        "counts_orientation": "genes_by_beads_in_file -> beads_by_genes_in_adata",
    }

    adata.uns.setdefault("spateo_io", {})
    adata.uns["spateo_io"].update(
        {
            "type": "slideseq",
            "sample": sample,
            "spatial_key": "spatial",
        }
    )
    
    _progress(f"Done (n_obs={adata.n_obs}, n_vars={adata.n_vars})", level="success")
    return adata
