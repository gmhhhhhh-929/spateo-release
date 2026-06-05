
"""
Data reading functions for seqFISH.

This module provides a reader for seqFISH dataset folders organized like::

    dataset/
    ├── SG_MouseKidneyDataRelease_CellCoordinates_section3.csv
    ├── SG_MouseKidneyDataRelease_Counts_section3.csv
    └── images/
        ├── SG_MouseKidneyDataRelease_CellMask_section3.tiff
        └── SG_MouseKidneyDataRelease_DAPI_section3.ome.tiff

The primary matrix is expected to be the cell-by-gene counts table.
Cell metadata coordinates are merged into ``adata.obs`` and ``adata.obsm["spatial"]``.
Cell masks and microscopy images are optional.
"""

from __future__ import annotations

import io
import logging
import os
import re
import sys
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix
from ....configuration import SKM

try:
    import tifffile
except Exception:  # pragma: no cover
    tifffile = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    from dask_image.imread import imread as dask_imread
except Exception:  # pragma: no cover
    dask_imread = None

try:
    import imageio.v3 as iio
except Exception:  # pragma: no cover
    iio = None

from ...._registry import register_function

try:
    from ...._settings import Colors
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
_TABLE_SUFFIXES = (".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt", ".txt.gz")


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
    text = f"[seqFISH]{tag} {message}"

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


@contextmanager
def _silence_image_backend_noise(level: int = logging.ERROR):
    """Silence noisy TIFF backends and warnings during best-effort image reads."""
    logger = logging.getLogger("tifffile")
    old_level = logger.level
    old_propagate = logger.propagate
    stderr_buf = io.StringIO()
    stdout_buf = io.StringIO()
    try:
        logger.setLevel(level)
        logger.propagate = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with redirect_stderr(stderr_buf), redirect_stdout(stdout_buf):
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


def _strip_known_suffixes(path: Path) -> str:
    name = path.name
    lower = name.lower()
    suffixes = tuple(sorted(_TABLE_SUFFIXES + _IMAGE_SUFFIXES, key=len, reverse=True))
    for suffix in suffixes:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _looks_like_table(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(s) for s in _TABLE_SUFFIXES)


def _looks_like_image(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(s) for s in _IMAGE_SUFFIXES)


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


def _guess_cell_id_column(df: pd.DataFrame) -> Optional[str]:
    priority = {
        "cellid", "cell", "cellindex", "entityid", "entity", "id", "label", "celllabel", "cell_id", "cell_index",
        "unnamed0", "index"
    }
    for col in df.columns:
        norm = _normalize_token(col)
        if norm in priority:
            return col
    for col in df.columns:
        series = df[col]
        if series.isna().any():
            continue
        as_str = series.astype(str)
        if as_str.nunique(dropna=True) == len(as_str):
            return col
    return None


def _guess_xy_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    norm_to_col = {_normalize_token(c): c for c in df.columns}

    x_aliases = ("centerx", "x", "xcoord", "coordx", "centroidx", "globalx", "positionx", "pixelx")
    y_aliases = ("centery", "y", "ycoord", "coordy", "centroidy", "globaly", "positiony", "pixely")
    z_aliases = ("centerz", "z", "zcoord", "coordz", "centroidz", "globalz", "positionz", "pixelz")

    x_col = next((norm_to_col[a] for a in x_aliases if a in norm_to_col), None)
    y_col = next((norm_to_col[a] for a in y_aliases if a in norm_to_col), None)
    z_col = next((norm_to_col[a] for a in z_aliases if a in norm_to_col), None)
    return x_col, y_col, z_col


def _guess_gene_column(df: pd.DataFrame) -> Optional[str]:
    aliases = (
        "gene", "genename", "genes", "target", "targetgene", "feature", "featurename", "name"
    )
    norm_to_col = {_normalize_token(c): c for c in df.columns}
    return next((norm_to_col[a] for a in aliases if a in norm_to_col), None)


def _ensure_unique_str_index(values: Sequence[Any], prefix: str) -> pd.Index:
    vals = pd.Index(pd.Series(list(values), dtype="object").astype(str))
    if vals.is_unique:
        return vals
    counts: Dict[str, int] = {}
    new_vals: List[str] = []
    for val in vals.tolist():
        n = counts.get(val, 0)
        if n == 0:
            new_vals.append(val)
        else:
            new_vals.append(f"{val}-{n}")
        counts[val] = n + 1
    return pd.Index(new_vals, name=prefix)


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.loc[:, ~out.columns.astype(str).str.startswith("Unnamed: ") | (out.notna().sum(axis=0) > 0)]
    out = out.dropna(axis=1, how="all")
    return out


def _parse_group_after_match(stem: str, match: re.Match) -> str:
    suffix = stem[match.end():]
    suffix = re.sub(r"^[\s_\-\.]+", "", suffix)
    suffix = re.sub(r"[\s_\-\.]+$", "", suffix)
    return suffix


def _match_role_and_group(path: Path) -> Tuple[Optional[str], str]:
    stem = _strip_known_suffixes(path)
    lower = stem.lower()

    patterns: List[Tuple[str, Sequence[str]]] = [
        ("meta", [r"cell[\s_\-]*coordinates?", r"cell[\s_\-]*coords?", r"metadata", r"meta"]),
        ("counts", [r"c[\s_\-]*x[\s_\-]*g", r"cell[\s_\-]*by[\s_\-]*gene", r"counts?"]),
        ("cell_mask", [r"cell[\s_\-]*mask", r"cell[\s_\-]*labels?", r"seg(?:mentation)?"]),
        ("dapi", [r"dapi"]),
    ]

    for role, regexes in patterns:
        for regex in regexes:
            m = re.search(regex, lower, flags=re.IGNORECASE)
            if m is not None:
                group = _parse_group_after_match(stem, m)
                return role, group

    return None, ""


def _collect_candidate_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(root.iterdir(), key=_natural_sort_key):
        if p.is_file() and (_looks_like_table(p) or _looks_like_image(p)):
            out.append(p)
        elif p.is_dir() and _normalize_token(p.name) == "images":
            for q in sorted(p.iterdir(), key=_natural_sort_key):
                if q.is_file() and _looks_like_image(q):
                    out.append(q)
    return out


def _discover_seqfish_assets(root: Path) -> Dict[str, Dict[str, Any]]:
    if root.is_file():
        root = root.parent
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"seqFISH reader expects a dataset directory: {root}")

    groups: Dict[str, Dict[str, Any]] = {}
    for path in _collect_candidate_files(root):
        role, group = _match_role_and_group(path)
        if role is None:
            continue

        groups.setdefault(group, {"images": {}, "files": []})
        groups[group]["files"].append(path)

        if role in {"cell_mask", "dapi"}:
            groups[group]["images"][role] = path
        else:
            groups[group][role] = path

    return groups


def _standardize_meta(df: pd.DataFrame) -> pd.DataFrame:
    df = _clean_dataframe(df)
    id_col = _guess_cell_id_column(df)
    if id_col is None:
        df = df.copy()
        df.insert(0, "cell_id", np.arange(df.shape[0]).astype(str))
        id_col = "cell_id"

    out = df.copy()
    out[id_col] = out[id_col].astype(str)
    out = out.drop_duplicates(subset=[id_col]).set_index(id_col)

    rename_map = {}
    x_col, y_col, z_col = _guess_xy_columns(out.reset_index())
    if x_col is not None:
        rename_map[x_col] = "center_x"
    if y_col is not None:
        rename_map[y_col] = "center_y"
    if z_col is not None:
        rename_map[z_col] = "center_z"

    norm_to_col = {_normalize_token(c): c for c in out.columns}
    for aliases, target in [
        (["fov"], "fov"),
        (["volume"], "volume"),
        (["area"], "area"),
        (["minx", "min_x"], "min_x"),
        (["maxx", "max_x"], "max_x"),
        (["miny", "min_y"], "min_y"),
        (["maxy", "max_y"], "max_y"),
    ]:
        for alias in aliases:
            if alias in norm_to_col:
                rename_map[norm_to_col[alias]] = target
                break

    out = out.rename(columns=rename_map)
    return out


def _is_likely_numeric_series(series: pd.Series) -> bool:
    try:
        pd.to_numeric(series.dropna().iloc[: min(50, max(1, series.dropna().shape[0]))], errors="raise")
        return True
    except Exception:
        return False


def _prepare_counts(path: Path, expected_ids: Optional[pd.Index] = None) -> AnnData:
    df = _clean_dataframe(_read_table_with_auto_sep(path))
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError(f"Empty counts table: {path}")

    id_col = _guess_cell_id_column(df)
    if id_col is None:
        first_col = df.columns[0]
        if not _is_likely_numeric_series(df[first_col]):
            id_col = first_col

    # Default orientation: rows=cells, cols=genes
    use_transpose = False
    if expected_ids is not None and len(expected_ids) > 0:
        if id_col is not None:
            row_ids = pd.Index(df[id_col].astype(str))
            col_ids = pd.Index(df.columns.drop(id_col).astype(str))
            row_overlap = len(set(row_ids).intersection(set(expected_ids.astype(str))))
            col_overlap = len(set(col_ids).intersection(set(expected_ids.astype(str))))
            if col_overlap > row_overlap and col_overlap >= max(1, int(0.3 * len(expected_ids))):
                use_transpose = True
        else:
            col_ids = pd.Index(df.columns.astype(str))
            col_overlap = len(set(col_ids).intersection(set(expected_ids.astype(str))))
            if col_overlap >= max(1, int(0.3 * len(expected_ids))):
                use_transpose = True

    if not use_transpose:
        obs_ids = df[id_col].astype(str).tolist() if id_col is not None else [str(i) for i in range(df.shape[0])]
        mat_df = df.drop(columns=[id_col]) if id_col is not None else df
        numeric = mat_df.apply(pd.to_numeric, errors="coerce").fillna(0)
        X = csr_matrix(numeric.to_numpy())
        adata = AnnData(X=X)
        adata.obs_names = _ensure_unique_str_index(obs_ids, "cell")
        adata.var_names = pd.Index(numeric.columns.astype(str))
        adata.obs["cell_id"] = adata.obs_names.astype(str)
        return adata

    # Transposed layout: rows=genes, cols=cells
    if id_col is None:
        raise ValueError(
            f"Counts table at {path.name} appears to require transposition, but no gene-id column could be identified."
        )
    gene_names = df[id_col].astype(str).tolist()
    numeric = df.drop(columns=[id_col]).apply(pd.to_numeric, errors="coerce").fillna(0)
    X = csr_matrix(numeric.to_numpy().T)
    adata = AnnData(X=X)
    adata.obs_names = _ensure_unique_str_index(list(numeric.columns.astype(str)), "cell")
    adata.var_names = pd.Index(gene_names)
    adata.obs["cell_id"] = adata.obs_names.astype(str)
    return adata


def _attach_meta(adata: AnnData, meta: pd.DataFrame, sample_key: str) -> None:
    if meta.empty:
        adata.obs["region"] = sample_key
        return

    meta_index = meta.index.astype(str)
    obs_index = adata.obs_names.astype(str)

    overlap = len(set(meta_index).intersection(set(obs_index)))
    if overlap > 0:
        aligned = meta.reindex(obs_index)
    elif len(meta) == adata.n_obs:
        aligned = meta.copy()
        aligned.index = obs_index
    else:
        warnings.warn(
            "Metadata table could not be aligned by cell IDs and has a different number of rows "
            f"({len(meta)} vs {adata.n_obs}). Metadata will not be attached to obs."
        )
        adata.obs["region"] = sample_key
        return

    for col in aligned.columns:
        adata.obs[col] = aligned[col].values

    adata.obs["region"] = sample_key
    if "center_x" in adata.obs.columns and "center_y" in adata.obs.columns:
        spatial = adata.obs[["center_x", "center_y"]].to_numpy(dtype=float)
        adata.obsm["spatial"] = spatial


def _normalize_raster_array(arr: Any) -> Any:
    """Normalize eager or lazy raster arrays for storage in ``uns['spatial']``."""
    shape = getattr(arr, "shape", None)
    if shape is None:
        return arr

    try:
        ndim = len(shape)
    except Exception:
        return arr

    if ndim == 0:
        return arr

    # Drop leading singleton dimensions, e.g. (1, y, x) -> (y, x).
    try:
        while len(getattr(arr, "shape", ())) > 2 and getattr(arr, "shape", ())[0] == 1:
            arr = arr[0]
    except Exception:
        return arr

    return arr


def _read_image(path: Path, *, raise_on_error: bool = False):
    """Best-effort microscopy image reader.

    This follows the general idea of ``spatialdata_io.seqfish``—prefer
    ``dask_image.imread`` for TIFF-like rasters—but differs in one practical
    way: if the file appears truncated or corrupted, we return ``None`` and let
    the caller keep only the file path in ``uns['spatial']`` instead of failing
    the whole reader.
    """
    errors: list[str] = []
    suffix = path.name.lower()
    is_tiff_like = suffix.endswith((".ome.tif", ".ome.tiff", ".tif", ".tiff"))

    if dask_imread is not None and is_tiff_like:
        with _silence_image_backend_noise():
            try:
                return _normalize_raster_array(dask_imread(str(path)))
            except Exception as e:
                errors.append(f"dask_image.imread: {e}")

    if tifffile is not None and is_tiff_like:
        with _silence_image_backend_noise():
            try:
                return _normalize_raster_array(tifffile.imread(str(path), is_ome=True))
            except Exception as e:
                errors.append(f"tifffile.imread(is_ome=True): {e}")

        with _silence_image_backend_noise():
            try:
                with tifffile.TiffFile(str(path), is_ome=True) as tif:
                    return _normalize_raster_array(tif.series[0].asarray())
            except Exception as e:
                errors.append(f"tifffile.TiffFile(is_ome=True): {e}")

        with _silence_image_backend_noise():
            try:
                with tifffile.TiffFile(str(path), is_mmstack=False) as tif:
                    return _normalize_raster_array(tif.series[0].asarray())
            except Exception as e:
                errors.append(f"tifffile.TiffFile(is_mmstack=False): {e}")

        with _silence_image_backend_noise():
            try:
                return _normalize_raster_array(tifffile.imread(str(path)))
            except Exception as e:
                errors.append(f"tifffile.imread: {e}")

    if iio is not None:
        with _silence_image_backend_noise():
            try:
                return _normalize_raster_array(iio.imread(path))
            except Exception as e:
                errors.append(f"imageio.v3.imread: {e}")

    if Image is not None:
        with _silence_image_backend_noise():
            try:
                with Image.open(path) as img:
                    return _normalize_raster_array(np.asarray(img))
            except Exception as e:
                errors.append(f"PIL.Image.open: {e}")

    detail = " | ".join(errors) if errors else "No image backend available."
    if raise_on_error:
        raise RuntimeError(
            f"Failed to read image '{path.name}'. Tried dask-image/tifffile/imageio/PIL. Errors: {detail}"
        )
    return None


def _init_spatial_slot(adata: AnnData, sample_key: str) -> MutableMapping[str, Any]:
    adata.uns.setdefault("spatial", {})
    adata.uns["spatial"].setdefault(sample_key, {})
    slot = adata.uns["spatial"][sample_key]
    slot.setdefault("images", {})
    slot.setdefault("image_files", {})
    slot.setdefault("metadata", {})
    return slot


@register_function(
    aliases=["read_seqfish", "seqfish reader", "读取seqfish", "seqfish io"],
    category="io",
    description="Read seqFISH cell-by-gene tables, cell coordinates, and optional microscopy images.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "adata = st.io.spatial.read_seqfish(",
        "    'seqfish_dir',",
        "    counts_file='SG_MouseKidneyDataRelease_Counts_section3.csv',",
        "    meta_file='SG_MouseKidneyDataRelease_CellCoordinates_section3.csv',",
        ")",
        "adata = st.io.spatial.read_seqfish(",
        "    'seqfish_dir',",
        "    counts_file='SG_MouseKidneyDataRelease_Counts_section3.csv',",
        "    meta_file='SG_MouseKidneyDataRelease_CellCoordinates_section3.csv',",
        "    load_images=False,",
        ")",
    ],
    related=["io.spatial.read_seqfish_plus", "io.spatial.read_merfish", "io.spatial.read_starmap_plus"],
)
def read_seqfish(
    path: Union[str, Path],
    *,
    counts_file: str,
    meta_file: str,
    load_images: bool = True,
    load_labels: bool = True,
) -> AnnData:
    r"""Read a seqFISH dataset folder.

    Arguments:
        path: Path to the directory containing seqFISH release files.
        counts_file: Cell-by-gene counts filename relative to ``path``.
        meta_file: Cell metadata filename relative to ``path``.
        load_images: Whether to load microscopy images (for example DAPI).
            When ``False``, image paths are still indexed under
            ``adata.uns['spatial'][region]['image_files']``.
        load_labels: Whether to load cell segmentation masks such as
            ``*_CellMask_*.tiff``. When ``False``, mask paths are still indexed.

    Returns:
        adata: AnnData with the following structure.

            - **X** – cell × gene count matrix from ``counts_file``
            - **obs** – cell metadata, including merged coordinates
            - **var_names** – gene names
            - **obsm['spatial']** – 2D cell coordinates ``(x, y)`` when available
            - **uns['spatial'][region]['images']** – loaded microscopy images
            - **uns['spatial'][region]['labels']** – loaded cell mask image
            - **uns['spateo_io']** – reader metadata and resolved file paths
    """
    root = Path(path).resolve()
    if root.is_file():
        root = root.parent

    _progress(f"Reading seqFISH dataset from: {root}")
    groups = _discover_seqfish_assets(root)
    counts_path = root / counts_file
    meta_path = root / meta_file
    if not counts_path.exists():
        raise FileNotFoundError(f"Counts file not found: {counts_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    counts_role, counts_group = _match_role_and_group(counts_path)
    meta_role, meta_group = _match_role_and_group(meta_path)
    if counts_role not in {"counts", None}:
        warnings.warn(f"`counts_file` does not look like a seqFISH counts table: {counts_path.name}")
    if meta_role not in {"meta", None}:
        warnings.warn(f"`meta_file` does not look like a seqFISH metadata table: {meta_path.name}")
    if counts_group and meta_group and counts_group != meta_group:
        raise ValueError(
            f"`counts_file` and `meta_file` appear to belong to different seqFISH groups: "
            f"{counts_group!r} vs {meta_group!r}."
        )

    group_key = counts_group or meta_group or ""
    if group_key in groups:
        bundle = groups[group_key]
    elif len(groups) == 1:
        group_key, bundle = next(iter(groups.items()))
    else:
        bundle = {"images": {}}
    sample_key = group_key or root.name

    _progress(f"Loading metadata: {meta_path.name}")
    meta = _standardize_meta(_read_table_with_auto_sep(meta_path))

    _progress(f"Loading counts matrix: {counts_path.name}")
    adata = _prepare_counts(counts_path, expected_ids=meta.index)

    #Set spateo keys
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    _progress(f"Set Spadeo-specific key values:adata.uns['__type'] and adata.uns['pp']",level='step')

    adata.X = csr_matrix(
        np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X, dtype=np.float32)
    )
    adata.obs["region"] = sample_key
    adata.obs["dataset"] = root.name
    adata.var_names = pd.Index(pd.Series(adata.var_names.astype(str)).astype(str))
    adata.var_names_make_unique()

    _attach_meta(adata, meta, sample_key=sample_key)

    spatial_slot = _init_spatial_slot(adata, sample_key)
    spatial_slot["metadata"].update(
        {
            "technology": "seqFISH",
            "region": sample_key,
            "dataset_root": str(root),
            "counts_file": str(counts_path),
            "meta_file": str(meta_path),
        }
    )

    image_files = bundle.get("images", {}) or {}
    for key, img_path in image_files.items():
        if key == "cell_mask":
            spatial_slot["metadata"]["cell_mask_file"] = str(img_path)
            spatial_slot.setdefault("label_files", {})["cell_mask"] = str(img_path)
            if load_labels:
                _progress(f"Loading cell mask: {img_path.name}")
                arr = _read_image(img_path, raise_on_error=False)
                if arr is not None:
                    spatial_slot["labels"] = arr
                else:
                    spatial_slot["metadata"].setdefault("failed_label_files", {})["cell_mask"] = str(img_path)
        else:
            spatial_slot["image_files"][key] = str(img_path)
            if load_images:
                _progress(f"Loading image: {img_path.name}")
                arr = _read_image(img_path, raise_on_error=False)
                if arr is not None:
                    spatial_slot["images"][key] = arr
                else:
                    spatial_slot["metadata"].setdefault("failed_image_files", {})[key] = str(img_path)

    adata.uns["spateo_io"] = {
        "reader": "read_seqfish",
        "technology": "seqFISH",
        "path": str(root),
        "region": sample_key,
        "files": {
            "counts_file": str(counts_path),
            "meta_file": str(meta_path),
            "images": {k: str(v) for k, v in image_files.items()},
        },
        "options": {
            "load_images": bool(load_images),
            "load_labels": bool(load_labels),
            "dtype": str(np.dtype(np.float32)),
        },
    }

    _progress(
        f"Done. AnnData shape = {adata.shape[0]} cells × {adata.shape[1]} genes for region '{sample_key}'.",
        level="success",
    )
    return adata
