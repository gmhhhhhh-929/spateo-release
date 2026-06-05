"""STARmap PLUS reader for Spateo spatial I/O."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from ...._registry import register_function
from ....configuration import SKM

try:
    from ...._settings import Colors
except Exception:  # pragma: no cover
    class Colors:
        """Fallback ANSI color codes when spateo._settings import is unavailable."""

        HEADER = "\033[95m"
        BLUE = "\033[94m"
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        WARNING = "\033[93m"
        FAIL = "\033[91m"
        ENDC = "\033[0m"
        BOLD = "\033[1m"
        UNDERLINE = "\033[4m"


def _progress(message: str, level: str = "info") -> None:
    color = Colors.CYAN
    if level == "success":
        color = Colors.GREEN
    elif level == "warn":
        color = Colors.WARNING
    print(f"{color}[STARmapPlus] {message}{Colors.ENDC}")


def _read_table(path: Path, **kwargs) -> pd.DataFrame:
    if ".parquet" in path.suffixes:
        return pd.read_parquet(path, **kwargs)

    name = path.name.lower()
    if name.endswith(".tsv") or name.endswith(".tsv.gz") or name.endswith(".txt") or name.endswith(".txt.gz"):
        return pd.read_csv(path, sep="\t", **kwargs)

    try:
        return pd.read_csv(path, sep=",", **kwargs)
    except Exception:
        try:
            return pd.read_csv(path, sep="\t", **kwargs)
        except Exception:
            return pd.read_csv(path, sep=None, engine="python", **kwargs)


def _resolve(root: Path, *candidates: str) -> Optional[Path]:
    seen = set()
    for name in candidates:
        for candidate in (name, f"{name}.gz") if name.endswith(".csv") else (name,):
            if candidate in seen:
                continue
            seen.add(candidate)
            path = root / candidate
            if path.exists():
                return path
    return None


def _infer_prefix(root: Path) -> str:
    prefixes = set()
    suffixes = (
        "processed_expression_pd.csv",
        "raw_expression_pd.csv",
        "spatial.csv",
        "spot_meta.csv",
    )

    for path in root.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if name.endswith(".gz"):
            name = name[:-3]
        for suffix in suffixes:
            if not name.endswith(suffix):
                continue
            prefix = name[: -len(suffix)].rstrip("_")
            prefixes.add(prefix)
            break

    if not prefixes:
        raise FileNotFoundError(
            f"Could not find STARmap PLUS files under {root}. "
            "Expected files like `spatial.csv` or `sample_spatial.csv`."
        )
    if len(prefixes) > 1:
        raise ValueError(
            f"Found multiple STARmap PLUS file groups under {root}: {sorted(prefixes)}. "
            "This reader only supports one group per directory."
        )
    return next(iter(prefixes))


def _build_candidates(prefix: str, stem: str) -> tuple[str, ...]:
    if not prefix:
        return (f"{stem}.csv",)
    return (f"{prefix}{stem}.csv", f"{prefix}_{stem}.csv")


def _normalize_names(index) -> pd.Index:
    values = []
    for value in index:
        values.append(None if pd.isna(value) else str(value).strip())
    return pd.Index(values, dtype="object")


def _find_name_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("NAME", "name", "Name", "cell_id", "CellID", "cellid", "spot_id", "SpotID", "spotid", "barcode", "Barcode"):
        if col in df.columns:
            return col
    return None


def _set_name_index(df: pd.DataFrame) -> pd.DataFrame:
    name_col = _find_name_col(df)
    if name_col is None:
        raise ValueError("Could not find NAME-like column.")
    out = df.copy()
    out[name_col] = out[name_col].astype(str)
    out = out.drop_duplicates(subset=[name_col]).set_index(name_col)
    out.index = _normalize_names(out.index)
    return out


def _find_coord_cols(df: pd.DataFrame) -> list[str]:
    lower_map = {str(col).lower(): col for col in df.columns}

    def _pick(*keys: str) -> Optional[str]:
        for key in keys:
            if key in lower_map:
                return lower_map[key]
        return None

    x_col = _pick("x", "global_x", "x_global", "xcoord", "x_coord", "x_location")
    y_col = _pick("y", "global_y", "y_global", "ycoord", "y_coord", "y_location")
    z_col = _pick("z", "global_z", "z_global", "zcoord", "z_coord", "z_location")

    if x_col is None or y_col is None:
        return []
    return [x_col, y_col] + ([z_col] if z_col is not None else [])


def _read_spatial_table(path: Path, reorient_xy: bool = False) -> pd.DataFrame:
    df = _read_table(path)
    coord_cols = _find_coord_cols(df)

    if len(coord_cols) < 2:
        try:
            header = list(_read_table(path, nrows=0).columns)
            alt = pd.read_csv(path, skiprows=1, header=None)
            if len(header) == alt.shape[1]:
                alt.columns = header
                df = alt
                coord_cols = _find_coord_cols(df)
        except Exception:
            pass

    if len(coord_cols) < 2:
        raise ValueError(
            f"Could not detect spatial coordinate columns in {path}. "
            "Expected columns like X/Y(/Z) or global_x/global_y."
        )

    if reorient_xy:
        x_col, y_col = coord_cols[:2]
        xx = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
        yy = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
        df = df.copy()
        df[x_col] = np.nanmax(yy) - yy
        df[y_col] = np.nanmax(xx) - xx

    return df


def _read_expression(path: Path, dtype: str = "float32") -> AnnData:
    df = _read_table(path)
    if df.shape[1] < 2:
        raise ValueError(f"Expression file has too few columns: {path}")

    first_col = str(df.columns[0])
    first_col_lower = first_col.lower()
    name_col = _find_name_col(df)
    gene_like = {"gene", "genes", "feature", "features", "unnamed: 0"}

    if first_col_lower in gene_like or (name_col is None and df.shape[0] < df.shape[1]):
        gene_names = df.iloc[:, 0].astype(str)
        obs_names = _normalize_names(df.columns[1:])
        matrix = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=dtype).T
        adata = AnnData(X=csr_matrix(matrix))
        adata.obs_names = obs_names
        adata.var_names = pd.Index(gene_names, dtype="object")
    else:
        obs_names = _normalize_names(df.iloc[:, 0])
        expr = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0)
        adata = AnnData(X=csr_matrix(expr.to_numpy(dtype=dtype)))
        adata.obs_names = obs_names
        adata.var_names = pd.Index(expr.columns.astype(str), dtype="object")

    adata.var_names_make_unique()
    if adata.obs_names.has_duplicates:
        raise ValueError(f"Expression file contains duplicated cell/spot names: {path}")
    return adata


def _merge_obs(
    obs: pd.DataFrame,
    extra: Optional[pd.DataFrame],
    label: str,
    *,
    allow_row_match: bool = False,
) -> pd.DataFrame:
    if extra is None or extra.empty:
        return obs

    try:
        extra = _set_name_index(extra)
    except ValueError:
        extra = extra.copy()
        extra.index = _normalize_names(extra.index)
        if extra.index.isin(obs.index).any():
            extra = extra.loc[~extra.index.duplicated(keep="first")]
        elif allow_row_match and len(extra) == len(obs):
            extra = extra.copy()
            extra.index = obs.index
        else:
            warnings.warn(f"Skipping `{label}` merge: no NAME-like column found.")
            return obs

    rename_map = {col: f"{col}_{label}" for col in extra.columns if col in obs.columns}
    if rename_map:
        extra = extra.rename(columns=rename_map)
    return obs.join(extra, how="left")


@register_function(
    aliases=["read_starmap_plus", "starmap plus", "读取starmap plus", "starmap_plus"],
    category="io",
    description="Read one STARmap PLUS file group with expression matrix, spatial coordinates, and optional metadata.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "adata = ov.io.spatial.read_starmap_plus(",
        "    'starmap_plus_dir',",
        "    counts_file='sample_counts.csv',",
        ")",
        "adata = ov.io.spatial.read_starmap_plus(",
        "    'starmap_plus_dir',",
        "    counts_file='sample_counts.csv',",
        "    spatial_file='sample_spatial.csv',",
        ")",
    ],
    related=["io.spatial.read_nanostring", "io.spatial.read_seqfish_plus", "io.spatial.read_merfish"],
)
def read_starmap_plus(
    path: Union[str, Path],
    *,
    counts_file: str,
    meta_file: str,
    spatial_file: str,
    reorient_xy: bool = False,
    dtype: str = "float32",
) -> AnnData:
    """Read a STARmap PLUS directory into a single AnnData object.

    Parameters
    ----------
    path
        Directory containing STARmap PLUS outputs.
    counts_file
        Expression matrix filename relative to ``path``.
    spatial_file
        Optional spatial coordinate filename relative to ``path``. When not
        provided, the reader will auto-discover the conventional STARmap PLUS
        spatial file.
    """
    root = Path(path).resolve()
    _progress(f"Reading STARmap PLUS data from: {root}")

    prefix = _infer_prefix(root)
    sample = root.name

    expr_path = root / counts_file
    spatial_path = (root / spatial_file) if spatial_file is not None else _resolve(
        root, *_build_candidates(prefix, "spatial")
    )
    spot_meta_path = _resolve(root, *_build_candidates(prefix, "spot_meta"))
    metadata_file = _resolve(root, meta_file)
    cluster_file = _resolve(root, "cluster.csv")

    if not expr_path.exists():
        raise FileNotFoundError(f"Counts file not found: {expr_path}")
    if spatial_path is None or not spatial_path.exists():
        raise FileNotFoundError(f"Spatial file not found under {root}")

    _progress(f"Loading expression matrix: {expr_path.name}")
    adata = _read_expression(expr_path, dtype=dtype)

    # Set spateo keys
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    _progress(f"Set Spadeo-specific key values:adata.uns['__type'] and adata.uns['pp']",level="step")

    adata.obs_names = _normalize_names(adata.obs_names)

    _progress(f"Loading spatial table: {spatial_path.name}")
    spatial = _read_spatial_table(spatial_path, reorient_xy=reorient_xy)
    try:
        spatial = _set_name_index(spatial)
    except ValueError:
        if len(spatial) != adata.n_obs:
            raise ValueError(
                f"Spatial file {spatial_path.name} has no NAME-like column and row number does not match expression matrix."
            )
        spatial = spatial.copy()
        spatial.index = adata.obs_names

    common_names = adata.obs_names[adata.obs_names.isin(spatial.index)]
    if len(common_names) == 0:
        raise ValueError(
            "No overlapping names between expression matrix and spatial table. "
            f"Example expression names: {list(adata.obs_names[:5])}; "
            f"example spatial names: {list(spatial.index[:5])}"
        )

    adata = adata[common_names].copy()
    adata.obs = _merge_obs(adata.obs, spatial.loc[common_names], "spatial", allow_row_match=True)

    coord_cols = _find_coord_cols(adata.obs)
    if len(coord_cols) < 2:
        raise ValueError(
            "Could not find coordinate columns after merging spatial table. "
            f"Available columns: {list(adata.obs.columns[:20])}"
        )
    adata.obsm["spatial"] = adata.obs[coord_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if spot_meta_path is not None:
        _progress(f"Loading spot metadata: {spot_meta_path.name}")
        adata.obs = _merge_obs(adata.obs, _read_table(spot_meta_path), "spot_meta", allow_row_match=True)

    if metadata_file is not None:
        _progress(f"Loading global metadata: {metadata_file.name}")
        adata.obs = _merge_obs(adata.obs, _read_table(metadata_file), "metadata")

    if cluster_file is not None:
        _progress(f"Loading cluster table: {cluster_file.name}")
        adata.obs = _merge_obs(adata.obs, _read_table(cluster_file), "cluster")

    adata.obs["sample"] = sample
    adata.obs["dataset"] = sample
    adata.obs["NAME"] = adata.obs_names.astype(str)

    group_key = prefix or sample
    adata.uns["spatial"] = {
        group_key: {
            "metadata": {
                "platform": "STARmap Plus",
                "sample": sample,
                "group_prefix": prefix,
                "expression_file": expr_path.name,
                "spatial_file": spatial_path.name,
                "spot_meta_file": spot_meta_path.name if spot_meta_path is not None else None,
                "coord_columns": coord_cols,
            }
        }
    }
    adata.uns["spateo_io"] = {
        "type": "starmap_plus",
        "sample": sample,
        "group_prefix": prefix,
        "spatial_key": "spatial",
    }

    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)

    _progress(f"Done (n_obs={adata.n_obs}, n_vars={adata.n_vars})", level="success")
    return adata


__all__ = ["read_starmap_plus"]
