"""Reader for standard 10x Genomics Visium Space Ranger outputs."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import h5py
import numpy as np
import pandas as pd
from anndata import AnnData
from PIL import Image

from ..._registry import register_function
from ...configuration import SKM
from ..single import read_10x_h5
from ._provenance import record_spatial_io, spatial_file_manifest

if TYPE_CHECKING:
    from os import PathLike


_POSITION_COLUMNS = [
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
]

_QC_IMAGE_NAMES = (
    "aligned_fiducials.jpg",
    "detected_tissue_image.jpg",
    "aligned_tissue_image.jpg",
    "cytassist_image.tiff",
    "cytassist_image.tif",
)


def _progress(message: str, level: str = "info") -> None:
    """Emit a compact progress message without terminal-specific colour codes."""
    tag = {"success": "OK", "warn": "WARN"}.get(level, "INFO")
    print(f"[Visium][{tag}] {message}")


def _resolve_visium_root(path: Union[str, "PathLike[str]"]) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a Space Ranger output directory, got: {root}")
    if (root / "outs").is_dir() and not (root / "spatial").is_dir():
        return root / "outs"
    return root


def _read_image(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    try:
        with Image.open(path) as image:
            return np.asarray(image)
    except (OSError, ValueError) as exc:
        warnings.warn(f"Could not load Visium image {path}: {exc}", UserWarning)
        return None


def _read_scalefactors(path: Path) -> dict[str, Any]:
    if not path.is_file():
        warnings.warn(f"Visium scale factors are missing: {path}", UserWarning)
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"Could not parse Visium scale factors {path}: {exc}", UserWarning)
        return {}
    return value if isinstance(value, dict) else {"value": value}


def _resolve_tissue_positions(spatial_dir: Path) -> Path:
    """Return the first existing positions file, preferring modern formats."""
    for name in ("tissue_positions.parquet", "tissue_positions.csv", "tissue_positions_list.csv"):
        candidate = spatial_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No tissue positions file found under {spatial_dir}. Expected "
        "tissue_positions.parquet, tissue_positions.csv, or tissue_positions_list.csv."
    )


def _read_tissue_positions(path: Path) -> pd.DataFrame:
    """Read both current headered and legacy headerless Space Ranger tables."""
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.name == "tissue_positions_list.csv":
        frame = pd.read_csv(path, header=None, names=_POSITION_COLUMNS)
    else:
        frame = pd.read_csv(path)
        if "barcode" not in frame.columns and frame.shape[1] == len(_POSITION_COLUMNS):
            # Some third-party pipelines use the modern filename but retain the
            # legacy headerless representation.
            frame = pd.read_csv(path, header=None, names=_POSITION_COLUMNS)

    if "barcode" not in frame.columns:
        if frame.index.name == "barcode":
            frame = frame.reset_index()
        else:
            raise ValueError(f"{path} does not contain a `barcode` column.")
    missing = [column for column in _POSITION_COLUMNS[1:] if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required Visium columns: {missing}")
    frame = frame[_POSITION_COLUMNS].copy()
    frame["barcode"] = frame["barcode"].astype(str)
    frame = frame.drop_duplicates("barcode", keep="first").set_index("barcode")
    for column in _POSITION_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _decode_h5_value(value: Any) -> Any:
    """Convert HDF5 attributes to H5AD-safe Python scalars and lists."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [_decode_h5_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _h5_metadata(path: Path) -> dict[str, Any]:
    with h5py.File(path, mode="r") as handle:
        return {str(key): _decode_h5_value(value) for key, value in handle.attrs.items()}


def _library_id(metadata: dict[str, Any], root: Path, requested: Optional[str]) -> str:
    if requested is not None:
        return str(requested)
    library_ids = metadata.get("library_ids")
    if isinstance(library_ids, list) and library_ids:
        return str(library_ids[0])
    if library_ids not in (None, ""):
        return str(library_ids)
    return root.parent.name if root.name == "outs" else root.name


@register_function(
    aliases=["read_visium", "visium reader", "读取visium", "10x visium", "spaceranger reader"],
    category="io",
    description="Read standard 10x Visium expression, coordinates, images, scale factors, and output provenance.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "adata = st.io.read_visium('/path/to/outs')",
        "adata = st.io.read_visium('/path/to/pipestance', load_images=False)",
        "adata = st.io.read_visium('/path/to/outs', load_qc_images=True)",
    ],
    related=["io.spatial.read_visium_hd", "io.single.read_10x_h5", "io.spatial.read_auto_spatial"],
)
def read_visium(
    path: Union[str, "PathLike[str]"],
    genome: Optional[str] = None,
    *,
    count_file: str = "filtered_feature_bc_matrix.h5",
    library_id: Optional[str] = None,
    load_images: bool = True,
    load_qc_images: bool = False,
    source_image_path: Optional[Union[str, "PathLike[str]"]] = None,
    hires_image_path: str = "spatial/tissue_hires_image.png",
    lowres_image_path: str = "spatial/tissue_lowres_image.png",
    scalefactors_path: str = "spatial/scalefactors_json.json",
) -> AnnData:
    """Read a standard Visium Space Ranger output directory.

    Spatial coordinates and scale factors are always read because they are
    small, essential assay metadata. ``load_images=False`` skips only pixel
    arrays. Optional diagnostic images are inventoried by default and loaded
    only when ``load_qc_images=True``.
    """
    root = _resolve_visium_root(path)
    h5_path = root / count_file
    if not h5_path.is_file():
        raise FileNotFoundError(f"Visium count file not found: {h5_path}")

    _progress(f"Reading {h5_path.name} from {root}")
    adata = read_10x_h5(h5_path, genome=genome)
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)

    h5_metadata = _h5_metadata(h5_path)
    sample = _library_id(h5_metadata, root, library_id)
    spatial_dir = root / "spatial"
    positions_path = _resolve_tissue_positions(spatial_dir)
    positions = _read_tissue_positions(positions_path)
    adata.obs = adata.obs.join(positions, how="left")

    coordinate_columns = ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    coordinates = adata.obs[coordinate_columns].to_numpy(dtype=np.float64)
    missing_coordinates = ~np.isfinite(coordinates).all(axis=1)
    if missing_coordinates.any():
        warnings.warn(
            f"{int(missing_coordinates.sum())} matrix barcodes have no finite Visium pixel coordinates.",
            UserWarning,
        )
    adata.obsm[SKM.OBSM_SPATIAL_KEY] = coordinates

    scale_path = root / scalefactors_path
    metadata: dict[str, Any] = {
        **h5_metadata,
        "positions_file": positions_path.relative_to(root).as_posix(),
        "coordinate_columns": coordinate_columns,
        "coordinate_system": "full-resolution image pixels (x=column, y=row)",
    }
    if source_image_path is not None:
        source = Path(source_image_path).expanduser().resolve()
        metadata["source_image_path"] = str(source)
        metadata["source_image_exists"] = source.is_file()

    images: dict[str, np.ndarray] = {}
    image_paths = {"hires": root / hires_image_path, "lowres": root / lowres_image_path}
    metadata["image_files"] = {
        key: candidate.relative_to(root).as_posix() for key, candidate in image_paths.items() if candidate.is_file()
    }
    if load_images:
        for key, candidate in image_paths.items():
            image = _read_image(candidate)
            if image is not None:
                images[key] = image

    qc_files = [spatial_dir / name for name in _QC_IMAGE_NAMES if (spatial_dir / name).is_file()]
    metadata["qc_image_files"] = [candidate.relative_to(root).as_posix() for candidate in qc_files]
    if load_qc_images:
        for candidate in qc_files:
            image = _read_image(candidate)
            if image is not None:
                images[f"qc_{candidate.stem}"] = image

    spatial_block: dict[str, Any] = {
        "images": images,
        "scalefactors": _read_scalefactors(scale_path),
        "metadata": metadata,
    }
    adata.uns[SKM.UNS_SPATIAL_KEY] = {sample: spatial_block}
    record_spatial_io(
        adata,
        technology="visium",
        source=root,
        reader="spateo.io.spatial.read_visium",
        evidence=(h5_path.name, positions_path.relative_to(root).as_posix()),
        reader_kwargs={
            "count_file": count_file,
            "load_images": load_images,
            "load_qc_images": load_qc_images,
        },
        manifest=spatial_file_manifest(root),
    )
    adata.uns["spateo_io"].update({"type": "visium", "library_id": sample})
    _progress(f"Done (n_obs={adata.n_obs}, n_vars={adata.n_vars})", level="success")
    return adata


__all__ = ["read_visium"]
