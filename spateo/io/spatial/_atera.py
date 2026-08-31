"""Reader for the current 10x Genomics Atera In Situ preview output."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from ..._registry import register_function
from ...configuration import SKM
from ..single import read_10x_h5
from ._provenance import record_spatial_io, spatial_file_manifest
from ._xenium import _boundaries_to_wkt, _read_cells_table, _resolve

PathLike = Union[str, Path]

_MORPHOLOGY_TAGS = {
    "dapi": ("dapi",),
    "boundary": ("atp1a1", "cd45", "e-cadherin", "ecadherin", "boundary"),
    "rna": ("18s", "rna"),
    "stroma": ("alphasma", "alpha-sma", "vimentin", "stroma"),
}


def _resolve_atera_root(path: PathLike) -> Path:
    root = Path(path).expanduser().resolve()
    if root.is_file():
        if root.suffix.lower() == ".zip":
            raise ValueError(
                "Atera outs archives can be tens of gigabytes. Extract the archive and pass "
                "the extracted `outs` directory instead of reading the ZIP in memory."
            )
        raise NotADirectoryError(f"Expected an Atera output directory, got file: {root}")
    if (root / "cell_feature_matrix.h5").is_file():
        return root
    if (root / "outs" / "cell_feature_matrix.h5").is_file():
        return root / "outs"
    return root


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"Could not parse {path.name}: {exc}", UserWarning)
        return {}
    return value if isinstance(value, dict) else {"value": value}


def _atera_evidence(root: Path) -> tuple[str, ...]:
    """Return positive evidence that a Xenium-shaped bundle is Atera."""
    evidence: list[str] = []
    metadata_path = root / "experiment.xenium"
    if metadata_path.is_file():
        metadata = _load_json(metadata_path)
        metadata_text = json.dumps(metadata, sort_keys=True, default=str).lower()
        if "atera" in metadata_text:
            evidence.append("experiment.xenium: Atera metadata")
        if "whole transcriptome" in metadata_text or "human wta" in metadata_text:
            evidence.append("experiment.xenium: whole-transcriptome panel")

    focus = root / "morphology_focus"
    channel_names = [path.name.lower() for path in focus.glob("ch*.ome.tif")] if focus.is_dir() else []
    named_stains = ("dapi", "atp1a1", "cd45", "18s", "alphasma", "vimentin")
    if sum(any(token in name for name in channel_names) for token in named_stains) >= 2:
        evidence.append("morphology_focus/: Atera named stain channels")
    return tuple(evidence)


def _morphology_channels(root: Path) -> list[Path]:
    focus = root / "morphology_focus"
    if not focus.is_dir():
        return []
    candidates = list(focus.glob("ch*.ome.tif")) + list(focus.glob("ch*.ome.tiff"))
    return sorted(set(candidates), key=lambda path: path.name.lower())


def _select_channel(channels: Sequence[Path], image_key: str) -> Optional[Path]:
    if not channels:
        return None
    key = str(image_key).strip().lower()
    if key.isdigit():
        index = int(key)
        return channels[index] if 0 <= index < len(channels) else None
    if key in {"morphology", "morphology_focus", "hires"}:
        return channels[0]
    tokens = _MORPHOLOGY_TAGS.get(key, (key,))
    return next((path for path in channels if any(token in path.name.lower() for token in tokens)), None)


def _load_pyramid_image(path: Path, max_dim: int) -> Optional[tuple[np.ndarray, float]]:
    if max_dim < 1:
        raise ValueError("`max_dim` must be a positive integer.")
    try:
        import tifffile
    except ImportError:
        warnings.warn("Install `tifffile` to load Atera OME-TIFF images.", UserWarning)
        return None

    try:
        # Atera's OME XML can cross-reference sibling files. Reading each file
        # as a standalone TIFF keeps its local pyramid available and bounded.
        with tifffile.TiffFile(path, is_ome=False) as tif:
            series = tif.series[0]
            levels = list(getattr(series, "levels", ()) or (series,))
            full_height = int(levels[0].shape[-2])
            chosen = levels[-1]
            for level in levels:
                if max(level.shape[-2:]) <= max_dim:
                    chosen = level
                    break
            image = chosen.asarray()
    except Exception as exc:
        warnings.warn(f"Failed to read Atera image {path.name}: {exc}", UserWarning)
        return None

    while image.ndim > 3:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 2, 3, 4) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image, 0, -1) if image.shape[0] in (3, 4) else image[0]
    downsample = float(image.shape[0] / full_height) if full_height else 1.0
    return image, downsample


def _find_companion(root: Path, explicit: Optional[PathLike], patterns: Sequence[str]) -> Optional[Path]:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Atera companion file not found: {path}")
        return path
    for base in (root, root.parent):
        for pattern in patterns:
            matches = sorted(base.glob(pattern), key=lambda path: path.name.lower())
            if matches:
                return matches[0]
    return None


def _load_affine(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    matrix = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"Expected a finite 3 x 3 H&E affine matrix in {path}, got {matrix.shape}.")
    return matrix


def _merge_cell_groups(adata: AnnData, path: Path) -> int:
    groups = pd.read_csv(path)
    id_col = next((name for name in ("cell_id", "cellID", "CellID") if name in groups), None)
    group_col = next((name for name in ("group", "cell_group", "cell_type", "cluster") if name in groups), None)
    color_col = next((name for name in ("color", "cell_group_color", "hex_color") if name in groups), None)
    if id_col is None or group_col is None:
        raise ValueError(f"{path.name} must contain a cell id and group column; found {list(groups.columns)}.")
    groups[id_col] = groups[id_col].astype(str)
    groups = groups.drop_duplicates(id_col).set_index(id_col)
    ids = pd.Index(adata.obs_names.astype(str))
    labels = groups[group_col].reindex(ids)
    adata.obs["cell_group"] = pd.Categorical(labels)
    if color_col is not None:
        adata.obs["cell_group_color"] = groups[color_col].reindex(ids).to_numpy()
    return int(labels.notna().sum())


def _match_cell_metadata(adata: AnnData, cells: pd.DataFrame) -> tuple[AnnData, tuple[str, str]]:
    if cells.empty:
        raise ValueError("Atera cells metadata is empty.")
    id_col = next((name for name in ("cell_id", "cellID", "CellID", "cell_ID") if name in cells), cells.columns[0])
    cells[id_col] = cells[id_col].astype(str)
    cells = cells.drop_duplicates(id_col).set_index(id_col)
    matrix_ids = pd.Index(adata.obs_names.astype(str))
    keep = matrix_ids.isin(cells.index)
    if not keep.all():
        warnings.warn(
            f"Dropping {int((~keep).sum())} matrix cells absent from Atera cells metadata.",
            UserWarning,
        )
        adata = adata[keep].copy()
        matrix_ids = pd.Index(adata.obs_names.astype(str))
    cells = cells.reindex(matrix_ids)
    xy = next(
        (
            pair
            for pair in (("x_centroid", "y_centroid"), ("CenterX_local_px", "CenterY_local_px"))
            if all(column in cells for column in pair)
        ),
        None,
    )
    if xy is None:
        raise ValueError(
            "Atera cells metadata lacks centroid columns. Expected `x_centroid`/`y_centroid`; "
            f"found {list(cells.columns)}."
        )
    coords = cells[list(xy)].to_numpy(dtype=np.float32)
    if not np.isfinite(coords).all():
        raise ValueError("Atera centroid coordinates contain NaN or infinite values.")
    adata.obs = cells.copy()
    adata.obsm[SKM.OBSM_SPATIAL_KEY] = coords
    return adata, xy


@register_function(
    aliases=["read_atera", "atera", "10x atera", "atera in situ", "读取atera"],
    category="io",
    description="Read current 10x Atera In Situ preview outputs, including cell metadata, boundaries, named morphology channels, and companion annotations.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "adata = st.io.read_atera('/path/to/extracted/outs')",
        "adata = st.io.read_atera('/path/to/outs', image_key='boundary', load_boundaries=False)",
    ],
    related=["io.spatial.read_xenium", "io.spatial.read_auto_spatial"],
)
def read_atera(
    path: PathLike,
    *,
    library_id: Optional[str] = None,
    load_image: bool = True,
    image_key: str = "dapi",
    image_max_dim: int = 4096,
    load_boundaries: bool = True,
    load_nucleus_boundaries: bool = True,
    load_cell_groups: bool = True,
    cell_groups_csv: Optional[PathLike] = None,
    load_he_image: bool = False,
    he_image: Optional[PathLike] = None,
    he_alignment_csv: Optional[PathLike] = None,
    he_max_dim: int = 4096,
    cache_file: Optional[PathLike] = None,
) -> AnnData:
    """Read an extracted 10x Atera output bundle into ``AnnData``.

    Notes
    -----
    The public 2026 Atera preview uses a Xenium v4-compatible core. 10x states
    that the final Atera format may change, so the reader records
    ``format_status='preview-xenium-v4'`` and validates Atera-specific evidence.
    Large transcript tables and full-resolution TIFFs are inventoried but are
    never loaded implicitly. Set ``load_boundaries=False`` for the lowest-memory
    cell-level load of very large slides.
    """
    root = _resolve_atera_root(path)
    cache_path = Path(cache_file).expanduser().resolve() if cache_file is not None else None
    if cache_path is not None and cache_path.is_file():
        from anndata import read_h5ad

        return read_h5ad(cache_path)

    matrix_path = _resolve(root, "cell_feature_matrix.h5")
    cells_path = _resolve(root, "cells.parquet", "cells.csv.gz", "cells.csv")
    if matrix_path is None or cells_path is None:
        raise FileNotFoundError("Atera requires `cell_feature_matrix.h5` and `cells.parquet` (or CSV) in " f"{root}.")
    evidence = _atera_evidence(root)
    if not evidence:
        warnings.warn(
            "This directory has a Xenium-compatible core but no explicit Atera marker. "
            "Proceeding because `read_atera` was requested directly.",
            UserWarning,
        )

    adata = read_10x_h5(matrix_path, gex_only=True)
    adata, xy = _match_cell_metadata(adata, _read_cells_table(cells_path))
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)

    metadata_path = _resolve(root, "experiment.xenium")
    metadata = _load_json(metadata_path) if metadata_path is not None else {}
    library_id = (
        str(library_id or metadata.get("region_name") or metadata.get("run_name") or root.name or "atera").strip()
        or "atera"
    )
    try:
        pixel_size = float(metadata.get("pixel_size", 0.2125))
    except (TypeError, ValueError):
        pixel_size = 0.2125
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError(f"Invalid Atera pixel_size in experiment metadata: {pixel_size!r}")

    mean_diameter_um = 15.0
    if "cell_area" in adata.obs:
        areas = pd.to_numeric(adata.obs["cell_area"], errors="coerce").to_numpy(dtype=float)
        mean_area = float(np.nanmean(areas))
        if np.isfinite(mean_area) and mean_area > 0:
            mean_diameter_um = float(2 * np.sqrt(mean_area / np.pi))

    channels = _morphology_channels(root)
    metadata = dict(metadata)
    metadata["morphology_channels"] = [channel.name for channel in channels]
    metadata["centroid_columns"] = list(xy)
    spatial_block: dict[str, Any] = {
        "images": {},
        "scalefactors": {
            "tissue_hires_scalef": float(1 / pixel_size),
            "spot_diameter_fullres": float(mean_diameter_um / pixel_size),
        },
        "metadata": metadata,
    }

    if load_image:
        selected = _select_channel(channels, image_key)
        if selected is None:
            warnings.warn(
                f"No Atera morphology channel matched {image_key!r}; available: "
                f"{[channel.name for channel in channels]}",
                UserWarning,
            )
        else:
            loaded = _load_pyramid_image(selected, image_max_dim)
            if loaded is not None:
                image, downsample = loaded
                spatial_block["images"]["hires"] = image
                spatial_block["images"][image_key] = image
                spatial_block["scalefactors"]["tissue_hires_scalef"] *= downsample
                spatial_block["scalefactors"]["spot_diameter_fullres"] *= downsample
                spatial_block["metadata"]["selected_morphology_channel"] = selected.name
                spatial_block["metadata"]["morphology_downsample"] = downsample

    cell_boundary_path = _resolve(root, "cell_boundaries.parquet", "cell_boundaries.csv.gz", "cell_boundaries.csv")
    nucleus_boundary_path = _resolve(
        root, "nucleus_boundaries.parquet", "nucleus_boundaries.csv.gz", "nucleus_boundaries.csv"
    )
    has_cell_geometry = False
    if load_boundaries and cell_boundary_path is not None:
        geometry = _boundaries_to_wkt(root, pd.Index(adata.obs_names.astype(str)))
        if geometry is not None:
            adata.obs["geometry"] = geometry.to_numpy()
            has_cell_geometry = bool((geometry != "").any())
    if load_nucleus_boundaries and nucleus_boundary_path is not None:
        nucleus_geometry = _boundaries_to_wkt(
            root,
            pd.Index(adata.obs_names.astype(str)),
            boundary_stem="nucleus_boundaries",
        )
        if nucleus_geometry is not None:
            adata.obs["nucleus_geometry"] = nucleus_geometry.to_numpy()

    group_path = (
        _find_companion(root, cell_groups_csv, ("*_cell_groups.csv", "cell_groups.csv")) if load_cell_groups else None
    )
    if group_path is not None:
        matched = _merge_cell_groups(adata, group_path)
        spatial_block["metadata"]["cell_groups_file"] = group_path.name
        spatial_block["metadata"]["cell_groups_matched"] = matched

    he_path = _find_companion(root, he_image, ("*_he_image.ome.tif", "*_he_image.ome.tiff"))
    affine_path = _find_companion(root, he_alignment_csv, ("*_he_alignment.csv", "he_alignment.csv"))
    if he_path is not None:
        spatial_block["metadata"]["he_image_file"] = he_path.name
    affine = _load_affine(affine_path)
    if affine is not None:
        spatial_block["metadata"]["he_alignment_file"] = affine_path.name
        spatial_block["scalefactors"]["he_affine"] = affine
    if load_he_image and he_path is not None:
        loaded_he = _load_pyramid_image(he_path, he_max_dim)
        if loaded_he is not None:
            he_array, he_downsample = loaded_he
            spatial_block["images"]["he"] = he_array
            spatial_block["scalefactors"]["he_downsample"] = he_downsample

    adata.uns[SKM.UNS_SPATIAL_KEY] = {library_id: spatial_block}
    record_spatial_io(
        adata,
        technology="atera",
        source=root,
        reader="spateo.io.spatial.read_atera",
        evidence=evidence,
        manifest=spatial_file_manifest(root),
        format_status="preview-xenium-v4",
    )
    adata.uns["spateo_io"].update(
        {
            "type": "atera_seg" if has_cell_geometry else "atera",
            "library_id": library_id,
        }
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(cache_path)
    return adata


__all__ = ["read_atera"]
