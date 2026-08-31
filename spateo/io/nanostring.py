"""Compatibility API for the maintained NanoString/CosMx reader."""

import re
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from ..configuration import SKM
from .spatial._nanostring import read_nanostring as _read_nanostring
from .utils import bin_indices, get_bin_props, get_points_props

FOV_PARSER = re.compile(r"^.+_F(?P<fov>[0-9]+)\..+$")


def read_nanostring_as_dataframe(path: str, label_columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Read a historical CosMx transcript or metadata CSV with standard names."""
    frame = pd.read_csv(path)
    if label_columns:
        missing = [column for column in label_columns if column not in frame]
        if missing:
            raise ValueError(f"NanoString label columns are missing: {missing}")
        labels = frame[label_columns[0]].astype(str)
        for column in label_columns[1:]:
            labels = labels.str.cat(frame[column].astype(str), sep="-")
        frame["label"] = labels
    numeric_casts = {
        "x_global_px": np.uint32,
        "y_global_px": np.uint32,
        "x_local_px": np.uint32,
        "y_local_px": np.uint32,
        "CenterX_local_px": np.uint32,
        "CenterY_local_px": np.uint32,
        "CenterX_global_px": np.uint32,
        "CenterY_global_px": np.uint32,
    }
    for column, dtype in numeric_casts.items():
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(dtype)
    return frame.rename(columns={"target": "gene", "x_global_px": "x", "y_global_px": "y"})


def stitch_images(stain_dir: str, positions_path: str, labels: bool = False) -> np.ndarray:
    """Stitch historical per-FOV CosMx images using global pixel offsets."""
    try:
        from skimage.io import imread
    except ImportError as exc:  # pragma: no cover - scikit-image is a core dependency
        raise ImportError("`stitch_images` requires scikit-image.") from exc
    root = Path(stain_dir)
    images: dict[int, np.ndarray] = {}
    for candidate in root.iterdir():
        match = FOV_PARSER.match(candidate.name)
        if match:
            fov = int(match.group("fov"))
            if fov in images:
                raise ValueError(f"Multiple images were found for FOV {fov} in {root}.")
            images[fov] = imread(candidate)
    positions = pd.read_csv(positions_path).set_index("fov")
    if set(positions.index.astype(int)) != set(images):
        raise ValueError("FOV position rows do not match the image filenames.")
    positions.index = positions.index.astype(int)
    x_offsets = positions["x_global_px"].astype(int)
    y_offsets = positions["y_global_px"].astype(int)
    x_min, y_min = int(x_offsets.min()), int(y_offsets.min())
    height = max(int(y_offsets[fov]) + image.shape[0] for fov, image in images.items()) - y_min
    width = max(int(x_offsets[fov]) + image.shape[1] for fov, image in images.items()) - x_min
    channels = next(iter(images.values())).shape[2:]
    dtype = np.uint64 if labels else next(iter(images.values())).dtype
    output = np.zeros((height, width, *channels), dtype=dtype)
    last_label = 0
    for fov, image in images.items():
        current = image.astype(dtype, copy=True)
        if labels:
            current[current > 0] += last_label
            last_label = int(current.max())
        x = int(x_offsets[fov]) - x_min
        y = int(y_offsets[fov]) - y_min
        output[y : y + current.shape[0], x : x + current.shape[1]] = current
    return output


def _read_nanostring_legacy(
    path: str,
    meta_path: Optional[str],
    binsize: Optional[int],
    label_columns: Optional[Union[str, Sequence[str]]],
    add_props: bool,
) -> AnnData:
    if (binsize is None) == (label_columns is None):
        raise ValueError("Exactly one of `binsize` and `label_columns` must be provided.")
    columns = [label_columns] if isinstance(label_columns, str) else label_columns
    data = read_nanostring_as_dataframe(path, columns)
    if "gene" not in data or "x" not in data or "y" not in data:
        raise ValueError("Legacy CosMx transcripts require target/x_global_px/y_global_px columns.")
    metadata = read_nanostring_as_dataframe(meta_path, columns) if meta_path and columns else None
    if columns:
        if "cell_ID" in data:
            data = data[data["cell_ID"] > 0].copy()
        effective_binsize = 1
        properties = get_points_props(data[["x", "y", "label"]]) if add_props else None
    else:
        effective_binsize = int(binsize)
        if effective_binsize < 1:
            raise ValueError("`binsize` must be a positive integer.")
        data["x"] = bin_indices(data["x"].to_numpy(), 0, effective_binsize)
        data["y"] = bin_indices(data["y"].to_numpy(), 0, effective_binsize)
        data["label"] = data["x"].astype(str).str.cat(data["y"].astype(str), sep="-")
        properties = (
            get_bin_props(data[["x", "y", "label"]].drop_duplicates(), effective_binsize) if add_props else None
        )
    counts = data.groupby(["label", "gene"], observed=True).size().rename("count").reset_index()
    labels = pd.Index(sorted(counts["label"].astype(str).unique()))
    genes = pd.Index(sorted(counts["gene"].astype(str).unique()))
    row = pd.Categorical(counts["label"].astype(str), categories=labels).codes
    column = pd.Categorical(counts["gene"].astype(str), categories=genes).codes
    adata = AnnData(
        X=csr_matrix((counts["count"].to_numpy(), (row, column)), shape=(len(labels), len(genes))),
        obs=pd.DataFrame(index=labels),
        var=pd.DataFrame(index=genes),
    )
    if metadata is not None:
        adata.obs = metadata.drop_duplicates("label").set_index("label").reindex(labels)
    if properties is not None:
        properties = properties.reindex(labels)
        adata.obs["area"] = properties["area"].to_numpy()
        adata.obsm[SKM.OBSM_SPATIAL_KEY] = properties.filter(regex="centroid-").to_numpy()
        adata.obsm["bbox"] = properties.filter(regex="bbox-").to_numpy()
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_BINSIZE_KEY, effective_binsize)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, 1.0)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, None)
    return adata


def read_nanostring(
    path: str,
    meta_path: Optional[str] = None,
    binsize: Optional[int] = None,
    label_columns: Optional[Union[str, Sequence[str]]] = None,
    add_props: bool = True,
    version: str = "cosmx",
    **kwargs,
) -> AnnData:
    """Read modern directory outputs or dispatch the historical transcript API."""
    if Path(path).is_dir() or "counts_file" in kwargs:
        return _read_nanostring(path, **kwargs)
    return _read_nanostring_legacy(path, meta_path, binsize, label_columns, add_props)


__all__ = ["FOV_PARSER", "read_nanostring", "read_nanostring_as_dataframe", "stitch_images"]
