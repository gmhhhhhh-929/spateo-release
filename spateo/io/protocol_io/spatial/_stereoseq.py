"""
Spateo reading functions for Stereo-seq.

This module provides functions for reading BGI / Stereo-seq spatial
transcriptomics outputs in a style aligned with the other Spateo spatial
readers.
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from os import PathLike
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix
from typing_extensions import Literal

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import skimage.io

from ...._registry import register_function
from ....configuration import SKM
from ._utils import (
    bin_indices,
    get_bin_props,
    get_coords_labels,
    get_label_props,
    get_points_props,
)

try:
    from ...._settings import Colors
except Exception:
    class Colors:
        """Fallback ANSI color codes when spatego._settings import is unavailable."""

        HEADER = "\033[95m"
        BLUE = "\033[94m"
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        WARNING = "\033[93m"
        FAIL = "\033[91m"
        ENDC = "\033[0m"
        BOLD = "\033[1m"
        UNDERLINE = "\033[4m"


try:
    import ngs_tools as ngs

    VERSIONS = {
        "stereo": ngs.chemistry.get_chemistry("Stereo-seq").resolution,
    }
except ModuleNotFoundError:

    class SpatialResolution(NamedTuple):
        scale: float = 1.0
        unit: Optional[Literal["nm", "um", "mm"]] = None

    VERSIONS = {"stereo": SpatialResolution(0.5, "um")}


COUNT_COLUMN_MAPPING = {
    SKM.X_LAYER: 3,
    SKM.SPLICED_LAYER_KEY: 4,
    SKM.UNSPLICED_LAYER_KEY: 5,
}


def _progress(message: str, level: str = "info") -> None:
    level_key = (level or "info").lower()
    if level_key == "info":
        msg = message.lower()
        if msg.startswith("reading"):
            level_key = "start"
        elif msg.startswith("loading") or msg.startswith("constructing"):
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
    text = f"[BGI]{tag} {message}"

    force_color = os.environ.get("FORCE_COLOR", "").strip() not in ("", "0", "false", "False")
    no_color = os.environ.get("NO_COLOR", "").strip() != ""
    supports_color = force_color or (hasattr(sys.stdout, "isatty") and sys.stdout.isatty())
    if no_color or not supports_color:
        print(text)
    else:
        print(f"{color}{text}{Colors.ENDC}")


def _resolve_scale(version: str) -> Tuple[float, Optional[str]]:
    scale, scale_unit = 1.0, None
    if version in VERSIONS:
        resolution = VERSIONS[version]
        scale, scale_unit = resolution.scale, resolution.unit
    return scale, scale_unit


def _initialize_spatial_metadata(
    adata: AnnData,
    *,
    binsize: int,
    version: str,
    io_type: str,
    source_path: Union[str, Path],
) -> None:
    scale, scale_unit = _resolve_scale(version)
    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_BINSIZE_KEY, binsize)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, scale)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, scale_unit)

    adata.uns.setdefault("spateo_io", {})
    adata.uns["spateo_io"].update(
        {
            "type": io_type,
            "platform": "BGI/Stereo-seq",
            "version": version,
            "source_path": str(source_path),
        }
    )


def _pad_image_xy(
    image: np.ndarray,
    *,
    pad_before_xy: Tuple[int, int] = (0, 0),
    target_xy_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Pad an image along its first two axes while preserving any channel axis."""
    pad_x_before, pad_y_before = pad_before_xy
    pad_width = [(pad_x_before, 0), (pad_y_before, 0)]
    if image.ndim > 2:
        pad_width.extend([(0, 0)] * (image.ndim - 2))
    image = np.pad(image, pad_width)

    if target_xy_shape is not None:
        pad_x_after = max(0, target_xy_shape[0] - image.shape[0])
        pad_y_after = max(0, target_xy_shape[1] - image.shape[1])
        if pad_x_after > 0 or pad_y_after > 0:
            pad_width = [(0, pad_x_after), (0, pad_y_after)]
            if image.ndim > 2:
                pad_width.extend([(0, 0)] * (image.ndim - 2))
            image = np.pad(image, pad_width)
    return image


def read_bgi_as_dataframe(
    path: Union[str, PathLike],
    label_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read a BGI / Stereo-seq read file as a standardized pandas DataFrame.

    Parameters
    ----------
    path
        Path to the tab-separated read file.
    label_column
        Optional column name containing positive cell labels.

    Returns
    -------
    pandas.DataFrame
        Dataframe with standardized columns:

        - ``geneID``: gene identifier/name
        - ``x``, ``y``: spatial coordinates
        - ``total``: total UMI/MID count
        - ``spliced`` / ``unspliced``: optional RNA species counts
        - ``label``: optional pre-existing cell label column
    """
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Read file not found: {file_path}")

    dtype = {
        "geneID": "category",
        "x": np.uint32,
        "y": np.uint32,
        "MIDCounts": np.uint16,
        "MIDCount": np.uint16,
        "UMICount": np.uint16,
        "UMICounts": np.uint16,
        "EXONIC": np.uint16,
        "INTRONIC": np.uint16,
    }
    rename = {
        "MIDCounts": "total",
        "MIDCount": "total",
        "UMICount": "total",
        "UMICounts": "total",
        "EXONIC": "spliced",
        "INTRONIC": "unspliced",
    }

    head = pd.read_csv(file_path, sep="\t", dtype=dtype, comment="#", nrows=10, compression="infer")

    if label_column is not None:
        dtype[label_column] = np.uint32
        rename[label_column] = "label"
        if label_column not in head.columns:
            raise IOError(f"Column `{label_column}` is not present in {file_path.name}.")

    rename_inverse: Dict[str, List[str]] = {}
    for original_name, standardized_name in rename.items():
        rename_inverse.setdefault(standardized_name, []).append(original_name)
    for standardized_name, candidate_columns in rename_inverse.items():
        if sum(col in head.columns for col in candidate_columns) > 1:
            raise IOError(f"Found multiple columns mapping to `{standardized_name}` in {file_path.name}.")

    df = pd.read_csv(file_path, sep="\t", dtype=dtype, comment="#", compression="infer").rename(columns=rename)

    required_columns = {"geneID", "x", "y", "total"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise IOError(
            f"Missing required columns in {file_path.name}: {sorted(missing)}. "
            "Expected at least geneID, x, y, and one total-count column."
        )

    return df


def dataframe_to_labels(
    df: pd.DataFrame,
    column: str,
    shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Convert a BGI dataframe containing cell labels to a sparse labels matrix.

    Parameters
    ----------
    df
        Read dataframe returned by :func:`read_bgi_as_dataframe`.
    column
        Column containing positive integer labels.
    shape
        Optional output shape. If omitted, inferred from ``x`` and ``y``.

    Returns
    -------
    numpy.ndarray
        Labels matrix.
    """
    shape = shape or (int(df["x"].max()) + 1, int(df["y"].max()) + 1)
    labels = np.zeros(shape, dtype=int)

    for label, sub_df in df.drop_duplicates(subset=[column, "x", "y"]).groupby(column):
        if label <= 0:
            continue
        labels[(sub_df["x"].values, sub_df["y"].values)] = label
    return labels


def dataframe_to_filled_labels(
    df: pd.DataFrame,
    column: str,
    shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Convert a BGI dataframe containing cell labels to a filled labels matrix.

    Parameters
    ----------
    df
        Read dataframe returned by :func:`read_bgi_as_dataframe`.
    column
        Column containing positive integer labels.
    shape
        Optional output shape. If omitted, inferred from ``x`` and ``y``.

    Returns
    -------
    numpy.ndarray
        Filled labels matrix.
    """
    shape = shape or (int(df["x"].max()) + 1, int(df["y"].max()) + 1)
    labels = np.zeros(shape, dtype=int)

    for label, sub_df in df.drop_duplicates(subset=[column, "x", "y"]).groupby(column):
        if label <= 0:
            continue
        points = sub_df[["x", "y"]].values.astype(int)
        min_offset = points.min(axis=0)
        max_offset = points.max(axis=0)
        xmin, ymin = min_offset
        xmax, ymax = max_offset
        points -= min_offset
        hull = cv2.convexHull(points, returnPoints=True)
        mask = cv2.fillConvexPoly(
            np.zeros((max_offset - min_offset + 1)[::-1], dtype=np.uint8),
            hull,
            color=1,
        ).T
        labels[xmin : xmax + 1, ymin : ymax + 1][mask == 1] = label
    return labels


@register_function(
    aliases=["read_bgi_agg", "bgi agg", "stereo agg", "读取bgi聚合", "读取stereo聚合"],
    category="io",
    description="Read BGI / Stereo-seq read-level data and aggregate counts onto a coordinate grid.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "adata = st.io.spatial.read_bgi_agg('reads.tsv')",
        "adata = st.io.spatial.read_bgi_agg('reads.tsv', stain_path='stain.tif', binsize=20)",
    ],
    related=["io.spatial.read_bgi"],
)
def read_bgi_agg(
    path: Union[str, PathLike],
    stain_path: Optional[Union[str, PathLike]] = None,
    binsize: int = 1,
    gene_agg: Optional[Dict[str, Union[List[str], Callable[[str], bool]]]] = None,
    prealigned: bool = False,
    label_column: Optional[str] = None,
    version: Literal["stereo"] = "stereo",
) -> AnnData:
    """
    Read BGI / Stereo-seq read-level data and aggregate UMIs per coordinate.

    Parameters
    ----------
    path
        Path to the BGI / Stereo-seq read file.
    stain_path
        Optional path to a stain image in the same coordinate system as the read file.
    binsize
        Spatial bin size used to aggregate counts.
    gene_agg
        Optional dictionary mapping layer names to gene lists or gene filter callables.
    prealigned
        Whether the stain image is already aligned to the minimum RNA coordinates.
    label_column
        Optional read-level column containing pre-existing cell labels.
    version
        Platform/version key used to attach spatial scale metadata.

    Returns
    -------
    anndata.AnnData
        Aggregated coordinate-grid AnnData. Total counts are stored in ``.X``;
        optional stain and label matrices are attached to ``.layers``.
    """
    file_path = Path(path).resolve()
    if binsize < 1 or int(binsize) != binsize:
        raise ValueError("`binsize` must be a positive integer.")

    _progress(f"Reading BGI aggregate data from: {file_path}")
    data = read_bgi_as_dataframe(file_path, label_column=label_column)

    x = data["x"].to_numpy(dtype=np.int64)
    y = data["y"].to_numpy(dtype=np.int64)
    x_min, y_min = int(x.min()), int(y.min())
    x_max, y_max = int(x.max()), int(y.max())
    shape = (x_max + 1, y_max + 1)

    layers = {}
    image = None
    labels = None

    if stain_path is not None:
        stain_file = Path(stain_path).resolve()
        if not stain_file.exists():
            raise FileNotFoundError(f"Stain image not found: {stain_file}")
        _progress(f"Loading stain image: {stain_file.name}")
        image = skimage.io.imread(stain_file)

        if prealigned:
            warnings.warn(
                "Assuming stain image is already aligned to the minimum RNA coordinates (prealigned=True)."
            )
            image = _pad_image_xy(image, pad_before_xy=(x_min, y_min))

        x_max = max(x_max, image.shape[0] - 1)
        y_max = max(y_max, image.shape[1] - 1)
        shape = (x_max + 1, y_max + 1)

        if image.shape[:2] != shape:
            warnings.warn(f"Padding stain image from {image.shape[:2]} to {shape} with zeros.")
            image = _pad_image_xy(image, target_xy_shape=shape)
        layers[SKM.STAIN_LAYER_KEY] = image

    if "label" in data.columns:
        _progress("Detected read-level labels; constructing labels matrix", level="warn")
        labels = dataframe_to_labels(data, "label", shape=shape)
        layers[SKM.LABELS_LAYER_KEY] = labels

    if binsize > 1:
        _progress(f"Binning counts with binsize={binsize}")
        shape = (math.ceil(shape[0] / binsize), math.ceil(shape[1] / binsize))
        x = bin_indices(x, 0, binsize)
        y = bin_indices(y, 0, binsize)
        x_min, y_min = int(x.min()), int(y.min())

        if image is not None:
            layers[SKM.STAIN_LAYER_KEY] = cv2.resize(image, shape[::-1])
        if labels is not None:
            warnings.warn(
                "Cell labels were provided and binsize > 1. Downsampled labels may contain slight inconsistencies."
            )
            layers[SKM.LABELS_LAYER_KEY] = labels[::binsize, ::binsize]

    _progress("Constructing count matrices")
    X = csr_matrix((data["total"].values, (x, y)), shape=shape, dtype=np.uint16)

    if "spliced" in data.columns:
        layers[SKM.SPLICED_LAYER_KEY] = csr_matrix(
            (data["spliced"].values, (x, y)),
            shape=shape,
            dtype=np.uint16,
        )
    if "unspliced" in data.columns:
        layers[SKM.UNSPLICED_LAYER_KEY] = csr_matrix(
            (data["unspliced"].values, (x, y)),
            shape=shape,
            dtype=np.uint16,
        )

    if gene_agg:
        _progress("Aggregating custom gene layers")
        for layer_name, genes in gene_agg.items():
            mask = data["geneID"].isin(genes) if isinstance(genes, list) else data["geneID"].map(genes)
            subset = data[mask]
            layers[layer_name] = csr_matrix(
                (subset["total"].values, (subset["x"].values, subset["y"].values)),
                shape=shape,
                dtype=np.uint16,
            )

    adata = AnnData(X=X, layers=layers)[x_min:, y_min:].copy()
    SKM.init_adata_type(adata, SKM.ADATA_AGG_TYPE)
    _initialize_spatial_metadata(
        adata,
        binsize=int(binsize),
        version=version,
        io_type="bgi_agg",
        source_path=file_path,
    )
    _progress(f"Set Spadeo-specific key values:adata.uns['__type'] and adata.uns['pp'].", level="step")
    _progress(f"Done (n_obs={adata.n_obs}, n_vars={adata.n_vars})", level="success")
    return adata


@register_function(
    aliases=["read_bgi", "bgi", "stereo", "读取bgi", "读取stereo"],
    category="io",
    description="Read BGI / Stereo-seq read-level data as bin-level or label-level AnnData.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "adata = st.io.spatial.read_bgi('reads.tsv', binsize=20)",
        "adata = st.io.spatial.read_bgi('reads.tsv', segmentation_adata=seg_adata, labels_layer='labels')",
        "adata = st.io.spatial.read_bgi('reads.tsv', label_column='cell_id')",
    ],
    related=["io.spatial.read_bgi_agg"],
)
@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE, "segmentation_adata", optional=True)
def read_bgi(
    path: Union[str, PathLike],
    binsize: Optional[int] = None,
    segmentation_adata: Optional[AnnData] = None,
    labels_layer: Optional[str] = None,
    labels: Optional[Union[np.ndarray, str, PathLike]] = None,
    seg_binsize: int = 1,
    label_column: Optional[str] = None,
    add_props: bool = True,
    version: Literal["stereo"] = "stereo",
) -> AnnData:
    """
    Read BGI / Stereo-seq read-level data as bin-by-gene or label-by-gene AnnData.

    Exactly one of ``binsize``, ``segmentation_adata``, ``labels``, or
    ``label_column`` must be provided.

    Parameters
    ----------
    path
        Path to the BGI / Stereo-seq read file.
    binsize
        Spatial bin size used to aggregate reads into pseudo-cells.
    segmentation_adata
        Aggregation-type AnnData containing segmentation results.
    labels_layer
        Layer name in ``segmentation_adata`` containing label masks.
    labels
        Numpy array or path to a ``.npy`` file containing labels.
    seg_binsize
        Bin size used when producing the segmentation labels.
    label_column
        Optional read-level column containing pre-existing cell labels.
    add_props
        Whether to attach label/bin properties such as area, centroid, contour,
        and bounding box.
    version
        Platform/version key used to attach spatial scale metadata.

    Returns
    -------
    anndata.AnnData
        Label-by-gene or bin-by-gene AnnData.
    """
    option_count = sum(
        [
            binsize is not None,
            segmentation_adata is not None,
            labels is not None,
            label_column is not None,
        ]
    )
    if option_count != 1:
        raise IOError("Exactly one of `segmentation_adata`, `binsize`, `labels`, or `label_column` must be provided.")
    if (segmentation_adata is None) ^ (labels_layer is None):
        raise IOError("Both `segmentation_adata` and `labels_layer` must be provided together.")
    if segmentation_adata is not None and SKM.get_adata_type(segmentation_adata) != SKM.ADATA_AGG_TYPE:
        raise IOError("Only `AGG` type AnnData objects are supported for `segmentation_adata`.")
    if binsize is not None and (int(binsize) != binsize or binsize < 1):
        raise IOError("`binsize` must be a positive integer.")
    if isinstance(labels, (str, PathLike)):
        labels = np.load(labels)

    file_path = Path(path).resolve()
    _progress(f"Reading BGI read-level data from: {file_path}")
    data = read_bgi_as_dataframe(file_path, label_column=label_column)

    uniq_gene = sorted(data["geneID"].unique())
    props = None

    if label_column is not None:
        _progress(f"Using precomputed read-level labels from column: {label_column}")
        binsize = 1
        data = data[data["label"] > 0].copy()
        if add_props:
            warnings.warn(
                "Using `label_column` with `add_props=True` may yield imperfect contours when labels are sparse."
            )
            props = get_points_props(data[["x", "y", "label"]])

    elif binsize is not None:
        _progress(f"Using binsize={binsize}")
        if binsize < 2:
            warnings.warn("Please consider using a larger `binsize` for BGI / Stereo-seq bin aggregation.")

        if binsize > 1:
            data = data.copy()
            data["x"] = bin_indices(data["x"].values, 0, binsize)
            data["y"] = bin_indices(data["y"].values, 0, binsize)

        data["label"] = data["x"].astype(str) + "-" + data["y"].astype(str)
        if add_props:
            props = get_bin_props(data[["x", "y", "label"]].drop_duplicates(), binsize)

    else:
        binsize = 1
        data_shape = (int(data["x"].max()) + 1, int(data["y"].max()) + 1)

        if labels is not None:
            _progress("Using labels provided via `labels`")
            if labels.shape != data_shape:
                warnings.warn(
                    f"Labels matrix shape {labels.shape} differs from inferred data extent {data_shape}. "
                    "Proceeding with the provided labels."
                )
        else:
            _progress("Using labels from `segmentation_adata` and `labels_layer`")
            labels = SKM.select_layer_data(segmentation_adata, labels_layer)

        label_coords = get_coords_labels(labels)

        if labels_layer is not None:
            seg_binsize = SKM.get_uns_spatial_attribute(segmentation_adata, SKM.UNS_SPATIAL_BINSIZE_KEY)
            x_min = int(segmentation_adata.obs_names[0]) * seg_binsize
            y_min = int(segmentation_adata.var_names[0]) * seg_binsize
            label_coords["x"] += x_min
            label_coords["y"] += y_min

        if seg_binsize > 1:
            warnings.warn("Binning was used during segmentation; expanding label coordinates to match read-level pixels.")
            coords_dfs = []
            for i in range(seg_binsize):
                for j in range(seg_binsize):
                    coords = label_coords.copy()
                    coords["x"] += i
                    coords["y"] += j
                    coords_dfs.append(coords)
            label_coords = pd.concat(coords_dfs, ignore_index=True)

        data = pd.merge(data, label_coords, on=["x", "y"], how="inner")
        if add_props:
            props = get_label_props(labels)

    uniq_cell = sorted(data["label"].unique())
    shape = (len(uniq_cell), len(uniq_gene))
    cell_dict = dict(zip(uniq_cell, range(len(uniq_cell))))
    gene_dict = dict(zip(uniq_gene, range(len(uniq_gene))))
    x_ind = data["label"].map(cell_dict).astype(int).values
    y_ind = data["geneID"].map(gene_dict).astype(int).values

    _progress("Constructing count matrices")
    X = csr_matrix((data["total"].values, (x_ind, y_ind)), shape=shape)
    layers = {}
    if "spliced" in data.columns:
        layers[SKM.SPLICED_LAYER_KEY] = csr_matrix((data["spliced"].values, (x_ind, y_ind)), shape=shape)
    if "unspliced" in data.columns:
        layers[SKM.UNSPLICED_LAYER_KEY] = csr_matrix((data["unspliced"].values, (x_ind, y_ind)), shape=shape)

    obs = pd.DataFrame(index=pd.Index(uniq_cell, dtype="object"))
    var = pd.DataFrame(index=pd.Index(uniq_gene, dtype="object"))
    adata = AnnData(X=X, obs=obs, var=var, layers=layers)

    #Set spateo keys
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    _initialize_spatial_metadata(
        adata,
        binsize=int(binsize),
        version=version,
        io_type="bgi",
        source_path=file_path,
    )
    _progress(f"Set Spadeo-specific key values:adata.uns['__type'] and adata.uns['pp'].", level="step")

    adata.obs["label"] = adata.obs_names.astype(str)

    if props is not None:
        ordered_props = props.reindex(adata.obs_names)
        if "area" in ordered_props.columns:
            adata.obs["area"] = ordered_props["area"].values
        centroid_cols = [c for c in ordered_props.columns if c.startswith("centroid-")]
        bbox_cols = [c for c in ordered_props.columns if c.startswith("bbox-")]
        if centroid_cols:
            adata.obsm["spatial"] = ordered_props[centroid_cols].to_numpy()
        if "contour" in ordered_props.columns:
            adata.obsm["contour"] = ordered_props["contour"].values
        if bbox_cols:
            adata.obsm["bbox"] = ordered_props[bbox_cols].to_numpy()
    
    _progress(f"Done (n_obs={adata.n_obs}, n_vars={adata.n_vars})", level="success")
    return adata
