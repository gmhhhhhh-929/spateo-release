"""Unified spatial preprocessing API with legacy compatibility exports."""

from . import auxseg, filter, image
from .feature import select_spatial_features
from .filter import filter_by_coordinates, filter_cells, filter_genes
from .graph import expression_neighbors, spatial_neighbors
from .normalization import calculate_size_factors, normalize_total
from .pca import pca
from .preprocessor import Preprocessor, SpatialPreprocessor, preprocess_spatial
from .qc import (
    calculate_spatial_qc,
    filter_genes_by_spatial_qc,
    filter_spots,
    flag_local_qc_outliers,
)
from .transform import log1p, log1p_layer, scale, scale_layer
from .utils import standardize_spatial_adata

__all__ = [
    "auxseg",
    "filter",
    "image",
    "SpatialPreprocessor",
    "Preprocessor",
    "preprocess_spatial",
    "standardize_spatial_adata",
    "calculate_spatial_qc",
    "filter_spots",
    "filter_genes_by_spatial_qc",
    "flag_local_qc_outliers",
    "calculate_size_factors",
    "normalize_total",
    "log1p_layer",
    "scale_layer",
    "select_spatial_features",
    "pca",
    "spatial_neighbors",
    "expression_neighbors",
    # Historical filtering/transformation functions retained for downstream modules.
    "filter_cells",
    "filter_genes",
    "filter_by_coordinates",
    "log1p",
    "scale",
]
