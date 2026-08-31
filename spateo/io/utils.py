"""Compatibility imports for maintained spatial I/O helpers."""

from .spatial._utils import (
    bin_indices,
    bin_matrix,
    centroids,
    contour_to_geo,
    get_bin_props,
    get_coords_labels,
    get_label_props,
    get_points_props,
    in_concave_hull,
    in_convex_hull,
)

__all__ = [
    "bin_indices",
    "centroids",
    "contour_to_geo",
    "get_points_props",
    "get_label_props",
    "get_bin_props",
    "in_concave_hull",
    "in_convex_hull",
    "bin_matrix",
    "get_coords_labels",
]
