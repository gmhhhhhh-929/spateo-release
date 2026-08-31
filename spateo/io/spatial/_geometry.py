"""Geometry helpers used by spatial readers and alignment tools."""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union

from ...logging import logger_manager as lm
from ._stereoseq import read_bgi_agg
from ._utils import centroids


def alpha_shape(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 1,
    buffer: float = 1,
    vectorize: bool = True,
) -> Tuple[Union[MultiPolygon, Polygon], List]:
    """Compute a robust two-dimensional alpha shape from point coordinates."""

    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be a finite positive value.")
    x_values = np.asarray(x, dtype=float).ravel()
    y_values = np.asarray(y, dtype=float).ravel()
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must contain the same number of coordinates.")
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    coords = np.column_stack((x_values[finite], y_values[finite]))
    if coords.size == 0:
        raise ValueError("At least one finite coordinate pair is required.")
    points = MultiPoint(coords)
    if len(points.geoms) < 4:
        hull = points.convex_hull
        return hull.buffer(buffer) if buffer else hull, []

    triangulation = Delaunay(coords)
    edge_points: List = []
    if vectorize:
        triangles = coords[triangulation.simplices]
        a = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1)
        b = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1)
        c = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)
        semiperimeter = (a + b + c) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            areas = np.sqrt(semiperimeter * (semiperimeter - a) * (semiperimeter - b) * (semiperimeter - c))
            circumradius = a * b * c / (4.0 * areas)
        filtered = triangles[np.isfinite(circumradius) & (circumradius < 1.0 / alpha)]
        if filtered.size:
            edge_points = np.concatenate(
                (filtered[:, (0, 1)], filtered[:, (1, 2)], filtered[:, (2, 0)]),
                axis=0,
            ).tolist()
    else:
        edges = set()

        def add_edge(first: int, second: int) -> None:
            if (first, second) in edges or (second, first) in edges:
                return
            edges.add((first, second))
            edge_points.append(coords[[first, second]])

        for first, second, third in triangulation.simplices:
            point_a, point_b, point_c = coords[first], coords[second], coords[third]
            a = math.dist(point_a, point_b)
            b = math.dist(point_b, point_c)
            c = math.dist(point_c, point_a)
            semiperimeter = (a + b + c) / 2.0
            area_squared = semiperimeter * (semiperimeter - a) * (semiperimeter - b) * (semiperimeter - c)
            if area_squared <= 0:
                continue
            circumradius = a * b * c / (4.0 * math.sqrt(area_squared))
            if circumradius < 1.0 / alpha:
                add_edge(first, second)
                add_edge(second, third)
                add_edge(third, first)

    polygons = list(polygonize(edge_points))
    alpha_hull = unary_union(polygons) if polygons else points.convex_hull
    if buffer:
        alpha_hull = alpha_hull.buffer(buffer)
    return alpha_hull, edge_points


def get_concave_hull(
    path: str,
    binsize: int = 20,
    min_agg_umi: Optional[int] = None,
    alpha: float = 1.0,
    buffer: Optional[float] = None,
) -> Tuple[Union[MultiPolygon, Polygon], List]:
    """Compute the concave hull of non-empty Stereo-seq aggregate bins."""

    adata = read_bgi_agg(path, binsize=binsize)
    threshold = binsize - 1 if min_agg_umi is None else min_agg_umi
    x_index, y_index = (adata.X > threshold).nonzero()
    if x_index.size == 0:
        raise ValueError("No Stereo-seq bins pass min_agg_umi.")

    x_min, y_min = int(adata.obs_names[0]), int(adata.var_names[0])
    if binsize != 1:
        x_index = centroids(x_index, coord_min=x_min, binsize=binsize)
        y_index = centroids(y_index, coord_min=y_min, binsize=binsize)
    else:
        x_index, y_index = x_index + x_min, y_index + y_min

    alpha_hull, edge_points = alpha_shape(
        x_index,
        y_index,
        alpha=alpha,
        buffer=binsize if buffer is None else buffer,
        vectorize=True,
    )
    if alpha_hull.is_empty:
        lm.main_warning(f"No alpha shape was identified; try alpha smaller than {alpha}.")
    return alpha_hull, edge_points


__all__ = ["alpha_shape", "get_concave_hull"]
