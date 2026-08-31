"""Spatial image loading and AnnData image-layer helpers."""

from typing import Optional

import cv2
import numpy as np
from anndata import AnnData


def add_image_layer(
    adata: AnnData,
    img: np.ndarray,
    scale_factor: float,
    slice: Optional[str] = None,
    img_layer: Optional[str] = None,
) -> AnnData:
    """Store an image and its spatial scale in an AnnData object."""

    spatial = adata.uns.setdefault("spatial", {})
    library = spatial.setdefault(slice, {})
    library.setdefault("images", {})[img_layer] = img
    library.setdefault("scalefactors", {})[img_layer] = scale_factor
    return adata


def read_image(
    adata: AnnData,
    filename: str,
    scale_factor: float,
    slice: Optional[str] = None,
    img_layer: Optional[str] = None,
) -> AnnData:
    """Read an image and store it under ``adata.uns['spatial']``."""

    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read spatial image {filename!r}.")
    return add_image_layer(
        adata=adata,
        img=image,
        scale_factor=scale_factor,
        slice=slice,
        img_layer=img_layer,
    )


__all__ = ["add_image_layer", "read_image"]
