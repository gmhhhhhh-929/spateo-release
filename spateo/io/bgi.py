"""Compatibility imports for the maintained Stereo-seq/BGI reader."""

from .spatial._stereoseq import (
    COUNT_COLUMN_MAPPING,
    VERSIONS,
    dataframe_to_filled_labels,
    dataframe_to_labels,
    read_bgi,
    read_bgi_agg,
    read_bgi_as_dataframe,
)

__all__ = [
    "COUNT_COLUMN_MAPPING",
    "VERSIONS",
    "read_bgi_as_dataframe",
    "dataframe_to_labels",
    "dataframe_to_filled_labels",
    "read_bgi_agg",
    "read_bgi",
]
