"""Automatic discovery delegates native schemas to independent platform modules."""

from .._native_readers import DOMESTIC, discover_domestic, get_reader

__all__ = ["DOMESTIC", "discover_domestic", "get_reader"]
