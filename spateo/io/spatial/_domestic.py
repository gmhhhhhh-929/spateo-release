"""Compatibility exports; platform implementations are in separate modules."""

from ._bmkmanu import read_bmkmanu
from ._salus import read_salus
from ._seekspace import read_seekspace
from ._singleron import read_singleron

__all__ = ["read_seekspace", "read_bmkmanu", "read_salus", "read_singleron"]
