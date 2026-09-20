"""Contract-based automatic spatial reading.

All automatic reader names return SpatialReadResult. Use read_spatial(load=False)
for discovery. The former scored detector API has been removed.
"""

from ._automatic import read_spatial
from ._result import SpatialDataset, SpatialReadResult

read_auto_spatial = read_spatial
read_spatial_auto = read_spatial

__all__ = ["read_spatial", "read_auto_spatial", "read_spatial_auto", "SpatialDataset", "SpatialReadResult"]
