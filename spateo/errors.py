class SpateoError(Exception):
    """Base exception for Spateo errors."""


class ConfigurationError(SpateoError):
    """Raised when Spateo receives an invalid configuration."""


class PreprocessingError(SpateoError):
    """Raised when preprocessing cannot be completed."""


class SpatialKeyError(PreprocessingError):
    """Raised when spatial coordinates are missing or invalid."""


class LayerKeyError(PreprocessingError):
    """Raised when a requested AnnData layer is missing."""


class PlottingError(SpateoError):
    """Raised when plotting cannot be completed."""


class SegmentationError(SpateoError):
    """Raised when segmentation cannot be completed."""
