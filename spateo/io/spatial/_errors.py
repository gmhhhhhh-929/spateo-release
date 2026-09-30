"""Failures shared by bounded spatial format adapters."""


class ContractError(ValueError):
    """Required data cannot satisfy the selected format contract."""


class ResourceDeferred(MemoryError):
    """Reading would exceed the configured resource budget."""
