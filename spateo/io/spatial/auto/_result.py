"""Explicit outcomes for score-free spatial reading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from anndata import AnnData

POLICY_VERSION = "spatial-contracts-v1"


@dataclass
class SpatialDataset:
    """One logical input, including unsuccessful or deferred inputs.

    ``load()`` retries only this input's resolved adapter. It never switches
    technology. Resource limits remain in effect unless explicitly overridden.
    """

    key: str
    technology: str
    source: str
    representation: str
    status: str = "discovered"
    adata: Optional[AnnData] = field(default=None, repr=False)
    diagnostics: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    estimated_bytes: int = 0
    _loader: Optional[Callable] = field(default=None, repr=False)

    def load(self, *, max_memory_bytes: Optional[int] = None) -> "SpatialDataset":
        """Load a deferred entry, preserving its identity and diagnostic history."""
        if self.status == "ready":
            return self
        if self.status != "deferred" or self._loader is None:
            raise ValueError(f"Entry {self.key!r} is {self.status}; only deferred entries can be loaded.")
        if max_memory_bytes is not None and max_memory_bytes <= 0:
            raise ValueError("max_memory_bytes must be positive")
        self._loader(self, max_memory_bytes)
        return self

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "technology": self.technology,
            "source": self.source,
            "representation": self.representation,
            "status": self.status,
            "shape": list(self.adata.shape) if self.adata is not None else None,
            "estimated_bytes": self.estimated_bytes,
            "evidence": self.evidence,
            "validation": self.validation,
            "diagnostics": self.diagnostics,
        }


@dataclass
class SpatialReadResult:
    """Stable return type for :func:`read_spatial`, for single and batch inputs.

    ``adata`` is available only for exactly one ready input and a complete
    discovery scope. Inspect ``datasets`` and ``report`` in every other case.
    """

    source: str
    datasets: Dict[str, SpatialDataset] = field(default_factory=dict)
    diagnostics: list = field(default_factory=list)
    discovery: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        states = [entry.status for entry in self.datasets.values()]
        scope_error = any(d.get("severity") == "error" for d in self.diagnostics)
        if states and all(s == "ready" for s in states) and not scope_error:
            return "ok"
        if "ready" in states:
            return "partial"
        if "deferred" in states:
            return "pending" if all(s == "deferred" for s in states) and not scope_error else "partial"
        return "failed"

    @property
    def adata(self) -> AnnData:
        if len(self.datasets) != 1 or self.status != "ok":
            raise ValueError("No unique complete AnnData; inspect result.datasets and result.report.")
        return next(iter(self.datasets.values())).adata

    @property
    def report(self) -> dict:
        """Fresh JSON-serializable report, including subsequent deferred loads."""
        return {
            "policy_version": POLICY_VERSION,
            "source": self.source,
            "status": self.status,
            "discovery": self.discovery,
            "diagnostics": self.diagnostics,
            "datasets": {key: entry.to_dict() for key, entry in self.datasets.items()},
        }

    def write_report(self, path) -> None:
        """Explicitly export diagnostics; reading itself never writes files."""
        import json

        Path(path).write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8")
