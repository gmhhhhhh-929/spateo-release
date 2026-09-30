"""Explicit outcomes and opt-in deferred materialization for spatial reading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Callable, Dict, Optional

from anndata import AnnData

from ._recovery import diagnostic_report

POLICY_VERSION = "spatial-contracts-v1"


@dataclass
class SpatialDataset:
    """One logical input, including unsuccessful or deferred inputs.

    ``load`` materializes the complete native matrix through its resolved reader;
    it is not a backed AnnData or an out-of-core slicing implementation. Metadata
    inspection and ``to_dict`` never materialize data. Retry requires an explicit
    request after an unsuccessful attempt, and never switches platform.
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
    lazy: bool = False
    materialization_attempts: int = 0
    required_files: list = field(default_factory=list)
    _loader: Optional[Callable] = field(default=None, repr=False)
    _memory_budget: int = field(default=0, repr=False)
    _last_budget: Optional[int] = field(default=None, repr=False)
    _lock: object = field(default_factory=RLock, repr=False, compare=False)

    def load(self, *, max_memory_bytes: Optional[int] = None, retry: bool = False) -> "SpatialDataset":
        """Materialize once; explicitly retry with ``retry=True`` or a new budget.

        Repeated calls after resource deferral with the same budget do no work.
        A failed read needs ``retry=True``; source mutation always requires a new
        ``read_spatial`` call so that discovery and reader selection are repeated.
        """
        if max_memory_bytes is not None and (
            isinstance(max_memory_bytes, bool) or not isinstance(max_memory_bytes, int) or max_memory_bytes <= 0
        ):
            raise ValueError("max_memory_bytes must be a positive integer")
        with self._lock:
            if self.status == "ready":
                return self
            if any(d.get("code") == "source_changed" for d in self.diagnostics):
                raise ValueError("Source changed; call read_spatial again before loading this input.")
            if self._loader is None or self.status not in ("deferred", "failed"):
                raise ValueError(f"Entry {self.key!r} is {self.status}; no resolved reader is available.")
            if self.status == "failed" and not retry:
                raise ValueError("Previous reading failed; inspect diagnostics, then explicitly set retry=True.")
            budget = self._memory_budget if max_memory_bytes is None else max_memory_bytes
            if (
                self.materialization_attempts
                and self.status == "deferred"
                and not retry
                and budget == self._last_budget
            ):
                return self
            self.materialization_attempts += 1
            self._memory_budget = budget
            self._last_budget = budget
            self._loader(self, budget)
            return self

    def materialize(self, *, max_memory_bytes: Optional[int] = None, retry: bool = False) -> AnnData:
        """Return the complete AnnData, or raise with the recorded failure status."""
        self.load(max_memory_bytes=max_memory_bytes, retry=retry)
        if self.status != "ready" or self.adata is None:
            raise ValueError(f"Entry {self.key!r} remains {self.status}; inspect its diagnostics and recovery actions.")
        return self.adata

    def to_dict(self) -> dict:
        validation = dict(self.validation)
        if "candidate_diagnostics" in validation:
            validation["candidate_diagnostics"] = [
                diagnostic_report(d, technology=self.technology, source=self.source, required_files=self.required_files)
                for d in validation["candidate_diagnostics"]
            ]
        return {
            "key": self.key,
            "technology": self.technology,
            "source": self.source,
            "representation": self.representation,
            "status": self.status,
            "shape": list(self.adata.shape) if self.adata is not None else None,
            "estimated_bytes": self.estimated_bytes,
            "lazy": self.lazy,
            "materialization_attempts": self.materialization_attempts,
            "last_memory_budget_bytes": self._last_budget,
            "required_files": self.required_files,
            "evidence": self.evidence,
            "validation": validation,
            "diagnostics": [
                diagnostic_report(d, technology=self.technology, source=self.source, required_files=self.required_files)
                for d in self.diagnostics
            ],
        }


@dataclass
class SpatialReadResult:
    """Named inputs and diagnostics for single or batch native spatial data.

    ``adata`` can materialize a unique entry only when ``lazy=True`` was requested
    and discovery is complete. ``load=False`` remains inspection-only. Reports
    contain metadata, never executable loaders or a resumable serialized object.
    """

    source: str
    datasets: Dict[str, SpatialDataset] = field(default_factory=dict)
    diagnostics: list = field(default_factory=list)
    discovery: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.datasets)

    def __iter__(self):
        return iter(self.datasets)

    def __getitem__(self, key):
        return self.datasets[key]

    def get(self, key, default=None):
        """Return a named entry without materializing it."""
        return self.datasets.get(key, default)

    def keys(self):
        return self.datasets.keys()

    def values(self):
        return self.datasets.values()

    def items(self):
        return self.datasets.items()

    def load(self, key=None, *, max_memory_bytes: Optional[int] = None, retry: bool = False) -> "SpatialReadResult":
        """Load one named entry, or all resolved entries when ``key`` is omitted.

        Already-ready entries are reused. Unresolved inputs remain in the report;
        failed reads are skipped unless ``retry=True``. This retains loaded arrays
        in the collection and respects its memory budget; it does not evict data.
        """
        selected = [self.datasets[key]] if key is not None else list(self.datasets.values())
        for entry in selected:
            if key is None and (entry._loader is None or (entry.status == "failed" and not retry)):
                continue
            if key is None and any(d.get("code") == "source_changed" for d in entry.diagnostics):
                continue
            entry.load(max_memory_bytes=max_memory_bytes, retry=retry)
        return self

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
        if len(self.datasets) == 1 and not any(d.get("severity") == "error" for d in self.diagnostics):
            entry = next(iter(self.datasets.values()))
            # Only the first lazy access may trigger work. Resource and validation
            # failures never cause a hidden second attempt on repeated access.
            if entry.lazy and entry.status == "deferred" and not entry.materialization_attempts:
                entry.load()
        if len(self.datasets) != 1 or self.status != "ok":
            raise ValueError("No unique complete AnnData; inspect result.datasets and result.report.")
        return next(iter(self.datasets.values())).adata

    @property
    def report(self) -> dict:
        """Fresh JSON-serializable report; inspecting it never loads matrix data."""
        return {
            "policy_version": POLICY_VERSION,
            "source": self.source,
            "status": self.status,
            "discovery": self.discovery,
            "diagnostics": [diagnostic_report(d, source=self.source) for d in self.diagnostics],
            "datasets": {key: entry.to_dict() for key, entry in self.datasets.items()},
        }

    def write_report(self, path) -> None:
        """Explicitly export diagnostics; reading itself never writes files."""
        import json

        Path(path).write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8")
