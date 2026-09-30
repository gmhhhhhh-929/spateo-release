"""Bounded inventories and resolved input descriptions shared by native readers."""

import os
from dataclasses import dataclass, field
from pathlib import Path

_SKIP_DIRS = {".git", "analysis", "images", "morphology_focus", "CellComposite", "CellLabels", "cell_boundaries"}


@dataclass
class Candidate:
    technology: str
    root: Path
    counts: Path
    metadata: Path
    representation: str
    options: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)

    @property
    def identity(self):
        # Representations of one source are grouped independently from platform claims.
        if self.technology in {"seekspace", "bmkmanu", "salus", "singleron"}:
            return str(self.counts), "domestic_native_matrix"
        return str(self.counts), self.representation


def inventory(path: Path, max_files: int, max_depth: int):
    """Inspect known roots and bounded sample containers without following symlinks."""
    files, roots, diagnostics = [], set(), []
    if path.is_file():
        root = path.parent
    else:
        root = path
    stack = [(root, 0)]
    visited = 0
    while stack:
        directory, depth = stack.pop()
        roots.add(directory)
        try:
            with os.scandir(directory) as iterator:
                children = []
                for item in iterator:
                    visited += 1
                    if visited > max_files:
                        diagnostics.append(
                            dict(
                                code="discovery_limit",
                                severity="error",
                                path=str(directory),
                                message="Inventory limit reached; discovery is incomplete.",
                            )
                        )
                        return sorted(files), sorted(roots), diagnostics
                    if item.is_symlink():
                        diagnostics.append(
                            dict(
                                code="symlink_skipped",
                                severity="error",
                                path=item.path,
                                message="Symlinks are not followed; pass the intended data directory directly.",
                            )
                        )
                        continue
                    p = Path(item.path)
                    if item.is_file():
                        files.append(p)
                    elif item.is_dir() and not item.name.startswith("."):
                        if item.name in _SKIP_DIRS:
                            continue
                        if depth < max_depth:
                            children.append((p, depth + 1))
                        else:
                            diagnostics.append(
                                dict(
                                    code="depth_limit",
                                    severity="error",
                                    path=str(p),
                                    message="Directory outside discovery depth; pass it directly.",
                                )
                            )
                stack.extend(sorted(children, reverse=True))
        except OSError as exc:
            diagnostics.append(dict(code="discovery_error", severity="error", path=str(directory), message=str(exc)))
    return sorted(files), sorted(roots), diagnostics
