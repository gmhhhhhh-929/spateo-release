#!/usr/bin/env python3
"""Audit Python and notebook code for legacy Spateo/Dynamo usage.

The script reads code only; notebook outputs are ignored.  Its JSON report is
also consumed by the figure-script migration workflow.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterator, NamedTuple


PATTERNS = {
    "dynamo_import": re.compile(r"^\s*(?:import\s+dynamo|from\s+dynamo\s+import)", re.MULTILINE),
    "dynamo_call": re.compile(r"\bdyn\."),
    "dynamo_size_factor": re.compile(r"\bnormalize_cell_expr_by_size_factors\s*\("),
    "numpy_log1p": re.compile(r"\bnp\.log1p\s*\("),
    "pot_monkeypatch": re.compile(r"\bot\.gromov\.cg\s*=\s*ot\.optim\.cg\b"),
    "cell_directions": re.compile(r"\bcell_directions\s*\("),
}


class CodeUnit(NamedTuple):
    path: Path
    cell: int | None
    source: str


def iter_code_units(root: Path) -> Iterator[CodeUnit]:
    for path in sorted(root.rglob("*")):
        if path.suffix == ".py":
            yield CodeUnit(path=path, cell=None, source=path.read_text(encoding="utf-8"))
        elif path.suffix == ".ipynb":
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") == "code":
                    yield CodeUnit(path=path, cell=index, source="".join(cell.get("source", [])))


def matching_lines(source: str, pattern: re.Pattern[str]) -> list[dict[str, object]]:
    lines = source.splitlines()
    matches: list[dict[str, object]] = []
    for match in pattern.finditer(source):
        line_number = source.count("\n", 0, match.start()) + 1
        start = max(0, line_number - 2)
        stop = min(len(lines), line_number + 2)
        matches.append(
            {
                "line": line_number,
                "context": "\n".join(lines[start:stop]),
            }
        )
    return matches


def audit(root: Path) -> dict[str, object]:
    findings: list[dict[str, object]] = []
    files: set[str] = set()
    notebooks: set[str] = set()
    for unit in iter_code_units(root):
        files.add(str(unit.path))
        if unit.path.suffix == ".ipynb":
            notebooks.add(str(unit.path))
        for name, pattern in PATTERNS.items():
            matches = matching_lines(unit.source, pattern)
            if matches:
                findings.append(
                    {
                        "kind": name,
                        "path": str(unit.path),
                        "cell": unit.cell,
                        "matches": matches,
                    }
                )
    return {
        "root": str(root),
        "script_files": len(files),
        "notebook_files": len(notebooks),
        "findings": findings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    result = audit(root)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
