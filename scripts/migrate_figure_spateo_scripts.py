#!/usr/bin/env python3
"""Create clean, Dynamo-free figure2/figure3 script copies for current Spateo.

Only Python scripts and Jupyter notebooks are copied.  Source files are never
modified.  Notebook outputs are cleared so stale tracebacks are not mistaken
for results from the migrated code.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any


CODE_SUFFIXES = {".py", ".ipynb"}
POT_PATCH = re.compile(r"ot\.gromov\.cg\s*=\s*ot\.optim\.cg")
DYNAMO_IMPORT = re.compile(
    r"^(?P<indent>\s*)(?P<statement>import\s+dynamo(?:\s+as\s+dyn)?|from\s+dynamo(?:\.[\w.]+)?\s+import\s+.+?)\s*$",
    re.MULTILINE,
)
PP_RESET = re.compile(
    r"^(?P<indent>\s*)(?P<statement>[A-Za-z_]\w*\.uns\s*\[\s*[\"']pp[\"']\s*\]\s*=\s*\{\s*\})\s*$",
    re.MULTILINE,
)
COUNT_X_ASSIGN = re.compile(
    r"^(?P<indent>\s*)(?P<statement>(?P<adata>[A-Za-z_]\w*)\.X\s*=\s*(?P=adata)\.layers\s*\[\s*[\"']counts_X[\"']\s*\]\.copy\(\))\s*$",
    re.MULTILINE,
)
COUNT_GENE_VIEW = re.compile(
    r"^(?P<indent>\s*)(?P<statement>(?P<adata>[A-Za-z_]\w*)\s*=\s*(?P<expression>(?P=adata)\[:,[^\n]+counts_X[^\n]+\]))\s*$",
    re.MULTILINE,
)


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def layer_reference(node: ast.AST) -> tuple[str, str] | None:
    if not isinstance(node, ast.Subscript):
        return None
    container = dotted_name(node.value)
    if not container or not container.endswith(".layers"):
        return None
    owner = container[: -len(".layers")]
    key_node = node.slice
    if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
        return owner, key_node.value
    return None


def keyword_expression(call: ast.Call, name: str) -> str | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return ast.unparse(keyword.value)
    return None


def commented_statement(original: list[str], indent: str, label: str) -> list[str]:
    result: list[str] = []
    for index, line in enumerate(original):
        content = line[len(indent) :] if line.startswith(indent) else line.lstrip()
        prefix = f"# OLD ({label}): " if index == 0 else "# "
        result.append(f"{indent}{prefix}{content}" if content else f"{indent}#")
    return result


def replace_legacy_statements(source: str) -> tuple[str, list[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, []

    lines = source.splitlines()
    replacements: list[tuple[int, int, list[str], str]] = []
    for node in ast.walk(tree):
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is None or end is None:
            continue

        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            function = dotted_name(call.func)
            adata = keyword_expression(call, "adata")
            indent = lines[start - 1][: len(lines[start - 1]) - len(lines[start - 1].lstrip())]
            old = lines[start - 1 : end]

            if function == "dyn.pp.normalize_cell_expr_by_size_factors" and adata:
                replacement = commented_statement(old, indent, "Dynamo")
                replacement.extend(
                    [
                        f'{indent}st.pp.normalize_total({adata}, layer="counts_X", out_layer="norm_X", target_sum=None, size_factor_key="Size_Factor", inplace=True)',
                        f'{indent}st.pp.log1p_layer({adata}, layer="norm_X", out_layer="log1p_X", set_X=True, inplace=True)',
                    ]
                )
                replacements.append((start - 1, end, replacement, "dynamo_size_factor_to_spateo"))
            elif function == "dyn.pp.normalize" and call.args:
                adata = ast.unparse(call.args[0])
                replacement = commented_statement(old, indent, "Dynamo")
                replacement.append(
                    f'{indent}st.pp.normalize_total({adata}, layer="X", out_layer="X", target_sum=None, size_factor_key="Size_Factor", inplace=True)'
                )
                replacements.append((start - 1, end, replacement, "dynamo_normalize_to_spateo"))
            elif function == "dyn.pp.log1p" and call.args:
                adata = ast.unparse(call.args[0])
                replacement = commented_statement(old, indent, "Dynamo")
                replacement.append(f"{indent}st.pp.log1p({adata}, copy=False)")
                replacements.append((start - 1, end, replacement, "dynamo_log1p_to_spateo"))

        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not isinstance(value, ast.Call) or dotted_name(value.func) != "np.log1p" or len(value.args) != 1:
                continue
            source_layer = layer_reference(value.args[0])
            target_layer = layer_reference(targets[0]) if len(targets) == 1 else None
            if source_layer is None or target_layer is None or source_layer[0] != target_layer[0]:
                continue
            owner = source_layer[0]
            indent = lines[start - 1][: len(lines[start - 1]) - len(lines[start - 1].lstrip())]
            old = lines[start - 1 : end]
            replacement = commented_statement(old, indent, "NumPy")
            replacement.append(
                f"{indent}st.pp.log1p_layer({owner}, layer={source_layer[1]!r}, out_layer={target_layer[1]!r}, set_X=False, inplace=True)"
            )
            replacements.append((start - 1, end, replacement, "numpy_layer_log1p_to_spateo"))

    changes: list[str] = []
    occupied: set[int] = set()
    for start, end, replacement, kind in sorted(replacements, reverse=True):
        span = set(range(start, end))
        if occupied.intersection(span):
            continue
        lines[start:end] = replacement
        occupied.update(span)
        changes.append(kind)
    trailing_newline = source.endswith("\n")
    rendered = "\n".join(lines)
    if trailing_newline:
        rendered += "\n"
    return rendered, sorted(changes)


def comment_block(source: str, reason: str) -> str:
    lines = [f"# {reason}", "# The original source remains unchanged in the source figure directory."]
    lines.extend(f"# OLD: {line}" if line else "#" for line in source.splitlines())
    return "\n".join(lines) + "\n"


def remove_dynamo_imports(source: str) -> tuple[str, bool]:
    changed = False

    def replacement(match: re.Match[str]) -> str:
        nonlocal changed
        changed = True
        return f'{match.group("indent")}# OLD (Dynamo dependency removed): {match.group("statement")}'

    return DYNAMO_IMPORT.sub(replacement, source), changed


def has_active_legacy_code(source: str) -> list[str]:
    active_lines = [line for line in source.splitlines() if not line.lstrip().startswith("#")]
    active = "\n".join(active_lines)
    issues: list[str] = []
    if re.search(r"^\s*(?:import\s+dynamo|from\s+dynamo\s+import)", active, re.MULTILINE):
        issues.append("active Dynamo import")
    if re.search(r"\bdyn\.", active):
        issues.append("active dyn call")
    if POT_PATCH.search(active):
        issues.append("active POT monkeypatch")
    if re.search(r"mesh_module\.fix_mesh\s*=", active):
        issues.append("active mesh monkeypatch")
    return issues


def transform_source(source: str) -> tuple[str, list[str]]:
    if "mesh_module.fix_mesh = safe_fix_mesh" in source:
        return (
            comment_block(
                source,
                "Current Spateo handles pymeshfix compatibility internally; this notebook monkeypatch is no longer used.",
            ),
            ["remove_meshfix_monkeypatch"],
        )

    changes: list[str] = []
    if POT_PATCH.search(source):
        source = comment_block(
            source,
            "Current Spateo resolves POT's conditional-gradient solver internally; no ot.gromov.cg monkeypatch is required.",
        )
        changes.append("remove_pot_monkeypatch")
        return source, changes

    source, statement_changes = replace_legacy_statements(source)
    changes.extend(statement_changes)

    if "dyn.tl.sample" in source:
        source = source.replace("dyn.tl.sample", "st.tl.sample")
        source = "# OLD (Dynamo): dyn.tl.sample; current Spateo: st.tl.sample\n" + source
        changes.append("dynamo_sample_to_spateo")

    if any(change.startswith("dynamo_") for change in changes):
        source, count = PP_RESET.subn(
            lambda match: (f'{match.group("indent")}# OLD (Dynamo metadata reset removed): {match.group("statement")}'),
            source,
        )
        if count:
            changes.append("preserve_spateo_preprocessing_metadata")
    if "dynamo_size_factor_to_spateo" in changes:
        source, count = COUNT_X_ASSIGN.subn(
            lambda match: (f'{match.group("indent")}# OLD (Dynamo input reset removed): {match.group("statement")}'),
            source,
        )
        if count:
            changes.append("read_counts_without_overwriting_x")

        def copy_gene_subset(match: re.Match[str]) -> str:
            return (
                f'{match.group("indent")}# OLD (AnnData view): {match.group("statement")}\n'
                f'{match.group("indent")}{match.group("adata")} = {match.group("expression")}.copy()'
            )

        source, count = COUNT_GENE_VIEW.subn(copy_gene_subset, source)
        if count:
            changes.append("materialize_anndata_subset")
    return source, sorted(set(changes))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def migrate_python(source_path: Path, output_path: Path) -> list[str]:
    source = source_path.read_text(encoding="utf-8")
    transformed, changes = transform_source(source)
    transformed, import_changed = remove_dynamo_imports(transformed)
    if import_changed:
        changes.append("remove_dynamo_import")
    issues = has_active_legacy_code(transformed)
    if issues:
        raise RuntimeError(f"{source_path}: {', '.join(issues)}")
    compile(transformed, str(output_path), "exec")
    output_path.write_text(transformed, encoding="utf-8")
    return sorted(set(changes))


def migrate_notebook(source_path: Path, output_path: Path) -> list[str]:
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    changes: list[str] = []
    code_cells: list[dict[str, Any]] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        code_cells.append(cell)
        source = "".join(cell.get("source", []))
        transformed, cell_changes = transform_source(source)
        cell["source"] = transformed.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
        changes.extend(cell_changes)

    if any(change.startswith("dynamo_") for change in changes):
        for cell in code_cells:
            source = "".join(cell.get("source", []))
            source, count = PP_RESET.subn(
                lambda match: (
                    f'{match.group("indent")}# OLD (Dynamo metadata reset removed): {match.group("statement")}'
                ),
                source,
            )
            if count:
                changes.append("preserve_spateo_preprocessing_metadata")
            cell["source"] = source.splitlines(keepends=True)

    combined = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
    if not re.search(
        r"\bdyn\.", "\n".join(line for line in combined.splitlines() if not line.lstrip().startswith("#"))
    ):
        for cell in code_cells:
            source = "".join(cell.get("source", []))
            source, import_changed = remove_dynamo_imports(source)
            if import_changed:
                changes.append("remove_dynamo_import")
            cell["source"] = source.splitlines(keepends=True)

    for index, cell in enumerate(code_cells):
        issues = has_active_legacy_code("".join(cell.get("source", [])))
        if issues:
            raise RuntimeError(f"{source_path}: code cell {index}: {', '.join(issues)}")

    notebook.setdefault("metadata", {})["spateo_migration"] = {
        "dynamo_required": False,
        "outputs_cleared": True,
        "workflow": "scripts/migrate_figure_spateo_scripts.py",
    }
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return sorted(set(changes + ["clear_notebook_outputs"]))


def migrate(source_roots: list[Path], output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent))
    records: list[dict[str, Any]] = []
    try:
        for source_root in source_roots:
            if not source_root.is_dir():
                raise NotADirectoryError(source_root)
            destination_root = staging / source_root.name
            for source_path in sorted(source_root.rglob("*")):
                if not source_path.is_file() or source_path.suffix not in CODE_SUFFIXES:
                    continue
                output_path = destination_root / source_path.relative_to(source_root)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if source_path.suffix == ".py":
                    changes = migrate_python(source_path, output_path)
                else:
                    changes = migrate_notebook(source_path, output_path)
                records.append(
                    {
                        "source": str(source_path),
                        "output": str(output_root / output_path.relative_to(staging)),
                        "source_sha256": sha256(source_path),
                        "output_sha256": sha256(output_path),
                        "changes": changes,
                    }
                )

        summary = {
            "status": "ok",
            "source_roots": [str(path) for path in source_roots],
            "output_root": str(output_root),
            "files": len(records),
            "dynamo_required": False,
            "records": records,
        }
        (staging / "MIGRATION_MANIFEST.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "README.md").write_text(
            "# Current Spateo figure scripts\n\n"
            "This directory is an immutable, code-only copy of figure2 and figure3. "
            "The source directories were not modified.\n\n"
            "- Dynamo preprocessing was replaced with Spateo-native normalization/log1p.\n"
            "- `dyn.tl.sample` was replaced with `st.tl.sample`.\n"
            "- POT and pymeshfix notebook monkeypatches were retired because current Spateo handles them internally.\n"
            "- Notebook outputs and execution counters were cleared.\n"
            "- See `MIGRATION_MANIFEST.json` for source/output hashes and per-file changes.\n",
            encoding="utf-8",
        )
        staging.rename(output_root)
    except Exception:
        raise
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_roots = [path.expanduser().resolve() for path in args.source]
    output_root = args.output_root.expanduser().resolve()
    result = migrate(source_roots, output_root)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
