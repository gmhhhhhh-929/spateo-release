import importlib.util
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "migrate_figure_spateo_scripts.py"
SPEC = importlib.util.spec_from_file_location("migrate_figure_spateo_scripts", SCRIPT_PATH)
MIGRATION = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MIGRATION)


def test_transform_current_spateo_preprocessing_and_pot_patch():
    source = """import dynamo as dyn
import spateo as st
import numpy as np
adata = adata[:, np.sum(adata.layers["counts_X"], axis=0) != 0]
adata.uns["pp"] = {}
adata.X = adata.layers["counts_X"].copy()
dyn.pp.normalize_cell_expr_by_size_factors(
    adata=adata,
    layers="X",
    X_total_layers=True,
)
"""
    transformed, changes = MIGRATION.transform_source(source)
    transformed, _ = MIGRATION.remove_dynamo_imports(transformed)

    assert "st.pp.normalize_total" in transformed
    assert "st.pp.log1p_layer" in transformed
    assert "adata = adata[:, np.sum" in transformed
    assert "].copy()" in transformed
    assert "# OLD (Dynamo input reset removed): adata.X" in transformed
    assert not MIGRATION.has_active_legacy_code(transformed)
    assert "dynamo_size_factor_to_spateo" in changes

    pot_source = """import ot
if not hasattr(ot.gromov, "cg") and hasattr(ot.optim, "cg"):
    ot.gromov.cg = ot.optim.cg
"""
    transformed, changes = MIGRATION.transform_source(pot_source)
    assert "no ot.gromov.cg monkeypatch is required" in transformed
    assert not MIGRATION.has_active_legacy_code(transformed)
    assert changes == ["remove_pot_monkeypatch"]


def test_migrate_notebook_clears_outputs_and_replaces_layer_log1p(tmp_path):
    source_path = tmp_path / "source.ipynb"
    output_path = tmp_path / "output.ipynb"
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {},
                "outputs": [{"output_type": "stream", "name": "stdout", "text": ["old"]}],
                "source": [
                    'adata.layers["counts_X"] = adata.X.copy()\n',
                    'adata.layers["log1p_X"] = np.log1p(adata.layers["counts_X"])\n',
                ],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    source_path.write_text(json.dumps(notebook), encoding="utf-8")

    changes = MIGRATION.migrate_notebook(source_path, output_path)
    migrated = json.loads(output_path.read_text(encoding="utf-8"))
    cell = migrated["cells"][0]
    source = "".join(cell["source"])
    assert "# OLD (NumPy):" in source
    assert "st.pp.log1p_layer" in source
    assert cell["outputs"] == []
    assert cell["execution_count"] is None
    assert migrated["metadata"]["spateo_migration"]["dynamo_required"] is False
    assert "numpy_layer_log1p_to_spateo" in changes
