"""Independent native readers share parsers with automatic IO, not its dispatcher.

The expected values below are specified independently of the parsing code. Every
fixture deliberately puts spatial records in a different order from barcodes and
uses duplicate gene symbols with distinct stable IDs.
"""

import ast
import gzip
import importlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from anndata import read_h5ad
from PIL import Image
from scipy.sparse import csc_matrix

import spateo.io as io

TECHNOLOGIES = ("seekspace", "bmkmanu", "salus", "singleron")
EXPECTED_COUNTS = np.array([[7, 0, 2], [0, 11, 0], [5, 1, 3]], dtype=np.int64)
EXPECTED_COORDS = np.array([[101.5, 200], [22, 405.25], [70, 35]], dtype=float)
EXPECTED_IMAGE = np.arange(36, dtype=np.uint8).reshape(6, 6)


def test_platform_dependency_graph_does_not_import_automatic_layer():
    """Guard the requested dependency direction, including captured import aliases."""
    modules = [importlib.import_module(f"spateo.io.spatial._{tech}") for tech in TECHNOLOGIES]
    folder = Path(modules[0].__file__).parent
    pending = [Path(module.__file__) for module in modules]
    checked = set()
    while pending:
        path = pending.pop()
        if path in checked:
            continue
        checked.add(path)
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            imports = []
            if isinstance(node, ast.ImportFrom):
                name = "." * node.level + (node.module or "")
                resolved = importlib.util.resolve_name(name, "spateo.io.spatial") if node.level else name
                imports.append(resolved)
                imports.extend(f"{resolved}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            for module in imports:
                assert not module.startswith("spateo.io.spatial.auto"), (path.name, module)
                prefix = "spateo.io.spatial."
                if module.startswith(prefix):
                    dependency = folder / (module[len(prefix) :].replace(".", "/") + ".py")
                    if dependency.is_file():
                        pending.append(dependency)
            if isinstance(node, ast.Call):
                callee = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                assert callee != "read_spatial", (path.name, node.lineno)
    assert len(checked) > len(TECHNOLOGIES), "The check must follow shared lower-level dependencies."


def native_fixture(root, technology):
    """Write complete source schemas; return the coordinate file for fault tests."""
    root.mkdir(parents=True, exist_ok=True)
    if technology == "singleron":
        matrix = csc_matrix(EXPECTED_COUNTS.T)
        with h5py.File(root / "filtered_feature_bc_matrix.h5", "w") as handle:
            handle.attrs["chemistry_description"] = "Spatial3"
            group = handle.create_group("matrix")
            for field in ("data", "indices", "indptr"):
                group[field] = getattr(matrix, field)
            group["shape"] = matrix.shape
            group["barcodes"] = np.array([b"spot-1", b"spot-2", b"spot-3"])
            group["features/id"] = np.array([b"EN_A", b"EN_B", b"EN_C"])
            group["features/name"] = np.array([b"Shared", b"Shared", b"Third"])
            group["features/feature_type"] = np.array([b"Gene Expression"] * 3)
        (root / "spatial").mkdir()
        coordinate = root / "spatial/positions_list.csv"
        coordinate.write_text("spot-3,1,2,0,35,70\nspot-1,1,0,0,200,101.5\nspot-2,0,1,0,405.25,22\n")
    else:
        content = {
            "matrix.mtx.gz": (
                "%%MatrixMarket matrix coordinate integer general\n"
                "% genes by observations\n3 3 6\n"
                "1 1 7\n3 1 2\n2 2 11\n1 3 5\n2 3 1\n3 3 3\n"
            ),
            "features.tsv.gz": "EN_A\tShared\tGene Expression\nEN_B\tShared\tGene Expression\nEN_C\tThird\tGene Expression\n",
            "barcodes.tsv.gz": "spot-1\nspot-2\nspot-3\n",
        }
        name, positions = {
            "seekspace": (
                "cell_locations.tsv.gz",
                "Cell_Barcode\tX\tY\nspot-3\t70\t35\nspot-1\t101.5\t200\nspot-2\t22\t405.25\n",
            ),
            "bmkmanu": ("barcodes_pos.tsv.gz", "spot-3\t70\t35\nspot-1\t101.5\t200\nspot-2\t22\t405.25\n"),
            "salus": ("spatial.txt.gz", "spot-3 70 35\nspot-1 101.5 200\nspot-2 22 405.25\n"),
        }[technology]
        content[name] = positions
        for filename, value in content.items():
            with gzip.open(root / filename, "wt") as handle:
                handle.write(value)
        coordinate = root / name
    Image.fromarray(EXPECTED_IMAGE).save(root / "native_stain.png")
    return coordinate


def assert_native_object(adata, technology, source, images=True):
    np.testing.assert_array_equal(adata.X.toarray(), EXPECTED_COUNTS)
    np.testing.assert_array_equal(adata.obsm["spatial"], EXPECTED_COORDS)
    assert adata.X.dtype == np.int64
    assert list(adata.obs_names) == ["spot-1", "spot-2", "spot-3"]
    assert list(adata.var_names) == ["EN_A", "EN_B", "EN_C"]
    assert list(adata.var["gene_name"]) == ["Shared", "Shared", "Third"]
    provenance = adata.uns["spateo_io"]
    assert provenance["technology"] == technology
    assert provenance["source"] == str(source.resolve())
    assert provenance["reader"] == f"spateo.io.spatial._{technology}.read_core"
    assert "confidence" not in provenance
    assert provenance["validation"]["identifiers"] == "complete"
    slot = next(iter(adata.uns["spatial"].values()))
    assert slot["metadata"]["image_registration"] == "not_established"
    if images:
        assert len(slot["images"]) == 1
        np.testing.assert_array_equal(next(iter(slot["images"].values())), EXPECTED_IMAGE)
    else:
        assert not slot["images"]


@pytest.mark.parametrize("technology", TECHNOLOGIES)
def test_auto_and_independent_reader_exact_objects_and_h5ad(tmp_path, technology):
    native_fixture(tmp_path, technology)
    auto = io.read_spatial(tmp_path)
    assert auto.status == "ok", auto.report
    direct = getattr(io, f"read_{technology}")(tmp_path)
    assert_native_object(auto.adata, technology, tmp_path)
    assert_native_object(direct, technology, tmp_path)
    pd.testing.assert_frame_equal(direct.obs, auto.adata.obs)
    pd.testing.assert_frame_equal(direct.var, auto.adata.var)
    direct.write_h5ad(tmp_path / "roundtrip.h5ad")
    assert_native_object(read_h5ad(tmp_path / "roundtrip.h5ad"), technology, tmp_path)


@pytest.mark.parametrize("technology", TECHNOLOGIES)
def test_explicit_reader_does_not_call_auto_discovery_or_parsing(tmp_path, monkeypatch, technology):
    native_fixture(tmp_path, technology)

    def forbidden(*args, **kwargs):
        raise AssertionError("Independent native reader invoked automatic orchestration")

    automatic = importlib.import_module("spateo.io.spatial.auto._automatic")
    contracts = importlib.import_module("spateo.io.spatial.auto._contracts")
    discovery = importlib.import_module("spateo.io.spatial.auto._discovery")
    monkeypatch.setattr(automatic, "read_spatial", forbidden)
    monkeypatch.setattr(contracts, "probe", forbidden)
    monkeypatch.setattr(contracts, "read_core", forbidden)
    monkeypatch.setattr(discovery, "discover", forbidden)
    # Imported callable aliases must not provide a back door to auto parsing.
    monkeypatch.setattr(automatic, "probe", forbidden)
    monkeypatch.setattr(automatic, "read_core", forbidden)
    monkeypatch.setattr(automatic, "discover", forbidden)
    direct = getattr(io, f"read_{technology}")(tmp_path)
    assert_native_object(direct, technology, tmp_path)


@pytest.mark.parametrize("technology", TECHNOLOGIES)
def test_automatic_and_explicit_use_same_platform_core(tmp_path, monkeypatch, technology):
    native_fixture(tmp_path, technology)
    platform = importlib.import_module(f"spateo.io.spatial._{technology}")
    core = platform.read_core
    calls = []

    def tracked(candidate, budget):
        calls.append((candidate.technology, str(candidate.counts)))
        return core(candidate, budget)

    monkeypatch.setattr(platform, "read_core", tracked)
    auto = io.read_spatial(tmp_path)
    assert auto.status == "ok", auto.report
    assert len(calls) == 1 and calls[0][0] == technology
    direct = getattr(io, f"read_{technology}")(tmp_path)
    assert len(calls) == 2 and calls[0] == calls[1]
    assert_native_object(direct, technology, tmp_path)


@pytest.mark.parametrize("technology", TECHNOLOGIES)
def test_direct_budget_deferral_resumes_without_changing_reader(tmp_path, technology):
    native_fixture(tmp_path, technology)
    result = getattr(io, f"read_{technology}")(tmp_path, return_result=True, max_memory_bytes=1, load_images=False)
    assert result.status == "pending", result.report
    assert len(result) == 1
    entry = next(iter(result.values()))
    assert entry.status == "deferred" and entry.adata is None
    assert "increase_budget_or_select_input" in json.dumps(result.report)
    adata = entry.materialize(max_memory_bytes=10**7)
    assert result.status == "ok"
    assert_native_object(adata, technology, tmp_path, images=False)
    attempts = entry.materialization_attempts
    assert entry.materialize() is adata
    assert entry.materialization_attempts == attempts


@pytest.mark.parametrize("technology", TECHNOLOGIES)
@pytest.mark.parametrize("fault", ("unknown_id", "duplicate_id"))
def test_invalid_coordinate_identifiers_rejected_by_both_entrypoints(tmp_path, technology, fault):
    coordinate = native_fixture(tmp_path, technology)
    opener = gzip.open if coordinate.suffix == ".gz" else open
    with opener(coordinate, "rt") as handle:
        original = handle.read()
    replacement = "absent" if fault == "unknown_id" else "spot-1"
    with opener(coordinate, "wt") as handle:
        handle.write(original.replace("spot-2", replacement))
    for result in (
        io.read_spatial(tmp_path),
        getattr(io, f"read_{technology}")(tmp_path, return_result=True),
    ):
        assert result.status == "failed", result.report
        assert all(entry.adata is None for entry in result.values())
        assert "recovery" in json.dumps(result.report)
        with pytest.raises(ValueError):
            _ = result.adata


@pytest.mark.parametrize("technology", TECHNOLOGIES)
def test_missing_coordinate_file_gives_no_invented_positions(tmp_path, technology):
    coordinate = native_fixture(tmp_path, technology)
    coordinate.unlink()
    for result in (
        io.read_spatial(tmp_path),
        getattr(io, f"read_{technology}")(tmp_path, return_result=True),
    ):
        assert result.status == "failed", result.report
        assert all(entry.adata is None for entry in result.values())
        assert "recovery" in json.dumps(result.report)
    with pytest.raises(ValueError):
        getattr(io, f"read_{technology}")(tmp_path)


def test_singleron_raw_and_filtered_are_distinct_preserved_inputs(tmp_path):
    native_fixture(tmp_path, "singleron")
    shutil.copyfile(tmp_path / "filtered_feature_bc_matrix.h5", tmp_path / "raw_feature_bc_matrix.h5")
    with h5py.File(tmp_path / "raw_feature_bc_matrix.h5", "r+") as handle:
        handle["matrix/data"][0] = 17
    for result in (io.read_spatial(tmp_path), io.read_singleron(tmp_path, return_result=True)):
        assert result.status == "ok" and len(result) == 2, result.report
        sums = sorted(int(entry.adata.X.sum()) for entry in result.values())
        assert sums == [29, 39]
        readers = {entry.adata.uns["spateo_io"]["reader"] for entry in result.values()}
        assert readers == {"spateo.io.spatial._singleron.read_core"}
        with pytest.raises(ValueError):
            _ = result.adata
    with pytest.raises(ValueError):
        io.read_singleron(tmp_path)


@pytest.mark.parametrize("technology", TECHNOLOGIES)
def test_direct_collection_does_not_take_one_sample_silently(tmp_path, technology):
    for sample in ("sample_1", "sample_2"):
        native_fixture(tmp_path / sample, technology)
    auto = io.read_spatial(tmp_path)
    direct = getattr(io, f"read_{technology}")(tmp_path, return_result=True)
    for result in (auto, direct):
        assert result.status == "ok" and len(result) == 2, result.report
        assert {entry.technology for entry in result.values()} == {technology}
        for entry in result.values():
            assert_native_object(entry.adata, technology, tmp_path / entry.source.split("/")[-1])
        with pytest.raises(ValueError):
            _ = result.adata
    with pytest.raises(ValueError):
        getattr(io, f"read_{technology}")(tmp_path)


@pytest.mark.parametrize("technology", TECHNOLOGIES)
def test_explicit_reader_selects_own_platform_in_mixed_parent(tmp_path, monkeypatch, technology):
    for native_technology in TECHNOLOGIES:
        native_fixture(tmp_path / native_technology, native_technology)
    automatic = io.read_spatial(tmp_path, technology=technology)
    assert automatic.status == "ok" and len(automatic) == 1, automatic.report

    def forbidden(*args, **kwargs):
        raise AssertionError("Explicit native reader called another platform's implementation")

    for other in set(TECHNOLOGIES) - {technology}:
        platform = importlib.import_module(f"spateo.io.spatial._{other}")
        for operation in ("discover", "probe", "read_core"):
            monkeypatch.setattr(platform, operation, forbidden)
    reader = getattr(io, f"read_{technology}")
    explicit = reader(tmp_path, return_result=True)
    assert explicit.status == "ok" and len(explicit) == 1, explicit.report
    assert explicit.discovery["scope"] == "explicit_platform"
    unclassified = [d for d in explicit.report["diagnostics"] if d["code"] == "unclassified_directory"]
    assert unclassified and all(d["severity"] == "info" for d in unclassified)
    for other in set(TECHNOLOGIES) - {technology}:
        assert str(tmp_path / other) in json.dumps(unclassified)
    for adata in (explicit.adata, reader(tmp_path)):
        assert_native_object(adata, technology, tmp_path / technology)
        pd.testing.assert_frame_equal(adata.obs, automatic.adata.obs)
        pd.testing.assert_frame_equal(adata.var, automatic.adata.var)
        np.testing.assert_array_equal(adata.X.toarray(), automatic.adata.X.toarray())
        np.testing.assert_array_equal(adata.obsm["spatial"], automatic.adata.obsm["spatial"])
        for other in set(TECHNOLOGIES) - {technology}:
            assert str(tmp_path / other) in json.dumps(adata.uns["spateo_io"]["unclassified_directories"])
