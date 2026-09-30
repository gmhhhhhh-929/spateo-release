"""Lazy native reads and diagnostic recovery: no silent retries or data repair."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from spateo.io import read_spatial
from spateo.io.spatial.auto import _automatic as automatic
from spateo.io.spatial.auto._errors import ResourceDeferred
from spateo.io.spatial.auto._recovery import diagnostic_report
from tests.io.test_automatic_reading import matrix, table_bundle, visium


def counted_reader(monkeypatch):
    calls = []
    original = automatic.read_core

    def read(candidate, budget):
        calls.append(str(candidate.counts))
        return original(candidate, budget)

    monkeypatch.setattr(automatic, "read_core", read)
    return calls


def test_lazy_report_is_nonloading_and_access_materializes_once(tmp_path, monkeypatch):
    visium(tmp_path)
    calls = counted_reader(monkeypatch)
    result = read_spatial(tmp_path, lazy=True)
    entry = next(iter(result.values()))
    assert result.status == "pending" and calls == [] and entry.adata is None
    assert result.get(entry.key) is result[entry.key]
    assert list(result) == list(result.keys())
    report = json.loads(json.dumps(result.report))
    assert report["datasets"][entry.key]["lazy"] is True
    assert report["datasets"][entry.key]["materialization_attempts"] == 0
    result.write_report(tmp_path / "audit.json")
    assert calls == []
    adata = result.adata
    assert adata is result.adata is entry.materialize()
    assert len(calls) == 1 and entry.materialization_attempts == 1
    np.testing.assert_array_equal(adata.X.toarray(), [[1, 2], [3, 4]])


def test_concurrent_access_same_handle_materializes_once(tmp_path, monkeypatch):
    visium(tmp_path)
    calls = counted_reader(monkeypatch)
    entry = next(iter(read_spatial(tmp_path, lazy=True).values()))
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda _: entry.materialize(), range(8)))
    assert len(calls) == 1 and all(a is outputs[0] for a in outputs)


def test_metadata_only_retains_old_no_implicit_load_contract(tmp_path, monkeypatch):
    visium(tmp_path)
    calls = counted_reader(monkeypatch)
    result = read_spatial(tmp_path, load=False)
    with pytest.raises(ValueError, match="No unique"):
        _ = result.adata
    assert calls == []
    result.load()
    assert result.status == "ok" and len(calls) == 1
    with pytest.raises(ValueError, match="not both"):
        read_spatial(tmp_path, lazy=True, load=False)


def test_load_one_key_does_not_load_siblings(tmp_path, monkeypatch):
    visium(tmp_path / "a")
    visium(tmp_path / "b")
    calls = counted_reader(monkeypatch)
    result = read_spatial(tmp_path, lazy=True)
    key = next(iter(result))
    with pytest.raises(ValueError):
        _ = result.adata
    assert not calls
    result.load(key)
    assert len(calls) == 1
    assert result[key].status == "ready" and result.status == "partial"
    assert {e.status for e in result.values()} == {"ready", "deferred"}
    result.load()
    assert result.status == "ok" and len(calls) == 2


def test_resource_failure_no_repeated_work_without_explicit_retry(tmp_path, monkeypatch):
    visium(tmp_path)
    calls = []
    original = automatic.read_core

    def deferred(candidate, budget):
        calls.append(budget)
        if len(calls) < 3:
            raise ResourceDeferred("Test transient allocation budget")
        return original(candidate, budget)

    monkeypatch.setattr(automatic, "read_core", deferred)
    result = read_spatial(tmp_path, lazy=True)
    entry = next(iter(result.values()))
    for _ in range(2):
        with pytest.raises(ValueError):
            _ = result.adata
    entry.load()
    assert len(calls) == 1
    entry.load(max_memory_bytes=2 * 1024**3)
    entry.load()
    assert len(calls) == 2 and entry.status == "deferred"
    entry.load(retry=True)
    assert len(calls) == 3 and result.status == "ok"


def test_real_memory_estimate_resumes_with_explicit_larger_budget(tmp_path):
    visium(tmp_path)
    result = read_spatial(tmp_path, lazy=True, max_memory_bytes=1)
    entry = next(iter(result.values()))
    with pytest.raises(ValueError):
        entry.materialize()
    assert entry.status == "deferred"
    assert "increase_budget_or_select_input" in str(result.report)
    a = entry.materialize(max_memory_bytes=10**7)
    assert a.shape == (2, 2)


def test_core_value_failure_lazy_attempt_is_not_silently_retried(tmp_path, monkeypatch):
    table_bundle(tmp_path, "merfish")
    path = tmp_path / "cell_by_gene_S1.csv"
    df = pd.read_csv(path)
    df["G1"] = df["G1"].astype(object)
    df.loc[1, "G1"] = "invalid"
    df.to_csv(path, index=False)
    calls = counted_reader(monkeypatch)
    result = read_spatial(tmp_path, lazy=True)
    entry = next(iter(result.values()))
    assert entry.status == "deferred"
    for _ in range(2):
        with pytest.raises(ValueError):
            _ = result.adata
    assert len(calls) == 1 and entry.status == "failed"
    with pytest.raises(ValueError, match="retry=True"):
        entry.load()
    entry.load(retry=True)
    assert len(calls) == 2 and entry.status == "failed"
    assert "repair_upstream_contract" in str(result.report)


def test_source_mutation_requires_rediscovery_even_explicit_retry(tmp_path, monkeypatch):
    visium(tmp_path)
    result = read_spatial(tmp_path, lazy=True)
    calls = counted_reader(monkeypatch)
    path = tmp_path / "spatial/tissue_positions.csv"
    path.write_text(path.read_text() + "\n")
    entry = next(iter(result.values()))
    entry.load()
    assert entry.status == "failed" and not calls
    assert "rediscover_changed_source" in str(result.report)
    with pytest.raises(ValueError, match="read_spatial again"):
        entry.load(retry=True)
    assert read_spatial(tmp_path).status == "ok"


def test_missing_companion_report_has_concrete_path_and_official_help(tmp_path):
    matrix(tmp_path / "filtered_feature_bc_matrix.h5")
    result = read_spatial(tmp_path)
    report = result.report
    entry = next(iter(report["datasets"].values()))
    fail = next(d for d in entry["diagnostics"] if d["code"] == "probe_failed")
    expected = str(tmp_path / "spatial/tissue_positions.csv")
    assert expected in fail["missing_files"]
    restore = next(a for a in fail["recovery"] if a["action"] == "restore_required_files")
    assert expected in restore["paths"]
    assert restore["links"][0]["url"].startswith("https://www.10xgenomics.com/")
    assert all(a["automatic"] is False for a in fail["recovery"])
    json.dumps(report)


def test_incomplete_mex_components_are_named(tmp_path):
    from scipy import sparse
    from scipy.io import mmwrite

    mex = tmp_path / "filtered_feature_bc_matrix"
    mex.mkdir()
    mmwrite(mex / "matrix.mtx", sparse.coo_matrix([[1, 2]]))
    result = read_spatial(tmp_path)
    fail = next(d for e in result.report["datasets"].values() for d in e["diagnostics"] if d["code"] == "probe_failed")
    assert str(mex / "barcodes.tsv") in fail["missing_files"]
    assert str(mex / "features.tsv") in fail["missing_files"]
    assert "restore_required_files" in str(fail)


def test_incomplete_scope_blocks_implicit_lazy_loading(tmp_path, monkeypatch):
    visium(tmp_path)
    nested = tmp_path / "nested" / "too_deep"
    nested.mkdir(parents=True)
    calls = counted_reader(monkeypatch)
    result = read_spatial(tmp_path, max_depth=1, lazy=True)
    assert result.discovery["complete"] is False
    with pytest.raises(ValueError):
        _ = result.adata
    assert calls == [] and "complete_inventory" in str(result.report)


@pytest.mark.parametrize("kind", ["gzip", "hdf5"])
def test_corrupt_source_recovery(tmp_path, kind):
    if kind == "gzip":
        (tmp_path / "native.gem.gz").write_bytes(b"not gzip")
    else:
        (tmp_path / "filtered_feature_bc_matrix.h5").write_bytes(b"not hdf5")
        (tmp_path / "spatial").mkdir()
        (tmp_path / "spatial/tissue_positions.csv").write_text(
            "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\na,1,0,0,1,1\n"
        )
    result = read_spatial(tmp_path)
    assert result.status == "failed"
    assert "verify_source_integrity" in str(result.report)


def test_image_budget_is_recoverable_without_failing_core(tmp_path, monkeypatch):
    visium(tmp_path)
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(tmp_path / "spatial/tissue_hires_image.png")
    monkeypatch.setattr(automatic, "_IMAGE_BUDGET", 10)
    result = read_spatial(tmp_path)
    assert result.status == "ok"
    assert "inspect_image_separately" in str(result.report)


def test_prefix_scopes_optional_images_to_one_sample(tmp_path):
    from anndata import AnnData

    from spateo.io.spatial.auto._discovery import Candidate

    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(tmp_path / "one_aligned_HE.jpg")
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(tmp_path / "two_aligned_HE.jpg")
    a = AnnData(np.zeros((1, 1)))
    a.uns["spatial"] = {"one": {"images": {}, "scalefactors": {}, "metadata": {}}}
    c = Candidate(
        "seekspace",
        tmp_path,
        tmp_path / "matrix.mtx",
        tmp_path / "cell_locations.tsv",
        "cells",
        {"image_prefix": "one_aligned_"},
    )
    automatic._assets(a, c, True, 10**6, [])
    assert set(a.uns["spatial"]["one"]["image_files"].values()) == {"one_aligned_HE.jpg"}


@pytest.mark.parametrize(
    "code,text,action",
    [
        ("path_missing", "absent", "locate_input"),
        ("permission_denied", "Permission denied", "restore_source_access"),
        ("dependency_missing", "No module named pyarrow", "install_reader_dependency"),
        ("unresolved_layout", "alternatives", "select_unambiguous_input"),
        ("optional_images_missing", "no image", "optional_assets"),
        ("unsupported_layout", "unsupported", "inspect_native_layout"),
    ],
)
def test_recovery_actions_are_json_and_never_automatic(code, text, action):
    d = diagnostic_report({"code": code, "message": text, "severity": "error"}, source="/native/sample")
    assert d["recovery"][0]["action"] == action
    assert d["recovery"][0]["automatic"] is False
    json.dumps(d)


def test_truncated_gzip_during_discovery_returns_recovery(tmp_path, monkeypatch):
    (tmp_path / "source.tsv.gz").write_bytes(b"\x1f\x8b")
    result = read_spatial(tmp_path, lazy=True)
    assert result.status == "failed" and "verify_source_integrity" in str(result.report)


def test_probe_deferred_recovery_present_inside_candidate_diagnostics(tmp_path):
    visium(tmp_path)
    result = read_spatial(tmp_path, lazy=True, max_memory_bytes=1)
    report = result.report
    entry = next(iter(report["datasets"].values()))
    # The table probe may complete under a tiny budget; an attempted core read is
    # still explicitly deferred with a resource action.
    if entry["validation"]["candidate_diagnostics"]:
        assert "increase_budget_or_select_input" in str(entry["validation"]["candidate_diagnostics"])
    else:
        result.load()
        assert "increase_budget_or_select_input" in str(result.report)
