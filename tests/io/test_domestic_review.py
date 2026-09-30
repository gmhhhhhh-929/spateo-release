"""Independent adversarial checks of domestic native file contracts."""

import gzip
import json

import h5py
import numpy as np
import pytest
from PIL import Image

from spateo.io import read_spatial
from tests.io.test_automatic_reading import matrix


def mex(root, records=(3,), field="integer", compression=False):
    root.mkdir(parents=True, exist_ok=True)
    text = "%%MatrixMarket matrix coordinate " + field + " general\n1 1 " + str(len(records)) + "\n"
    text += "".join("1 1 " + str(v) + "\n" for v in records)
    if compression:
        with gzip.open(root / "matrix.mtx.gz", "wt") as f:
            f.write(text)
    else:
        (root / "matrix.mtx").write_text(text)
    (root / "barcodes.tsv").write_text("bc1\n")
    (root / "features.tsv").write_text("stable_gene_id\tDisplayName\tGene Expression\n")
    (root / "cell_locations.tsv").write_text("Cell_Barcode\tX\tY\nbc1\t1\t2\n")
    return root


def test_same_matrix_competing_domestic_platforms_are_unresolved(tmp_path):
    mex(tmp_path)
    (tmp_path / "barcodes_pos.tsv").write_text("bc1\t1\t2\n")
    (tmp_path / "spatial.txt").write_text("bc1 1 2\n")
    result = read_spatial(tmp_path, load_images=False)
    assert len(result) == 1 and result.status == "failed", result.report
    assert next(iter(result.values())).status == "unresolved"
    assert not any(e.adata is not None for e in result.values())
    # Explicit experiment knowledge can restrict discovery, never a score.
    result = read_spatial(tmp_path, technology="salus", load_images=False)
    assert result.status == "ok", result.report
    assert next(iter(result.values())).representation == "spatial_barcodes"


def test_resolved_second_alternative_has_correct_representation(tmp_path):
    mex(tmp_path)
    (tmp_path / "cell_locations.tsv").write_text("wrong\tfields\na\tb\n")
    (tmp_path / "barcodes_pos.tsv").write_text("bc1 1 2\n")
    result = read_spatial(tmp_path, load_images=False)
    assert result.status == "ok", result.report
    entry = next(iter(result.values()))
    assert (entry.technology, entry.representation) == ("bmkmanu", "aggregated_bins")
    assert entry.adata.uns["native_spatial_export"]["observation_type"] == "aggregated_bin"


@pytest.mark.parametrize(
    "records,field",
    [
        ([-1, 2], "integer"),
        (["0.5", "0.5"], "real"),
        ([2**62] * 4, "integer"),
        (["9007199254740993"], "real"),
    ],
)
@pytest.mark.parametrize("compression", [False, True])
def test_invalid_raw_counts_not_hidden_by_sparse_aggregation(tmp_path, records, field, compression):
    mex(tmp_path, records, field, compression)
    result = read_spatial(tmp_path, load_images=False)
    assert result.status == "failed", result.report
    assert next(iter(result.values())).adata is None


@pytest.mark.parametrize("records,expected", [([2**53 + 1], 2**53 + 1), ([3, 5], 8)])
def test_integer_counts_are_exact_and_duplicate_totals_preserved(tmp_path, records, expected):
    mex(tmp_path, records)
    result = read_spatial(tmp_path, load_images=False)
    assert result.status == "ok", result.report
    assert int(result.adata.X[0, 0]) == expected
    assert str(result.adata.X.dtype) == "int64"
    assert list(result.adata.var_names) == ["stable_gene_id"]
    assert list(result.adata.var["gene_name"]) == ["DisplayName"]
    assert list(result.adata.var["feature_types"]) == ["Gene Expression"]


def test_malformed_sibling_preserves_healthy_input_and_reports_incomplete_scope(tmp_path):
    mex(tmp_path / "good")
    (tmp_path / "bad.tsv.gz").write_bytes(b"\x1f\x8b")
    result = read_spatial(tmp_path, load_images=False)
    assert result.status == "partial", result.report
    assert len(result) == 1 and next(iter(result.values())).status == "ready"
    assert result.discovery["complete"] is False
    assert "verify_source_integrity" in str(result.report)


def test_seekspace_parent_images_and_metadata_are_sample_specific(tmp_path):
    mex(tmp_path / "one_filtered_feature_bc_matrix")
    mex(tmp_path / "two_filtered_feature_bc_matrix")
    for prefix, value in (("one", 20), ("two", 80)):
        Image.fromarray(np.full((3, 3, 3), value, dtype=np.uint8)).save(tmp_path / f"{prefix}_aligned_HE.png")
    result = read_spatial(tmp_path)
    assert result.status == "ok" and len(result) == 2, result.report
    for entry in result.values():
        prefix = "one" if "one_filtered" in entry.key else "two"
        slot = next(iter(entry.adata.uns["spatial"].values()))
        assert set(slot["image_files"].values()) == {f"{prefix}_aligned_HE.png"}
        assert len(slot["images"]) == 1


def test_stable_feature_ids_not_duplicate_display_names(tmp_path):
    mex(tmp_path)
    (tmp_path / "matrix.mtx").write_text("%%MatrixMarket matrix coordinate integer general\n2 1 2\n1 1 3\n2 1 5\n")
    (tmp_path / "features.tsv").write_text("id1\tSameName\tGene Expression\nid2\tSameName\tGene Expression\n")
    result = read_spatial(tmp_path, load_images=False)
    assert result.status == "ok", result.report
    assert list(result.adata.var_names) == ["id1", "id2"]
    assert list(result.adata.var["gene_name"]) == ["SameName", "SameName"]
    assert "Variable names are not unique" not in str(result.report)
    np.testing.assert_array_equal(result.adata.X.toarray(), [[3, 5]])


def test_duplicate_native_ids_and_missing_metadata_are_not_repaired(tmp_path):
    mex(tmp_path)
    path = tmp_path / "cell_locations.tsv"
    path.write_text("Cell_Barcode\tX\tY\nbc1\t1\t2\nbc1\t3\t4\n")
    assert read_spatial(tmp_path).status == "failed"
    path.write_text("Cell_Barcode\tX\tY\nother\t1\t2\n")
    result = read_spatial(tmp_path)
    assert result.status == "failed" and "lack metadata" in str(result.report)


def test_native_matrix_components_are_source_mutation_guarded(tmp_path):
    mex(tmp_path)
    result = read_spatial(tmp_path, lazy=True)
    (tmp_path / "features.tsv").write_text("new_gene_id\tNewName\tGene Expression\n")
    result.load()
    assert result.status == "failed" and "source_changed" in str(result.report)


def test_incomplete_identified_singleron_names_native_missing_positions(tmp_path):
    path = tmp_path / "filtered_feature_bc_matrix.h5"
    matrix(path)
    with h5py.File(path, "a") as handle:
        handle.attrs["chemistry_description"] = "Spatial3"
    result = read_spatial(tmp_path)
    assert result.status == "failed"
    entry = next(iter(result.values()))
    assert entry.technology == "singleron", result.report
    assert str(tmp_path / "spatial/positions_list.csv") in str(result.report)


def visium_with_seekspace(root, seekspace_valid):
    mex(root / "filtered_feature_bc_matrix")
    (root / "spatial").mkdir()
    (root / "spatial/tissue_positions.csv").write_text(
        "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\nbc1,1,0,0,2,1\n"
    )
    if not seekspace_valid:
        (root / "filtered_feature_bc_matrix/cell_locations.tsv").write_text("wrong\tfields\nx\ty\n")


def test_invalid_domestic_companion_does_not_hide_valid_visium(tmp_path):
    visium_with_seekspace(tmp_path, False)
    result = read_spatial(tmp_path, load_images=False)
    assert result.status == "ok", result.report
    assert len(result) == 1
    assert next(iter(result.values())).technology == "visium"


def test_valid_competing_visium_and_domestic_claim_is_unresolved(tmp_path):
    visium_with_seekspace(tmp_path, True)
    result = read_spatial(tmp_path, load_images=False)
    assert len(result) == 1 and next(iter(result.values())).status == "unresolved", result.report


def test_unsupported_raw_mex_alias_does_not_hide_required_matrix_path(tmp_path):
    mex(tmp_path)
    (tmp_path / "matrix.mtx").rename(tmp_path / "matrix.tsv")
    result = read_spatial(tmp_path)
    assert result.status == "failed"
    diagnostic = next(
        d for e in result.report["datasets"].values() for d in e["diagnostics"] if d["code"] == "probe_failed"
    )
    assert str(tmp_path / "matrix.mtx") in diagnostic["missing_files"]


def singleron_raw_h5(root, values, dtype):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "filtered_feature_bc_matrix.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["chemistry_description"] = "Spatial3"
        group = handle.create_group("matrix")
        group["data"] = np.asarray(values, dtype=dtype)
        group["indices"] = np.zeros(len(values), dtype=np.int64)
        group["indptr"] = np.asarray([0, len(values)], dtype=np.int64)
        group["shape"] = np.asarray([1, 1], dtype=np.int64)
        group["barcodes"] = np.asarray(["bc1"], dtype="S")
        group["features/id"] = np.asarray(["stable_id"], dtype="S")
        group["features/name"] = np.asarray(["DisplayName"], dtype="S")
    (root / "spatial").mkdir()
    (root / "spatial/positions_list.csv").write_text("bc1,1,0,0,2,1\n")


@pytest.mark.parametrize(
    "values,dtype",
    [
        ([0.5, 0.5], np.float64),
        ([-1, 2], np.int64),
        ([2**62] * 4, np.int64),
        ([2**63], np.uint64),
        ([float(2**53 + 1)], np.float64),
    ],
)
def test_singleron_invalid_raw_h5_values_fail_before_sparse_conversion(tmp_path, monkeypatch, values, dtype):
    from spateo.io.spatial.auto import _contracts

    singleron_raw_h5(tmp_path, values, dtype)
    original = _contracts.sparse.csc_matrix
    conversions = []

    def tracked(*args, **kwargs):
        conversions.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(_contracts.sparse, "csc_matrix", tracked)
    result = read_spatial(tmp_path, load_images=False)
    assert result.status == "failed", result.report
    assert next(iter(result.values())).technology == "singleron"
    assert conversions == [], "Invalid raw values reached sparse conversion/coalescing"


def test_singleron_integer_h5_above_float_precision_remains_exact(tmp_path):
    singleron_raw_h5(tmp_path, [2**53 + 1], np.int64)
    result = read_spatial(tmp_path, load_images=False)
    assert result.status == "ok", result.report
    assert int(result.adata.X[0, 0]) == 2**53 + 1
    assert list(result.adata.var_names) == ["stable_id"]
    assert list(result.adata.var["gene_name"]) == ["DisplayName"]


def test_domestic_h5_duplicate_gene_symbols_do_not_warn_or_replace_stable_ids(tmp_path):
    path = tmp_path / "filtered_feature_bc_matrix.h5"
    matrix(path)
    with h5py.File(path, "a") as handle:
        handle.attrs["chemistry_description"] = "Spatial3"
        del handle["matrix/features/name"]
        handle["matrix/features/name"] = np.asarray(["SameName", "SameName"], dtype="S")
    (tmp_path / "spatial").mkdir()
    (tmp_path / "spatial/positions_list.csv").write_text("c2,1,1,1,11,21\nc1,1,0,0,10,20\n")
    result = read_spatial(tmp_path, load_images=False)
    assert result.status == "ok", result.report
    assert list(result.adata.var_names) == ["id1", "id2"]
    assert list(result.adata.var["gene_name"]) == ["SameName", "SameName"]
    assert "Variable names are not unique" not in str(result.report)
    np.testing.assert_array_equal(result.adata.obsm["spatial"], [[20, 10], [21, 11]])


@pytest.mark.parametrize("name", ["preview.png", "README.txt", "notes.tsv"])
def test_explicit_noncore_file_does_not_select_neighboring_domestic_matrix(tmp_path, name):
    mex(tmp_path)
    requested = tmp_path / name
    requested.write_bytes(b"optional content")
    result = read_spatial(requested, load_images=False)
    assert result.status == "failed" and len(result) == 0, result.report
    assert "unsupported_layout" in str(result.report)
    assert read_spatial(tmp_path, load_images=False).status == "ok"


@pytest.mark.parametrize("name", ["matrix.mtx", "features.tsv", "barcodes.tsv", "cell_locations.tsv"])
def test_explicit_native_component_selects_its_bundle(tmp_path, name):
    mex(tmp_path)
    result = read_spatial(tmp_path / name, load_images=False)
    assert result.status == "ok" and len(result) == 1, result.report
    assert result.adata.shape == (1, 1)
