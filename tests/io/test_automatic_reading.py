"""Behavioral tests for score-free, lossless core reading."""

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from anndata import read_h5ad
from PIL import Image
from scipy import sparse

from spateo.io import SpatialReadResult, read_spatial


def matrix(path, ids=("c1", "c2"), values=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    X = sparse.csc_matrix(np.array([[1, 3], [2, 4]], dtype=np.int32) if values is None else values)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        for key, arr in [
            ("data", X.data),
            ("indices", X.indices),
            ("indptr", X.indptr),
            ("shape", np.array(X.shape)),
            ("barcodes", np.asarray(ids, dtype="S")),
        ]:
            g.create_dataset(key, data=arr)
        feats = g.create_group("features")
        for key, arr in [("id", ["id1", "id2"]), ("name", ["G1", "G2"]), ("feature_type", ["Gene Expression"] * 2)]:
            feats.create_dataset(key, data=np.asarray(arr, dtype="S"))


def visium(root, ids=("c1", "c2"), header=True):
    matrix(root / "filtered_feature_bc_matrix.h5")
    (root / "spatial").mkdir(exist_ok=True)
    f = pd.DataFrame(
        [[i, 1, n, n, 10 + n, 20 + n] for n, i in enumerate(ids)],
        columns=["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"],
    )
    # Deliberately reverse positions order to test ID-based joining.
    f.iloc[::-1].to_csv(
        root / "spatial" / ("tissue_positions.csv" if header else "tissue_positions_list.csv"),
        index=False,
        header=header,
    )
    return root


def cell_bundle(root, atera=False):
    matrix(root / "cell_feature_matrix.h5")
    pd.DataFrame({"cell_id": ["c2", "c1"], "x_centroid": [21, 20], "y_centroid": [11, 10]}).to_csv(
        root / "cells.csv", index=False
    )
    if atera:
        (root / "experiment.xenium").write_text(json.dumps({"platform": "Atera", "run_name": "sample"}))
    return root


def table_bundle(root, tech):
    root.mkdir(parents=True, exist_ok=True)
    counts = pd.DataFrame({"cell_id": ["c1", "c2"], "G1": [1, 3], "G2": [2, 4]})
    meta = pd.DataFrame({"cell_id": ["c2", "c1"], "center_x": [21, 20], "center_y": [11, 10]})
    if tech == "merfish":
        counts.to_csv(root / "cell_by_gene_S1.csv", index=False)
        meta.to_csv(root / "cell_metadata_S1.csv", index=False)
    elif tech == "seqfish":
        counts.to_csv(root / "SG_Counts_S1.csv", index=False)
        meta.to_csv(root / "SG_CellCoordinates_S1.csv", index=False)
    elif tech == "slideseq":
        pd.DataFrame({"gene": ["G1", "G2"], "c1": [1, 2], "c2": [3, 4]}).to_csv(root / "MappedDGEForR.csv", index=False)
        meta.rename(columns={"cell_id": "barcodes", "center_x": "xcoord", "center_y": "ycoord"}).to_csv(
            root / "BeadLocationsForR.csv", index=False
        )
    elif tech == "starmap_plus":
        counts.rename(columns={"cell_id": "NAME"}).to_csv(root / "sample_processed_expression_pd.csv", index=False)
        meta.rename(columns={"cell_id": "NAME", "center_x": "X", "center_y": "Y"}).to_csv(
            root / "sample_spatial.csv", index=False
        )
    elif tech == "nanostring":
        counts["fov"] = 1
        meta["fov"] = 1
        counts.to_csv(root / "sample_exprMat_file.csv", index=False)
        meta.rename(columns={"center_x": "CenterX_local_px", "center_y": "CenterY_local_px"}).to_csv(
            root / "sample_metadata_file.csv", index=False
        )
    return root


def test_visium_exact_counts_ids_coordinates_and_roundtrip(tmp_path):
    root = visium(tmp_path / "sample" / "outs", header=False)
    result = read_spatial(root.parent)
    assert isinstance(result, SpatialReadResult) and result.status == "ok", result.report
    a = result.adata
    assert list(a.obs_names) == ["c1", "c2"]
    np.testing.assert_array_equal(a.X.toarray(), [[1, 2], [3, 4]])
    np.testing.assert_array_equal(a.obsm["spatial"], [[20, 10], [21, 11]])
    assert "confidence" not in a.uns["spateo_io"]
    assert a.uns["spateo_io"]["validation"]["content"] == "passed"
    a.write_h5ad(tmp_path / "result.h5ad")
    back = read_h5ad(tmp_path / "result.h5ad")
    np.testing.assert_array_equal(back.X.toarray(), a.X.toarray())
    assert back.uns["spateo_io"]["policy_version"] == "spatial-contracts-v1"
    result.write_report(tmp_path / "report.json")
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "ok"


@pytest.mark.parametrize("tech", ["xenium", "atera", "merfish", "seqfish", "slideseq", "starmap_plus", "nanostring"])
def test_platform_contracts_are_lossless(tmp_path, tech):
    root = cell_bundle(tmp_path, tech == "atera") if tech in ("xenium", "atera") else table_bundle(tmp_path, tech)
    r = read_spatial(root)
    assert r.status == "ok", r.report
    assert len(r.datasets) == 1 and next(iter(r.datasets.values())).technology == tech
    np.testing.assert_array_equal(r.adata.X.toarray(), [[1, 2], [3, 4]])
    np.testing.assert_array_equal(r.adata.obsm["spatial"], [[20, 10], [21, 11]])


def test_seqfish_vendor_label_identifier(tmp_path):
    table_bundle(tmp_path, "seqfish")
    p = tmp_path / "SG_CellCoordinates_S1.csv"
    frame = pd.read_csv(p).rename(columns={"cell_id": "label"})
    frame.to_csv(p, index=False)
    a = read_spatial(tmp_path).adata
    assert list(a.obs_names) == ["c1", "c2"]
    np.testing.assert_array_equal(a.obsm["spatial"], [[20, 10], [21, 11]])


@pytest.mark.parametrize("declaration", ["numeric", "invalid", "nan"])
def test_starmap_type_declaration_is_not_an_observation(tmp_path, declaration):
    table_bundle(tmp_path, "starmap_plus")
    p = tmp_path / "sample_spatial.csv"
    frame = pd.read_csv(p)
    frame = pd.concat([pd.DataFrame([{"NAME": "TYPE", "X": declaration, "Y": "numeric"}]), frame])
    frame.to_csv(p, index=False)
    result = read_spatial(tmp_path)
    if declaration == "numeric":
        np.testing.assert_array_equal(result.adata.obsm["spatial"], [[20, 10], [21, 11]])
        assert "TYPE" not in result.adata.obs_names
    else:
        assert result.status == "failed"


def test_hd_all_resolutions_and_cellseg_returned(tmp_path):
    outs = tmp_path / "outs"
    for size in ("008", "016"):
        visium(outs / "binned_outputs" / f"square_{size}um")
    seg = outs / "segmented_outputs"
    matrix(seg / "filtered_feature_cell_matrix.h5", ids=("cellid_000000001-1", "cellid_000000002-1"))
    feats = []
    for i in (1, 2):
        feats.append(
            {
                "type": "Feature",
                "properties": {"cell_id": i},
                "geometry": {"type": "Polygon", "coordinates": [[[i, 0], [i + 1, 0], [i, 1], [i, 0]]]},
            }
        )
    (seg / "cell_segmentations.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    r = read_spatial(tmp_path)
    assert r.status == "ok" and len(r.datasets) == 3, r.report
    assert {e.representation for e in r.datasets.values()} == {"cells", "bin:8um/filtered", "bin:16um/filtered"}
    with pytest.raises(ValueError):
        _ = r.adata


def test_native_gem_counts_not_uint16_truncated(tmp_path):
    (tmp_path / "test.gem").write_text("geneID\tx\ty\tMIDCount\nG1\t0\t1\t70000\nG2\t0\t1\t2\nG1\t0\t1\t3\n")
    r = read_spatial(tmp_path)
    assert r.status == "ok", r.report
    np.testing.assert_array_equal(r.adata.X.toarray(), [[70003, 2]])
    np.testing.assert_array_equal(r.adata.obsm["spatial"], [[0, 1]])


def test_empty_file_names_do_not_pass_contract(tmp_path):
    (tmp_path / "spatial").mkdir()
    (tmp_path / "filtered_feature_bc_matrix.h5").touch()
    (tmp_path / "spatial/tissue_positions.csv").touch()
    result = read_spatial(tmp_path)
    assert result.status == "failed"
    assert next(iter(result.datasets.values())).adata is None


def test_missing_positions_is_explicit(tmp_path):
    matrix(tmp_path / "filtered_feature_bc_matrix.h5")
    r = read_spatial(tmp_path)
    assert r.status == "failed" and "Missing required" in str(r.report)


@pytest.mark.parametrize("bad", ["missing", "duplicate", "nonfinite"])
def test_bad_ids_and_coordinates_are_not_repaired(tmp_path, bad):
    visium(tmp_path)
    f = tmp_path / "spatial/tissue_positions.csv"
    df = pd.read_csv(f)
    if bad == "missing":
        df = df.iloc[:1]
    elif bad == "duplicate":
        df.loc[:, "barcode"] = "c1"
    else:
        df.loc[0, "pxl_row_in_fullres"] = "bad"
    df.to_csv(f, index=False)
    r = read_spatial(tmp_path)
    assert r.status == "failed" and all(e.adata is None for e in r.datasets.values()), r.report


def test_late_bad_table_values_cannot_be_zero_filled(tmp_path):
    root = table_bundle(tmp_path, "merfish")
    count = root / "cell_by_gene_S1.csv"
    df = pd.read_csv(count)
    df.loc[1, "G1"] = "invalid"
    df.to_csv(count, index=False)
    r = read_spatial(root)
    assert r.status == "failed" and "Non-numeric" in str(r.report)


def test_no_row_order_fallback(tmp_path):
    root = table_bundle(tmp_path, "seqfish")
    meta = root / "SG_CellCoordinates_S1.csv"
    df = pd.read_csv(meta)
    df["cell_id"] = ["z1", "z2"]
    df.to_csv(meta, index=False)
    r = read_spatial(root)
    assert r.status == "failed", r.report


def test_multiple_groups_keep_all_inputs(tmp_path):
    root = table_bundle(tmp_path, "merfish")
    for name in ("cell_by_gene", "cell_metadata"):
        (root / f"{name}_S2.csv").write_bytes((root / f"{name}_S1.csv").read_bytes())
    r = read_spatial(root)
    assert r.status == "ok" and len(r.datasets) == 2, r.report


def test_failed_sample_does_not_hide_success(tmp_path):
    visium(tmp_path / "good")
    visium(tmp_path / "bad", ids=("absent",))
    r = read_spatial(tmp_path)
    assert r.status == "partial" and {e.status for e in r.datasets.values()} == {"ready", "failed"}, r.report
    with pytest.raises(ValueError):
        _ = r.adata


def test_memory_deferral_can_resume_and_report_updates(tmp_path):
    visium(tmp_path)
    r = read_spatial(tmp_path, max_memory_bytes=1)
    assert r.status == "pending"
    entry = next(iter(r.datasets.values()))
    assert entry.adata is None
    entry.load(max_memory_bytes=10**7)
    assert r.status == "ok", r.report
    assert r.report["datasets"][entry.key]["validation"]["content"] == "passed"


def test_inspection_is_not_full_validation(tmp_path):
    visium(tmp_path)
    r = read_spatial(tmp_path, load=False)
    assert r.status == "pending" and next(iter(r.datasets.values())).validation["content"] == "not_loaded"
    next(iter(r.datasets.values())).load()
    assert r.status == "ok"


def test_optional_bad_image_does_not_fail_core(tmp_path):
    visium(tmp_path)
    (tmp_path / "spatial/tissue_hires_image.png").write_text("invalid")
    r = read_spatial(tmp_path)
    assert r.status == "ok" and "optional_image_error" in str(r.report)
    assert next(iter(r.adata.uns["spatial"].values()))["images"] == {}


def test_images_scalefactors_and_no_source_writes(tmp_path):
    visium(tmp_path)
    folder = tmp_path / "spatial"
    Image.fromarray(np.ones((4, 5, 3), dtype=np.uint8)).save(folder / "tissue_hires_image.png")
    (folder / "scalefactors_json.json").write_text(json.dumps({"tissue_hires_scalef": 0.5}))
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    r = read_spatial(tmp_path)
    assert r.status == "ok", r.report
    slot = next(iter(r.adata.uns["spatial"].values()))
    assert slot["images"]["hires"].shape == (4, 5, 3)
    assert slot["scalefactors"]["tissue_hires_scalef"] == 0.5
    assert before == {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_companion_conflict_is_not_broken_by_file_order(tmp_path):
    visium(tmp_path)
    p = tmp_path / "spatial/tissue_positions.csv"
    pd.read_csv(p).to_csv(tmp_path / "spatial/tissue_positions_list.csv", header=False, index=False)
    r = read_spatial(tmp_path)
    assert r.status == "failed" and next(iter(r.datasets.values())).status == "unresolved"


def test_score_ranker_has_been_removed(tmp_path):
    import spateo.io.spatial.auto as automatic

    visium(tmp_path)
    assert not hasattr(automatic, "_rank_matches")
    assert read_spatial(tmp_path).status == "ok"


def test_symlink_not_followed(tmp_path):
    outside = tmp_path / "outside"
    visium(outside)
    root = tmp_path / "root"
    root.mkdir()
    (root / "sample").symlink_to(outside, target_is_directory=True)
    r = read_spatial(root)
    assert r.status == "failed" and "symlink_skipped" in str(r.report)


def test_scope_limit_never_claims_complete_success(tmp_path):
    visium(tmp_path)
    r = read_spatial(tmp_path, max_files=1)
    assert r.status != "ok" and not r.discovery["complete"]


def test_path_missing_and_bad_limits(tmp_path):
    assert read_spatial(tmp_path / "absent").status == "failed"
    with pytest.raises(ValueError):
        read_spatial(tmp_path, max_memory_bytes=0)
    with pytest.raises(ValueError):
        read_spatial(tmp_path, technology="made_up")


@pytest.mark.skipif(
    not os.environ.get("SPATEO_VISIUM_DATA"), reason="Set SPATEO_VISIUM_DATA to a real Visium directory"
)
def test_real_visium_against_source_matrix():
    path = Path(os.environ["SPATEO_VISIUM_DATA"])
    r = read_spatial(path)
    assert r.status == "ok", r.report
    a = r.adata
    with h5py.File(path / "filtered_feature_bc_matrix.h5") as f:
        g = f["matrix"]
        original = sparse.csc_matrix(
            (g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g["shape"][:])
        ).T.tocsr()
        assert (a.X != original).nnz == 0
        assert list(a.obs_names) == list(g["barcodes"].asstr()[:])
    positions = pd.read_csv(
        path / "spatial/tissue_positions_list.csv",
        header=None,
        names=["barcode", "in_tissue", "array_row", "array_col", "y", "x"],
    ).set_index("barcode")
    np.testing.assert_array_equal(a.obsm["spatial"], positions.loc[a.obs_names, ["x", "y"]].to_numpy())


def test_mex_only_bin_layout(tmp_path):
    from scipy.io import mmwrite

    root = tmp_path / "binned_outputs/square_008um"
    mex = root / "filtered_feature_bc_matrix"
    mex.mkdir(parents=True)
    mmwrite(mex / "matrix.mtx", sparse.coo_matrix([[1, 3], [2, 4]]))
    (mex / "barcodes.tsv").write_text("c1\nc2\n")
    (mex / "features.tsv").write_text("id1\tG1\tGene Expression\nid2\tG2\tGene Expression\n")
    (root / "spatial").mkdir()
    pd.DataFrame(
        [["c1", 1, 0, 0, 10, 20], ["c2", 1, 1, 1, 11, 21]],
        columns=["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"],
    ).to_csv(root / "spatial/tissue_positions.csv", index=False)
    r = read_spatial(tmp_path)
    assert r.status == "ok", r.report
    np.testing.assert_array_equal(r.adata.X.toarray(), [[1, 2], [3, 4]])


def test_raw_and_filtered_populations_not_merged(tmp_path):
    visium(tmp_path)
    matrix(tmp_path / "raw_feature_bc_matrix.h5")
    r = read_spatial(tmp_path)
    assert r.status == "ok" and len(r.datasets) == 2, r.report


def test_h5_and_mex_encodings_not_assumed_equal(tmp_path):
    visium(tmp_path)
    from scipy.io import mmwrite

    mex = tmp_path / "filtered_feature_bc_matrix"
    mex.mkdir()
    mmwrite(mex / "matrix.mtx", sparse.coo_matrix([[1, 3], [2, 4]]))
    (mex / "barcodes.tsv").write_text("c1\nc2\n")
    (mex / "features.tsv").write_text("id1\tG1\nid2\tG2\n")
    r = read_spatial(tmp_path)
    assert r.status == "failed" and next(iter(r.datasets.values())).status == "unresolved", r.report


def test_parquet_centroids_and_z_preserved(tmp_path):
    root = table_bundle(tmp_path, "merfish")
    meta = root / "cell_metadata_S1.csv"
    df = pd.read_csv(meta)
    df["center_z"] = [5, 6]
    df.to_parquet(root / "cell_metadata_S1.parquet", index=False)
    meta.unlink()
    r = read_spatial(root)
    assert r.status == "ok", r.report
    np.testing.assert_array_equal(r.adata.obsm["spatial"], [[20, 10, 6], [21, 11, 5]])


def test_corrupt_sparse_indices_fail_full_read(tmp_path):
    visium(tmp_path)
    with h5py.File(tmp_path / "filtered_feature_bc_matrix.h5", "a") as f:
        f["matrix/indices"][0] = 100
    r = read_spatial(tmp_path)
    assert r.status == "failed" and "Invalid sparse" in str(r.report)


def test_unrecognized_sibling_is_not_silent_success(tmp_path):
    visium(tmp_path / "good")
    other = tmp_path / "unknown"
    other.mkdir()
    (other / "unknown.csv").write_text("a,b\n1,2\n")
    r = read_spatial(tmp_path)
    assert r.status == "partial" and "unrecognized_input_directory" in str(r.report)


def test_expected_companion_symlink_cannot_escape(tmp_path):
    root = tmp_path / "sample"
    matrix(root / "filtered_feature_bc_matrix.h5")
    outside = tmp_path / "outside"
    visium(outside)
    (root / "spatial").symlink_to(outside / "spatial", target_is_directory=True)
    r = read_spatial(root)
    assert r.status == "failed" and all(e.adata is None for e in r.datasets.values())


def test_duplicate_table_metadata_never_dropped(tmp_path):
    table_bundle(tmp_path, "slideseq")
    path = tmp_path / "BeadLocationsForR.csv"
    df = pd.read_csv(path)
    pd.concat([df, df.iloc[:1]]).to_csv(path, index=False)
    r = read_spatial(tmp_path)
    assert r.status == "failed" and "duplicated identifiers" in str(r.report)


def test_deferred_source_change_requires_rediscovery(tmp_path):
    visium(tmp_path)
    r = read_spatial(tmp_path, load=False)
    p = tmp_path / "spatial/tissue_positions.csv"
    p.write_text(p.read_text() + "\n")
    next(iter(r.datasets.values())).load()
    assert r.status == "failed" and "Source changed" in str(r.report)


def test_bgi_tsv_header_supported(tmp_path):
    (tmp_path / "molecules.tsv").write_text("geneID\tx\ty\tMIDCounts\nG1\t1\t2\t3\n")
    r = read_spatial(tmp_path)
    assert r.status == "ok" and next(iter(r.datasets.values())).technology == "bgi", r.report


def test_library_id_metadata_preserved(tmp_path):
    visium(tmp_path)
    with h5py.File(tmp_path / "filtered_feature_bc_matrix.h5", "a") as f:
        f.attrs["library_ids"] = np.asarray(["library-A"], dtype="S")
    r = read_spatial(tmp_path)
    assert r.status == "ok" and "library-A" in r.adata.uns["spatial"]


def test_cosmx_multiple_file_groups_not_cross_paired(tmp_path):
    table_bundle(tmp_path, "nanostring")
    for suffix in ("exprMat_file", "metadata_file"):
        (tmp_path / f"other_{suffix}.csv").write_bytes((tmp_path / f"sample_{suffix}.csv").read_bytes())
    r = read_spatial(tmp_path)
    assert r.status == "ok" and len(r.datasets) == 2, r.report


def test_single_matrix_path_does_not_load_siblings(tmp_path):
    visium(tmp_path / "one")
    visium(tmp_path / "two")
    r = read_spatial(tmp_path / "one/filtered_feature_bc_matrix.h5")
    assert r.status == "ok" and len(r.datasets) == 1, r.report


def test_explicit_technology_restricts_intended_scope(tmp_path):
    visium(tmp_path / "visium")
    cell_bundle(tmp_path / "xenium")
    r = read_spatial(tmp_path, technology="visium")
    assert r.status == "ok" and len(r.datasets) == 1, r.report
    assert r.discovery["excluded_by_technology"] >= 1


def test_standard_images_take_budget_priority_over_qc(tmp_path, monkeypatch):
    import spateo.io.spatial.auto._automatic as automatic

    visium(tmp_path)
    for name in ("tissue_hires_image.png", "aligned_fiducials.jpg", "detected_tissue_image.jpg"):
        Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(tmp_path / "spatial" / name)
    Image.fromarray(np.zeros((3, 3, 3), dtype=np.uint8)).save(tmp_path / "spatial/tissue_lowres_image.png")
    monkeypatch.setattr(automatic, "_IMAGE_BUDGET", 100)
    r = read_spatial(tmp_path)
    assert r.status == "ok"
    assert set(next(iter(r.adata.uns["spatial"].values()))["images"]) == {"hires", "lowres"}
