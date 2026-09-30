"""Native vendor-schema fixtures: independent expected values, IDs and coordinates."""

import gzip
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from anndata import read_h5ad
from PIL import Image

import spateo.io as io


def bundle(root, technology):
    root.mkdir(parents=True, exist_ok=True)
    counts = np.array([[2, 0, 7], [0, 5, 1]], dtype=np.int64)  # genes x spots
    if technology == "singleron":
        from scipy.sparse import csc_matrix

        matrix = csc_matrix(counts)
        with h5py.File(root / "filtered_feature_bc_matrix.h5", "w") as f:
            f.attrs["chemistry_description"] = "Spatial3"
            g = f.create_group("matrix")
            for key in ("data", "indices", "indptr"):
                g[key] = getattr(matrix, key)
            g["shape"] = matrix.shape
            g["barcodes"] = np.array([b"b1", b"b2", b"b3"])
            g["features/id"] = np.array([b"EN1", b"EN2"])
            g["features/name"] = np.array([b"Shared", b"Shared"])
        (root / "spatial").mkdir()
        (root / "spatial/positions_list.csv").write_text("b3,1,2,0,60,30\nb1,1,0,0,20,10\nb2,0,1,0,40,20\n")
        return
    for name, value in {
        "matrix.mtx.gz": "%%MatrixMarket matrix coordinate integer general\n2 3 4\n1 1 2\n2 2 5\n1 3 7\n2 3 1\n",
        "features.tsv.gz": "EN1\tShared\tGene Expression\nEN2\tShared\tGene Expression\n",
        "barcodes.tsv.gz": "b1\nb2\nb3\n",
    }.items():
        with gzip.open(root / name, "wt") as f:
            f.write(value)
    filename, value = {
        "seekspace": ("cell_locations.tsv.gz", "Cell_Barcode\tX\tY\nb3\t30\t60\nb1\t10\t20\nb2\t20\t40\n"),
        "bmkmanu": ("barcodes_pos.tsv.gz", "b3\t30\t60\nb1\t10\t20\nb2\t20\t40\n"),
        "salus": ("spatial.txt.gz", "b3 30 60\nb1 10 20\nb2 20 40\n"),
    }[technology]
    with gzip.open(root / filename, "wt") as f:
        f.write(value)


@pytest.mark.parametrize("technology", ["seekspace", "bmkmanu", "salus", "singleron"])
def test_auto_explicit_and_roundtrip_preserve_all_source_counts(tmp_path, technology):
    bundle(tmp_path, technology)
    result = io.read_spatial(tmp_path, load_images=False)
    assert result.status == "ok", result.report
    entry = next(iter(result.values()))
    assert entry.technology == technology
    a = result.adata
    np.testing.assert_array_equal(a.X.toarray(), [[2, 0], [0, 5], [7, 1]])
    np.testing.assert_array_equal(a.obsm["spatial"], [[10, 20], [20, 40], [30, 60]])
    assert list(a.obs_names) == ["b1", "b2", "b3"]
    assert list(a.var_names) == ["EN1", "EN2"]
    assert list(a.var.gene_name) == ["Shared", "Shared"]
    assert a.X.dtype == np.int64 and a.n_obs == 3
    direct = getattr(io, "read_" + technology)(tmp_path, load_images=False)
    assert (direct.X != a.X).nnz == 0
    assert getattr(io, "read_" + technology)(tmp_path, load_images=False, return_result=True).status == "ok"
    a.write_h5ad(tmp_path / "result.h5ad")
    saved = read_h5ad(tmp_path / "result.h5ad")
    assert (saved.X != a.X).nnz == 0
    np.testing.assert_array_equal(saved.obsm["spatial"], a.obsm["spatial"])
    assert saved.uns["spateo_io"]["technology"] == technology


@pytest.mark.parametrize("technology", ["seekspace", "bmkmanu", "salus", "singleron"])
def test_domestic_lazy_preserves_report_and_selection(tmp_path, technology):
    bundle(tmp_path / technology, technology)
    result = io.read_spatial(tmp_path, lazy=True, load_images=False)
    entry = next(iter(result.values()))
    assert entry.materialization_attempts == 0
    json.dumps(result.report)
    assert entry.materialization_attempts == 0
    assert result.adata.shape == (3, 2)
    assert entry.materialization_attempts == 1
    assert result.adata is entry.adata
    assert entry.materialization_attempts == 1


@pytest.mark.parametrize("filename", ["matrix.mtx.gz", "features.tsv.gz", "barcodes.tsv.gz"])
def test_incomplete_mex_names_missing_component(tmp_path, filename):
    bundle(tmp_path, "bmkmanu")
    (tmp_path / filename).unlink()
    result = io.read_spatial(tmp_path)
    assert result.status == "failed"
    report = json.dumps(result.report)
    assert filename in report and "recovery" in report


@pytest.mark.parametrize(
    "technology,filename",
    [("seekspace", "cell_locations.tsv.gz"), ("bmkmanu", "barcodes_pos.tsv.gz"), ("salus", "spatial.txt.gz")],
)
def test_missing_or_duplicate_coordinate_ids_never_row_order_join(tmp_path, technology, filename):
    bundle(tmp_path, technology)
    with gzip.open(tmp_path / filename, "rt") as f:
        text = f.read()
    with gzip.open(tmp_path / filename, "wt") as f:
        f.write(text.replace("b2", "b1"))
    result = io.read_spatial(tmp_path)
    assert result.status == "failed"
    assert next(iter(result.values())).adata is None


def test_bmk_raw_chip_index_file_is_not_xy(tmp_path):
    bundle(tmp_path, "bmkmanu")
    (tmp_path / "barcodes_pos.tsv.gz").unlink()
    (tmp_path / "barcode_pos.tsv").write_text("b1\t1\t2\t3\t4\n")
    result = io.read_spatial(tmp_path)
    assert result.status == "failed"
    assert "BSTMatrix aggregated export" in str(result.report)


def test_singleron_is_not_inferred_from_generic_visium_h5(tmp_path):
    bundle(tmp_path, "singleron")
    with h5py.File(tmp_path / "filtered_feature_bc_matrix.h5", "r+") as f:
        del f.attrs["chemistry_description"]
    result = io.read_spatial(tmp_path, technology="singleron")
    assert result.status == "failed" and not result.datasets


def test_two_seekspace_samples_only_load_matching_images(tmp_path):
    for name, value in [("A", 10), ("B", 20)]:
        bundle(tmp_path / (name + "_filtered_feature_bc_matrix"), "seekspace")
        Image.fromarray(np.full((5, 5), value, dtype=np.uint8)).save(tmp_path / (name + "_aligned_DAPI.png"))
    result = io.read_spatial(tmp_path)
    assert result.status == "ok" and len(result.datasets) == 2, result.report
    for entry in result.values():
        slot = next(iter(entry.adata.uns["spatial"].values()))
        assert len(slot["images"]) == 1
        name = next(iter(slot["images"]))
        assert name[0] in entry.key and name.endswith("_aligned_DAPI.png")


def test_distinct_samples_are_all_retained(tmp_path):
    for technology in ("seekspace", "bmkmanu", "salus", "singleron"):
        bundle(tmp_path / technology, technology)
    result = io.read_spatial(tmp_path, load_images=False)
    assert result.status == "ok", result.report
    assert {e.technology for e in result.values()} == {"seekspace", "bmkmanu", "salus", "singleron"}
    with pytest.raises(ValueError):
        _ = result.adata
