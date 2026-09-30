"""Independent count/coordinate oracles for sequencing readers and scope limits."""

import gzip

import numpy as np
import pytest
from anndata import read_h5ad
from PIL import Image
from scipy import sparse
from scipy.io import mmwrite

from spateo.io import (
    read_bgi,
    read_seqscope,
    read_slideseq,
    read_spatial,
    read_stereoseq,
)


def slide_fixture(root, compressed=False):
    root.mkdir(exist_ok=True)
    files = {
        "MappedDGEForR.csv": "gene,0002,0001,0003\nG2,0,2147483651,7\nG0,0,0,0\nG1,3,2,9\n",
        "BeadLocationsForR.csv": "barcodes,xcoord,ycoord\n0003,8.125,9.25\n0001,1.75,2.5\n0002,3.25,4.5\n",
    }
    for name, text in files.items():
        if compressed:
            with gzip.open(root / (name + ".gz"), "wt") as f:
                f.write(text)
        else:
            (root / name).write_text(text)
    return root


def assert_slide(a):
    assert list(a.obs_names) == ["0002", "0001", "0003"]
    assert list(a.var_names) == ["G2", "G0", "G1"]
    assert a.X.dtype == np.dtype("int64")
    np.testing.assert_array_equal(a.X.toarray(), [[0, 0, 3], [2147483651, 0, 2], [7, 0, 9]])
    np.testing.assert_array_equal(a.obsm["spatial"], [[3.25, 4.5], [1.75, 2.5], [8.125, 9.25]])


@pytest.mark.parametrize("compressed", [False, True])
def test_slideseq_both_routes_exact_native_counts_and_images(tmp_path, compressed):
    root = slide_fixture(tmp_path / "puck", compressed)
    pixels = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    Image.fromarray(pixels).save(root / "tissue.png")
    direct = read_slideseq(root)
    result = read_spatial(root)
    assert result.status == "ok", result.report
    assert next(iter(result.values())).technology == "slideseq"
    for i, a in enumerate((direct, result.adata)):
        assert_slide(a)
        images = [
            img
            for slot in a.uns["spatial"].values()
            if isinstance(slot, dict)
            for img in slot.get("images", {}).values()
        ]
        assert any(np.array_equal(img, pixels) for img in images)
        a.write_h5ad(tmp_path / f"slide_{i}.h5ad")
        assert_slide(read_h5ad(tmp_path / f"slide_{i}.h5ad"))


@pytest.mark.parametrize(
    "problem",
    [
        "bad_count",
        "negative",
        "fraction",
        "precision",
        "duplicate_gene",
        "duplicate_barcode",
        "duplicate_position",
        "missing_position",
        "nan_coordinate",
        "missing_file",
    ],
)
def test_slideseq_both_routes_reject_invalid_native_data(tmp_path, problem):
    root = slide_fixture(tmp_path)
    counts, coords = root / "MappedDGEForR.csv", root / "BeadLocationsForR.csv"
    text = counts.read_text()
    changes = {"bad_count": "bad", "negative": "-1", "fraction": "1.5", "precision": "9223372036854775808"}
    if problem in changes:
        counts.write_text(text.replace("2147483651", changes[problem]))
    elif problem == "duplicate_gene":
        counts.write_text(text.replace("G0,", "G2,"))
    elif problem == "duplicate_barcode":
        counts.write_text(text.replace("gene,0002,0001,0003", "gene,0002,0002,0003"))
    elif problem == "duplicate_position":
        coords.write_text(coords.read_text() + "0001,99,99\n")
    elif problem == "missing_position":
        coords.write_text(coords.read_text().replace("0001,1.75,2.5\n", ""))
    elif problem == "nan_coordinate":
        coords.write_text(coords.read_text().replace("1.75", "NaN"))
    else:
        coords.unlink()
    with pytest.raises((ValueError, OSError)):
        read_slideseq(root)
    result = read_spatial(root)
    assert result.status == "failed", result.report
    assert all(e.adata is None for e in result.values())


def test_slideseq_direct_does_not_call_automatic_dispatch(tmp_path, monkeypatch):
    import spateo.io.spatial.auto._automatic as automatic

    def forbidden(*args, **kwargs):
        raise AssertionError("Direct reading called automatic dispatch")

    monkeypatch.setattr(automatic, "read_spatial", forbidden)
    assert_slide(read_slideseq(slide_fixture(tmp_path), load_images=False))


def stereo_gem(path, modern, cell):
    schema = "#FileFormat=GEMv0.2\n#OffsetX=100\n#OffsetY=200\n" if modern else ""
    if cell:
        schema += "#BinType=CellBin\n#BinSize=Cell\n"
    header = "geneID\t" + ("geneName\t" if modern else "") + "x\ty\tMIDCount\tExonCount" + ("\tCellID" if cell else "")
    records = [
        ("G2", "same", 1, 2, 70000, 60000, 0),
        ("G1", "same", 1, 2, 3, 2, 0),
        ("G2", "same", 3, 4, 5, 4, 0),
        ("G1", "same", 50, 60, 9, 8, 7),
    ]
    rows = []
    for gene, symbol, x, y, count, exon, label in records:
        row = [gene] + ([symbol] if modern else []) + [x, y, count, exon] + ([label] if cell else [])
        rows.append("\t".join(map(str, row)))
    text = schema + header + "\n" + "\n".join(rows) + "\n"
    with gzip.open(path, "wt") as f:
        f.write(text)
    return path


@pytest.mark.parametrize("chemistry", ["V1", "V2"])
@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("cell", [False, True])
def test_stereoseq_gem_both_routes_exact_counts_layers_coordinates(tmp_path, chemistry, modern, cell):
    path = stereo_gem(tmp_path / "sample.gem.gz", modern, cell)
    options = {"stereoseq_chemistry": chemistry, "stereoseq_bin_size": None if cell else 50}
    auto = read_spatial(path, load_images=False, **options).adata
    direct = read_stereoseq(path, chemistry=chemistry, bin_size=None if cell else 50, load_images=False)
    expected_xy = [[2, 3], [50, 60]] if cell else [[0, 0], [50, 50]]
    for i, a in enumerate((auto, direct)):
        assert list(a.var_names) == ["G2", "G1"]
        np.testing.assert_array_equal(a.X.toarray(), [[70005, 3], [0, 9]])
        np.testing.assert_array_equal(a.layers["exon"].toarray(), [[60004, 2], [0, 8]])
        np.testing.assert_array_equal(a.obsm["spatial"], expected_xy)
        assert a.uns["stereoseq"]["chemistry"] == chemistry
        if modern:
            np.testing.assert_array_equal(a.obsm["spatial_global"], np.array(expected_xy) + [100, 200])
        a.write_h5ad(tmp_path / f"stereo_{i}.h5ad")
        np.testing.assert_array_equal(read_h5ad(tmp_path / f"stereo_{i}.h5ad").X.toarray(), [[70005, 3], [0, 9]])


@pytest.mark.parametrize("cell", [False, True])
def test_stereoseq_gef_both_routes_exact(tmp_path, cell):
    from tests.io.test_stereoseq_native import gef

    path = gef(tmp_path, cell=cell, modern=True)
    auto = read_spatial(path, load_images=False).adata
    direct = read_stereoseq(path, load_images=False)
    for a in (auto, direct):
        np.testing.assert_array_equal(a.X.toarray(), [[70000, 3], [0, 4]] if cell else [[70000, 3], [4, 0]])
        np.testing.assert_array_equal(a.obsm["spatial"], [[10, 20], [30, 40]])
        assert list(a.var_names) == ["ENSG1", "ENSG2"]
        assert int(a.layers["exon"].sum()) == 60003


def test_stereo_legacy_binning_semantics_are_explicit(tmp_path):
    path = stereo_gem(tmp_path / "sample.gem.gz", modern=True, cell=False)
    native = read_stereoseq(path, bin_size=50, load_images=False)
    legacy = read_bgi(path, binsize=50, add_props=True)
    np.testing.assert_array_equal(legacy[:, ["G2", "G1"]].X.toarray(), native.X.toarray())
    # Legacy outputs bin indices as IDs and centroids as coordinates; native uses bin origins.
    assert list(legacy.obs_names) == ["0-0", "1-1"]
    np.testing.assert_array_equal(legacy.obsm["spatial"], native.obsm["spatial"] + 25)


def seqscope_fixture(root):
    root.mkdir(exist_ok=True)
    (root / "features.tsv").write_text("id2\tSame\tGene Expression\nid1\tSame\tGene Expression\n")
    (root / "barcodes.tsv").write_text("0002\n0001\n0003\n")
    mmwrite(root / "matrix.mtx", sparse.coo_matrix([[0, 70000, 7], [3, 2, 9]], dtype=np.int64))
    path = root / "positions.txt"
    path.write_text("0003 1 1101 25 31\n0001 1 1101 12 14\n0002 1 1101 11 13\n")
    return path


@pytest.mark.parametrize("binsize", [None, 10])
def test_seqscope_direct_exact_and_auto_explicitly_unsupported(tmp_path, binsize):
    path = seqscope_fixture(tmp_path)
    a = read_seqscope(tmp_path, path, binsize=binsize, add_props=False)
    assert list(a.var_names) == ["id2", "id1"]
    if binsize is None:
        assert list(a.obs_names) == ["0002", "0001", "0003"]
        np.testing.assert_array_equal(a.X.toarray(), [[0, 3], [70000, 2], [7, 9]])
        np.testing.assert_array_equal(a.obsm["spatial"], [[11, 13], [12, 14], [25, 31]])
    else:
        np.testing.assert_array_equal(a.X.toarray(), [[70000, 5], [7, 9]])
        np.testing.assert_array_equal(a.obsm["spatial"], [[15, 15], [25, 35]])
    result = read_spatial(tmp_path)
    assert result.status != "ok" and not any(e.status == "ready" for e in result.values())


@pytest.mark.parametrize(
    "problem", ["missing_position", "duplicate_position", "negative", "fraction", "negative_count", "duplicate_barcode"]
)
def test_seqscope_direct_rejects_lossy_input(tmp_path, problem):
    path = seqscope_fixture(tmp_path)
    if problem == "missing_position":
        path.write_text(path.read_text().replace("0001 1 1101 12 14\n", ""))
    elif problem == "duplicate_position":
        path.write_text(path.read_text() + "0001 1 1101 8 9\n")
    elif problem in ("negative", "fraction"):
        path.write_text(path.read_text().replace("12 14", "-1 14" if problem == "negative" else "1.5 14"))
    elif problem == "negative_count":
        mmwrite(tmp_path / "matrix.mtx", sparse.coo_matrix([[0, -1, 7], [3, 2, 9]]))
    else:
        (tmp_path / "barcodes.tsv").write_text("0002\n0002\n0003\n")
    with pytest.raises(ValueError):
        read_seqscope(tmp_path, path, binsize=None)


def test_slideseq_ambiguous_files_require_explicit_choice(tmp_path):
    slide_fixture(tmp_path)
    with gzip.open(tmp_path / "MappedDGEForR.csv.gz", "wt") as handle:
        handle.write((tmp_path / "MappedDGEForR.csv").read_text())
    with pytest.raises(ValueError, match="Ambiguous"):
        read_slideseq(tmp_path)
    assert_slide(read_slideseq(tmp_path, counts_file="MappedDGEForR.csv"))
    result = read_spatial(tmp_path)
    assert result.status == "ok" and len(result.datasets) == 2
    for entry in result.values():
        assert_slide(entry.adata)
    with pytest.raises(ValueError):
        _ = result.adata


def test_slideseq_large_integer_text_is_exact(tmp_path):
    slide_fixture(tmp_path)
    path = tmp_path / "MappedDGEForR.csv"
    path.write_text(path.read_text().replace("2147483651", "9007199254740993"))
    for a in (read_slideseq(tmp_path), read_spatial(tmp_path).adata):
        assert int(a.X[1, 0]) == 9007199254740993
        assert a.X.dtype == np.dtype("int64")
