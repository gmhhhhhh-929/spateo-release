"""Native SAW schemas, exact count conservation, and explicit failure cases."""

import gzip

import h5py
import numpy as np
import pytest

from spateo.io import read_spatial, read_stereoseq
from spateo.io.spatial._stereoseq import read_bgi, read_bgi_as_dataframe


def gem(tmp_path, text, compressed=False):
    p = tmp_path / ("sample.gem.gz" if compressed else "sample.gem")
    if compressed:
        with gzip.open(p, "wt") as f:
            f.write(text)
    else:
        p.write_text(text)
    return p


def gef(tmp_path, cell=False, modern=True):
    p = tmp_path / ("test.cellbin.gef" if cell else "test.gef")
    with h5py.File(p, "w") as f:
        f.attrs["version"] = 4 if modern else 2
        f.attrs["omics"] = "Transcriptomics"
        g = f.create_group("cellBin" if cell else "geneExp/bin1")
        gene_fields = (
            [("geneID", "S16"), ("geneName", "S16")] if modern else [(("geneName" if cell else "gene"), "S16")]
        )
        if cell:
            genes = [(b"ENSG1", b"Same"), (b"ENSG2", b"Same")] if modern else [(b"g1",), (b"g2",)]
            g.create_dataset("gene", data=np.array(genes, dtype=gene_fields))
            g.create_dataset(
                "cell",
                data=np.array(
                    [(7, 10, 20, 0, 2, 70003), (9, 30, 40, 2, 1, 4)],
                    dtype=[
                        ("id", "u4"),
                        ("x", "i4"),
                        ("y", "i4"),
                        ("offset", "u4"),
                        ("geneCount", "u4"),
                        ("expCount", "u4"),
                    ],
                ),
            )
            g.create_dataset(
                "cellExp", data=np.array([(0, 70000), (1, 3), (1, 4)], dtype=[("geneID", "u4"), ("count", "u4")])
            )
            g.create_dataset("cellExpExon", data=np.array([60000, 2, 1], dtype="u4"))
            f.attrs["resolution"] = 715
        else:
            genes = [(b"ENSG1", b"Same", 0, 2), (b"ENSG2", b"Same", 2, 1)] if modern else [(b"g1", 0, 2), (b"g2", 2, 1)]
            g.create_dataset("gene", data=np.array(genes, dtype=gene_fields + [("offset", "u4"), ("count", "u4")]))
            d = g.create_dataset(
                "expression",
                data=np.array(
                    [(10, 20, 70000), (30, 40, 4), (10, 20, 3)], dtype=[("x", "u4"), ("y", "u4"), ("count", "u4")]
                ),
            )
            d.attrs["resolution"] = 500
            g.create_dataset("exon", data=np.array([60000, 1, 2], dtype="u4"))
    return p


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("chemistry", [None, "V1", "V2"])
def test_gem_modern_retains_ids_total_rna_exon_and_offsets(tmp_path, compressed, chemistry):
    p = gem(
        tmp_path,
        "#FileFormat=GEMv0.2\n#BinType=Bin\n#BinSize=1\n#OffsetX=100\n#OffsetY=200\ngeneID\tgeneName\tx\ty\tMIDCount\tExonCount\nENSG1\tSame\t10\t20\t70000\t60000\nENSG1\tSame\t10\t20\t2\t1\nENSG2\tSame\t30\t40\t3\t2\nNONPOLYA\tRNA\t30\t40\t4\t0\n",
        compressed,
    )
    a = read_stereoseq(p, chemistry=chemistry, load_images=False)
    np.testing.assert_array_equal(a.X.toarray(), [[70002, 0, 0], [0, 3, 4]])
    np.testing.assert_array_equal(a.layers["exon"].toarray(), [[60001, 0, 0], [0, 2, 0]])
    np.testing.assert_array_equal(a.obsm["spatial_global"], [[110, 220], [130, 240]])
    assert list(a.var_names) == ["ENSG1", "ENSG2", "NONPOLYA"]
    assert list(a.var.gene_name) == ["Same", "Same", "RNA"]
    assert a.uns["stereoseq"]["chemistry"] == (chemistry or "unspecified")
    assert "spliced" not in a.layers
    q = tmp_path / "roundtrip.h5ad"
    a.write_h5ad(q)
    from anndata import read_h5ad

    b = read_h5ad(q)
    assert b.uns["stereoseq"]["file_format"] == "GEMv0.2"


@pytest.mark.parametrize("cell", [False, True])
@pytest.mark.parametrize("modern", [False, True])
def test_gef_old_new_bin_cell_roundtrip(tmp_path, cell, modern):
    p = gef(tmp_path, cell, modern)
    a = read_stereoseq(p, load_images=False)
    expected = [[70000, 3], [0, 4]] if cell else [[70000, 3], [4, 0]]
    np.testing.assert_array_equal(a.X.toarray(), expected)
    assert int(a.layers["exon"].sum()) == 60003
    assert a.uns["stereoseq"]["chemistry"] == "unspecified"  # GEF schema 2 != chemistry V2
    assert a.uns["stereoseq"]["pitch_nm"] == (715 if cell else 500)
    assert list(a.obs_names) == (["7", "9"] if cell else ["10_20", "30_40"])
    a.write_h5ad(tmp_path / "roundtrip.h5ad")


def test_gem_explicit_bin_aggregation_conserves_all_counts(tmp_path):
    p = gem(
        tmp_path,
        "geneID\tx\ty\tMIDCounts\tEXONIC\tINTRONIC\ng1\t1\t2\t7\t4\t3\ng1\t3\t4\t5\t2\t3\ng2\t50\t50\t9\t7\t2\n",
    )
    a = read_stereoseq(p, bin_size=50, load_images=False)
    np.testing.assert_array_equal(a.X.toarray(), [[12, 0], [0, 9]])
    np.testing.assert_array_equal(a.obsm["spatial"], [[0, 0], [50, 50]])
    assert a.uns["stereoseq"]["input_records"] == 3
    assert int(a.layers["spliced"].sum()) == 13


def test_cell_gem_uses_unique_dnb_centroid_includes_zero_id(tmp_path):
    p = gem(
        tmp_path,
        "#BinType=CellBin\n#BinSize=Cell\ngeneID\tx\ty\tMIDCount\tCellID\ng1\t1\t2\t1\t0\ng2\t1\t2\t2\t0\ng1\t3\t4\t3\t0\ng2\t20\t30\t4\t9\n",
    )
    a = read_stereoseq(p, load_images=False)
    np.testing.assert_array_equal(a.obsm["spatial"], [[2, 3], [20, 30]])
    np.testing.assert_array_equal(a.X.toarray(), [[4, 2], [0, 4]])
    assert a.uns["stereoseq"]["observation_type"] == "cell"


@pytest.mark.parametrize(
    "body",
    [
        "geneID\tx\ty\tMIDCount\ng1\t1\t2\t-1\n",
        "geneID\tx\ty\tMIDCount\ng1\t1.5\t2\t1\n",
        "geneID\tx\ty\tMIDCount\ng1\t1\tNaN\t1\n",
        "geneID\tx\ty\tMIDCount\n\t1\t2\t1\n",
        "geneID\tx\ty\tMIDCount\tMIDCounts\ng1\t1\t2\t1\t1\n",
        "geneID\tx\ty\tMIDCount\tExonCount\ng1\t1\t2\t1\t2\n",
        "geneID\tx\ty\tMIDCount\ng1\t1\t2\t9223372036854775808\n",
        "geneID\tx\ty\tMIDCount\ng1\t1\t2\t9223372036854775807\ng1\t1\t2\t1\n",
        "#FileFormat=GEMv9.0\ngeneID\tx\ty\tMIDCount\ng1\t1\t2\t1\n",
        "#Omics=Proteomics\ngeneID\tx\ty\tMIDCount\ng1\t1\t2\t1\n",
        "geneID\tgeneName\tx\ty\tMIDCount\ng1\tA\t1\t2\t1\ng1\tB\t3\t4\t1\n",
        "geneID\tx\ty\tMIDCount\tCellID\ng1\t1\t2\t1\t1\ng2\t1\t2\t1\t2\n",
    ],
)
def test_malformed_gem_is_failed_not_repaired(tmp_path, body):
    result = read_spatial(gem(tmp_path, body), load_images=False)
    assert [x.status for x in result.datasets.values()] == ["failed"]
    assert all(x.adata is None for x in result.datasets.values())


@pytest.mark.parametrize("mutation", ["offset", "gene_index", "negative", "exon", "total", "duplicate_id"])
def test_cell_gef_corruption(tmp_path, mutation):
    p = gef(tmp_path, True)
    with h5py.File(p, "r+") as f:
        g = f["cellBin"]
        if mutation in ["offset", "total", "duplicate_id"]:
            a = g["cell"][:]
            a["offset" if mutation == "offset" else "expCount" if mutation == "total" else "id"][1] = 0
            if mutation == "duplicate_id":
                a["id"][1] = 7
            g["cell"][:] = a
        elif mutation == "gene_index":
            a = g["cellExp"][:]
            a["geneID"][0] = 99
            g["cellExp"][:] = a
        elif mutation == "negative":
            a = g["cell"][:]
            a["x"][0] = -1
            g["cell"][:] = a
        else:
            g["cellExpExon"][0] = 80000
    assert next(iter(read_spatial(p).datasets.values())).status == "failed"


def test_resource_deferred_no_partial_object(tmp_path):
    p = gef(tmp_path)
    r = read_spatial(p, max_memory_bytes=1024, load_images=False)
    assert next(iter(r.datasets.values())).status == "deferred"
    assert next(iter(r.datasets.values())).adata is None
    p.unlink()
    p = gem(tmp_path, "geneID\tx\ty\tMIDCount\ng1\t1\t2\t1\n")
    r = read_spatial(p, max_memory_bytes=1024, load_images=False)
    assert next(iter(r.datasets.values())).status == "deferred"


def test_old_direct_reader_no_uint16_overflow(tmp_path):
    p = gem(
        tmp_path, "geneID\tgeneName\tx\ty\tMIDCount\tExonCount\ng1\tA\t1\t2\t70000\t65000\ng1\tA\t1\t2\t10000\t9000\n"
    )
    a = read_bgi(p, binsize=50, add_props=False)
    assert int(a.X.sum()) == 80000
    assert int(a.layers["exon"].sum()) == 74000
    assert a.var.gene_name.iloc[0] == "A"


@pytest.mark.parametrize("bin_size", [0, -1, True, 1.5])
def test_invalid_resolution_option(tmp_path, bin_size):
    with pytest.raises(ValueError):
        read_spatial(tmp_path, stereoseq_bin_size=bin_size)


def test_gef_resolution_missing_and_cell_rebin_rejected(tmp_path):
    p = gef(tmp_path)
    assert next(iter(read_spatial(p, stereoseq_bin_size=50).datasets.values())).status == "failed"
    p.unlink()
    p = gef(tmp_path, True)
    assert next(iter(read_spatial(p, stereoseq_bin_size=50).datasets.values())).status == "failed"


def test_v2_paper_gem2_and_legacy_negative_coordinate(tmp_path):
    p = gem(tmp_path, "geneID\tx\ty\tMIDCount\tEXONIC\tINTRONIC\ng1\t1\t2\t8\t3\t5\n", True)
    q = p.with_name("paper.gem2.gz")
    p.rename(q)
    a = read_stereoseq(q, chemistry="V2", load_images=False)
    assert int(a.layers["unspliced"].sum()) == 5
    assert int(a.layers["spliced"].sum()) == 3
    q.unlink()
    p = gem(tmp_path, "geneID\tx\ty\tMIDCount\ng1\t-1\t2\t1\n")
    with pytest.raises(ValueError):
        read_bgi_as_dataframe(p)


def test_multiresolution_gef_defaults_finest_selects_explicit(tmp_path):
    p = gef(tmp_path)
    with h5py.File(p, "r+") as f:
        f.copy("geneExp/bin1", "geneExp/bin50")
    assert read_stereoseq(p, load_images=False).uns["stereoseq"]["bin_size"] == 1
    assert read_stereoseq(p, bin_size=50, load_images=False).uns["stereoseq"]["bin_size"] == 50


def test_gem_coarser_native_cannot_upsample(tmp_path):
    p = gem(tmp_path, "#BinSize=50\ngeneID\tx\ty\tMIDCount\ng1\t0\t50\t4\n")
    assert read_stereoseq(p, load_images=False).uns["stereoseq"]["bin_size"] == 50
    assert next(iter(read_spatial(p, stereoseq_bin_size=1).datasets.values())).status == "failed"


def test_integer_valued_decimal_text_keeps_exact_large_counts(tmp_path):
    p = gem(tmp_path, "geneID\tx\ty\tMIDCount\ng1\t1.0\t2.00\t9007199254740993.0\n")
    a = read_stereoseq(p, load_images=False)
    assert a.X[0, 0] == 9007199254740993
    np.testing.assert_array_equal(a.obsm["spatial"], [[1, 2]])
