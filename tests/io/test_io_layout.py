import importlib.util
from pathlib import Path

import numpy as np
import scipy.io

import spateo.io as io
import spateo.io.spatial as spatial_io

LEGACY_FLAT_MODULES = {
    "bbs",
    "bgi",
    "image",
    "image_utils",
    "merfish",
    "nanostring",
    "seqfish",
    "seqscope",
    "slideseq",
    "starmap",
    "tenx",
    "utils",
}


def test_io_uses_only_subpackages_for_implementations():
    io_root = Path(io.__file__).parent
    assert {path.name for path in io_root.glob("*.py")} == {"__init__.py"}
    for module_name in LEGACY_FLAT_MODULES:
        assert importlib.util.find_spec(f"spateo.io.{module_name}") is None


def test_relocated_internal_capabilities_are_available_from_spatial_api():
    expected = (
        spatial_io.read_bgi,
        spatial_io.read_merfish,
        spatial_io.read_nanostring,
        spatial_io.read_seqfish,
        spatial_io.read_seqscope,
        spatial_io.read_slideseq,
        spatial_io.read_starmap_plus,
        spatial_io.read_image,
        spatial_io.add_image_layer,
        spatial_io.alpha_shape,
        spatial_io.bin_matrix,
    )
    assert all(callable(function) for function in expected)


def test_seqscope_reader_preserves_or_aggregates_barcodes(tmp_path):
    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()
    (matrix_dir / "barcodes.tsv").write_text("bc1\nbc2\nbc3\n", encoding="utf-8")
    (matrix_dir / "features.tsv").write_text(
        "gene1\tGene 1\tGene Expression\ngene2\tGene 2\tGene Expression\n",
        encoding="utf-8",
    )
    scipy.io.mmwrite(matrix_dir / "matrix.mtx", np.asarray([[1, 0, 2], [0, 3, 1]]))
    positions = tmp_path / "positions.txt"
    positions.write_text("bc1 1 1 1 1\nbc2 1 1 2 2\nbc3 1 1 12 12\n", encoding="utf-8")

    barcode_level = spatial_io.read_seqscope(matrix_dir, positions, binsize=None)
    assert barcode_level.shape == (3, 2)
    np.testing.assert_array_equal(barcode_level.obsm["spatial"], [[1, 1], [2, 2], [12, 12]])

    aggregated = spatial_io.read_seqscope(matrix_dir, positions, binsize=10, add_props=False)
    assert aggregated.shape == (2, 2)
    np.testing.assert_array_equal(np.asarray(aggregated.X.sum(axis=0)).ravel(), [3, 4])


def test_alpha_shape_handles_small_and_regular_point_sets():
    triangle, triangle_edges = spatial_io.alpha_shape(
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        buffer=0,
    )
    assert not triangle.is_empty
    assert triangle_edges == []

    square, square_edges = spatial_io.alpha_shape(
        np.asarray([0.0, 1.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0, 1.0]),
        alpha=1.0,
        buffer=0,
    )
    assert not square.is_empty
    assert square_edges
