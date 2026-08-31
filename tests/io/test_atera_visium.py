"""Regression tests for current 10x spatial output contracts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse

from spateo.io import read_atera, read_visium
from spateo.io.spatial.auto import detect_spatial_technology


def _write_10x_h5(path: Path, counts: np.ndarray, barcodes: list[str], genes: list[str]) -> None:
    """Write the small subset of the 10x v3 HDF5 contract used by readers."""
    matrix = sparse.csc_matrix(np.asarray(counts, dtype=np.int32))
    with h5py.File(path, "w") as handle:
        group = handle.create_group("matrix")
        group.create_dataset("barcodes", data=np.asarray(barcodes, dtype="S"))
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices)
        group.create_dataset("indptr", data=matrix.indptr)
        group.create_dataset("shape", data=np.asarray(matrix.shape, dtype=np.int64))
        features = group.create_group("features")
        features.create_dataset("id", data=np.asarray([f"id-{gene}" for gene in genes], dtype="S"))
        features.create_dataset("name", data=np.asarray(genes, dtype="S"))
        features.create_dataset("feature_type", data=np.asarray(["Gene Expression"] * len(genes), dtype="S"))


class TestAteraIO(TestCase):
    def test_auto_detect_and_read_preview_bundle(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            focus = root / "morphology_focus"
            focus.mkdir()
            _write_10x_h5(
                root / "cell_feature_matrix.h5",
                np.array([[1, 0, 2], [0, 3, 1]]),
                ["cell-1", "cell-2", "cell-3"],
                ["GENE1", "GENE2"],
            )
            pd.DataFrame(
                {
                    "cell_id": ["cell-1", "cell-2", "cell-3"],
                    "x_centroid": [1.0, 2.0, 3.0],
                    "y_centroid": [4.0, 5.0, 6.0],
                    "cell_area": [50.0, 60.0, 70.0],
                }
            ).to_csv(root / "cells.csv", index=False)
            vertices = []
            for cell_id, x, y in (("cell-1", 1, 4), ("cell-2", 2, 5), ("cell-3", 3, 6)):
                vertices.extend((cell_id, x + dx, y + dy) for dx, dy in ((0, 0), (1, 0), (0, 1)))
            pd.DataFrame(vertices, columns=["cell_id", "vertex_x", "vertex_y"]).to_csv(
                root / "cell_boundaries.csv", index=False
            )
            (root / "experiment.xenium").write_text(
                json.dumps({"run_name": "Atera WTA preview", "panel": {"type": "whole transcriptome"}}),
                encoding="utf-8",
            )
            # Detection uses explicit Atera stain names; arrays are not loaded in this test.
            (focus / "ch0000_dapi.ome.tif").touch()
            (focus / "ch0001_atp1a1_cd45.ome.tif").touch()

            match = detect_spatial_technology(root)
            cache_path = root / "atera-cache.h5ad"
            adata = read_atera(
                root,
                load_image=False,
                load_nucleus_boundaries=False,
                load_cell_groups=False,
                cell_groups_csv=root / "intentionally-missing.csv",
                cache_file=cache_path,
            )
            cached = read_atera(root, cache_file=cache_path)
            self.assertTrue(cache_path.is_file())

        self.assertEqual("atera", match.technology)
        self.assertEqual((3, 2), adata.shape)
        np.testing.assert_allclose(adata.obsm["spatial"], [[1, 4], [2, 5], [3, 6]])
        self.assertTrue(adata.obs["geometry"].str.startswith("POLYGON").all())
        self.assertEqual("preview-xenium-v4", adata.uns["spateo_io"]["format_status"])
        self.assertEqual("atera_seg", adata.uns["spateo_io"]["type"])
        self.assertIn("cell_feature_matrix.h5", adata.uns["spateo_io"]["manifest"]["paths"])
        self.assertEqual(adata.shape, cached.shape)


class TestVisiumIO(TestCase):
    def test_coordinates_do_not_depend_on_image_loading(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "sample" / "outs"
            spatial_dir = root / "spatial"
            spatial_dir.mkdir(parents=True)
            matrix_path = root / "filtered_feature_bc_matrix.h5"
            _write_10x_h5(matrix_path, np.array([[1, 0], [0, 2]]), ["spot-1", "spot-2"], ["G1", "G2"])
            with h5py.File(matrix_path, "a") as handle:
                handle.attrs["library_ids"] = np.asarray(["library-A"], dtype="S")
                handle.attrs["chemistry_description"] = "Visium"
            pd.DataFrame(
                [
                    ["spot-1", 1, 0, 1, 10.0, 20.0],
                    ["spot-2", 0, 1, 0, 30.0, 40.0],
                ]
            ).to_csv(spatial_dir / "tissue_positions_list.csv", index=False, header=False)
            (spatial_dir / "scalefactors_json.json").write_text(
                json.dumps({"tissue_hires_scalef": 0.5, "spot_diameter_fullres": 100}), encoding="utf-8"
            )
            Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(spatial_dir / "tissue_hires_image.png")

            adata_no_pixels = read_visium(root.parent, load_images=False)
            adata_hires = read_visium(root, load_images=True)
            roundtrip_path = root / "visium.h5ad"
            adata_hires.write_h5ad(roundtrip_path)
            self.assertTrue(roundtrip_path.is_file())

        # obsm uses image x/y, i.e. full-resolution column then row.
        np.testing.assert_allclose(adata_no_pixels.obsm["spatial"], [[20, 10], [40, 30]])
        self.assertEqual({}, adata_no_pixels.uns["spatial"]["library-A"]["images"])
        self.assertEqual((4, 5, 3), adata_hires.uns["spatial"]["library-A"]["images"]["hires"].shape)
        self.assertNotIn("lowres", adata_hires.uns["spatial"]["library-A"]["images"])
        self.assertEqual("visium", adata_hires.uns["spateo_io"]["technology"])
