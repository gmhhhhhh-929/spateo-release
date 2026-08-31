from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from spateo.io.spatial.auto import (
    detect_spatial_technologies,
    detect_spatial_technology,
)


class TestSpatialAuto(TestCase):
    def _tmpdir(self):
        return TemporaryDirectory()

    def test_detect_xenium_layout(self):
        with self._tmpdir() as tmp:
            root = Path(tmp)
            (root / "cell_feature_matrix.h5").touch()
            (root / "cells.csv.gz").touch()
            (root / "experiment.xenium").write_text("{}", encoding="utf-8")

            match = detect_spatial_technology(root)

        self.assertEqual("xenium", match.technology)
        self.assertEqual(root.resolve(), match.path)

    def test_detect_visium_hd_bin_from_outs(self):
        with self._tmpdir() as tmp:
            root = Path(tmp)
            square = root / "binned_outputs" / "square_016um"
            (square / "spatial").mkdir(parents=True)
            (square / "filtered_feature_bc_matrix.h5").touch()
            (square / "spatial" / "tissue_positions.parquet").touch()

            match = detect_spatial_technology(root)

        self.assertEqual("visium_hd_bin", match.technology)
        self.assertEqual(square.resolve(), match.path)
        self.assertEqual({"data_type": "bin", "binsize": 16}, dict(match.kwargs))

    def test_visium_hd_outs_with_bin_and_cellseg_is_ambiguous(self):
        with self._tmpdir() as tmp:
            root = Path(tmp)
            square = root / "binned_outputs" / "square_008um"
            (square / "spatial").mkdir(parents=True)
            (square / "filtered_feature_bc_matrix.h5").touch()
            (square / "spatial" / "tissue_positions.parquet").touch()

            seg = root / "segmented_outputs"
            seg.mkdir()
            (seg / "filtered_feature_cell_matrix.h5").touch()
            (seg / "graphclust_annotated_cell_segmentations.geojson").touch()

            with self.assertRaises(ValueError):
                detect_spatial_technology(root)

            match = detect_spatial_technology(root, technology="visium_hd_bin")

        self.assertEqual("visium_hd_bin", match.technology)
        self.assertEqual({"data_type": "bin", "binsize": 8}, dict(match.kwargs))

    def test_detect_merfish_layout(self):
        with self._tmpdir() as tmp:
            root = Path(tmp)
            (root / "cell_by_gene_S1R1.csv").touch()
            (root / "cell_metadata_S1R1.csv").touch()
            (root / "detected_transcripts_S1R1.csv").touch()
            (root / "images").mkdir()

            match = detect_spatial_technology(root)

        self.assertEqual("merfish", match.technology)
        self.assertEqual("cell_by_gene_S1R1.csv", match.kwargs["counts_file"])
        self.assertEqual("cell_metadata_S1R1.csv", match.kwargs["meta_file"])

    def test_detect_nanostring_layout(self):
        with self._tmpdir() as tmp:
            root = Path(tmp)
            (root / "sample_exprMat_file.csv").touch()
            (root / "sample_metadata_file.csv").touch()
            (root / "sample_fov_positions_file.csv").touch()
            (root / "CellComposite").mkdir()
            (root / "CellLabels").mkdir()

            match = detect_spatial_technology(root)

        self.assertEqual("nanostring", match.technology)
        self.assertEqual("sample_exprMat_file.csv", match.kwargs["counts_file"])
        self.assertEqual("sample_metadata_file.csv", match.kwargs["meta_file"])
        self.assertEqual("sample_fov_positions_file.csv", match.kwargs["fov_file"])

    def test_detect_bgi_file_and_directory(self):
        with self._tmpdir() as tmp:
            root = Path(tmp)
            gem = root / "reads.gem"
            gem.write_text("geneID\tx\ty\tMIDCounts\nGeneA\t1\t2\t3\n", encoding="utf-8")

            file_match = detect_spatial_technology(gem)
            dir_match = detect_spatial_technology(root)

        self.assertEqual("bgi", file_match.technology)
        self.assertEqual(gem.resolve(), file_match.path)
        self.assertEqual("bgi", dir_match.technology)
        self.assertEqual({"binsize": 1}, dict(dir_match.kwargs))

    def test_list_candidates_keeps_all_seqfish_groups(self):
        with self._tmpdir() as tmp:
            root = Path(tmp)
            (root / "SG_Counts_section1.csv").touch()
            (root / "SG_CellCoordinates_section1.csv").touch()
            (root / "SG_Counts_section2.csv").touch()
            (root / "SG_CellCoordinates_section2.csv").touch()

            matches = detect_spatial_technologies(root, technology="seqfish")

        self.assertEqual(2, len(matches))
        self.assertEqual({"seqfish"}, {m.technology for m in matches})
