"""Import compatibility checks for public flat IO modules."""

from unittest import TestCase


class TestLegacyIOImports(TestCase):
    def test_historical_helpers_remain_importable(self):
        from spateo.io.merfish import (
            read_merfish_as_anndata,
            read_merfish_positions_as_dataframe,
        )
        from spateo.io.nanostring import (
            FOV_PARSER,
            read_nanostring_as_dataframe,
            stitch_images,
        )
        from spateo.io.seqfish import read_seqfish_meta_as_dataframe
        from spateo.io.slideseq import (
            read_slideseq_as_dataframe,
            read_slideseq_beads_as_dataframe,
        )
        from spateo.io.starmap import (
            read_starmap_as_anndata,
            read_starmap_positions_as_dataframe,
        )
        from spateo.io.utils import bin_indices, bin_matrix, get_bin_props

        callables = (
            read_merfish_as_anndata,
            read_merfish_positions_as_dataframe,
            read_nanostring_as_dataframe,
            stitch_images,
            read_seqfish_meta_as_dataframe,
            read_slideseq_as_dataframe,
            read_slideseq_beads_as_dataframe,
            read_starmap_as_anndata,
            read_starmap_positions_as_dataframe,
            bin_indices,
            bin_matrix,
            get_bin_props,
        )
        self.assertTrue(all(callable(function) for function in callables))
        self.assertIsNotNone(FOV_PARSER.match("sample_F001.tif"))
