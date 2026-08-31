"""Tests for the promoted spatial preprocessing pipeline."""

from unittest import TestCase

import numpy as np
from anndata import AnnData
from scipy import sparse

from spateo.preprocessing import normalize_total, preprocess_spatial


class TestSpatialPreprocessing(TestCase):
    def test_auto_atera_recipe_preserves_counts_and_input(self):
        random = np.random.default_rng(42)
        adata = AnnData(sparse.csr_matrix(random.poisson(2, size=(30, 12))))
        adata.var_names = [f"GENE{i}" for i in range(adata.n_vars)]
        adata.obsm["spatial"] = random.uniform(0, 100, size=(adata.n_obs, 2))
        adata.uns["spateo_io"] = {"technology": "atera"}

        result = preprocess_spatial(
            adata,
            inplace=False,
            n_top_genes=8,
            n_pca_components=5,
            build_expression_graph=True,
        )

        self.assertIsNotNone(result)
        self.assertEqual([], list(adata.layers.keys()))
        self.assertEqual((30, 12), adata.shape)
        self.assertEqual("standard", result.uns["pp"]["spatial_preprocess"]["recipe"])
        self.assertEqual("atera", result.uns["pp"]["spatial_preprocess"]["technology"])
        self.assertEqual({"counts", "norm", "log1p_norm"}, set(result.layers.keys()))
        np.testing.assert_array_equal(result.layers["counts"].toarray(), adata.X.toarray())
        np.testing.assert_allclose(np.asarray(result.layers["norm"].sum(axis=1)).ravel(), 1e4)
        self.assertEqual((result.n_obs, 5), result.obsm["X_pca"].shape)
        self.assertGreater(result.obsp["spatial_connectivities"].nnz, 0)
        self.assertGreater(result.obsp["connectivities"].nnz, 0)

    def test_count_contract_can_be_strict(self):
        adata = AnnData(np.array([[0.1, 1.0], [2.0, 3.0]], dtype=float))
        adata.obsm["spatial"] = np.array([[0, 0], [1, 1]], dtype=float)
        with self.assertRaisesRegex(ValueError, "integer-like"):
            preprocess_spatial(
                adata,
                recipe="standard",
                validate_counts="error",
                build_spatial_graph=False,
                run_pca=False,
            )

    def test_median_size_factor_normalization_can_replace_x(self):
        counts = sparse.csr_matrix([[1, 1], [2, 4], [0, 0]], dtype=int)
        adata = AnnData(counts.copy())

        normalize_total(adata, layer="X", out_layer="X", target_sum=None)

        np.testing.assert_allclose(np.asarray(adata.X.sum(axis=1)).ravel(), [4.0, 4.0, 0.0])
        np.testing.assert_allclose(adata.obs["size_factor"].to_numpy(), [0.5, 1.5, 1.0])
        self.assertEqual(4.0, adata.uns["pp"]["normalize_total"]["resolved_target_sum"]["all"])

    def test_library_aware_median_targets(self):
        adata = AnnData(sparse.csr_matrix([[1, 1], [3, 3], [5, 5], [10, 10]], dtype=int))
        adata.obs["library"] = ["a", "a", "b", "b"]

        result = normalize_total(
            adata,
            layer="X",
            out_layer="norm",
            target_sum=None,
            library_key="library",
            inplace=False,
        )

        np.testing.assert_allclose(np.asarray(result.layers["norm"].sum(axis=1)).ravel(), [4, 4, 15, 15])
        np.testing.assert_array_equal(adata.X.toarray(), [[1, 1], [3, 3], [5, 5], [10, 10]])

    def test_raw_recipe_only_runs_qc_and_spatial_graph(self):
        adata = AnnData(sparse.csr_matrix([[1, 0], [0, 2], [1, 1]], dtype=int))
        adata.obsm["spatial"] = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)

        result = preprocess_spatial(adata, recipe="raw", min_cells=1, inplace=False)

        self.assertEqual({"counts"}, set(result.layers))
        self.assertNotIn("X_pca", result.obsm)
        self.assertEqual("raw", result.uns["pp"]["spatial_preprocess"]["recipe"])

    def test_sctransform_recipe_is_not_silently_approximated(self):
        adata = AnnData(sparse.csr_matrix([[1, 0], [0, 1]], dtype=int))
        adata.obsm["spatial"] = np.array([[0, 0], [1, 1]], dtype=float)
        with self.assertRaisesRegex(ValueError, "no longer a production preprocessing recipe"):
            preprocess_spatial(adata, recipe="sctransform", inplace=False)
