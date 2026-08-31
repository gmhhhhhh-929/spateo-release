"""Tests for the promoted spatial preprocessing pipeline."""

from unittest import TestCase

import numpy as np
from anndata import AnnData
from scipy import sparse

from spateo.preprocessing import preprocess_spatial


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
        self.assertEqual("atera", result.uns["pp"]["spatial_preprocess"]["recipe"])
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
                recipe="generic",
                validate_counts="error",
                build_spatial_graph=False,
                run_pca=False,
            )
