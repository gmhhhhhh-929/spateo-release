import numpy as np
from anndata import AnnData

import spateo.alignment.methods.paste as paste_module
from spateo.tdr.morphometrics.morphofield.sparsevfc import cell_directions


def _sample(expression: np.ndarray, coordinates: np.ndarray) -> AnnData:
    adata = AnnData(X=expression.astype(np.float64))
    adata.var_names = [f"gene_{i}" for i in range(expression.shape[1])]
    adata.layers["normalized"] = adata.X.copy()
    adata.obsm["align_spatial"] = coordinates.astype(np.float64)
    return adata


def test_paste_pairwise_align_uses_current_pot_cg(monkeypatch):
    """POT >=0.9.6 must work without creating ``ot.gromov.cg``."""
    sample_a = _sample(
        np.array([[1, 3], [2, 2], [4, 1]]),
        np.array([[0, 0], [1, 0], [0, 1]]),
    )
    sample_b = _sample(
        np.array([[1, 4], [3, 2], [5, 1]]),
        np.array([[0.1, 0], [1.1, 0], [0.1, 1]]),
    )
    monkeypatch.delattr(paste_module.ot.gromov, "cg", raising=False)

    pi, objective = paste_module.paste_pairwise_align(
        sampleA=sample_a,
        sampleB=sample_b,
        layer="normalized",
        spatial_key="align_spatial",
        alpha=0.1,
        numItermax=5,
        numItermaxEmd=1000,
        dtype="float64",
        device="cpu",
        verbose=False,
    )

    assert pi.shape == (3, 3)
    np.testing.assert_allclose(pi.sum(), 1.0)
    assert np.isfinite(objective)
    assert not hasattr(paste_module.ot.gromov, "cg")


def test_cell_directions_forwards_layer_as_rep_layer(monkeypatch):
    """Regression test for ``align_preprocess(layer=...)`` TypeError."""
    sample_a = _sample(
        np.array([[1, 3], [2, 2], [4, 1]]),
        np.array([[0, 0], [1, 0], [0, 1]]),
    )
    sample_b = _sample(
        np.array([[1, 4], [3, 2], [5, 1]]),
        np.array([[0.1, 0], [1.1, 0], [0.1, 1]]),
    )
    monkeypatch.delattr(paste_module.ot.gromov, "cg", raising=False)

    _, pi = cell_directions(
        adataA=sample_a,
        adataB=sample_b,
        layer="normalized",
        spatial_key="align_spatial",
        key_added="cells_mapping",
        alpha=0.1,
        numItermax=5,
        numItermaxEmd=1000,
        dtype="float64",
        device="cpu",
        inplace=True,
    )

    assert pi.shape == (3, 3)
    assert sample_a.obsm["X_cells_mapping"].shape == (3, 2)
    assert sample_a.obsm["V_cells_mapping"].shape == (3, 2)
