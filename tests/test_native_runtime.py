import ast
from pathlib import Path

import numpy as np
from anndata import AnnData
from scipy import sparse

from spateo._native import (
    SparseVectorField,
    fetch_X_data,
    neighbors,
    predict_fate,
    sample,
    sparse_vector_field,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_source_and_requirements_do_not_import_dynamo():
    imported_from = []
    for source_path in (REPOSITORY_ROOT / "spateo").rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_from.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_from.append(node.module)
    assert not [name for name in imported_from if name.split(".")[0] == "dynamo"]

    for requirements_name in ("requirements.txt", "win-requirements.txt"):
        requirements = (REPOSITORY_ROOT / requirements_name).read_text(encoding="utf-8").splitlines()
        declared = [line.strip().lower() for line in requirements if line.strip() and not line.startswith("#")]
        assert not any(line.startswith(("dynamo", "dynamo-release")) for line in declared)


def test_fetch_x_data_preserves_requested_gene_order_and_sparse_matrix():
    adata = AnnData(
        sparse.csr_matrix([[1, 2, 3], [4, 5, 6]]),
        var={"gene": ["a", "b", "c"]},
    )
    adata.var_names = ["a", "b", "c"]
    genes, matrix = fetch_X_data(adata, genes=["c", "missing", "a"])
    assert genes == ["c", "a"]
    assert sparse.issparse(matrix)
    np.testing.assert_array_equal(matrix.toarray(), [[3, 1], [6, 4]])


def test_native_neighbor_graph_uses_expected_anndata_keys():
    coordinates = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1], [5, 5]], dtype=float)
    adata = AnnData(np.ones((coordinates.shape[0], 2)))
    neighbors(adata, X_data=coordinates, n_neighbors=2, result_prefix="spatial")

    distances = adata.obsp["spatial_distances"]
    connectivities = adata.obsp["spatial_connectivities"]
    assert distances.shape == (5, 5)
    assert connectivities.shape == (5, 5)
    assert (distances != distances.T).nnz == 0
    assert (connectivities != connectivities.T).nnz == 0
    assert adata.uns["spatial_neighbors"]["indices"].shape == (5, 2)
    assert adata.uns["spatial_neighbors"]["params"]["method"] == "spateo"


def test_spatial_sampling_is_deterministic_unique_and_balanced():
    coordinates = np.column_stack((np.linspace(0, 100, 101), np.zeros(101)))
    values = np.arange(coordinates.shape[0])
    selected_a = sample(values, n=5, method="trn", X=coordinates, seed=7)
    selected_b = sample(values, n=5, method="trn", X=coordinates, seed=7)

    np.testing.assert_array_equal(selected_a, selected_b)
    assert len(np.unique(selected_a)) == 5
    assert np.ptp(coordinates[selected_a, 0]) >= 90


def test_native_sparse_vector_field_and_fate_contract():
    rng = np.random.default_rng(4)
    coordinates = rng.normal(size=(50, 2))
    velocity = np.tile(np.asarray([0.8, -0.25]), (coordinates.shape[0], 1))
    grid = coordinates[:6]

    result = sparse_vector_field(coordinates, velocity, Grid=grid, M=15, lambda_=1e-4, seed=3)
    expected_keys = {
        "X",
        "Y",
        "valid_ind",
        "X_ctrl",
        "ctrl_idx",
        "beta",
        "C",
        "V",
        "grid",
        "grid_V",
        "method",
    }
    assert expected_keys.issubset(result)
    assert result["method"] == "sparsevfc"
    np.testing.assert_allclose(result["grid_V"], velocity[:6], atol=0.06)

    vector_field = SparseVectorField(result)
    jacobian = vector_field.get_Jacobian()(grid)
    assert jacobian.shape == (2, 2, 6)
    assert np.isfinite(jacobian).all()

    fate = predict_fate(result, grid[:2], direction="forward", interpolation_num=8, t_end=1.0)
    assert len(fate["prediction"]) == 2
    assert fate["prediction"][0].shape == (2, 8)
    displacement = fate["prediction"][0][:, -1] - fate["prediction"][0][:, 0]
    np.testing.assert_allclose(displacement, [0.8, -0.25], atol=0.08)

    backward = predict_fate(result, grid[:1], direction="backward", interpolation_num=8, t_end=1.0)
    assert np.all(np.diff(backward["t"][0]) > 0)
    backward_displacement = backward["prediction"][0][:, -1] - backward["prediction"][0][:, 0]
    np.testing.assert_allclose(backward_displacement, [0.8, -0.25], atol=0.08)


def test_public_morphofield_and_interpolation_paths_are_native():
    from spateo.tdr.interpolations.interpolation_sparseVFC import kernel_interpolation
    from spateo.tdr.morphometrics.morphofield.sparsevfc import _morphofield_sparsevfc
    from spateo.tdr.morphometrics.morphofield.trajectory import morphopath
    from spateo.tdr.morphometrics.morphofield_dg.differential_geometry import (
        morphofield_divergence,
        morphofield_jacobian,
        morphofield_velocity,
    )

    rng = np.random.default_rng(12)
    coordinates = rng.normal(size=(20, 3))
    velocity = np.tile(np.asarray([0.2, -0.1, 0.05]), (20, 1))
    field = _morphofield_sparsevfc(
        coordinates,
        velocity,
        NX=coordinates[:5],
        M=8,
        lambda_=1e-4,
        restart_num=0,
    )
    adata = AnnData(np.ones((20, 2)), obsm={"spatial": coordinates})
    adata.uns["VecFld_morpho"] = field

    morphofield_velocity(adata)
    morphofield_divergence(adata)
    morphofield_jacobian(adata)
    morphopath(adata, direction="forward", interpolation_num=5, t_end=1.0)
    assert adata.obsm["velocity"].shape == (20, 3)
    assert adata.uns["jacobian"].shape == (3, 3, 20)
    assert len(adata.uns["fate_morpho"]["prediction"]) == 20

    source = AnnData(
        rng.uniform(0, 5, size=(20, 2)),
        obsm={"spatial": coordinates},
    )
    source.var_names = ["gene_a", "gene_b"]
    interpolated = kernel_interpolation(
        source,
        target_points=coordinates[:4],
        keys=["gene_a"],
        M=8,
        lambda_=1e-4,
    )
    assert interpolated.shape == (4, 1)
    assert interpolated.obsm["spatial"].shape == (4, 3)
