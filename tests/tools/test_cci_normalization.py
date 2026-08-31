"""Tests for CCI-only count-model normalization factors."""

import numpy as np
from scipy import sparse

from spateo.tools.CCI_effects_modeling._normalization import calc_tmmwsp_factors


def test_tmmwsp_factors_are_finite_and_geometrically_centered():
    counts = sparse.csr_matrix(
        [
            [10, 0, 4, 0, 1],
            [20, 1, 8, 0, 2],
            [3, 6, 0, 2, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=float,
    )

    factors = calc_tmmwsp_factors(counts)

    assert factors.shape == (4,)
    assert np.isfinite(factors).all()
    assert (factors > 0).all()
    np.testing.assert_allclose(np.exp(np.mean(np.log(factors))), 1.0)


def test_identical_profiles_have_unit_tmmwsp_factors():
    counts = np.tile(np.array([0, 2, 5, 0, 1]), (3, 1))
    np.testing.assert_allclose(calc_tmmwsp_factors(counts), np.ones(3))
