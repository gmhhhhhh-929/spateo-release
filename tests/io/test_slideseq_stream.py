"""Strict streaming regressions for native Slide-seq matrices."""

import gzip

import numpy as np
import pytest

from spateo.io.spatial.auto._contracts import ContractError, ResourceDeferred, _slideseq_counts


@pytest.mark.parametrize("compressed", [False, True])
def test_stream_values_zero_genes_and_order(tmp_path, compressed):
    p = tmp_path / ("counts.csv.gz" if compressed else "counts.csv")
    text = "Row,b2,b1\nG2,0,5\nG0,0,0\nG1,3,2\n"
    if compressed:
        with gzip.open(p, "wt") as f:
            f.write(text)
    else:
        p.write_text(text)
    a = _slideseq_counts(p, full=True)
    assert list(a.obs_names) == ["b2", "b1"]
    assert list(a.var_names) == ["G2", "G0", "G1"]
    np.testing.assert_array_equal(a.X.toarray(), [[0, 0, 3], [5, 0, 2]])


@pytest.mark.parametrize("bad", ["bad,1", "NaN,2", "-1,2", "1", "1,2,3"])
def test_late_invalid_row_is_not_hidden_by_probe(tmp_path, bad):
    p = tmp_path / "counts.csv"
    p.write_text("Row,b1,b2\nG1,1,2\nG2," + bad + "\n")
    assert _slideseq_counts(p)["n_obs"] == 2
    with pytest.raises(ContractError):
        _slideseq_counts(p, full=True)


@pytest.mark.parametrize("text", ["Row,b1,b1\nG1,1,2\n", "Row,b1\nG1,1\nG1,2\n", "Row,b1\n", "Row,b1\n,1\n"])
def test_invalid_identifiers_and_empty_matrix(tmp_path, text):
    p = tmp_path / "counts.csv"
    p.write_text(text)
    with pytest.raises(ContractError):
        _slideseq_counts(p, full=True)


def test_growing_storage_is_bounded(tmp_path):
    p = tmp_path / "counts.csv"
    p.write_text("Row,b1,b2\nG1,1,2\nG2,3,4\n")
    assert _slideseq_counts(p, budget=4500)["n_obs"] == 2
    with pytest.raises(ResourceDeferred):
        _slideseq_counts(p, full=True, budget=4500)
