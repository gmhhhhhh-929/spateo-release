"""Public API migration and score-free discovery regressions."""

import importlib

import numpy as np
import pytest
from anndata import AnnData

from spateo.io import read_spatial, SpatialReadResult
from spateo.io.spatial._provenance import record_spatial_io


@pytest.fixture
def gem(tmp_path):
    path = tmp_path / "reads.gem"
    path.write_text("geneID\tx\ty\tMIDCounts\nGeneA\t1\t2\t3\nGeneB\t4\t5\t7\n")
    return path


@pytest.mark.parametrize("namespace", ["spateo.io", "spateo.io.spatial", "spateo.io.spatial.auto"])
def test_all_public_names_use_one_reader(namespace, gem):
    module = importlib.import_module(namespace)
    for name in ("read_spatial", "read_auto_spatial", "read_spatial_auto"):
        reader = getattr(module, name)
        assert reader is read_spatial
        result = reader(gem)
        assert isinstance(result, SpatialReadResult)
        assert result.status == "ok", result.report
        np.testing.assert_array_equal(result.adata.X.toarray(), [[3, 0], [0, 7]])
        assert "confidence" not in result.adata.uns["spateo_io"]
    for name in ("SpatialReadMatch", "detect_spatial_technology", "detect_spatial_technologies"):
        assert not hasattr(module, name)
        assert name not in module.__all__


@pytest.mark.parametrize("alias", ["read_spatial", "read_auto_spatial", "read_spatial_auto"])
@pytest.mark.parametrize("option", ["min_confidence", "strict", "return_match"])
def test_retired_options_are_rejected(alias, option, gem):
    import spateo.io as io

    with pytest.raises(TypeError, match=option):
        getattr(io, alias)(gem, **{option: True})


def test_discovery_then_load_preserves_identity_and_values(gem):
    result = read_spatial(gem, load=False)
    assert len(result.datasets) == 1
    entry = next(iter(result.datasets.values()))
    assert entry.status == "deferred" and entry.adata is None
    assert entry.technology == "bgi"
    entry.load()
    assert entry.status == "ready", result.report
    np.testing.assert_array_equal(entry.adata.X.toarray(), [[3, 0], [0, 7]])
    assert "confidence" not in entry.adata.uns["spateo_io"]


def test_new_provenance_removes_stale_score(gem):
    adata = AnnData(np.ones((1, 1)))
    adata.uns["spateo_io"] = {"confidence": 0.92, "custom_note": "preserve"}
    record_spatial_io(adata, technology="bgi", source=gem, reader="test")
    assert "confidence" not in adata.uns["spateo_io"]
    assert adata.uns["spateo_io"]["custom_note"] == "preserve"
