"""Independent native imaging fixtures: exact expectations for both public routes."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from spateo.io import (
    read_merfish,
    read_nanostring,
    read_seqfish,
    read_spatial,
    read_starmap_plus,
)

TECHS = ("merfish", "seqfish", "nanostring", "starmap_plus")
COUNTS = np.array([[2**53 + 1, 0, 7], [3, 2**31 + 3, 0], [0, 11, 5]], dtype=np.int64)
XYZ = np.array([[1.25, -2.5, 0.75], [9.5, 4.125, 1.25], [-3.75, 8.25, 2.5]])
GENES = ["Gad1", "Slc17a7", "Mbp"]
IDS = ["001", "010", "100"]


def fixture(root: Path, tech: str, *, processed=False):
    root.mkdir(parents=True)
    ids = IDS if tech != "nanostring" else ["1_1", "1_2", "2_1"]
    values = COUNTS.copy() if not processed else np.array([[0.25, -1.5, 7.125], [3.25, 2.5, 0], [0, -11.75, 5.5]])
    counts = pd.DataFrame(values, columns=GENES)
    counts.insert(0, "cell_id", IDS)
    meta = pd.DataFrame({"cell_id": IDS, "center_x": XYZ[:, 0], "center_y": XYZ[:, 1], "center_z": XYZ[:, 2]})
    if tech == "merfish":
        cf, mf = "cell_by_gene_S1.csv", "cell_metadata_S1.csv"
    elif tech == "seqfish":
        cf, mf = "SG_Counts_S1.csv", "SG_CellCoordinates_S1.csv"
    elif tech == "nanostring":
        cf, mf = "sample_exprMat_file.csv", "sample_metadata_file.csv"
        counts["cell_id"], meta["cell_id"] = ["1", "1", "2"], ["1", "1", "2"]
        counts["fov"], meta["fov"] = ["1", "2", "1"], ["1", "2", "1"]
        meta = meta.rename(columns={"center_x": "CenterX_local_px", "center_y": "CenterY_local_px"}).drop(
            columns="center_z"
        )
        meta["CenterX_global_px"], meta["CenterY_global_px"] = XYZ[:, 0] + 100, XYZ[:, 1] + 200
    else:
        cf = "sample_processed_expression_pd.csv" if processed else "sample_raw_expression_pd.csv"
        mf = "sample_spatial.csv"
        counts = counts.rename(columns={"cell_id": "NAME"})
        meta = meta.rename(columns={"cell_id": "NAME", "center_x": "X", "center_y": "Y", "center_z": "Z"})
    counts.to_csv(root / cf, index=False)
    meta.iloc[[2, 0, 1]].to_csv(root / mf, index=False)
    return root, cf, mf, ids, values


def direct(bundle, tech):
    root, cf, mf, _, _ = bundle
    if tech == "merfish":
        return read_merfish(root, counts_file=cf, meta_file=mf, load_images=False, load_boundaries=False)
    if tech == "seqfish":
        return read_seqfish(root, counts_file=cf, meta_file=mf, load_images=False, load_labels=False)
    if tech == "nanostring":
        return read_nanostring(root, counts_file=cf, meta_file=mf)
    return read_starmap_plus(root, counts_file=cf, meta_file="missing_optional_meta.csv", spatial_file=mf)


def check_core(obj, bundle, tech):
    _, _, _, ids, expected = bundle
    assert set(obj.obs_names) == set(ids)
    assert obj.n_obs == 3 and obj.n_vars == 3
    assert list(obj.var_names) == GENES
    aligned = obj[ids]
    np.testing.assert_array_equal(aligned.X.toarray(), expected)
    assert expected.dtype.kind != "i" or aligned.X.dtype.kind in "iu"
    np.testing.assert_array_equal(aligned.obsm["spatial"], XYZ[:, :2] if tech == "nanostring" else XYZ)


@pytest.mark.parametrize("tech", TECHS)
def test_exact_integer_counts_shuffled_coordinates_and_roundtrip(tmp_path, tech):
    bundle = fixture(tmp_path / tech, tech)
    automatic = read_spatial(bundle[0], load_images=False)
    assert automatic.status == "ok", automatic.report
    explicit = direct(bundle, tech)
    for name, obj in (("automatic", automatic.adata), ("direct", explicit)):
        check_core(obj, bundle, tech)
        output = tmp_path / f"{tech}_{name}.h5ad"
        obj.write_h5ad(output)
        check_core(ad.read_h5ad(output), bundle, tech)
    if tech == "nanostring":
        np.testing.assert_array_equal(explicit[bundle[3]].obsm["spatial_fov"], XYZ[:, :2] + [100, 200])


@pytest.mark.parametrize("tech", TECHS)
@pytest.mark.parametrize(
    "damage",
    ("missing_id", "duplicate_id", "nonfinite_coordinate", "nonnumeric_count", "negative_count", "fractional_count"),
)
def test_invalid_core_is_rejected_by_both_routes(tmp_path, tech, damage):
    bundle = fixture(tmp_path / tech, tech)
    root, cf, mf, _, _ = bundle
    if damage in ("nonnumeric_count", "negative_count", "fractional_count"):
        table = pd.read_csv(root / cf, dtype=str, keep_default_na=False)
        table.loc[0, "Gad1"] = {"nonnumeric_count": "broken", "negative_count": "-1", "fractional_count": "0.5"}[damage]
        table.to_csv(root / cf, index=False)
    else:
        table = pd.read_csv(root / mf, dtype=str, keep_default_na=False)
        if damage == "missing_id":
            table = table.iloc[:-1]
        elif damage == "duplicate_id":
            table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
        else:
            column = "CenterX_local_px" if tech == "nanostring" else "X" if tech == "starmap_plus" else "center_x"
            table.loc[0, column] = "inf"
        table.to_csv(root / mf, index=False)
    automatic = read_spatial(root, load_images=False)
    assert automatic.status == "failed", automatic.report
    with pytest.raises((ValueError, KeyError)):
        direct(bundle, tech)


def test_starmap_processed_real_values_and_type_row(tmp_path):
    bundle = fixture(tmp_path / "star", "starmap_plus", processed=True)
    root, _, mf, _, _ = bundle
    spatial = pd.read_csv(root / mf, dtype=str)
    schema = pd.DataFrame([{c: "TYPE" if c == "NAME" else "numeric" for c in spatial}])
    pd.concat([schema, spatial], ignore_index=True).to_csv(root / mf, index=False)
    result = read_spatial(root, load_images=False)
    assert result.status == "ok", result.report
    check_core(result.adata, bundle, "starmap_plus")
    check_core(direct(bundle, "starmap_plus"), bundle, "starmap_plus")


@pytest.mark.parametrize("tech", TECHS)
def test_direct_does_not_call_automatic_entrypoint(tmp_path, monkeypatch, tech):
    import spateo.io.spatial.auto._automatic as automatic_module

    bundle = fixture(tmp_path / tech, tech)

    def forbidden(*args, **kwargs):
        raise AssertionError("Direct reader called automatic reader")

    monkeypatch.setattr(automatic_module, "read_spatial", forbidden)
    check_core(direct(bundle, tech), bundle, tech)


@pytest.mark.parametrize("tech", TECHS)
@pytest.mark.parametrize("damage", ("missing_file", "duplicate_expression_id", "wrong_ids_same_length"))
def test_missing_files_and_identifier_integrity(tmp_path, tech, damage):
    bundle = fixture(tmp_path / tech, tech)
    root, cf, mf, _, _ = bundle
    if damage == "missing_file":
        (root / mf).unlink()
    elif damage == "duplicate_expression_id":
        frame = pd.read_csv(root / cf, dtype=str, keep_default_na=False)
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        frame.to_csv(root / cf, index=False)
    else:
        frame = pd.read_csv(root / mf, dtype=str, keep_default_na=False)
        frame.iloc[:, 0] = ["wrong_1", "wrong_2", "wrong_3"]
        frame.to_csv(root / mf, index=False)
    result = read_spatial(root, load_images=False)
    assert result.status == "failed", result.report
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        direct(bundle, tech)


@pytest.mark.parametrize("tech", ("seqfish", "starmap_plus"))
def test_gene_by_cell_orientation_uses_identifier_evidence(tmp_path, tech):
    bundle = fixture(tmp_path / tech, tech)
    root, cf, _, ids, values = bundle
    counts = pd.DataFrame(values.T, columns=ids)
    counts.insert(0, "gene", GENES)
    counts.to_csv(root / cf, index=False)
    result = read_spatial(root, load_images=False)
    assert result.status == "ok", result.report
    check_core(result.adata, bundle, tech)
    check_core(direct(bundle, tech), bundle, tech)


def image_arrays(obj):
    def visit(value):
        if isinstance(value, np.ndarray):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from visit(child)

    for slot in obj.uns["spatial"].values():
        yield from visit(slot.get("images", {}))


@pytest.mark.parametrize("tech", ("merfish", "seqfish", "nanostring"))
def test_optional_raster_pixels_are_preserved(tmp_path, tech):
    bundle = fixture(tmp_path / tech, tech)
    root, cf, mf, _, _ = bundle
    pixels = np.arange(30, dtype=np.uint8).reshape(5, 6)
    if tech == "merfish":
        path = root / "images/mosaic_DAPI_z3.tif"
    elif tech == "seqfish":
        path = root / "images/SG_DAPI_S1.tiff"
    else:
        path = root / "CellComposite/sample_F001.png"
    path.parent.mkdir()
    Image.fromarray(pixels).save(path)
    result = read_spatial(root, load_images=True)
    assert result.status == "ok", result.report
    if tech == "merfish":
        explicit = read_merfish(root, counts_file=cf, meta_file=mf, load_images=True, load_boundaries=False)
    elif tech == "seqfish":
        explicit = read_seqfish(root, counts_file=cf, meta_file=mf, load_images=True, load_labels=False)
    else:
        explicit = direct(bundle, tech)
    for obj in (result.adata, explicit):
        check_core(obj, bundle, tech)
        assert any(np.array_equal(array, pixels) for array in image_arrays(obj))


def test_starmap_explicit_lossy_dtype_is_rejected(tmp_path):
    root, cf, mf, _, _ = fixture(tmp_path / "star", "starmap_plus")
    with pytest.raises(ValueError, match="precision"):
        read_starmap_plus(root, counts_file=cf, meta_file="optional.csv", spatial_file=mf, dtype="float32")


@pytest.mark.parametrize("tech", ("merfish", "seqfish", "nanostring"))
def test_corrupt_optional_image_keeps_exact_core(tmp_path, tech):
    bundle = fixture(tmp_path / tech, tech)
    root, cf, mf, _, _ = bundle
    path = (
        root
        / {
            "merfish": "images/mosaic_DAPI_z3.tif",
            "seqfish": "images/SG_DAPI_S1.tiff",
            "nanostring": "CellComposite/sample_F001.png",
        }[tech]
    )
    path.parent.mkdir()
    path.write_bytes(b"not an image")
    result = read_spatial(root, load_images=True)
    assert result.status == "ok", result.report
    assert any(d["code"] == "optional_image_error" for d in next(iter(result.datasets.values())).diagnostics)
    if tech == "merfish":
        with pytest.warns(UserWarning, match="Failed to load image"):
            explicit = read_merfish(root, counts_file=cf, meta_file=mf, load_images=True, load_boundaries=False)
    elif tech == "seqfish":
        explicit = read_seqfish(root, counts_file=cf, meta_file=mf, load_images=True, load_labels=False)
        assert any(slot["metadata"].get("failed_image_files") for slot in explicit.uns["spatial"].values())
    else:
        with pytest.warns(UserWarning, match="Failed to load optional image"):
            explicit = direct(bundle, tech)
        assert any(slot.get("failed_image_files") for slot in explicit.uns["spatial"].values())
    check_core(result.adata, bundle, tech)
    check_core(explicit, bundle, tech)


@pytest.mark.parametrize("tech", TECHS)
def test_missing_xy_field_is_rejected(tmp_path, tech):
    bundle = fixture(tmp_path / tech, tech)
    root, _, mf, _, _ = bundle
    metadata = pd.read_csv(root / mf, dtype=str, keep_default_na=False)
    column = "CenterX_local_px" if tech == "nanostring" else "X" if tech == "starmap_plus" else "center_x"
    metadata.drop(columns=column).to_csv(root / mf, index=False)
    result = read_spatial(root, load_images=False)
    assert result.status == "failed", result.report
    with pytest.raises((ValueError, KeyError)):
        direct(bundle, tech)


def test_cosmx_fov_metadata_does_not_translate_local_coordinates(tmp_path):
    bundle = fixture(tmp_path / "cosmx", "nanostring")
    root, cf, mf, ids, _ = bundle
    pd.DataFrame({"fov": [1, 2], "x_global_px": [1000, 2000], "y_global_px": [3000, 4000]}).to_csv(
        root / "sample_fov_positions_file.csv", index=False
    )
    explicit = read_nanostring(root, counts_file=cf, meta_file=mf, fov_file="sample_fov_positions_file.csv")
    automatic = read_spatial(root, load_images=False)
    assert automatic.status == "ok", automatic.report
    for obj in (explicit, automatic.adata):
        check_core(obj, bundle, "nanostring")
        np.testing.assert_array_equal(obj[ids].obsm["spatial_fov"], XYZ[:, :2] + [100, 200])
    assert explicit.uns["spatial"]["1"]["metadata"]["x_global_px"] == 1000
    assert explicit.uns["spatial"]["2"]["metadata"]["y_global_px"] == 4000


def test_starmap_reorientation_is_explicit_and_z_is_unchanged(tmp_path):
    bundle = fixture(tmp_path / "star", "starmap_plus")
    root, cf, mf, ids, _ = bundle
    explicit = read_starmap_plus(root, counts_file=cf, meta_file="optional.csv", spatial_file=mf, reorient_xy=True)
    expected = np.column_stack([XYZ[:, 1].max() - XYZ[:, 1], XYZ[:, 0].max() - XYZ[:, 0], XYZ[:, 2]])
    np.testing.assert_array_equal(explicit[ids].obsm["spatial"], expected)
    np.testing.assert_array_equal(explicit[ids].X.toarray(), COUNTS)
    default_auto = read_spatial(root, load_images=False)
    assert default_auto.status == "ok", default_auto.report
    check_core(default_auto.adata, bundle, "starmap_plus")


def test_invalid_optional_cosmx_global_coordinates_warn_and_keep_local_core(tmp_path):
    bundle = fixture(tmp_path / "cosmx", "nanostring")
    root, _, mf, _, _ = bundle
    metadata = pd.read_csv(root / mf, dtype=str, keep_default_na=False)
    metadata.loc[0, "CenterX_global_px"] = "inf"
    metadata.to_csv(root / mf, index=False)
    automatic = read_spatial(root, load_images=False)
    assert automatic.status == "ok", automatic.report
    assert any(d["code"] == "reader_warning" for d in next(iter(automatic.datasets.values())).diagnostics)
    with pytest.warns(UserWarning, match="Optional global FOV coordinates are invalid"):
        explicit = direct(bundle, "nanostring")
    for obj in (automatic.adata, explicit):
        check_core(obj, bundle, "nanostring")
        assert "spatial_fov" not in obj.obsm
        assert obj.uns["spateo_io"]["optional_global_coordinates"] == "invalid"
