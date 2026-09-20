#!/usr/bin/env python3
"""Verify the public mouse-brain Visium bundle without modifying input files.

Run from the repository checkout:
  python scripts/verify_automatic_spatial_reading.py DATASET --output REPORT_DIR
"""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata import read_h5ad
from scipy import sparse

# Prefer the checkout containing this script to a different installed checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spateo.io import read_spatial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.dataset.resolve(), args.output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Choose an output directory outside the input dataset")
    result = read_spatial(source)
    if result.status != "ok":
        raise RuntimeError(json.dumps(result.report, ensure_ascii=False, indent=2))
    a = result.adata
    with h5py.File(source / "filtered_feature_bc_matrix.h5") as f:
        g = f["matrix"]
        expected = sparse.csc_matrix(
            (g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g["shape"][:])
        ).T.tocsr()
        assert a.shape == expected.shape and (a.X != expected).nnz == 0
        assert a.obs_names.tolist() == g["barcodes"].asstr()[:].tolist()
        assert a.var["gene_ids"].tolist() == g["features/id"].asstr()[:].tolist()
        assert a.var_names.tolist() == g["features/name"].asstr()[:].tolist()
    pos = pd.read_csv(
        source / "spatial/tissue_positions_list.csv",
        header=None,
        names=["barcode", "in_tissue", "array_row", "array_col", "y", "x"],
    ).set_index("barcode")
    np.testing.assert_array_equal(a.obsm["spatial"], pos.loc[a.obs_names, ["x", "y"]].to_numpy())
    output.mkdir(parents=True, exist_ok=True)
    target = output / "V1_Adult_Mouse_Brain_score_free.h5ad"
    a.write_h5ad(target, compression="gzip")
    back = read_h5ad(target)
    assert (back.X != a.X).nnz == 0
    pd.testing.assert_frame_equal(back.obs, a.obs)
    pd.testing.assert_frame_equal(back.var, a.var)
    np.testing.assert_array_equal(back.obsm["spatial"], a.obsm["spatial"])
    for library, slot in a.uns["spatial"].items():
        for name, image in slot["images"].items():
            np.testing.assert_array_equal(back.uns["spatial"][library]["images"][name], image)
        assert back.uns["spatial"][library]["scalefactors"] == slot["scalefactors"]
    assert "confidence" not in back.uns["spateo_io"]
    report = {
        "dataset": "V1_Adult_Mouse_Brain",
        "source_url": "https://www.10xgenomics.com/datasets/mouse-brain-section-coronal-1-standard-1-1-0",
        "shape": list(a.shape),
        "nnz": int(a.X.nnz),
        "total_umi": int(a.X.sum()),
        "dtype": str(a.X.dtype),
        "status": result.status,
        "policy_version": a.uns["spateo_io"]["policy_version"],
        "checks": {
            "all_matrix_values": "passed",
            "all_barcodes_and_order": "passed",
            "all_gene_ids_and_symbols": "passed",
            "all_xy_by_barcode": "passed",
            "h5ad_matrix_obs_var_xy_images_scales_roundtrip": "passed",
        },
        "images": {
            lib: {key: list(img.shape) for key, img in slot["images"].items()} for lib, slot in a.uns["spatial"].items()
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "anndata", "scipy", "h5py", "pyarrow"]
        },
        "python": sys.version.split()[0],
        "core_sha256": {},
    }
    for name in ["filtered_feature_bc_matrix.h5", "spatial/tissue_positions_list.csv"]:
        with (source / name).open("rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024**2), b""):
                digest.update(chunk)
        report["core_sha256"][name] = digest.hexdigest()
    (output / "real_visium_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    result.write_report(output / "spatial_read_report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
